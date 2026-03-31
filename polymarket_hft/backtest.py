"""
Event-Driven Backtester — Phase 2.5 (Tick-Level + Parameter Optimization).

Replays raw historical tick data (with real aggressor side) through
the Phase 1 signal logic, reconstructing CVD and Z-Score exactly as
the live UnifiedOracle does.

Features:
    - Dynamic Volatility Quoting: spread adapts to Z-Score regime.
    - Configurable directional thresholds and stop-loss.
    - Parameter sweep mode: runs 3 configs and outputs comparative tear sheet.

Usage:
    python -m scripts.fetch_historical       # Step 1: download tick data
    python backtest.py                       # Step 2: single run (baseline)
    python backtest.py --sweep               # Step 3: parameter sweep (3 configs)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from simulator.clob_env import CLOBSimulator

logger = logging.getLogger("backtest")

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Tick loader
# ---------------------------------------------------------------------------

@dataclass
class Tick:
    """Single raw trade with aggressor side."""
    timestamp_ms: int
    price: float
    quantity: float
    side: str       # "B" = buyer aggressor (hit ask), "A" = seller aggressor
    trade_id: int


def load_ticks(filepath: Path) -> list[Tick]:
    """Load tick data from compressed CSV."""
    ticks: list[Tick] = []

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticks.append(Tick(
                timestamp_ms=int(row["timestamp_ms"]),
                price=float(row["price"]),
                quantity=float(row["quantity"]),
                side=row["side"],
                trade_id=int(row["trade_id"]),
            ))

    ticks.sort(key=lambda t: t.timestamp_ms)
    logger.info("Loaded %d ticks from %s.", len(ticks), filepath)
    return ticks


# ---------------------------------------------------------------------------
# Offline Oracle — tick-level replay matching live UnifiedOracle logic
# ---------------------------------------------------------------------------

class OfflineOracle:
    """
    Replays raw tick data through the exact same signal math as the live
    UnifiedOracle:

    CVD: Real aggressor side from tick data.
        side=B -> signed_volume = +notional (buyer hit ask)
        side=A -> signed_volume = -notional (seller hit bid)
        Rolling 60-second window, pruned by timestamp.

    Z-Score: 100ms fixed-rate sampling of fair price into deque(maxlen=600).
        Exactly 60 seconds of normalized data. Welford recurrence.
        Sampling uses simulated time from tick timestamps.

    Fair Price: In production, fair = (HL_mid * 0.65) + (BN_mid * 0.35).
        For backtest with single-venue ticks, we use tick price directly
        as fair price (equivalent to both venues quoting the same price).
    """

    CVD_WINDOW_MS = 60_000          # 60 seconds in ms
    ZSCORE_SAMPLE_INTERVAL_MS = 100 # 100ms sampling
    ZSCORE_WINDOW_SIZE = 600        # 600 samples = 60 seconds
    ZSCORE_MIN_SAMPLES = 30         # Minimum before Z valid

    def __init__(self) -> None:
        # CVD: (timestamp_ms, signed_volume)
        self._cvd_window: deque[tuple[int, float]] = deque()
        self.cvd_delta: float = 0.0

        # Z-Score: fixed-rate sampling at 100ms intervals of simulated time.
        self._zscore_window: deque[float] = deque(maxlen=self.ZSCORE_WINDOW_SIZE)
        self._last_sample_ms: int = 0
        self.z_score: float = 0.0

        # Current fair price.
        self.fair_price: float = 0.0

    def process_tick(self, tick: Tick) -> None:
        """
        Process one raw tick: update CVD, sample Z-Score, update fair price.
        """
        self.fair_price = tick.price

        # --- CVD: real aggressor side ---
        notional = tick.price * tick.quantity
        if tick.side == "B":
            signed_vol = notional
        elif tick.side == "A":
            signed_vol = -notional
        else:
            return  # Skip unknown side.

        self._cvd_window.append((tick.timestamp_ms, signed_vol))

        # Prune CVD window to 60 seconds.
        cutoff = tick.timestamp_ms - self.CVD_WINDOW_MS
        while self._cvd_window and self._cvd_window[0][0] < cutoff:
            self._cvd_window.popleft()

        # Recompute CVD sum.
        self.cvd_delta = sum(v for _, v in self._cvd_window)

        # --- Z-Score: sample at 100ms intervals of simulated time ---
        if self._last_sample_ms == 0:
            self._last_sample_ms = tick.timestamp_ms

        while tick.timestamp_ms >= self._last_sample_ms + self.ZSCORE_SAMPLE_INTERVAL_MS:
            self._last_sample_ms += self.ZSCORE_SAMPLE_INTERVAL_MS
            self._zscore_window.append(self.fair_price)

        self._recompute_zscore()

    def _recompute_zscore(self) -> None:
        """Welford recurrence — identical to live oracle."""
        n = len(self._zscore_window)
        if n < self.ZSCORE_MIN_SAMPLES:
            self.z_score = 0.0
            return

        mean = 0.0
        m2 = 0.0
        for i, val in enumerate(self._zscore_window, 1):
            delta = val - mean
            mean += delta / i
            delta2 = val - mean
            m2 += delta * delta2

        variance = m2 / (n - 1) if n > 1 else 0.0
        std = math.sqrt(variance) if variance > 0 else 0.0
        current = self._zscore_window[-1]
        self.z_score = (current - mean) / std if std > 0 else 0.0


# ---------------------------------------------------------------------------
# Probability Mapper
# ---------------------------------------------------------------------------

def price_to_probability(
    price: float,
    price_floor: float,
    price_ceiling: float,
) -> float:
    """
    Map a USD price to a Polymarket probability (0.01 - 0.99).

    price_floor maps to ~0.05, price_ceiling maps to ~0.95.
    """
    if price_ceiling <= price_floor:
        return 0.50
    raw = (price - price_floor) / (price_ceiling - price_floor)
    return max(0.01, min(0.99, 0.05 + raw * 0.90))


# ---------------------------------------------------------------------------
# Dynamic Spread Selection (Task 2.5)
# ---------------------------------------------------------------------------

def compute_dynamic_spread(
    z_score: float,
    has_inventory: bool,
    spread_tiers: tuple[float, float, float] | None = None,
) -> float | None:
    """
    Adapt maker spread to volatility regime based on Z-Score.

    Returns spread in probability units, or None to pause quoting.

    Args:
        spread_tiers: (calm, normal, volatile) spreads in probability units.
            Defaults to (0.005, 0.015, 0.030).

    Regime thresholds (absolute Z-Score):
        |Z| < 1.0  (Quiet):    calm_spread (default 0.5c) — farm rebates
        1.0 <= |Z| <= 2.0:     normal_spread (default 1.5c) — normal
        |Z| > 2.0  (Volatile): volatile_spread (default 3.0c) — wide, or None to pause
    """
    calm, normal, volatile = spread_tiers or (0.005, 0.015, 0.030)
    abs_z = abs(z_score)

    if abs_z < 1.0:
        return calm
    elif abs_z <= 2.0:
        return normal
    else:
        # Volatile: widen spread. Pause entirely if carrying inventory.
        if has_inventory:
            return None  # Pause quoting — avoid adverse selection.
        return volatile


# ---------------------------------------------------------------------------
# Main Backtest Engine
# ---------------------------------------------------------------------------

# We process ticks individually but only re-evaluate strategy logic
# at a configurable interval to avoid placing/cancelling thousands of
# maker orders per second. This models realistic order management cadence.
STRATEGY_EVAL_INTERVAL_MS = 1_000  # Re-evaluate strategy every 1 second.


@dataclass
class BacktestResult:
    """Container for a single backtest run result."""
    sim: CLOBSimulator
    final_fair_prob: float
    num_ticks: int
    span_hours: float
    config_name: str = ""
    daily_metrics: list[dict] = field(default_factory=list)


def run_backtest(
    ticks: list[Tick],
    starting_capital: float = 500.0,
    maker_spread: float = 0.015,
    maker_size_usdc: float = 25.0,
    directional_size_usdc: float = 30.0,
    cvd_threshold: float = 5_000_000.0,
    zscore_threshold: float = 2.5,
    max_exposure: float = 75.0,
    stop_loss_pct: float = 0.02,
    dynamic_spread: bool = False,
    inventory_skew: bool = False,
    inventory_limit: float = 40.0,
    max_skew_cents: float = 0.03,
    global_stop_loss: bool = False,
    global_stop_distance: float = 0.08,
    cooldown_duration_ms: int = 0,
    macro_breaker_enabled: bool = False,
    macro_breaker_limit: float = -25.0,
    daily_macro_reset: bool = False,
    spread_tiers: tuple[float, float, float] | None = None,
    config_name: str = "",
    quiet: bool = False,
) -> BacktestResult:
    """
    Run the full event-driven tick-level backtest.

    Every tick updates CVD and Z-Score. Strategy decisions (place orders,
    enter directional) are evaluated every STRATEGY_EVAL_INTERVAL_MS.
    Fill checks happen on every tick (a resting order fills the instant
    the price crosses it).

    Args:
        dynamic_spread: If True, spread adapts to Z-Score regime (Task 2.5).
        stop_loss_pct: Runner stop-loss distance from entry (default 2%).
        inventory_skew: If True, apply asymmetric quoting (Task 2.8).
        inventory_limit: Max inventory USDC before bid-side pauses.
        max_skew_cents: Maximum skew applied at full inventory.
        global_stop_loss: If True, flatten maker inventory at 8c loss (Task 2.9).
        global_stop_distance: Distance in probability units for global stop.
        cooldown_duration_ms: Post-stop cooldown in ms (0=disabled, 300000=5min) (Task 2.11).
        macro_breaker_enabled: If True, halt trading at realized PnL limit (Task 2.12).
        macro_breaker_limit: Realized PnL threshold for macro breaker (default -$25).
        quiet: Suppress periodic progress logging (for sweep mode).
    """
    sim = CLOBSimulator(
        starting_capital=starting_capital,
        max_exposure_usdc=max_exposure,
        stop_loss_pct=stop_loss_pct,
        inventory_limit=inventory_limit,
        max_skew_cents=max_skew_cents,
        global_stop_distance=global_stop_distance,
        cooldown_duration_ms=cooldown_duration_ms,
        macro_breaker_limit=macro_breaker_limit,
    )
    oracle = OfflineOracle()

    # Rolling price range (no lookahead bias).
    # Use first 5 minutes of ticks to establish initial range.
    warmup_end_ms = ticks[0].timestamp_ms + 5 * 60_000
    warmup_prices = [t.price for t in ticks if t.timestamp_ms < warmup_end_ms]
    if not warmup_prices:
        warmup_prices = [ticks[0].price]
    price_floor = min(warmup_prices)
    price_ceiling = max(warmup_prices)
    if price_ceiling <= price_floor:
        price_ceiling = price_floor + 1.0

    total_ticks = len(ticks)
    first_ts = ticks[0].timestamp_ms
    last_ts = ticks[-1].timestamp_ms
    span_hours = (last_ts - first_ts) / 3_600_000

    if not quiet:
        logger.info(
            "Backtest [%s]: %d ticks over %.1f hours, capital=$%.0f, "
            "spread=%s, exposure=$%.0f, stop=%.1f%%",
            config_name or "default", total_ticks, span_hours,
            starting_capital,
            "dynamic" if dynamic_spread else f"{maker_spread:.3f}",
            max_exposure, stop_loss_pct * 100,
        )

    in_directional_mode = False
    last_strategy_eval_ms = 0
    tick_idx = 0
    final_fair_prob = 0.50
    log_interval_ms = 600_000  # Log progress every 10 minutes of sim time.
    next_log_ms = first_ts + log_interval_ms

    # Daily PnL tracking (for daily_macro_reset mode).
    daily_metrics: list[dict] = []
    prev_day: int = first_ts // 86_400_000
    day_start_equity: float = starting_capital
    day_global_stops: int = 0

    for tick in ticks:
        # Feed every tick into the oracle.
        oracle.process_tick(tick)
        tick_idx += 1

        # --- Daily macro breaker reset at UTC midnight ---
        current_day = tick.timestamp_ms // 86_400_000
        if daily_macro_reset and current_day > prev_day:
            day_end_equity = sim.get_total_equity(final_fair_prob)
            daily_metrics.append({
                "date": datetime.fromtimestamp(
                    prev_day * 86_400, tz=timezone.utc
                ).strftime("%Y-%m-%d"),
                "start_equity": day_start_equity,
                "end_equity": day_end_equity,
                "pnl": day_end_equity - day_start_equity,
                "maker_rebates": sim.account.total_maker_rebates,
                "directional_pnl": sim.account.total_directional_pnl,
                "breaker_tripped": sim.macro_breaker_tripped,
                "global_stops": day_global_stops,
            })
            sim.reset_daily_pnl()
            day_start_equity = day_end_equity
            day_global_stops = 0
            prev_day = current_day

            if not quiet:
                logger.info(
                    "=== DAY BOUNDARY: %s | equity=$%.2f ===",
                    daily_metrics[-1]["date"], day_end_equity,
                )

        # Expand price range on new extremes (forward-only).
        if tick.price < price_floor:
            price_floor = tick.price
        if tick.price > price_ceiling:
            price_ceiling = tick.price

        fair_prob = price_to_probability(tick.price, price_floor, price_ceiling)
        final_fair_prob = fair_prob

        # --- Fill check on every tick ---
        sim.process_price_update(
            price_low=fair_prob,
            price_high=fair_prob,
            idx=tick_idx,
        )

        # --- Global portfolio stop-loss on every tick (Task 2.9) ---
        if global_stop_loss:
            stop_exits = sim.check_global_stop_loss(fair_prob, tick_idx, tick.timestamp_ms)
            if daily_macro_reset and stop_exits:
                day_global_stops += len(stop_exits)

        # --- Check runner stops on every tick ---
        sim.check_runner_stops(fair_prob, tick_idx)

        # --- Check scale-out on every tick ---
        sim.check_scale_out(fair_prob, tick_idx)

        # --- Macro circuit breaker check (Task 2.12) ---
        if macro_breaker_enabled:
            if sim.check_macro_breaker(tick_idx):
                # Breaker just tripped — skip all strategy logic.
                sim.update_drawdown(fair_prob)
                continue
            if sim.macro_breaker_tripped:
                # Breaker was previously tripped — skip all strategy logic.
                continue

        # --- Strategy evaluation at 1-second intervals ---
        if tick.timestamp_ms < last_strategy_eval_ms + STRATEGY_EVAL_INTERVAL_MS:
            continue
        last_strategy_eval_ms = tick.timestamp_ms

        cvd = oracle.cvd_delta
        z = oracle.z_score

        # State B: Directional entry (Task 2.6 — loosened thresholds).
        # Trigger: Z > threshold AND |CVD| > cvd_threshold.
        if (
            z > zscore_threshold
            and abs(cvd) > cvd_threshold
            and not in_directional_mode
        ):
            entry = sim.enter_directional(
                probability=fair_prob,
                size_usdc=directional_size_usdc,
                idx=tick_idx,
            )
            if entry:
                in_directional_mode = True
                sim.cancel_ask_orders()
                logger.info(
                    "DIRECTIONAL ENTRY @ tick %d: CVD=$%s, Z=%+.2f, "
                    "prob=%.4f, price=$%.2f",
                    tick_idx, f"{cvd:+,.0f}", z, fair_prob, tick.price,
                )

        # Reset directional mode when flat.
        if in_directional_mode and not sim.account.positions:
            in_directional_mode = False

        # State A: Maker quoting (when not directional).
        if not in_directional_mode:
            # Determine effective spread.
            if dynamic_spread:
                # Task 2.5: Spread adapts to Z-Score regime.
                has_inventory = sim.account.yes_shares > 0
                effective_spread = compute_dynamic_spread(z, has_inventory, spread_tiers)
                if effective_spread is None:
                    # Volatile + inventory: pause quoting.
                    sim.cancel_all_orders()
                    effective_spread = None  # Signal: no quoting this tick.
            else:
                effective_spread = maker_spread

            # Place orders (skewed or symmetric).
            if effective_spread is not None:
                if inventory_skew:
                    # Task 2.8: Asymmetric quoting with inventory skew.
                    sim.place_skewed_maker_orders(
                        fair_price=fair_prob,
                        spread=effective_spread,
                        size_usdc=maker_size_usdc,
                        idx=tick_idx,
                        current_time_ms=tick.timestamp_ms,
                    )
                else:
                    sim.place_maker_orders(
                        fair_price=fair_prob,
                        spread=effective_spread,
                        size_usdc=maker_size_usdc,
                        idx=tick_idx,
                        current_time_ms=tick.timestamp_ms,
                    )

        # Update drawdown.
        sim.update_drawdown(fair_prob)

        # Periodic progress logging.
        if not quiet and tick.timestamp_ms >= next_log_ms:
            elapsed_min = (tick.timestamp_ms - first_ts) / 60_000
            equity = sim.get_total_equity(fair_prob)
            logger.info(
                "  [%.0fmin] tick %d/%d | equity=$%.2f | bal=$%.2f | "
                "rebates=$%.4f | dir_pnl=$%.4f | Z=%+.2f | CVD=$%s",
                elapsed_min, tick_idx, total_ticks, equity,
                sim.account.usdc_balance,
                sim.account.total_maker_rebates,
                sim.account.total_directional_pnl,
                z, f"{cvd:+,.0f}",
            )
            next_log_ms += log_interval_ms

    # Record final (partial) day if in daily reset mode.
    if daily_macro_reset:
        day_end_equity = sim.get_total_equity(final_fair_prob)
        daily_metrics.append({
            "date": datetime.fromtimestamp(
                prev_day * 86_400, tz=timezone.utc
            ).strftime("%Y-%m-%d"),
            "start_equity": day_start_equity,
            "end_equity": day_end_equity,
            "pnl": day_end_equity - day_start_equity,
            "maker_rebates": sim.account.total_maker_rebates,
            "directional_pnl": sim.account.total_directional_pnl,
            "breaker_tripped": sim.macro_breaker_tripped,
            "global_stops": day_global_stops,
        })

    return BacktestResult(
        sim=sim,
        final_fair_prob=final_fair_prob,
        num_ticks=total_ticks,
        span_hours=span_hours,
        config_name=config_name,
        daily_metrics=daily_metrics,
    )


# ---------------------------------------------------------------------------
# Tear Sheet (Task 2.4 — mark to final fair_prob, not 0.50)
# ---------------------------------------------------------------------------

def print_tear_sheet(result: BacktestResult, starting_capital: float = 500.0) -> None:
    """Print the Quant Tear Sheet to terminal, marking open positions at final fair_prob."""
    sim = result.sim
    acct = sim.account
    # Task 2.4: Mark open inventory at the final tick's fair probability,
    # NOT hardcoded 0.50.
    final_equity = sim.get_total_equity(result.final_fair_prob)

    maker_fills = [
        t for t in acct.trades
        if t.action.startswith("MAKER_") and "NO_INVENTORY" not in t.action
    ]
    directional_entries = [
        t for t in acct.trades if t.action == "DIRECTIONAL_ENTRY"
    ]
    scale_outs = [
        t for t in acct.trades if t.action == "SCALE_OUT_50PCT"
    ]
    runner_stops = [
        t for t in acct.trades if t.action == "RUNNER_STOP_EXIT"
    ]
    global_stops = [
        t for t in acct.trades if t.action == "GLOBAL_STOP_LOSS"
    ]

    net_pnl = final_equity - starting_capital
    roi_pct = (net_pnl / starting_capital) * 100 if starting_capital > 0 else 0.0

    border = "=" * 62
    divider = "-" * 62

    print(f"\n{border}")
    print("        POLYMARKET HFT — PHASE 2.5 QUANT TEAR SHEET")
    if result.config_name:
        print(f"        Config: {result.config_name}")
    print(border)
    print(f"  Data Points (raw ticks):         {result.num_ticks:>12,d}")
    print(f"  Simulation Window:               {result.span_hours:>11.1f}h")
    print(f"  Data Source:                     {'Binance aggTrades':>18s}")
    print(f"  Final Mark Price (prob):         {result.final_fair_prob:>11.4f}")
    print(divider)

    print("  CAPITAL")
    print(f"    Starting Capital:              ${starting_capital:>11,.2f}")
    print(f"    Final Equity:                  ${final_equity:>11,.2f}")
    print(f"    Net PnL:                       ${net_pnl:>+11,.4f}")
    print(f"    ROI:                           {roi_pct:>+11.4f}%")
    print(divider)

    print("  VOLUME & FEES")
    print(f"    Total Simulated Volume:        ${acct.total_volume:>11,.2f}")
    print(f"    Total Maker Rebates Earned:    ${acct.total_maker_rebates:>+11,.4f}")
    print(f"    Total Taker Fees Paid:         ${acct.total_taker_fees:>11,.4f}")
    print(f"    Net Fee Impact:                ${acct.total_maker_rebates - acct.total_taker_fees:>+11,.4f}")
    print(divider)

    print("  DIRECTIONAL PnL (Runners)")
    print(f"    Directional PnL:               ${acct.total_directional_pnl:>+11,.4f}")
    print(f"    Directional Entries:           {len(directional_entries):>12d}")
    print(f"    Scale-Outs (50%):              {len(scale_outs):>12d}")
    print(f"    Runner Stops:                  {len(runner_stops):>12d}")
    print(f"    Global Stops (8c):             {len(global_stops):>12d}")
    if global_stops:
        global_stop_pnl = sum(t.pnl_usdc for t in global_stops)
        print(f"    Global Stop PnL:               ${global_stop_pnl:>+11,.4f}")
    if directional_entries:
        dir_wins = len([t for t in scale_outs if t.pnl_usdc > 0])
        dir_total = len(directional_entries)
        print(f"    Directional Win Rate:          {dir_wins / dir_total * 100 if dir_total else 0:>11.1f}%")
    print(divider)

    print("  RISK")
    print(f"    Maximum Drawdown:              {acct.max_drawdown_pct * 100:>11.2f}%")
    print(f"    Open Positions:                {len(acct.positions):>12d}")
    print(f"    Open YES Shares:               {acct.yes_shares:>12.2f}")
    print(divider)

    print("  CIRCUIT BREAKERS (Tasks 2.11-2.12)")
    cooldown_ms = sim.cooldown_duration_ms
    print(f"    Cooldown Duration:             {'OFF' if cooldown_ms == 0 else f'{cooldown_ms // 1000}s':>12s}")
    print(f"    Global Stops (cooldown events):{len(global_stops):>12d}")
    breaker_status = "TRIPPED" if sim.macro_breaker_tripped else "OK"
    print(f"    Macro Breaker Status:          {breaker_status:>12s}")
    print(f"    Macro Breaker Limit:           ${sim.macro_breaker_limit:>11,.2f}")
    print(f"    Realized PnL (breaker input):  ${acct.total_directional_pnl:>+11,.4f}")
    print(divider)

    print("  TRADE COUNTS")
    print(f"    Total Trades:                  {len(acct.trades):>12d}")
    print(f"    Maker Fills:                   {len(maker_fills):>12d}")
    print(f"    Directional Trades:            {len(directional_entries) + len(scale_outs) + len(runner_stops):>12d}")
    print(border)
    print()


# ---------------------------------------------------------------------------
# Parameter Sweep (Task 2.7)
# ---------------------------------------------------------------------------

def run_parameter_sweep(
    ticks: list[Tick],
    starting_capital: float = 500.0,
) -> list[BacktestResult]:
    """
    Run 5 backtest configurations and return results for comparison.

    Run 1: Baseline — static 1.5c spread, Z>2.5 & CVD>$5M, stop 2%.
    Run 2: Dynamic Quoting + Loosened Directional — dynamic spread,
           Z>2.0 & |CVD|>$3M, stop 2%.
    Run 3: Hyper-Defensive — max exposure $40, stop 1%, static 1.5c.
    Run 4: Skewed & Stopped — dynamic spread, inventory skew, global 8c stop.
    Run 5: The Surviving Maker — Run 4 + 5-min cooldown + $25 macro breaker.
    """
    configs = [
        {
            "config_name": "Run 1: Baseline",
            "maker_spread": 0.015,
            "dynamic_spread": False,
            "inventory_skew": False,
            "global_stop_loss": False,
            "cvd_threshold": 5_000_000.0,
            "zscore_threshold": 2.5,
            "max_exposure": 75.0,
            "stop_loss_pct": 0.02,
            "inventory_limit": 40.0,
        },
        {
            "config_name": "Run 2: Dynamic + Loosened",
            "maker_spread": 0.015,
            "dynamic_spread": True,
            "inventory_skew": False,
            "global_stop_loss": False,
            "cvd_threshold": 3_000_000.0,
            "zscore_threshold": 2.0,
            "max_exposure": 75.0,
            "stop_loss_pct": 0.02,
            "inventory_limit": 40.0,
        },
        {
            "config_name": "Run 3: Hyper-Defensive",
            "maker_spread": 0.015,
            "dynamic_spread": False,
            "inventory_skew": False,
            "global_stop_loss": False,
            "cvd_threshold": 5_000_000.0,
            "zscore_threshold": 2.5,
            "max_exposure": 40.0,
            "stop_loss_pct": 0.01,
            "inventory_limit": 40.0,
        },
        {
            "config_name": "Run 4: Skewed & Stopped",
            "maker_spread": 0.015,
            "dynamic_spread": True,
            "inventory_skew": True,
            "global_stop_loss": True,
            "cvd_threshold": 3_000_000.0,
            "zscore_threshold": 2.0,
            "max_exposure": 75.0,
            "stop_loss_pct": 0.02,
            "inventory_limit": 40.0,
        },
        {
            "config_name": "Run 5: The Surviving Maker",
            "maker_spread": 0.015,
            "dynamic_spread": True,
            "inventory_skew": True,
            "global_stop_loss": True,
            "cvd_threshold": 3_000_000.0,
            "zscore_threshold": 2.0,
            "max_exposure": 75.0,
            "stop_loss_pct": 0.02,
            "inventory_limit": 40.0,
            "cooldown_duration_ms": 300_000,
            "macro_breaker_enabled": True,
            "macro_breaker_limit": -25.0,
        },
    ]

    results: list[BacktestResult] = []

    for cfg in configs:
        logger.info("=" * 50)
        logger.info("SWEEP: Starting %s", cfg["config_name"])
        logger.info("=" * 50)

        result = run_backtest(
            ticks=ticks,
            starting_capital=starting_capital,
            maker_spread=cfg["maker_spread"],
            dynamic_spread=cfg["dynamic_spread"],
            inventory_skew=cfg["inventory_skew"],
            inventory_limit=cfg["inventory_limit"],
            global_stop_loss=cfg["global_stop_loss"],
            cvd_threshold=cfg["cvd_threshold"],
            zscore_threshold=cfg["zscore_threshold"],
            max_exposure=cfg["max_exposure"],
            stop_loss_pct=cfg["stop_loss_pct"],
            cooldown_duration_ms=cfg.get("cooldown_duration_ms", 0),
            macro_breaker_enabled=cfg.get("macro_breaker_enabled", False),
            macro_breaker_limit=cfg.get("macro_breaker_limit", -25.0),
            config_name=cfg["config_name"],
            quiet=True,
        )
        results.append(result)

    return results


def print_sweep_tear_sheet(
    results: list[BacktestResult],
    starting_capital: float = 500.0,
) -> None:
    """Print a comparative side-by-side sweep tear sheet."""
    n_cols = len(results)
    col_width = 20
    total_width = 32 + n_cols * (col_width + 2)
    border = "=" * total_width
    divider = "-" * total_width

    print(f"\n{border}")
    print("           POLYMARKET HFT — PHASE 2.85 PARAMETER SWEEP TEAR SHEET")
    print(border)

    # Header row.
    col_labels = [r.config_name.replace("Run ", "R") for r in results]
    print(f"  {'Metric':<30s}", end="")
    for label in col_labels:
        print(f"  {label:>{col_width}s}", end="")
    print()
    print(divider)

    # Extract metrics for each run.
    rows: list[tuple[str, list[str]]] = []

    # Capital section.
    final_equities = []
    net_pnls = []
    rois = []
    for r in results:
        eq = r.sim.get_total_equity(r.final_fair_prob)
        final_equities.append(eq)
        pnl = eq - starting_capital
        net_pnls.append(pnl)
        rois.append(pnl / starting_capital * 100)

    rows.append(("Starting Capital", [f"${starting_capital:,.2f}"] * n_cols))
    rows.append(("Final Equity", [f"${eq:,.2f}" for eq in final_equities]))
    rows.append(("Net PnL", [f"${pnl:+,.4f}" for pnl in net_pnls]))
    rows.append(("ROI", [f"{roi:+.4f}%" for roi in rois]))
    rows.append(("---", []))

    # Fees.
    rows.append(("Total Volume", [
        f"${r.sim.account.total_volume:,.2f}" for r in results
    ]))
    rows.append(("Maker Rebates", [
        f"${r.sim.account.total_maker_rebates:+,.4f}" for r in results
    ]))
    rows.append(("Taker Fees Paid", [
        f"${r.sim.account.total_taker_fees:,.4f}" for r in results
    ]))
    rows.append(("Net Fee Impact", [
        f"${r.sim.account.total_maker_rebates - r.sim.account.total_taker_fees:+,.4f}"
        for r in results
    ]))
    rows.append(("---", []))

    # Directional & Stops.
    rows.append(("Directional PnL", [
        f"${r.sim.account.total_directional_pnl:+,.4f}" for r in results
    ]))
    rows.append(("Directional Entries", [
        str(len([t for t in r.sim.account.trades if t.action == "DIRECTIONAL_ENTRY"]))
        for r in results
    ]))
    rows.append(("Scale-Outs", [
        str(len([t for t in r.sim.account.trades if t.action == "SCALE_OUT_50PCT"]))
        for r in results
    ]))
    rows.append(("Runner Stops", [
        str(len([t for t in r.sim.account.trades if t.action == "RUNNER_STOP_EXIT"]))
        for r in results
    ]))
    rows.append(("Global Stops (8c)", [
        str(len([t for t in r.sim.account.trades if t.action == "GLOBAL_STOP_LOSS"]))
        for r in results
    ]))
    rows.append(("---", []))

    # Risk.
    rows.append(("Max Drawdown", [
        f"{r.sim.account.max_drawdown_pct * 100:.2f}%" for r in results
    ]))
    rows.append(("Open Positions", [
        str(len(r.sim.account.positions)) for r in results
    ]))
    rows.append(("Open YES Shares", [
        f"{r.sim.account.yes_shares:.2f}" for r in results
    ]))
    rows.append(("---", []))

    # Circuit breakers (Tasks 2.11-2.12).
    rows.append(("Cooldown Duration", [
        "OFF" if r.sim.cooldown_duration_ms == 0 else f"{r.sim.cooldown_duration_ms // 1000}s"
        for r in results
    ]))
    rows.append(("Macro Breaker", [
        "TRIPPED" if r.sim.macro_breaker_tripped else "OK"
        for r in results
    ]))
    rows.append(("---", []))

    # Trade counts.
    rows.append(("Total Trades", [
        str(len(r.sim.account.trades)) for r in results
    ]))
    rows.append(("Maker Fills", [
        str(len([
            t for t in r.sim.account.trades
            if t.action.startswith("MAKER_") and "NO_INVENTORY" not in t.action
        ]))
        for r in results
    ]))

    # Print rows.
    for label, values in rows:
        if label == "---":
            print(divider)
            continue
        print(f"  {label:<30s}", end="")
        for val in values:
            print(f"  {val:>{col_width}s}", end="")
        print()

    print(border)
    print()

    # --- Focused Run 3 vs Run 4 vs Run 5 comparison ---
    if n_cols >= 5:
        r3, r4, r5 = results[2], results[3], results[4]
        eq3 = r3.sim.get_total_equity(r3.final_fair_prob)
        eq4 = r4.sim.get_total_equity(r4.final_fair_prob)
        eq5 = r5.sim.get_total_equity(r5.final_fair_prob)
        pnl3 = eq3 - starting_capital
        pnl4 = eq4 - starting_capital
        pnl5 = eq5 - starting_capital
        dd3 = r3.sim.account.max_drawdown_pct * 100
        dd4 = r4.sim.account.max_drawdown_pct * 100
        dd5 = r5.sim.account.max_drawdown_pct * 100
        gs4 = len([t for t in r4.sim.account.trades if t.action == "GLOBAL_STOP_LOSS"])
        gs5 = len([t for t in r5.sim.account.trades if t.action == "GLOBAL_STOP_LOSS"])

        focus_border = "=" * 78
        focus_div = "-" * 78
        print(f"\n{focus_border}")
        print("  FOCUSED COMPARISON: Run 3 (Hyper-Def) vs Run 4 (Smart) vs Run 5 (Surviving)")
        print(focus_border)
        print(f"  {'Metric':<30s}  {'Run 3':>12s}  {'Run 4':>12s}  {'Run 5':>12s}")
        print(focus_div)
        print(f"  {'Net PnL':<30s}  ${pnl3:>+11,.4f}  ${pnl4:>+11,.4f}  ${pnl5:>+11,.4f}")
        print(f"  {'ROI':<30s}  {pnl3/starting_capital*100:>+11.4f}%  {pnl4/starting_capital*100:>+11.4f}%  {pnl5/starting_capital*100:>+11.4f}%")
        print(f"  {'Max Drawdown':<30s}  {dd3:>11.2f}%  {dd4:>11.2f}%  {dd5:>11.2f}%")
        print(f"  {'Open Positions':<30s}  {len(r3.sim.account.positions):>12d}  {len(r4.sim.account.positions):>12d}  {len(r5.sim.account.positions):>12d}")
        print(f"  {'Open YES Shares':<30s}  {r3.sim.account.yes_shares:>12.2f}  {r4.sim.account.yes_shares:>12.2f}  {r5.sim.account.yes_shares:>12.2f}")
        print(f"  {'Global Stops':<30s}  {'N/A':>12s}  {gs4:>12d}  {gs5:>12d}")
        print(f"  {'Cooldown':<30s}  {'N/A':>12s}  {'OFF':>12s}  {'300s':>12s}")
        print(f"  {'Macro Breaker':<30s}  {'N/A':>12s}  {'N/A':>12s}  {'TRIPPED' if r5.sim.macro_breaker_tripped else 'OK':>12s}")
        print(f"  {'Maker Rebates':<30s}  ${r3.sim.account.total_maker_rebates:>+11,.4f}  ${r4.sim.account.total_maker_rebates:>+11,.4f}  ${r5.sim.account.total_maker_rebates:>+11,.4f}")
        print(f"  {'Total Trades':<30s}  {len(r3.sim.account.trades):>12d}  {len(r4.sim.account.trades):>12d}  {len(r5.sim.account.trades):>12d}")
        print(focus_border)
        print()
    elif n_cols >= 4:
        r3, r4 = results[2], results[3]
        eq3 = r3.sim.get_total_equity(r3.final_fair_prob)
        eq4 = r4.sim.get_total_equity(r4.final_fair_prob)
        pnl3 = eq3 - starting_capital
        pnl4 = eq4 - starting_capital
        dd3 = r3.sim.account.max_drawdown_pct * 100
        dd4 = r4.sim.account.max_drawdown_pct * 100
        gs4 = len([t for t in r4.sim.account.trades if t.action == "GLOBAL_STOP_LOSS"])

        focus_border = "=" * 62
        focus_div = "-" * 62
        print(f"\n{focus_border}")
        print("  FOCUSED COMPARISON: Run 3 (Hyper-Def) vs Run 4 (Smart Maker)")
        print(focus_border)
        print(f"  {'Metric':<30s}  {'Run 3':>12s}  {'Run 4':>12s}")
        print(focus_div)
        print(f"  {'Net PnL':<30s}  ${pnl3:>+11,.4f}  ${pnl4:>+11,.4f}")
        print(f"  {'ROI':<30s}  {pnl3/starting_capital*100:>+11.4f}%  {pnl4/starting_capital*100:>+11.4f}%")
        print(f"  {'Max Drawdown':<30s}  {dd3:>11.2f}%  {dd4:>11.2f}%")
        print(f"  {'Global Stops Triggered':<30s}  {'N/A':>12s}  {gs4:>12d}")
        print(f"  {'Maker Rebates':<30s}  ${r3.sim.account.total_maker_rebates:>+11,.4f}  ${r4.sim.account.total_maker_rebates:>+11,.4f}")
        print(f"  {'Total Trades':<30s}  {len(r3.sim.account.trades):>12d}  {len(r4.sim.account.trades):>12d}")
        print(focus_border)
        print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket HFT Backtester — Phase 2.5 (Tick-Level + Optimization)"
    )
    parser.add_argument(
        "--data",
        default=str(DATA_DIR / "btc_ticks.csv.gz"),
        help="Path to tick data CSV.GZ file.",
    )
    parser.add_argument(
        "--capital", type=float, default=500.0,
        help="Starting capital in USDC (default: 500).",
    )
    parser.add_argument(
        "--spread", type=float, default=0.015,
        help="Maker spread in probability units (default: 0.015 = 1.5 cents).",
    )
    parser.add_argument(
        "--maker-size", type=float, default=25.0,
        help="USDC per side for maker orders (default: 25).",
    )
    parser.add_argument(
        "--dir-size", type=float, default=30.0,
        help="USDC for directional entries (default: 30).",
    )
    parser.add_argument(
        "--cvd-threshold", type=float, default=5_000_000.0,
        help="CVD delta threshold for directional entry (default: 5M USD).",
    )
    parser.add_argument(
        "--zscore-threshold", type=float, default=2.5,
        help="Z-Score threshold for directional entry (default: 2.5).",
    )
    parser.add_argument(
        "--max-exposure", type=float, default=75.0,
        help="Max exposure in USDC (default: 75).",
    )
    parser.add_argument(
        "--stop-loss-pct", type=float, default=0.02,
        help="Runner stop-loss pct from entry (default: 0.02 = 2%%).",
    )
    parser.add_argument(
        "--dynamic-spread", action="store_true",
        help="Enable dynamic volatility quoting (Task 2.5).",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run 3-config parameter sweep instead of single backtest.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
        stream=sys.stdout,
    )

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(
            "Data file not found: %s\n"
            "Run 'python -m scripts.fetch_historical' first.",
            data_path,
        )
        sys.exit(1)

    ticks = load_ticks(data_path)
    if len(ticks) < 100:
        logger.error("Insufficient tick data (%d). Need at least 100.", len(ticks))
        sys.exit(1)

    if args.sweep:
        # Task 2.7: Parameter sweep.
        results = run_parameter_sweep(ticks, starting_capital=args.capital)

        # Print individual tear sheets.
        for result in results:
            print_tear_sheet(result, starting_capital=args.capital)

        # Print comparative sweep tear sheet.
        print_sweep_tear_sheet(results, starting_capital=args.capital)
    else:
        # Single run.
        result = run_backtest(
            ticks=ticks,
            starting_capital=args.capital,
            maker_spread=args.spread,
            maker_size_usdc=args.maker_size,
            directional_size_usdc=args.dir_size,
            cvd_threshold=args.cvd_threshold,
            zscore_threshold=args.zscore_threshold,
            max_exposure=args.max_exposure,
            stop_loss_pct=args.stop_loss_pct,
            dynamic_spread=args.dynamic_spread,
        )
        print_tear_sheet(result, starting_capital=args.capital)


if __name__ == "__main__":
    main()

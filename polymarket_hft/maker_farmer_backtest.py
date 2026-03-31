"""
Pure Market Maker Farmer Backtest -- Tasks 2.21-2.24

Abandons all directional taker strategies. Pivots to 'Path B: The Pure Market
Maker'. Targets high-volume ultra-short-term markets and farms Maker Rebates
while actively dodging latency snipers.

Rules:
    Rule 1 -- 100% Maker:
        Only POST_ONLY logic for entries. Never cross the spread to open.
        Quote symmetric BID/ASK at fair_prob +/- half_spread.

    Rule 2 -- Toxicity Shield (Anti-Sniper, dynamic):
        If abs(CVD) > 15% of Rolling_10m_Volume OR abs(Z) > 2.0:
        cancel all quotes, halt quoting for 15 seconds.

    Rule 3 -- TTE Killswitch (Time-to-Expiry):
        Simulate 5-minute market cycles. If time remaining in current
        5-min window < 60 seconds: cancel all orders, flatten inventory
        via Taker market order, halt until next window.

    Rule 4 -- Hyper-Skew:
        Inventory_Limit = $50. If > $50: taker flatten + 15s cooldown.
        Skew_Threshold = $5. If > $5: pause increasing side, exit at +/-0.001.
        If <= $5: normal symmetric at fair +/- 0.0025.

    Rule 5 -- Grind Shield (Micro-Trend Filter, dynamic):
        Rolling 60s CVD_Grind (reuses OfflineOracle cvd_delta).
        If CVD_Grind > +5% of Rolling_10m_Volume: pause ASKs.
        If CVD_Grind < -5% of Rolling_10m_Volume: pause BIDs.
        Only applies to normal symmetric quoting; exit modes are immune.

    Rule 6 -- Time-Decay Inventory Limits (cross-timeframe):
        If TTE > 50% of window: Limits normal ($50 hard / $5 skew).
        If TTE <= 50% of window: BOTH limits slam to $5.
        If inventory > $5 at decay trigger, aggressive maker exit at +/-0.001.

    Rule 7 -- Zero-Edge Exit (cross-timeframe):
        If inventory > $1 AND TTE < 40% of window: price maker exit
        at exactly fair_prob (0.000 spread). Most aggressive maker
        exit before TTE killswitch taker flatten.

Risk Controls:
    $5 Daily MTM Circuit Breaker (mark-to-market, resets at 00:00 UTC).
    Evaluates total open risk on every tick:
        Daily_MTM_PnL = current_equity - day_start_equity
    If Daily_MTM_PnL <= -$5: flatten all, halt until midnight.

Usage:
    python maker_farmer_backtest.py
    python maker_farmer_backtest.py --data data/btc_ticks_20260301_20260307.csv.gz
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backtest import OfflineOracle, Tick, load_ticks, price_to_probability

logger = logging.getLogger("maker_farmer")

DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_EVAL_INTERVAL_MS = 1_000       # Re-evaluate quoting every 1 second.
TOXICITY_SHIELD_DURATION_MS = 15_000    # 15-second pause after toxic flow.
INVENTORY_BREACH_COOLDOWN_MS = 15_000   # 15s cooldown after breach flatten.
VOLUME_WINDOW_MS = 600_000              # 10-minute rolling window for avg volume.
TIME_DECAY_INVENTORY_LIMIT = 5.0        # $5 effective limit in time-decay zone.

# Window-relative fractions (replace hardcoded thresholds).
TTE_KILLSWITCH_FRAC = 0.20             # Taker flatten at 20% remaining.
TIME_DECAY_FRAC = 0.50                 # Skew slam at 50% remaining.
ZERO_EDGE_FRAC = 0.40                  # Zero-spread exit at 40% remaining.
TOXICITY_VOLUME_PCT = 0.15             # Shield if CVD > 15% of 10m volume.
GRIND_VOLUME_PCT = 0.05                # Pause side if CVD > 5% of 10m volume.


# ---------------------------------------------------------------------------
# Polymarket Fee Structure (March 2026)
# ---------------------------------------------------------------------------

def taker_fee_rate(prob: float) -> float:
    """Taker fee: 2 * 0.0156 * min(p, 1-p). Peaks at 1.56% at 50/50."""
    return 2.0 * 0.0156 * min(prob, 1.0 - prob)


def maker_rebate_rate(prob: float) -> float:
    """Maker rebate = 20% of the taker fee collected from counterparty."""
    return 0.20 * taker_fee_rate(prob)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MakerOrder:
    """A resting maker order (BID or ASK)."""
    side: str           # "BID" or "ASK"
    price: float        # Probability price level
    size_usdc: float    # USDC size of the order


@dataclass
class FillRecord:
    """A single maker fill event."""
    side: str           # "BID" or "ASK"
    fill_price: float   # Probability at fill
    size_usdc: float    # USDC notional
    shares: float       # YES shares transacted
    rebate: float       # Maker rebate earned
    idx: int            # Tick index
    time_ms: int        # Timestamp


@dataclass
class FlattenRecord:
    """A taker flatten event (TTE / inventory breach / breaker / EOD)."""
    reason: str         # "TTE", "INVENTORY_BREACH", "BREAKER", "EOD"
    inventory_shares: float
    flatten_prob: float
    balance_change: float   # Signed: + for long (sell), - for short (buy back)
    taker_fee: float
    pnl_usdc: float
    notional: float         # Absolute notional value of the flatten trade
    idx: int
    time_ms: int


@dataclass
class DailyMetric:
    """Per-day stats."""
    date: str
    start_equity: float
    end_equity: float
    pnl: float
    maker_fills: int
    maker_rebates: float
    taker_fees: float
    shield_triggers: int
    tte_flattens: int
    inventory_flattens: int
    breaker_tripped: bool
    grind_pauses: int = 0
    time_decay_exits: int = 0
    zero_edge_quotes: int = 0


@dataclass
class FarmerResult:
    """Full backtest output."""
    fills: list[FillRecord] = field(default_factory=list)
    flattens: list[FlattenRecord] = field(default_factory=list)
    daily_metrics: list[DailyMetric] = field(default_factory=list)
    starting_capital: float = 500.0
    final_equity: float = 500.0
    peak_equity: float = 500.0
    max_drawdown_pct: float = 0.0
    total_maker_rebates: float = 0.0
    total_taker_fees: float = 0.0
    total_volume: float = 0.0
    total_shield_triggers: int = 0
    total_tte_flattens: int = 0
    total_inventory_flattens: int = 0
    total_grind_pauses: int = 0
    total_time_decay_exits: int = 0
    total_zero_edge_quotes: int = 0
    window_minutes: float = 5.0
    span_hours: float = 0.0
    num_ticks: int = 0


# ---------------------------------------------------------------------------
# Maker Farmer Backtest Engine
# ---------------------------------------------------------------------------

def run_maker_farmer_backtest(
    ticks: list[Tick],
    starting_capital: float = 500.0,
    maker_size_usdc: float = 10.0,
    half_spread: float = 0.0025,
    zscore_toxicity_threshold: float = 2.0,
    inventory_skew_threshold: float = 5.0,
    inventory_hard_limit: float = 50.0,
    skew_exit_offset: float = 0.001,
    macro_breaker_limit: float = -5.0,
    time_decay_inventory_limit: float = 5.0,
    # Cross-timeframe: window-relative fractions.
    window_minutes: float = 5.0,
    time_decay_frac: float = 0.50,
    zero_edge_frac: float = 0.40,
    tte_killswitch_frac: float = 0.20,
    # Cross-asset: volume-relative thresholds.
    toxicity_volume_pct: float = 0.15,
    grind_volume_pct: float = 0.05,
) -> FarmerResult:
    """
    Run the Pure Market Maker Farmer backtest.

    All entries are POST_ONLY maker orders (earn rebates).
    Exits happen via:
        - Opposing maker fills (earn rebates)
        - TTE flatten (taker, pay fee)
        - Inventory breach flatten (taker, pay fee)
        - Macro breaker flatten (taker, pay fee)
    """
    oracle = OfflineOracle()

    # --- Derive ms thresholds from window-relative fractions ---
    window_ms = int(window_minutes * 60_000)
    tte_killswitch_ms = int(window_ms * tte_killswitch_frac)
    time_decay_ms = int(window_ms * time_decay_frac)
    zero_edge_ms = int(window_ms * zero_edge_frac)

    balance = starting_capital
    inventory_shares: float = 0.0       # + = long YES, - = short (owe YES)
    inventory_cost_basis: float = 0.0   # Total USDC spent acquiring inventory

    fills: list[FillRecord] = []
    flattens: list[FlattenRecord] = []
    total_maker_rebates = 0.0
    total_taker_fees = 0.0
    total_volume = 0.0

    # Drawdown tracking.
    peak_equity = starting_capital
    max_drawdown_pct = 0.0

    # Daily macro breaker.
    daily_realized_pnl = 0.0
    macro_breaker_tripped = False

    # Resting maker orders — at most one BID and one ASK.
    bid_order: MakerOrder | None = None
    ask_order: MakerOrder | None = None

    # Toxicity shield state.
    shield_active = False
    shield_end_ms = 0
    total_shield_triggers = 0

    # TTE killswitch state.
    tte_halted = False
    current_tte_window_start_ms = 0     # Start of current window.

    # Rolling 10-minute volume tracker (cross-asset dynamic thresholds).
    vol_window: deque[tuple[int, float]] = deque()
    vol_running_sum: float = 0.0
    rolling_10m_volume: float = 0.0

    # Inventory breach cooldown.
    inventory_cooldown_end_ms = 0

    # Counters.
    total_tte_flattens = 0
    total_inventory_flattens = 0
    total_grind_pauses = 0
    total_time_decay_exits = 0
    total_zero_edge_quotes = 0

    # Price range for probability mapping.
    warmup_end_ms = ticks[0].timestamp_ms + 5 * 60_000
    warmup_prices = [t.price for t in ticks if t.timestamp_ms < warmup_end_ms]
    if not warmup_prices:
        warmup_prices = [ticks[0].price]
    price_floor = min(warmup_prices)
    price_ceiling = max(warmup_prices)
    if price_ceiling <= price_floor:
        price_ceiling = price_floor + 1.0

    # Timing.
    total_ticks = len(ticks)
    first_ts = ticks[0].timestamp_ms
    last_ts = ticks[-1].timestamp_ms
    span_hours = (last_ts - first_ts) / 3_600_000

    # Align first TTE window.
    current_tte_window_start_ms = (first_ts // window_ms) * window_ms

    # Day boundary tracking.
    prev_day = first_ts // 86_400_000
    day_start_equity = starting_capital
    day_maker_fills = 0
    day_maker_rebates = 0.0
    day_taker_fees = 0.0
    day_shield_triggers = 0
    day_tte_flattens = 0
    day_inventory_flattens = 0
    day_grind_pauses = 0
    day_time_decay_exits = 0
    day_zero_edge_quotes = 0
    daily_metrics: list[DailyMetric] = []

    # Strategy eval pacing.
    last_strategy_eval_ms = 0
    tick_idx = 0
    fair_prob = 0.50

    logger.info(
        "Maker Farmer: %d ticks over %.1f hours, capital=$%.0f, "
        "size=$%.0f, spread=%.1fc, window=%.0fm, "
        "toxicity=%.0f%%vol|Z>%.1f, grind=%.0f%%vol, "
        "skew>$%.0f, limit=$%.0f, "
        "decay=%.0f%%→$%.0f, zero_edge=%.0f%%, kill=%.0f%%",
        total_ticks, span_hours, starting_capital, maker_size_usdc,
        half_spread * 2 * 100, window_minutes,
        toxicity_volume_pct * 100, zscore_toxicity_threshold,
        grind_volume_pct * 100,
        inventory_skew_threshold, inventory_hard_limit,
        time_decay_frac * 100, time_decay_inventory_limit,
        zero_edge_frac * 100, tte_killswitch_frac * 100,
    )

    for tick in ticks:
        oracle.process_tick(tick)
        tick_idx += 1

        # --- Rolling 10-minute volume tracker (O(1) amortized) ---
        abs_notional = tick.price * tick.quantity
        vol_window.append((tick.timestamp_ms, abs_notional))
        vol_running_sum += abs_notional
        vol_cutoff = tick.timestamp_ms - VOLUME_WINDOW_MS
        while vol_window and vol_window[0][0] < vol_cutoff:
            vol_running_sum -= vol_window.popleft()[1]
        rolling_10m_volume = vol_running_sum

        # --- Day boundary: reset macro breaker ---
        current_day = tick.timestamp_ms // 86_400_000
        if current_day > prev_day:
            day_end_equity = _compute_equity(
                balance, inventory_shares, fair_prob,
            )
            daily_metrics.append(DailyMetric(
                date=datetime.fromtimestamp(
                    prev_day * 86_400, tz=timezone.utc,
                ).strftime("%Y-%m-%d"),
                start_equity=day_start_equity,
                end_equity=day_end_equity,
                pnl=day_end_equity - day_start_equity,
                maker_fills=day_maker_fills,
                maker_rebates=day_maker_rebates,
                taker_fees=day_taker_fees,
                shield_triggers=day_shield_triggers,
                tte_flattens=day_tte_flattens,
                inventory_flattens=day_inventory_flattens,
                breaker_tripped=macro_breaker_tripped,
                grind_pauses=day_grind_pauses,
                time_decay_exits=day_time_decay_exits,
                zero_edge_quotes=day_zero_edge_quotes,
            ))
            daily_realized_pnl = 0.0
            macro_breaker_tripped = False
            day_start_equity = day_end_equity
            day_maker_fills = 0
            day_maker_rebates = 0.0
            day_taker_fees = 0.0
            day_shield_triggers = 0
            day_tte_flattens = 0
            day_inventory_flattens = 0
            day_grind_pauses = 0
            day_time_decay_exits = 0
            day_zero_edge_quotes = 0
            prev_day = current_day
            logger.info(
                "=== DAY BOUNDARY: %s | equity=$%.2f ===",
                daily_metrics[-1].date, day_end_equity,
            )

        # Update price range.
        if tick.price < price_floor:
            price_floor = tick.price
        if tick.price > price_ceiling:
            price_ceiling = tick.price
        fair_prob = price_to_probability(tick.price, price_floor, price_ceiling)

        # Clamp fair_prob to tradeable range.
        fair_prob = max(0.02, min(0.98, fair_prob))

        # --- MTM Macro breaker check (every tick) ---
        # Mark-to-market: track TOTAL daily PnL including unrealized inventory.
        equity = _compute_equity(balance, inventory_shares, fair_prob)
        daily_mtm_pnl = equity - day_start_equity

        if daily_mtm_pnl <= macro_breaker_limit and not macro_breaker_tripped:
            macro_breaker_tripped = True
            bid_order = None
            ask_order = None
            if inventory_shares != 0.0:
                fr = _flatten_inventory(
                    inventory_shares, inventory_cost_basis,
                    fair_prob, "BREAKER", tick_idx, tick.timestamp_ms,
                )
                balance += fr.balance_change
                total_taker_fees += fr.taker_fee
                day_taker_fees += fr.taker_fee
                total_volume += fr.notional
                daily_realized_pnl += fr.pnl_usdc
                flattens.append(fr)
                inventory_shares = 0.0
                inventory_cost_basis = 0.0
            logger.warning(
                "MTM BREAKER at tick %d: mtm_pnl=$%.2f (realized=$%.2f)",
                tick_idx, daily_mtm_pnl, daily_realized_pnl,
            )

        # Drawdown tracking (uses equity computed above).
        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity
            if dd > max_drawdown_pct:
                max_drawdown_pct = dd

        if macro_breaker_tripped:
            continue

        # ---------------------------------------------------------------
        # Rule 3: TTE Killswitch — check every tick
        # ---------------------------------------------------------------
        # Determine which window we're in.
        tick_window_start = (tick.timestamp_ms // window_ms) * window_ms

        # New window started — resume trading.
        if tick_window_start > current_tte_window_start_ms:
            current_tte_window_start_ms = tick_window_start
            tte_halted = False

        time_remaining_ms = (
            current_tte_window_start_ms + window_ms - tick.timestamp_ms
        )

        if not tte_halted and time_remaining_ms < tte_killswitch_ms:
            # TTE killswitch fires.
            tte_halted = True
            bid_order = None
            ask_order = None
            if inventory_shares != 0.0:
                fr = _flatten_inventory(
                    inventory_shares, inventory_cost_basis,
                    fair_prob, "TTE", tick_idx, tick.timestamp_ms,
                )
                balance += fr.balance_change
                total_taker_fees += fr.taker_fee
                day_taker_fees += fr.taker_fee
                total_volume += fr.notional
                daily_realized_pnl += fr.pnl_usdc
                flattens.append(fr)
                inventory_shares = 0.0
                inventory_cost_basis = 0.0
                total_tte_flattens += 1
                day_tte_flattens += 1
                logger.debug(
                    "TTE FLATTEN at tick %d: pnl=$%+.4f, fee=$%.4f",
                    tick_idx, fr.pnl_usdc, fr.taker_fee,
                )
            continue  # Skip everything else until new window.

        if tte_halted:
            # Still in the final 60s — drawdown already tracked above.
            continue

        # ---------------------------------------------------------------
        # Rule 2: Toxicity Shield — check every tick (dynamic threshold)
        # ---------------------------------------------------------------
        cvd = oracle.cvd_delta
        z = oracle.z_score
        toxicity_trigger = rolling_10m_volume * toxicity_volume_pct

        if abs(cvd) > toxicity_trigger or abs(z) > zscore_toxicity_threshold:
            if not shield_active:
                shield_active = True
                shield_end_ms = tick.timestamp_ms + TOXICITY_SHIELD_DURATION_MS
                total_shield_triggers += 1
                day_shield_triggers += 1
                bid_order = None
                ask_order = None
                logger.debug(
                    "TOXICITY SHIELD at tick %d: CVD=$%s, Z=%+.2f",
                    tick_idx, f"{cvd:+,.0f}", z,
                )
            else:
                # Extend the shield if toxic flow persists.
                shield_end_ms = tick.timestamp_ms + TOXICITY_SHIELD_DURATION_MS

        # Check if shield expired.
        if shield_active and tick.timestamp_ms >= shield_end_ms:
            shield_active = False

        # ---------------------------------------------------------------
        # Fill Check — every tick (O(1))
        # ---------------------------------------------------------------
        if not shield_active:
            # BID fill: price dropped to our bid level.
            if bid_order is not None and fair_prob <= bid_order.price:
                fill_price = bid_order.price
                shares_bought = bid_order.size_usdc / fill_price
                rebate = bid_order.size_usdc * maker_rebate_rate(fill_price)

                # Deduct cost from balance, add rebate.
                balance -= bid_order.size_usdc
                balance += rebate
                total_maker_rebates += rebate
                day_maker_rebates += rebate
                total_volume += bid_order.size_usdc

                # Update inventory.
                inventory_shares += shares_bought
                inventory_cost_basis += bid_order.size_usdc

                fills.append(FillRecord(
                    side="BID", fill_price=fill_price,
                    size_usdc=bid_order.size_usdc, shares=shares_bought,
                    rebate=rebate, idx=tick_idx, time_ms=tick.timestamp_ms,
                ))
                day_maker_fills += 1
                bid_order = None  # Order consumed.

            # ASK fill: price rose to our ask level.
            if ask_order is not None and fair_prob >= ask_order.price:
                fill_price = ask_order.price
                shares_sold = ask_order.size_usdc / fill_price
                rebate = ask_order.size_usdc * maker_rebate_rate(fill_price)

                # Receive proceeds + rebate.
                balance += ask_order.size_usdc
                balance += rebate
                total_maker_rebates += rebate
                day_maker_rebates += rebate
                total_volume += ask_order.size_usdc

                # Update inventory.
                inventory_shares -= shares_sold
                # Reduce cost basis proportionally.
                if inventory_shares > 0 and (inventory_shares + shares_sold) > 0:
                    ratio = inventory_shares / (inventory_shares + shares_sold)
                    inventory_cost_basis *= ratio
                elif inventory_shares <= 0:
                    # Went flat or short: realize PnL.
                    inventory_cost_basis = 0.0
                    if inventory_shares < 0:
                        # Now short: cost basis tracks the short side.
                        inventory_cost_basis = abs(
                            inventory_shares
                        ) * fill_price

                fills.append(FillRecord(
                    side="ASK", fill_price=fill_price,
                    size_usdc=ask_order.size_usdc, shares=shares_sold,
                    rebate=rebate, idx=tick_idx, time_ms=tick.timestamp_ms,
                ))
                day_maker_fills += 1
                ask_order = None  # Order consumed.

        # ---------------------------------------------------------------
        # Rule 4: Inventory Breach Check — after fills
        # ---------------------------------------------------------------
        inventory_usdc = abs(inventory_shares * fair_prob)

        if inventory_usdc > inventory_hard_limit:
            # HARD BREACH: taker flatten + 15s cooldown.
            bid_order = None
            ask_order = None
            fr = _flatten_inventory(
                inventory_shares, inventory_cost_basis,
                fair_prob, "INVENTORY_BREACH", tick_idx, tick.timestamp_ms,
            )
            balance += fr.balance_change
            total_taker_fees += fr.taker_fee
            day_taker_fees += fr.taker_fee
            total_volume += fr.notional
            daily_realized_pnl += fr.pnl_usdc
            flattens.append(fr)
            inventory_shares = 0.0
            inventory_cost_basis = 0.0
            total_inventory_flattens += 1
            day_inventory_flattens += 1
            inventory_cooldown_end_ms = (
                tick.timestamp_ms + INVENTORY_BREACH_COOLDOWN_MS
            )
            logger.debug(
                "INVENTORY BREACH at tick %d: pnl=$%+.4f, fee=$%.4f",
                tick_idx, fr.pnl_usdc, fr.taker_fee,
            )

        # ---------------------------------------------------------------
        # Strategy Eval — every 1 second
        # ---------------------------------------------------------------
        if tick.timestamp_ms < last_strategy_eval_ms + STRATEGY_EVAL_INTERVAL_MS:
            continue
        last_strategy_eval_ms = tick.timestamp_ms

        # Skip if shield active.
        if shield_active:
            continue

        # Skip if in inventory breach cooldown.
        if tick.timestamp_ms < inventory_cooldown_end_ms:
            continue

        # Skip if insufficient balance.
        if balance < maker_size_usdc:
            continue

        # --- Compute effective limits (Rule 6: Time-Decay) ---
        # At TTE ≤ decay fraction, BOTH the hard limit AND skew slam to $5.
        in_time_decay = time_remaining_ms <= time_decay_ms
        effective_limit = (
            time_decay_inventory_limit if in_time_decay
            else inventory_hard_limit
        )
        effective_skew = (
            time_decay_inventory_limit if in_time_decay
            else inventory_skew_threshold
        )

        inventory_usdc = abs(inventory_shares * fair_prob)

        # --- Determine exit mode ---
        # Rule 7: Zero-Edge Exit (TTE < zero_edge fraction, inventory > $1)
        zero_edge_active = (
            time_remaining_ms < zero_edge_ms
            and inventory_usdc > 1.0
        )

        # Rule 6: Time-Decay Exit (TTE ≤ decay fraction, inventory > $5)
        time_decay_active = (
            in_time_decay
            and inventory_usdc > time_decay_inventory_limit
        )

        # --- Place/update maker orders based on inventory + TTE state ---
        if inventory_usdc > inventory_hard_limit:
            # Should have been flattened above — safety guard.
            bid_order = None
            ask_order = None

        elif zero_edge_active:
            # Rule 7: Price exit at EXACTLY fair_prob (zero spread).
            # Most aggressive maker exit before TTE killswitch at 60s.
            if inventory_shares > 0:
                bid_order = None
                ask_order = MakerOrder(
                    side="ASK",
                    price=max(fair_prob, 0.01),
                    size_usdc=min(maker_size_usdc, inventory_usdc),
                )
            else:
                ask_order = None
                bid_order = MakerOrder(
                    side="BID",
                    price=min(fair_prob, 0.99),
                    size_usdc=min(maker_size_usdc, inventory_usdc),
                )
            total_zero_edge_quotes += 1
            day_zero_edge_quotes += 1

        elif time_decay_active:
            # Rule 6: Aggressive maker exit at +/- 0.001 (same as hyper-skew).
            # Inventory exceeds time-decay limit of $5.
            if inventory_shares > 0:
                bid_order = None
                ask_price = fair_prob + skew_exit_offset
                ask_price = min(ask_price, 0.99)
                ask_order = MakerOrder(
                    side="ASK",
                    price=ask_price,
                    size_usdc=min(maker_size_usdc, inventory_usdc),
                )
            else:
                ask_order = None
                bid_price = fair_prob - skew_exit_offset
                bid_price = max(bid_price, 0.01)
                bid_order = MakerOrder(
                    side="BID",
                    price=bid_price,
                    size_usdc=min(maker_size_usdc, inventory_usdc),
                )
            total_time_decay_exits += 1
            day_time_decay_exits += 1

        elif inventory_usdc > effective_skew:
            # Rule 4: HYPER-SKEW — pause increasing side, ultra-aggressive exit.
            # effective_skew = $25 normally, slams to $5 at TTE ≤ 150s.
            if inventory_shares > 0:
                bid_order = None
                ask_price = fair_prob + skew_exit_offset
                ask_price = min(ask_price, 0.99)
                ask_order = MakerOrder(
                    side="ASK",
                    price=ask_price,
                    size_usdc=min(maker_size_usdc, inventory_usdc),
                )
            else:
                ask_order = None
                bid_price = fair_prob - skew_exit_offset
                bid_price = max(bid_price, 0.01)
                bid_order = MakerOrder(
                    side="BID",
                    price=bid_price,
                    size_usdc=min(maker_size_usdc, inventory_usdc),
                )

        else:
            # NORMAL: symmetric quoting at fair +/- half_spread.
            bid_price = fair_prob - half_spread
            ask_price = fair_prob + half_spread
            bid_price = max(bid_price, 0.01)
            ask_price = min(ask_price, 0.99)

            bid_order = MakerOrder(
                side="BID", price=bid_price, size_usdc=maker_size_usdc,
            )
            ask_order = MakerOrder(
                side="ASK", price=ask_price, size_usdc=maker_size_usdc,
            )

            # Rule 5: Grind Shield — only applies to normal symmetric quoting.
            # Exit modes (zero-edge, time-decay, hyper-skew) are immune.
            grind_cvd = oracle.cvd_delta
            grind_trigger = rolling_10m_volume * grind_volume_pct
            if grind_cvd > grind_trigger and ask_order is not None:
                # Market buying aggressively → pause ASKs (avoid going short).
                ask_order = None
                total_grind_pauses += 1
                day_grind_pauses += 1
            elif grind_cvd < -grind_trigger and bid_order is not None:
                # Market selling aggressively → pause BIDs (avoid going long).
                bid_order = None
                total_grind_pauses += 1
                day_grind_pauses += 1

    # --- End of data: flatten any remaining inventory ---
    if inventory_shares != 0.0:
        fr = _flatten_inventory(
            inventory_shares, inventory_cost_basis,
            fair_prob, "EOD", tick_idx, ticks[-1].timestamp_ms,
        )
        balance += fr.balance_change
        total_taker_fees += fr.taker_fee
        total_volume += fr.notional
        flattens.append(fr)
        inventory_shares = 0.0
        inventory_cost_basis = 0.0

    # Record final partial day.
    daily_metrics.append(DailyMetric(
        date=datetime.fromtimestamp(
            prev_day * 86_400, tz=timezone.utc,
        ).strftime("%Y-%m-%d"),
        start_equity=day_start_equity,
        end_equity=balance,
        pnl=balance - day_start_equity,
        maker_fills=day_maker_fills,
        maker_rebates=day_maker_rebates,
        taker_fees=day_taker_fees,
        shield_triggers=day_shield_triggers,
        tte_flattens=day_tte_flattens,
        inventory_flattens=day_inventory_flattens,
        breaker_tripped=macro_breaker_tripped,
        grind_pauses=day_grind_pauses,
        time_decay_exits=day_time_decay_exits,
        zero_edge_quotes=day_zero_edge_quotes,
    ))

    return FarmerResult(
        fills=fills,
        flattens=flattens,
        daily_metrics=daily_metrics,
        starting_capital=starting_capital,
        final_equity=balance,
        peak_equity=peak_equity,
        max_drawdown_pct=max_drawdown_pct,
        total_maker_rebates=total_maker_rebates,
        total_taker_fees=total_taker_fees,
        total_volume=total_volume,
        total_shield_triggers=total_shield_triggers,
        total_tte_flattens=total_tte_flattens,
        total_inventory_flattens=total_inventory_flattens,
        total_grind_pauses=total_grind_pauses,
        total_time_decay_exits=total_time_decay_exits,
        total_zero_edge_quotes=total_zero_edge_quotes,
        window_minutes=window_minutes,
        span_hours=span_hours,
        num_ticks=total_ticks,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_inventory(
    inventory_shares: float,
    cost_basis: float,
    fair_prob: float,
    reason: str,
    idx: int,
    time_ms: int,
) -> FlattenRecord:
    """
    Flatten inventory via taker market order.

    If long (inventory_shares > 0): sell YES shares at fair_prob.
        balance_change = +(shares * fair_prob) - fee  (cash inflow)
        pnl = balance_change - cost_basis

    If short (inventory_shares < 0): buy YES shares at fair_prob to cover.
        balance_change = -(|shares| * fair_prob) - fee  (cash outflow)
        pnl = cost_basis + balance_change
             = cost_basis - |shares| * fair_prob - fee
    """
    # Notional = absolute trade value (for volume tracking).
    notional = abs(inventory_shares) * fair_prob

    if inventory_shares > 0:
        # LONG: sell YES shares → cash inflow.
        fee = notional * taker_fee_rate(fair_prob)
        bal_change = notional - fee
        pnl = bal_change - cost_basis
    else:
        # SHORT: buy back YES to cover → cash outflow.
        fee = notional * taker_fee_rate(fair_prob)
        bal_change = -(notional + fee)
        # We received cost_basis when selling short; now we spend notional+fee.
        pnl = cost_basis + bal_change  # = cost_basis - notional - fee

    return FlattenRecord(
        reason=reason,
        inventory_shares=inventory_shares,
        flatten_prob=fair_prob,
        balance_change=bal_change,
        taker_fee=fee,
        pnl_usdc=pnl,
        notional=notional,
        idx=idx,
        time_ms=time_ms,
    )


def _compute_equity(
    balance: float,
    inventory_shares: float,
    fair_prob: float,
) -> float:
    """
    Balance + mark-to-market inventory value.

    Long: we own shares worth shares * p → positive MTM.
    Short: we owe shares costing |shares| * p to buy back → negative MTM.
    """
    if inventory_shares > 0:
        # Long YES: asset worth shares * p.
        mtm = inventory_shares * fair_prob
    elif inventory_shares < 0:
        # Short YES: liability costing |shares| * p to cover.
        mtm = -(abs(inventory_shares) * fair_prob)
    else:
        mtm = 0.0
    return balance + mtm


# ---------------------------------------------------------------------------
# Tear Sheet (Task 2.22)
# ---------------------------------------------------------------------------

def print_farmer_tear_sheet(result: FarmerResult) -> None:
    """Print the Pure Market Maker Farmer tear sheet."""
    net_pnl = result.final_equity - result.starting_capital
    roi = net_pnl / result.starting_capital * 100

    total_fills = len(result.fills)
    bid_fills = sum(1 for f in result.fills if f.side == "BID")
    ask_fills = sum(1 for f in result.fills if f.side == "ASK")

    # Sharpe from daily metrics.
    daily_returns = []
    for d in result.daily_metrics:
        if d.start_equity > 0:
            daily_returns.append(d.pnl / d.start_equity)
    if len(daily_returns) >= 2:
        mean_r = sum(daily_returns) / len(daily_returns)
        var = sum((r - mean_r) ** 2 for r in daily_returns) / (
            len(daily_returns) - 1
        )
        std_r = math.sqrt(var) if var > 0 else 0.0
        sharpe = (mean_r / std_r) * math.sqrt(365) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    net_fee = result.total_maker_rebates - result.total_taker_fees
    breaker_days = sum(1 for d in result.daily_metrics if d.breaker_tripped)

    border = "=" * 62
    divider = "-" * 62

    print(f"\n{border}")
    print("    PURE MARKET MAKER FARMER -- HOT POTATO PROTOCOL")
    print(border)
    print(f"  Data Points (raw ticks):         {result.num_ticks:>12,d}")
    print(f"  Simulation Window:               {result.span_hours:>11.1f}h")
    days = len(result.daily_metrics)
    print(f"  Trading Days:                    {days:>12d}")
    cycles = int(result.span_hours * 60 / result.window_minutes)
    print(f"  {result.window_minutes:.0f}-Minute Cycles:                 {cycles:>12,d}")
    print(divider)

    print("  CAPITAL")
    print(f"    Starting Capital:              ${result.starting_capital:>11,.2f}")
    print(f"    Final Equity:                  ${result.final_equity:>11,.2f}")
    print(f"    Total Net PnL:                 ${net_pnl:>+11,.4f}")
    print(f"    ROI:                           {roi:>+11.4f}%")
    print(divider)

    print("  FEE CAPTURE (PRIMARY REVENUE)")
    print(f"    Total Maker Rebates Earned:    ${result.total_maker_rebates:>+11,.4f}")
    print(f"    Total Taker Fees Paid:         ${result.total_taker_fees:>11,.4f}")
    print(f"    Net Fee Impact:                ${net_fee:>+11,.4f}")
    if total_fills > 0:
        avg_rebate = result.total_maker_rebates / total_fills
        print(f"    Avg Rebate per Fill:           ${avg_rebate:>+11,.6f}")
    print(f"    Total Volume Traded:           ${result.total_volume:>11,.2f}")
    print(divider)

    print("  MAKER ACTIVITY")
    print(f"    Total Maker Fills:             {total_fills:>12,d}")
    print(f"    BID Fills (bought YES):        {bid_fills:>12,d}")
    print(f"    ASK Fills (sold YES):          {ask_fills:>12,d}")
    print(f"    Fill Imbalance:                {bid_fills - ask_fills:>+12,d}")
    print(divider)

    print("  TOXICITY SHIELD (Rule 2)")
    print(f"    Total Shield Triggers:         {result.total_shield_triggers:>12,d}")
    if result.span_hours > 0:
        per_hour = result.total_shield_triggers / result.span_hours
        print(f"    Triggers / Hour:               {per_hour:>11.1f}")
    print(divider)

    print("  TTE KILLSWITCH (Rule 3)")
    print(f"    TTE Flattens:                  {result.total_tte_flattens:>12,d}")
    tte_fees = sum(
        f.taker_fee for f in result.flattens if f.reason == "TTE"
    )
    tte_pnl = sum(
        f.pnl_usdc for f in result.flattens if f.reason == "TTE"
    )
    print(f"    TTE Flatten Fees:              ${tte_fees:>11,.4f}")
    print(f"    TTE Flatten PnL:               ${tte_pnl:>+11,.4f}")
    print(divider)

    print("  INVENTORY MANAGEMENT (Rule 4)")
    print(f"    Inventory Breach Flattens:     {result.total_inventory_flattens:>12,d}")
    inv_fees = sum(
        f.taker_fee for f in result.flattens if f.reason == "INVENTORY_BREACH"
    )
    inv_pnl = sum(
        f.pnl_usdc for f in result.flattens if f.reason == "INVENTORY_BREACH"
    )
    print(f"    Breach Flatten Fees:           ${inv_fees:>11,.4f}")
    print(f"    Breach Flatten PnL:            ${inv_pnl:>+11,.4f}")
    print(divider)

    print("  GRIND SHIELD (Rule 5)")
    print(f"    Grind Side-Pauses:             {result.total_grind_pauses:>12,d}")
    if result.span_hours > 0:
        gp_hour = result.total_grind_pauses / result.span_hours
        print(f"    Pauses / Hour:                 {gp_hour:>11.1f}")
    print(divider)

    print("  TIME-DECAY LIMITS (Rule 6)")
    print(f"    Time-Decay Exit Evals:         {result.total_time_decay_exits:>12,d}")
    print(divider)

    print("  ZERO-EDGE EXIT (Rule 7)")
    print(f"    Zero-Edge Quote Evals:         {result.total_zero_edge_quotes:>12,d}")
    print(divider)

    print("  RISK")
    print(f"    Max Drawdown:                  {result.max_drawdown_pct * 100:>11.2f}%")
    print(f"    Sharpe Ratio (ann.):           {sharpe:>+11.2f}")
    print(f"    Breaker Days:                  {breaker_days:>12d}")
    print(border)

    # --- Daily breakdown ---
    if result.daily_metrics:
        print(f"\n{border}")
        print("    DAILY BREAKDOWN")
        print(border)
        print(
            f"  {'Date':<12s}  {'PnL':>9s}  {'Fills':>5s}  "
            f"{'Rebates':>8s}  {'TkrFee':>7s}  {'Shld':>4s}  "
            f"{'TTE':>3s}  {'InvB':>4s}  {'Grnd':>4s}  "
            f"{'TDec':>4s}  {'ZEdg':>4s}  {'Brk':>3s}"
        )
        print(divider)
        for d in result.daily_metrics:
            brk = "X" if d.breaker_tripped else " "
            print(
                f"  {d.date:<12s}  ${d.pnl:>+8,.2f}  {d.maker_fills:>5d}  "
                f"${d.maker_rebates:>+7,.4f}  ${d.taker_fees:>6,.4f}  "
                f"{d.shield_triggers:>4d}  "
                f"{d.tte_flattens:>3d}  {d.inventory_flattens:>4d}  "
                f"{d.grind_pauses:>4d}  {d.time_decay_exits:>4d}  "
                f"{d.zero_edge_quotes:>4d}  [{brk}]"
            )
        print(divider)
        tot_pnl = sum(d.pnl for d in result.daily_metrics)
        tot_fills = sum(d.maker_fills for d in result.daily_metrics)
        tot_reb = sum(d.maker_rebates for d in result.daily_metrics)
        tot_fee = sum(d.taker_fees for d in result.daily_metrics)
        tot_shld = sum(d.shield_triggers for d in result.daily_metrics)
        tot_tte = sum(d.tte_flattens for d in result.daily_metrics)
        tot_inv = sum(d.inventory_flattens for d in result.daily_metrics)
        tot_grnd = sum(d.grind_pauses for d in result.daily_metrics)
        tot_tdec = sum(d.time_decay_exits for d in result.daily_metrics)
        tot_zedg = sum(d.zero_edge_quotes for d in result.daily_metrics)
        print(
            f"  {'TOTAL':<12s}  ${tot_pnl:>+8,.2f}  {tot_fills:>5d}  "
            f"${tot_reb:>+7,.4f}  ${tot_fee:>6,.4f}  "
            f"{tot_shld:>4d}  {tot_tte:>3d}  {tot_inv:>4d}  "
            f"{tot_grnd:>4d}  {tot_tdec:>4d}  {tot_zedg:>4d}"
        )
        print(border)
        print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    default_data = DATA_DIR / "btc_ticks_20260301_20260307.csv.gz"
    if not default_data.exists():
        default_data = DATA_DIR / "btc_ticks.csv.gz"

    parser = argparse.ArgumentParser(
        description="Pure Market Maker Farmer -- Rebate Farming Backtest"
    )
    parser.add_argument(
        "--data", default=str(default_data),
        help="Path to tick data CSV.GZ file.",
    )
    parser.add_argument(
        "--capital", type=float, default=500.0,
        help="Starting capital in USDC (default: 500).",
    )
    parser.add_argument(
        "--maker-size", type=float, default=10.0,
        help="USDC per maker order side (default: 10).",
    )
    parser.add_argument(
        "--window-minutes", type=float, default=5.0,
        help="Market cycle window in minutes (default: 5).",
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
            "Run 'python -m scripts.fetch_historical "
            "--start-date YYYY-MM-DD --end-date YYYY-MM-DD' first.",
            data_path,
        )
        sys.exit(1)

    ticks = load_ticks(data_path)
    if len(ticks) < 100:
        logger.error("Insufficient tick data (%d).", len(ticks))
        sys.exit(1)

    first_ts = ticks[0].timestamp_ms
    last_ts = ticks[-1].timestamp_ms
    span_days = (last_ts - first_ts) / 86_400_000
    logger.info(
        "Loaded %d ticks spanning %.1f days from %s.",
        len(ticks), span_days, data_path.name,
    )

    result = run_maker_farmer_backtest(
        ticks,
        starting_capital=args.capital,
        maker_size_usdc=args.maker_size,
        window_minutes=args.window_minutes,
    )

    print_farmer_tear_sheet(result)


if __name__ == "__main__":
    main()

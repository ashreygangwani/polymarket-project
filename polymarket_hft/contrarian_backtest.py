"""
Contrarian Taker Backtest — Tasks 2.17 & 2.18

Completely abandons the symmetric Maker strategy. We are now a directional,
contrarian Taker bot that fades retail crowd momentum.

Core Logic (Fading the Crowd):
    CVD > +$5M AND Z > 2.0  ->  SHORT (buy NO) — crowd panic-buying the top.
    CVD < -$5M AND Z < -2.0 ->  LONG  (buy YES) — crowd panic-selling the bottom.

Trade Management:
    Take Profit: +4.0 cents (2:1 reward).
    Stop Loss:   -2.0 cents (risk unit).

Risk Controls:
    $25 Daily Macro Breaker (resets at 00:00 UTC).
    5-minute post-trade cooldown (no re-entry immediately after a close).

Usage:
    python contrarian_backtest.py
    python contrarian_backtest.py --data data/btc_ticks_20260301_20260307.csv.gz
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backtest import OfflineOracle, Tick, load_ticks, price_to_probability

logger = logging.getLogger("contrarian_backtest")

DATA_DIR = Path(__file__).resolve().parent / "data"

# Strategy evaluation: check entry signals every 1 second.
# TP/SL checks happen on every tick (instant fills on cross).
STRATEGY_EVAL_INTERVAL_MS = 1_000


# ---------------------------------------------------------------------------
# Polymarket Taker Fee (March 2026)
# ---------------------------------------------------------------------------

def taker_fee_rate(prob: float) -> float:
    """Taker fee: 2 * 0.0156 * min(p, 1-p). Peaks at 1.56% at 50/50."""
    return 2.0 * 0.0156 * min(prob, 1.0 - prob)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ContrarianPosition:
    """An open contrarian position."""
    direction: str          # "LONG" (bought YES) or "SHORT" (bought NO)
    entry_prob: float       # Probability at entry
    size_usdc: float        # USDC risked (before fees)
    shares: float           # YES shares (LONG) or NO shares (SHORT)
    entry_fee: float        # Taker fee paid on entry
    entry_idx: int
    entry_time_ms: int
    tp_prob: float          # Take-profit trigger probability
    sl_prob: float          # Stop-loss trigger probability


@dataclass
class TradeResult:
    """A completed trade record."""
    direction: str          # "LONG" or "SHORT"
    entry_prob: float
    exit_prob: float
    pnl_usdc: float         # Net PnL after all fees
    entry_fee: float
    exit_fee: float
    outcome: str            # "TP", "SL", "BREAKER", "EOD"
    entry_idx: int
    exit_idx: int


@dataclass
class DailyMetric:
    """Per-day stats."""
    date: str
    start_equity: float
    end_equity: float
    pnl: float
    trades: int
    wins: int
    losses: int
    breaker_tripped: bool


@dataclass
class ContrarianResult:
    """Full backtest output."""
    trades: list[TradeResult] = field(default_factory=list)
    daily_metrics: list[DailyMetric] = field(default_factory=list)
    starting_capital: float = 500.0
    final_equity: float = 500.0
    peak_equity: float = 500.0
    max_drawdown_pct: float = 0.0
    total_fees: float = 0.0
    span_hours: float = 0.0
    num_ticks: int = 0


# ---------------------------------------------------------------------------
# Contrarian Backtest Engine
# ---------------------------------------------------------------------------

def run_contrarian_backtest(
    ticks: list[Tick],
    starting_capital: float = 500.0,
    trade_size_usdc: float = 30.0,
    cvd_threshold: float = 5_000_000.0,
    zscore_threshold: float = 2.0,
    take_profit_cents: float = 0.04,
    stop_loss_cents: float = 0.02,
    cooldown_ms: int = 300_000,
    macro_breaker_limit: float = -25.0,
) -> ContrarianResult:
    """
    Run the contrarian taker backtest.

    Checks TP/SL on every tick. Evaluates entry signals every 1 second.
    Daily macro breaker resets at 00:00 UTC.
    """
    oracle = OfflineOracle()
    balance = starting_capital
    position: ContrarianPosition | None = None
    trades: list[TradeResult] = []
    total_fees = 0.0

    # Drawdown tracking.
    peak_equity = starting_capital
    max_drawdown_pct = 0.0

    # Daily macro breaker.
    daily_realized_pnl = 0.0
    macro_breaker_tripped = False

    # Cooldown.
    last_trade_close_ms = 0

    # Price range for probability mapping (same as main backtest).
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

    # Day boundary tracking.
    prev_day = first_ts // 86_400_000
    day_start_equity = starting_capital
    day_trades = 0
    day_wins = 0
    day_losses = 0
    daily_metrics: list[DailyMetric] = []

    # Strategy eval pacing.
    last_strategy_eval_ms = 0
    tick_idx = 0
    fair_prob = 0.50

    logger.info(
        "Contrarian Backtest: %d ticks over %.1f hours, capital=$%.0f, "
        "trade_size=$%.0f, TP=+%.1fc, SL=-%.1fc, CVD>$%.0fM, Z>%.1f",
        total_ticks, span_hours, starting_capital, trade_size_usdc,
        take_profit_cents * 100, stop_loss_cents * 100,
        cvd_threshold / 1e6, zscore_threshold,
    )

    for tick in ticks:
        oracle.process_tick(tick)
        tick_idx += 1

        # --- Day boundary: reset macro breaker ---
        current_day = tick.timestamp_ms // 86_400_000
        if current_day > prev_day:
            # Record completed day.
            day_end_equity = balance
            if position is not None:
                day_end_equity = _mark_to_market(
                    balance, position, fair_prob,
                )
            daily_metrics.append(DailyMetric(
                date=datetime.fromtimestamp(
                    prev_day * 86_400, tz=timezone.utc,
                ).strftime("%Y-%m-%d"),
                start_equity=day_start_equity,
                end_equity=day_end_equity,
                pnl=day_end_equity - day_start_equity,
                trades=day_trades,
                wins=day_wins,
                losses=day_losses,
                breaker_tripped=macro_breaker_tripped,
            ))
            # Reset for new day.
            daily_realized_pnl = 0.0
            macro_breaker_tripped = False
            last_trade_close_ms = 0
            day_start_equity = day_end_equity
            day_trades = 0
            day_wins = 0
            day_losses = 0
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

        # --- TP/SL check on EVERY tick (instant fill) ---
        if position is not None:
            exit_result = _check_exit(position, fair_prob)
            if exit_result is not None:
                trade, pnl, exit_fee = _close_position(
                    position, fair_prob, exit_result, tick_idx, balance,
                )
                balance += pnl + position.size_usdc  # Return capital + PnL.
                total_fees += exit_fee
                daily_realized_pnl += trade.pnl_usdc
                trades.append(trade)
                day_trades += 1
                if trade.outcome == "TP":
                    day_wins += 1
                else:
                    day_losses += 1
                last_trade_close_ms = tick.timestamp_ms
                position = None

                logger.info(
                    "  %s %s @ tick %d: entry=%.4f exit=%.4f pnl=$%+.4f "
                    "bal=$%.2f",
                    trade.outcome, trade.direction, tick_idx,
                    trade.entry_prob, trade.exit_prob,
                    trade.pnl_usdc, balance,
                )

        # --- Macro breaker check ---
        if daily_realized_pnl <= macro_breaker_limit and not macro_breaker_tripped:
            macro_breaker_tripped = True
            # Force-close any open position.
            if position is not None:
                trade, pnl, exit_fee = _close_position(
                    position, fair_prob, "BREAKER", tick_idx, balance,
                )
                balance += pnl + position.size_usdc
                total_fees += exit_fee
                daily_realized_pnl += trade.pnl_usdc
                trades.append(trade)
                day_trades += 1
                day_losses += 1
                position = None
            logger.warning(
                "MACRO BREAKER TRIPPED at tick %d: daily_pnl=$%.2f. "
                "Halting for rest of day.",
                tick_idx, daily_realized_pnl,
            )

        # --- Drawdown tracking ---
        current_equity = balance
        if position is not None:
            current_equity = _mark_to_market(balance, position, fair_prob)
        if current_equity > peak_equity:
            peak_equity = current_equity
        if peak_equity > 0:
            dd = (peak_equity - current_equity) / peak_equity
            if dd > max_drawdown_pct:
                max_drawdown_pct = dd

        # Skip strategy eval if breaker tripped or not at 1s interval.
        if macro_breaker_tripped:
            continue
        if tick.timestamp_ms < last_strategy_eval_ms + STRATEGY_EVAL_INTERVAL_MS:
            continue
        last_strategy_eval_ms = tick.timestamp_ms

        # --- Cooldown check ---
        if tick.timestamp_ms < last_trade_close_ms + cooldown_ms:
            continue

        # --- Entry signal check (only if flat) ---
        if position is not None:
            continue

        cvd = oracle.cvd_delta
        z = oracle.z_score

        # Insufficient balance for a trade.
        estimated_fee = trade_size_usdc * taker_fee_rate(fair_prob)
        if balance < trade_size_usdc + estimated_fee:
            continue

        # LONG signal: crowd panic-selling the bottom.
        if cvd < -cvd_threshold and z < -zscore_threshold:
            position = _open_long(
                fair_prob, trade_size_usdc, tick_idx, tick.timestamp_ms,
                take_profit_cents, stop_loss_cents,
            )
            balance -= (position.size_usdc + position.entry_fee)
            total_fees += position.entry_fee
            logger.info(
                "LONG ENTRY @ tick %d: CVD=$%s, Z=%+.2f, prob=%.4f, "
                "TP=%.4f, SL=%.4f",
                tick_idx, f"{cvd:+,.0f}", z, fair_prob,
                position.tp_prob, position.sl_prob,
            )

        # SHORT signal: crowd panic-buying the top.
        elif cvd > cvd_threshold and z > zscore_threshold:
            position = _open_short(
                fair_prob, trade_size_usdc, tick_idx, tick.timestamp_ms,
                take_profit_cents, stop_loss_cents,
            )
            balance -= (position.size_usdc + position.entry_fee)
            total_fees += position.entry_fee
            logger.info(
                "SHORT ENTRY @ tick %d: CVD=$%s, Z=%+.2f, prob=%.4f, "
                "TP=%.4f, SL=%.4f",
                tick_idx, f"{cvd:+,.0f}", z, fair_prob,
                position.tp_prob, position.sl_prob,
            )

    # --- End of data: close any open position ---
    if position is not None:
        trade, pnl, exit_fee = _close_position(
            position, fair_prob, "EOD", tick_idx, balance,
        )
        balance += pnl + position.size_usdc
        total_fees += exit_fee
        daily_realized_pnl += trade.pnl_usdc
        trades.append(trade)
        day_trades += 1
        if trade.pnl_usdc > 0:
            day_wins += 1
        else:
            day_losses += 1
        position = None

    # Record final partial day.
    daily_metrics.append(DailyMetric(
        date=datetime.fromtimestamp(
            prev_day * 86_400, tz=timezone.utc,
        ).strftime("%Y-%m-%d"),
        start_equity=day_start_equity,
        end_equity=balance,
        pnl=balance - day_start_equity,
        trades=day_trades,
        wins=day_wins,
        losses=day_losses,
        breaker_tripped=macro_breaker_tripped,
    ))

    return ContrarianResult(
        trades=trades,
        daily_metrics=daily_metrics,
        starting_capital=starting_capital,
        final_equity=balance,
        peak_equity=peak_equity,
        max_drawdown_pct=max_drawdown_pct,
        total_fees=total_fees,
        span_hours=span_hours,
        num_ticks=total_ticks,
    )


# ---------------------------------------------------------------------------
# Position Helpers
# ---------------------------------------------------------------------------

def _open_long(
    prob: float,
    size_usdc: float,
    idx: int,
    time_ms: int,
    tp_cents: float,
    sl_cents: float,
) -> ContrarianPosition:
    """Open a LONG (buy YES) position."""
    shares = size_usdc / prob
    entry_fee = size_usdc * taker_fee_rate(prob)
    return ContrarianPosition(
        direction="LONG",
        entry_prob=prob,
        size_usdc=size_usdc,
        shares=shares,
        entry_fee=entry_fee,
        entry_idx=idx,
        entry_time_ms=time_ms,
        tp_prob=prob + tp_cents,
        sl_prob=prob - sl_cents,
    )


def _open_short(
    prob: float,
    size_usdc: float,
    idx: int,
    time_ms: int,
    tp_cents: float,
    sl_cents: float,
) -> ContrarianPosition:
    """Open a SHORT (buy NO) position."""
    no_price = 1.0 - prob
    no_shares = size_usdc / no_price
    entry_fee = size_usdc * taker_fee_rate(prob)
    return ContrarianPosition(
        direction="SHORT",
        entry_prob=prob,
        size_usdc=size_usdc,
        shares=no_shares,
        entry_fee=entry_fee,
        entry_idx=idx,
        entry_time_ms=time_ms,
        tp_prob=prob - tp_cents,   # Price drops = NO gains.
        sl_prob=prob + sl_cents,   # Price rises = against our short.
    )


def _check_exit(pos: ContrarianPosition, fair_prob: float) -> str | None:
    """Check if TP or SL is hit. Returns outcome string or None."""
    if pos.direction == "LONG":
        if fair_prob >= pos.tp_prob:
            return "TP"
        if fair_prob <= pos.sl_prob:
            return "SL"
    else:  # SHORT
        if fair_prob <= pos.tp_prob:
            return "TP"
        if fair_prob >= pos.sl_prob:
            return "SL"
    return None


def _close_position(
    pos: ContrarianPosition,
    exit_prob: float,
    outcome: str,
    idx: int,
    balance: float,
) -> tuple[TradeResult, float, float]:
    """
    Close a position and compute PnL.

    Returns (TradeResult, net_pnl_change_to_balance, exit_fee).
    The caller adds back size_usdc + pnl to balance.
    """
    if pos.direction == "LONG":
        # Sell YES shares at exit_prob.
        gross_proceeds = pos.shares * exit_prob
        exit_fee = gross_proceeds * taker_fee_rate(exit_prob)
        net_proceeds = gross_proceeds - exit_fee
        pnl = net_proceeds - pos.size_usdc - pos.entry_fee
    else:
        # Sell NO shares at (1 - exit_prob).
        no_exit_price = 1.0 - exit_prob
        gross_proceeds = pos.shares * no_exit_price
        exit_fee = gross_proceeds * taker_fee_rate(exit_prob)
        net_proceeds = gross_proceeds - exit_fee
        pnl = net_proceeds - pos.size_usdc - pos.entry_fee

    trade = TradeResult(
        direction=pos.direction,
        entry_prob=pos.entry_prob,
        exit_prob=exit_prob,
        pnl_usdc=pnl,
        entry_fee=pos.entry_fee,
        exit_fee=exit_fee,
        outcome=outcome,
        entry_idx=pos.entry_idx,
        exit_idx=idx,
    )
    # Return the net change to apply to balance (caller adds back size_usdc).
    return trade, pnl, exit_fee


def _mark_to_market(
    balance: float,
    pos: ContrarianPosition,
    fair_prob: float,
) -> float:
    """Mark open position to current price for equity tracking."""
    if pos.direction == "LONG":
        mtm = pos.shares * fair_prob
    else:
        mtm = pos.shares * (1.0 - fair_prob)
    return balance + mtm


# ---------------------------------------------------------------------------
# Tear Sheet (Task 2.18)
# ---------------------------------------------------------------------------

def print_contrarian_tear_sheet(result: ContrarianResult) -> None:
    """Print the Contrarian Taker tear sheet."""
    trades = result.trades
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.outcome == "TP")
    losses = sum(1 for t in trades if t.outcome in ("SL", "BREAKER"))
    eod_closes = sum(1 for t in trades if t.outcome == "EOD")
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    net_pnl = result.final_equity - result.starting_capital
    roi = net_pnl / result.starting_capital * 100

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

    avg_win = 0.0
    avg_loss = 0.0
    if wins > 0:
        avg_win = sum(t.pnl_usdc for t in trades if t.outcome == "TP") / wins
    if losses > 0:
        avg_loss = sum(
            t.pnl_usdc for t in trades if t.outcome in ("SL", "BREAKER")
        ) / losses

    border = "=" * 62
    divider = "-" * 62

    print(f"\n{border}")
    print("    CONTRARIAN TAKER BACKTEST — TASKS 2.17 & 2.18")
    print(border)
    print(f"  Data Points (raw ticks):         {result.num_ticks:>12,d}")
    print(f"  Simulation Window:               {result.span_hours:>11.1f}h")
    days = len(result.daily_metrics)
    print(f"  Trading Days:                    {days:>12d}")
    print(divider)

    print("  CAPITAL")
    print(f"    Starting Capital:              ${result.starting_capital:>11,.2f}")
    print(f"    Final Equity:                  ${result.final_equity:>11,.2f}")
    print(f"    Total Net PnL:                 ${net_pnl:>+11,.4f}")
    print(f"    ROI:                           {roi:>+11.4f}%")
    print(divider)

    print("  TRADE PERFORMANCE")
    print(f"    Total Trades Taken:            {total_trades:>12d}")
    print(f"    Wins (Take Profit):            {wins:>12d}")
    print(f"    Losses (Stop Loss):            {losses:>12d}")
    print(f"    EOD Closes:                    {eod_closes:>12d}")
    print(f"    Win Rate:                      {win_rate:>11.1f}%")
    print(f"    Avg Win:                       ${avg_win:>+11,.4f}")
    print(f"    Avg Loss:                      ${avg_loss:>+11,.4f}")
    if avg_loss != 0:
        print(f"    Avg Win / Avg Loss:            {abs(avg_win / avg_loss):>11.2f}x")
    print(divider)

    print("  RISK")
    print(f"    Max Drawdown:                  {result.max_drawdown_pct * 100:>11.2f}%")
    print(f"    Total Taker Fees Paid:         ${result.total_fees:>11,.4f}")
    print(f"    Sharpe Ratio (ann.):           {sharpe:>+11.2f}")
    breaker_days = sum(1 for d in result.daily_metrics if d.breaker_tripped)
    print(f"    Breaker Days:                  {breaker_days:>12d}")
    print(divider)

    # Direction breakdown.
    longs = [t for t in trades if t.direction == "LONG"]
    shorts = [t for t in trades if t.direction == "SHORT"]
    long_wins = sum(1 for t in longs if t.outcome == "TP")
    short_wins = sum(1 for t in shorts if t.outcome == "TP")

    print("  DIRECTION BREAKDOWN")
    print(f"    LONG trades:                   {len(longs):>12d}")
    if longs:
        print(f"    LONG win rate:                 "
              f"{long_wins / len(longs) * 100:>11.1f}%")
        print(f"    LONG PnL:                      "
              f"${sum(t.pnl_usdc for t in longs):>+11,.4f}")
    print(f"    SHORT trades:                  {len(shorts):>12d}")
    if shorts:
        print(f"    SHORT win rate:                "
              f"{short_wins / len(shorts) * 100:>11.1f}%")
        print(f"    SHORT PnL:                     "
              f"${sum(t.pnl_usdc for t in shorts):>+11,.4f}")
    print(border)

    # --- Daily breakdown ---
    if result.daily_metrics:
        print(f"\n{border}")
        print("    DAILY BREAKDOWN")
        print(border)
        print(f"  {'Date':<12s}  {'PnL':>10s}  {'Trades':>6s}  "
              f"{'W':>3s}  {'L':>3s}  {'WR%':>5s}  {'Brk':>3s}")
        print(divider)
        for d in result.daily_metrics:
            wr = (d.wins / d.trades * 100) if d.trades > 0 else 0
            brk = "X" if d.breaker_tripped else " "
            print(
                f"  {d.date:<12s}  ${d.pnl:>+9,.2f}  {d.trades:>6d}  "
                f"{d.wins:>3d}  {d.losses:>3d}  {wr:>4.0f}%  [{brk}]"
            )
        print(divider)
        total_pnl = sum(d.pnl for d in result.daily_metrics)
        total_t = sum(d.trades for d in result.daily_metrics)
        total_w = sum(d.wins for d in result.daily_metrics)
        total_l = sum(d.losses for d in result.daily_metrics)
        total_wr = (total_w / total_t * 100) if total_t > 0 else 0
        print(
            f"  {'TOTAL':<12s}  ${total_pnl:>+9,.2f}  {total_t:>6d}  "
            f"{total_w:>3d}  {total_l:>3d}  {total_wr:>4.0f}%"
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
        description="Contrarian Taker Backtest — Fade the Crowd"
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
        "--trade-size", type=float, default=30.0,
        help="USDC per trade (default: 30).",
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

    result = run_contrarian_backtest(
        ticks,
        starting_capital=args.capital,
        trade_size_usdc=args.trade_size,
    )

    print_contrarian_tear_sheet(result)


if __name__ == "__main__":
    main()

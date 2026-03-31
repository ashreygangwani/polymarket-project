"""
Regime Backtester -- Multi-Day Parameter Sweep with Daily Macro Reset.

Runs the HFT strategy over multi-day tick data with per-day macro breaker
resets. Losing $25 on Tuesday shuts down Tuesday, but trading resumes
Wednesday morning.

Compares 3 configurations:
    Config A (Baseline): exact Run 5 params -- spread tiers [0.5c, 1.5c, 3.0c]
    Config B (Hyper-Aggressive): tight spreads [0.5c, 0.5c, 1.5c] to farm volume
    Config C (Directional Bias): same spreads as A, lower Z/CVD thresholds

Usage:
    python -m scripts.fetch_historical --start-date 2026-03-01 --end-date 2026-03-07
    python regime_backtest.py --data data/btc_ticks_20260301_20260307.csv.gz
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from backtest import BacktestResult, load_ticks, run_backtest

logger = logging.getLogger("regime_backtest")

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Config Definitions (Task 2.16)
# ---------------------------------------------------------------------------

REGIME_CONFIGS = [
    {
        "config_name": "Config A: Baseline (Run 5)",
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
        "daily_macro_reset": True,
        "spread_tiers": (0.005, 0.015, 0.030),
    },
    {
        "config_name": "Config B: Hyper-Aggressive",
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
        "daily_macro_reset": True,
        "spread_tiers": (0.005, 0.005, 0.015),
    },
    {
        "config_name": "Config C: Directional Bias",
        "maker_spread": 0.015,
        "dynamic_spread": True,
        "inventory_skew": True,
        "global_stop_loss": True,
        "cvd_threshold": 2_000_000.0,
        "zscore_threshold": 1.5,
        "max_exposure": 75.0,
        "stop_loss_pct": 0.02,
        "inventory_limit": 40.0,
        "cooldown_duration_ms": 300_000,
        "macro_breaker_enabled": True,
        "macro_breaker_limit": -25.0,
        "daily_macro_reset": True,
        "spread_tiers": (0.005, 0.015, 0.030),
    },
]


# ---------------------------------------------------------------------------
# Regime Sweep Runner
# ---------------------------------------------------------------------------

def run_regime_sweep(
    ticks: list,
    starting_capital: float = 500.0,
) -> list[BacktestResult]:
    """Run all 3 regime configs and return results."""
    results: list[BacktestResult] = []

    for cfg in REGIME_CONFIGS:
        logger.info("=" * 60)
        logger.info("REGIME SWEEP: Starting %s", cfg["config_name"])
        logger.info("=" * 60)

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
            cooldown_duration_ms=cfg["cooldown_duration_ms"],
            macro_breaker_enabled=cfg["macro_breaker_enabled"],
            macro_breaker_limit=cfg["macro_breaker_limit"],
            daily_macro_reset=cfg["daily_macro_reset"],
            spread_tiers=cfg["spread_tiers"],
            config_name=cfg["config_name"],
            quiet=True,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Daily Analytics Helpers
# ---------------------------------------------------------------------------

def _compute_daily_stats(
    result: BacktestResult,
) -> dict:
    """Compute daily-level analytics from a backtest result."""
    dm = result.daily_metrics
    total_days = len(dm)
    profitable_days = sum(1 for d in dm if d["pnl"] > 0)
    losing_days = sum(1 for d in dm if d["pnl"] < 0)
    win_rate = (profitable_days / total_days * 100) if total_days > 0 else 0.0

    # Annualized Sharpe from daily returns.
    daily_returns = []
    for d in dm:
        if d["start_equity"] > 0:
            daily_returns.append(d["pnl"] / d["start_equity"])

    if len(daily_returns) >= 2:
        mean_ret = sum(daily_returns) / len(daily_returns)
        variance = sum(
            (r - mean_ret) ** 2 for r in daily_returns
        ) / (len(daily_returns) - 1)
        std_ret = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = (mean_ret / std_ret) * math.sqrt(365) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    breaker_days = sum(1 for d in dm if d["breaker_tripped"])
    total_global_stops = sum(d["global_stops"] for d in dm)

    # Per-day maker rebates (deltas from cumulative).
    total_rebates_earned = dm[-1]["maker_rebates"] if dm else 0.0

    return {
        "total_days": total_days,
        "profitable_days": profitable_days,
        "losing_days": losing_days,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "breaker_days": breaker_days,
        "total_global_stops": total_global_stops,
        "total_rebates_earned": total_rebates_earned,
    }


# ---------------------------------------------------------------------------
# Multi-Day Tear Sheet (Task 2.16)
# ---------------------------------------------------------------------------

def print_regime_tear_sheet(
    results: list[BacktestResult],
    starting_capital: float = 500.0,
) -> None:
    """Print multi-day comparative tear sheet with daily analytics."""
    n_cols = len(results)
    col_width = 22
    label_width = 34
    total_width = label_width + n_cols * (col_width + 2)
    border = "=" * total_width
    divider = "-" * total_width

    print(f"\n{border}")
    print("       POLYMARKET HFT — MULTI-DAY REGIME BACKTEST TEAR SHEET")
    print(border)

    # Header row.
    col_labels = [r.config_name.split(":")[0].strip() for r in results]
    print(f"  {'Metric':<{label_width - 2}s}", end="")
    for label in col_labels:
        print(f"  {label:>{col_width}s}", end="")
    print()
    print(divider)

    # Pre-compute stats.
    stats = [_compute_daily_stats(r) for r in results]
    final_eqs = [r.sim.get_total_equity(r.final_fair_prob) for r in results]
    net_pnls = [eq - starting_capital for eq in final_eqs]

    rows: list[tuple[str, list[str]]] = []

    # --- Simulation Info ---
    rows.append(("Simulation Window", [f"{r.span_hours:.1f}h" for r in results]))
    rows.append(("Total Ticks", [f"{r.num_ticks:,d}" for r in results]))
    rows.append(("Trading Days", [str(s["total_days"]) for s in stats]))
    rows.append(("---", []))

    # --- Capital ---
    rows.append(("Starting Capital", [f"${starting_capital:,.2f}"] * n_cols))
    rows.append(("Final Equity", [f"${eq:,.2f}" for eq in final_eqs]))
    rows.append(("Total Net PnL", [f"${pnl:+,.4f}" for pnl in net_pnls]))
    rows.append(("ROI", [
        f"{pnl / starting_capital * 100:+.4f}%" for pnl in net_pnls
    ]))
    rows.append(("---", []))

    # --- Daily Analytics ---
    rows.append(("Win Rate (days)", [
        f"{s['profitable_days']}/{s['total_days']} ({s['win_rate']:.0f}%)"
        for s in stats
    ]))
    rows.append(("Sharpe Ratio (ann.)", [f"{s['sharpe']:+.2f}" for s in stats]))
    rows.append(("---", []))

    # --- Volume & Fees ---
    rows.append(("Total Maker Rebates", [
        f"${r.sim.account.total_maker_rebates:+,.4f}" for r in results
    ]))
    rows.append(("Total Taker Fees", [
        f"${r.sim.account.total_taker_fees:,.4f}" for r in results
    ]))
    rows.append(("Total Volume", [
        f"${r.sim.account.total_volume:,.2f}" for r in results
    ]))
    rows.append(("---", []))

    # --- Risk ---
    rows.append(("Max Drawdown", [
        f"{r.sim.account.max_drawdown_pct * 100:.2f}%" for r in results
    ]))
    rows.append(("Breaker Days", [str(s["breaker_days"]) for s in stats]))
    rows.append(("Global Stops (total)", [
        str(s["total_global_stops"]) for s in stats
    ]))
    rows.append(("---", []))

    # --- Trade Counts ---
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
    rows.append(("Directional Entries", [
        str(len([t for t in r.sim.account.trades if t.action == "DIRECTIONAL_ENTRY"]))
        for r in results
    ]))
    rows.append(("---", []))

    # --- Config Parameters ---
    rows.append(("Spread Tiers (c/n/v)", [
        "/".join(f"{s * 100:.1f}c" for s in cfg["spread_tiers"])
        for cfg in REGIME_CONFIGS
    ]))
    rows.append(("CVD Threshold", [
        f"${cfg['cvd_threshold'] / 1e6:.0f}M" for cfg in REGIME_CONFIGS
    ]))
    rows.append(("Z-Score Threshold", [
        f"{cfg['zscore_threshold']:.1f}" for cfg in REGIME_CONFIGS
    ]))

    # Print rows.
    for label, values in rows:
        if label == "---":
            print(divider)
            continue
        print(f"  {label:<{label_width - 2}s}", end="")
        for val in values:
            print(f"  {val:>{col_width}s}", end="")
        print()

    print(border)

    # --- Per-Day Breakdown ---
    if results[0].daily_metrics:
        print(f"\n{border}")
        print("       DAILY PnL BREAKDOWN")
        print(border)

        # Header.
        print(f"  {'Date':<12s}", end="")
        for r in results:
            label = r.config_name.split(":")[0].strip()
            print(f"  {'PnL':>10s} {'Brk':>3s}", end="")
        print()
        print(divider)

        max_days = max(len(r.daily_metrics) for r in results)
        for i in range(max_days):
            if i < len(results[0].daily_metrics):
                date_str = results[0].daily_metrics[i]["date"]
            else:
                date_str = "?"
            print(f"  {date_str:<12s}", end="")
            for r in results:
                if i < len(r.daily_metrics):
                    d = r.daily_metrics[i]
                    brk = "X" if d["breaker_tripped"] else " "
                    print(f"  ${d['pnl']:>+8,.2f} [{brk}]", end="")
                else:
                    print(f"  {'N/A':>10s}    ", end="")
            print()

        print(divider)

        # Daily totals row.
        print(f"  {'TOTAL':<12s}", end="")
        for r in results:
            total_pnl = sum(d["pnl"] for d in r.daily_metrics)
            print(f"  ${total_pnl:>+8,.2f}    ", end="")
        print()

        print(border)
        print()

    # --- Verdict ---
    best_sharpe_idx = max(range(n_cols), key=lambda i: stats[i]["sharpe"])
    best_pnl_idx = max(range(n_cols), key=lambda i: net_pnls[i])

    print(f"  VERDICT:")
    print(
        f"    Highest Sharpe:  {results[best_sharpe_idx].config_name} "
        f"(Sharpe={stats[best_sharpe_idx]['sharpe']:+.2f})"
    )
    print(
        f"    Highest PnL:     {results[best_pnl_idx].config_name} "
        f"(PnL=${net_pnls[best_pnl_idx]:+,.4f})"
    )
    print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket HFT Regime Backtester — Multi-Day Sweep"
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
        logger.error("Insufficient tick data (%d). Need at least 100.", len(ticks))
        sys.exit(1)

    # Log data window.
    first_ts = ticks[0].timestamp_ms
    last_ts = ticks[-1].timestamp_ms
    span_hours = (last_ts - first_ts) / 3_600_000
    span_days = span_hours / 24
    logger.info(
        "Loaded %d ticks spanning %.1f hours (%.1f days).",
        len(ticks), span_hours, span_days,
    )

    # Run all 3 regime configs.
    results = run_regime_sweep(ticks, starting_capital=args.capital)

    # Print the regime tear sheet.
    print_regime_tear_sheet(results, starting_capital=args.capital)


if __name__ == "__main__":
    main()

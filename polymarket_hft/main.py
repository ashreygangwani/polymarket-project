"""
Polymarket HFT Bot — Phase 1 Entry Point.

Launches the UnifiedOracle and RiskMonitor as concurrent asyncio tasks
within a single event loop. Includes a heartbeat logger for real-time
system state inspection.

Usage:
    python main.py
    python main.py --assets BTC ETH SOL
    python main.py --assets BTC --log-level DEBUG
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time

from config import TRACKED_ASSETS
from risk_monitor import RiskMonitor
from unified_oracle import UnifiedOracle

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-14s | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging with millisecond timestamps."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stdout,
    )
    # Suppress noisy third-party loggers.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

HEARTBEAT_INTERVAL_S = 5.0


async def heartbeat(oracle: UnifiedOracle, risk_monitor: RiskMonitor) -> None:
    """
    Periodically log the system state for observability.
    Runs every HEARTBEAT_INTERVAL_S seconds.
    """
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

            lines = ["--- HEARTBEAT ---"]

            # Oracle state per asset.
            for asset in oracle.assets:
                s = oracle.state[asset]
                ready = oracle.is_ready(asset)
                sfp = s["synthetic_fair_price"]
                cvd = s["cvd_delta"]
                z = s["z_score"]
                bn_mid = s["binance_mid"]
                hl_mid = s["hl_mid"]

                lines.append(
                    f"  {asset:>5s} | ready={ready} | "
                    f"BN_mid={bn_mid:>12,.2f} | HL_mid={hl_mid:>12,.2f} | "
                    f"SFP={sfp:>12,.2f} | CVD={cvd:>+14,.0f} | Z={z:>+6.2f}"
                )

            # Risk state.
            toxic = risk_monitor.toxic_flow_active
            liq_total = risk_monitor.get_window_total_usd()
            lines.append(
                f"  RISK  | TOXIC_FLOW={'ACTIVE' if toxic else 'CLEAR':>6s} | "
                f"liq_window=${liq_total:>12,.0f}"
            )

            logger.info("\n".join(lines))

    except asyncio.CancelledError:
        logger.info("Heartbeat stopped.")


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


async def shutdown(
    oracle: UnifiedOracle,
    risk_monitor: RiskMonitor,
    heartbeat_task: asyncio.Task,
) -> None:
    """Orderly shutdown: stop oracle, risk monitor, heartbeat."""
    logger.info("Shutdown initiated...")

    heartbeat_task.cancel()
    await asyncio.gather(heartbeat_task, return_exceptions=True)

    await oracle.stop()
    await risk_monitor.stop()

    logger.info("All systems shut down cleanly.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(assets: list[str]) -> None:
    """Launch oracle + risk monitor + heartbeat, handle signals."""
    oracle = UnifiedOracle(assets=assets)
    risk_monitor = RiskMonitor()

    # Launch all three subsystems as independent background tasks.
    await oracle.start()
    await risk_monitor.start()
    heartbeat_task = asyncio.create_task(heartbeat(oracle, risk_monitor), name="heartbeat")

    logger.info(
        "Phase 1 systems online. Tracking: %s. Press Ctrl+C to stop.",
        ", ".join(assets),
    )

    # Wait for shutdown signal.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("Received shutdown signal.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()
    await shutdown(oracle, risk_monitor, heartbeat_task)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polymarket HFT Bot — Phase 1: UnifiedOracle + RiskMonitor"
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=list(TRACKED_ASSETS),
        help="Base assets to track (e.g., BTC ETH SOL). Default: %(default)s",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    logger.info("Starting Polymarket HFT Phase 1...")
    try:
        asyncio.run(main(args.assets))
    except KeyboardInterrupt:
        pass

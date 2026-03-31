"""
Fetch historical raw tick data with aggressor side for backtesting.

Data Source: Binance Futures aggTrades (BTCUSDT perpetual).
    - Real tick-level trades with true aggressor side (m field).
    - Full pagination via startTime/endTime, 1000 trades per request.
    - Hyperliquid's recentTrades only returns ~10 trades with no pagination,
      and their S3 archive is Requester Pays (needs AWS credentials).
    - Binance BTC tick data is structurally correlated with Hyperliquid BTC
      for CVD signal backtesting (same asset, similar microstructure).

Fault-Tolerant Design (Task 2.14b):
    - Incremental saving: each 1000-trade chunk is appended to a plain CSV
      immediately and flushed to disk. Zero data held in RAM.
    - Auto-resume: if the target CSV already exists, the script reads the
      last trade's timestamp and resumes from there. Re-running the exact
      same command after a crash picks up where it left off.
    - Exponential backoff: 10 retries with 5/10/30/60s sleep on failure.
    - Final output is gzip-compressed for the backtester.

Output: data/btc_ticks.csv.gz (or btc_ticks_YYYYMMDD_YYYYMMDD.csv.gz for date ranges)
    Columns: timestamp_ms, price, quantity, side, trade_id
    side: B = buyer aggressor (hit ask), A = seller aggressor (hit bid)

Usage:
    python -m scripts.fetch_historical                                          # Last 6 hours
    python -m scripts.fetch_historical --hours 12                               # Last 12 hours
    python -m scripts.fetch_historical --start-date 2026-03-01 --end-date 2026-03-07  # Full week
    python -m scripts.fetch_historical --start-date 2026-03-01                  # From date to now
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger("fetch_historical")

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
AGGTRADES_ENDPOINT = "/fapi/v1/aggTrades"
TRADES_PER_REQUEST = 1000

# --- Fault-tolerance constants (Task 2.14b) ---
REQUEST_TIMEOUT_S = 30          # Up from 10s.
MAX_RETRIES = 10                # Up from 3.
BACKOFF_SCHEDULE_S = [5, 10, 30, 30, 60, 60, 60, 60, 60, 60]
RATE_LIMIT_PAUSE_S = 0.1       # 100ms between successful calls.

CSV_FIELDNAMES = ["timestamp_ms", "price", "quantity", "side", "trade_id"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Auto-Resume: read checkpoint from existing CSV
# ---------------------------------------------------------------------------

def get_resume_timestamp(csv_path: Path) -> int | None:
    """
    Read the last complete line of an existing CSV to find the resume point.

    Returns the timestamp_ms of the last recorded trade, or None if the
    file doesn't exist or has no data rows.
    """
    if not csv_path.exists():
        return None

    file_size = csv_path.stat().st_size
    if file_size == 0:
        return None

    # Read the tail of the file to find the last complete line.
    # For large files, only read the last 4KB.
    read_size = min(4096, file_size)
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            f.seek(max(0, file_size - read_size))
            tail = f.read()
    except OSError as exc:
        logger.warning("Could not read checkpoint file %s: %s", csv_path, exc)
        return None

    lines = tail.strip().split("\n")
    # Walk backwards to find the last line that looks like a data row.
    for line in reversed(lines):
        parts = line.split(",")
        if len(parts) >= 5:
            try:
                ts = int(parts[0])
                if ts > 1_000_000_000_000:  # Sanity: after year 2001 in ms.
                    return ts
            except ValueError:
                continue  # Header row or corrupted line.

    return None


# ---------------------------------------------------------------------------
# Incremental CSV Writer
# ---------------------------------------------------------------------------

class IncrementalCSVWriter:
    """
    Append-mode CSV writer that flushes each chunk to disk immediately.

    Opens the file in append mode. Writes a header only if the file is
    new or empty. Each call to write_chunk() appends rows and flushes.
    """

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
        self._file = open(csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        if needs_header:
            self._writer.writerow(CSV_FIELDNAMES)
            self._file.flush()

        self.rows_written = 0

    def write_chunk(self, raw_trades: list[dict]) -> None:
        """Normalize a Binance batch and append to CSV immediately."""
        for t in raw_trades:
            side = "A" if t["m"] else "B"
            self._writer.writerow([t["T"], t["p"], t["q"], side, t["a"]])
        self._file.flush()
        self.rows_written += len(raw_trades)

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Gzip Compression
# ---------------------------------------------------------------------------

def compress_csv_to_gz(csv_path: Path, gz_path: Path) -> None:
    """Compress a plain CSV to .csv.gz and remove the original."""
    logger.info("Compressing %s → %s ...", csv_path.name, gz_path.name)
    with open(csv_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    gz_kb = gz_path.stat().st_size / 1024
    csv_kb = csv_path.stat().st_size / 1024
    ratio = gz_kb / csv_kb * 100 if csv_kb > 0 else 0
    logger.info(
        "Compressed: %.1f KB → %.1f KB (%.0f%% of original).",
        csv_kb, gz_kb, ratio,
    )
    csv_path.unlink()
    logger.info("Removed intermediate CSV: %s", csv_path.name)


# ---------------------------------------------------------------------------
# Streaming Fetch with Checkpointing (Tasks 1-4)
# ---------------------------------------------------------------------------

def fetch_and_stream(
    symbol: str,
    start_ms: int,
    end_ms: int,
    writer: IncrementalCSVWriter,
) -> int:
    """
    Fetch aggregate trades from Binance and stream each chunk to disk.

    Returns the total number of trades written in this session.
    """
    cursor = start_ms
    request_count = 0
    session_trades = 0
    fetch_start_time = time.time()

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": TRADES_PER_REQUEST,
        }

        # --- Retry loop with exponential backoff (Task 2.14b-3) ---
        batch = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(
                    f"{BINANCE_FUTURES_BASE}{AGGTRADES_ENDPOINT}",
                    params=params,
                    timeout=REQUEST_TIMEOUT_S,
                )
                # Handle rate-limit and server errors with backoff.
                if resp.status_code in (429, 500, 502, 503):
                    backoff_s = BACKOFF_SCHEDULE_S[
                        min(attempt, len(BACKOFF_SCHEDULE_S) - 1)
                    ]
                    logger.warning(
                        "HTTP %d on attempt %d/%d. Sleeping %ds...",
                        resp.status_code, attempt + 1, MAX_RETRIES, backoff_s,
                    )
                    time.sleep(backoff_s)
                    continue

                resp.raise_for_status()
                batch = resp.json()
                break

            except (requests.RequestException, ValueError) as exc:
                backoff_s = BACKOFF_SCHEDULE_S[
                    min(attempt, len(BACKOFF_SCHEDULE_S) - 1)
                ]
                logger.warning(
                    "Attempt %d/%d failed: %s. Sleeping %ds...",
                    attempt + 1, MAX_RETRIES, exc, backoff_s,
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(backoff_s)
                else:
                    raise RuntimeError(
                        f"Binance API unavailable after {MAX_RETRIES} retries: "
                        f"{exc}"
                    )

        if not batch:
            logger.info("No more trades returned.")
            break

        # --- Incremental save: append chunk to CSV immediately (Task 2.14b-1) ---
        writer.write_chunk(batch)
        request_count += 1
        session_trades += len(batch)

        # Advance cursor past the last trade's timestamp.
        last_ts = batch[-1]["T"]
        if last_ts <= cursor:
            # No forward progress — break to avoid infinite loop.
            break
        cursor = last_ts + 1

        # Progress logging every 50 requests.
        if request_count % 50 == 0:
            pct = (cursor - start_ms) / max(1, end_ms - start_ms) * 100
            elapsed_s = time.time() - fetch_start_time
            eta_min = (elapsed_s * (100 - pct) / pct / 60) if pct > 0 else 0.0
            dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            logger.info(
                "  %d requests, %d trades (session), %.1f%% complete "
                "(up to %s) | elapsed=%.0fs, ETA=%.1fmin",
                request_count, session_trades, pct,
                dt.strftime("%Y-%m-%d %H:%M:%S UTC"), elapsed_s, eta_min,
            )

        # --- Rate-limit pause between successful calls (Task 2.14b-4) ---
        time.sleep(RATE_LIMIT_PAUSE_S)

    logger.info(
        "Session complete: %d trades in %d API requests.",
        session_trades, request_count,
    )
    return session_trades


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical tick data from Binance Futures for backtesting."
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT",
        help="Binance Futures symbol (default: BTCUSDT).",
    )
    parser.add_argument(
        "--hours", type=int, default=6,
        help="Hours of history to fetch (default: 6). Ignored if --start-date is set.",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date in YYYY-MM-DD format (UTC). Overrides --hours.",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date in YYYY-MM-DD format (UTC). Defaults to now if --start-date is set.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        stream=sys.stdout,
    )

    symbol = args.symbol.upper()

    # Determine time window.
    if args.start_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        start_ms = int(start_dt.timestamp() * 1000)

        if args.end_date:
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            # End date is inclusive: end of that day.
            end_dt = end_dt + timedelta(days=1) - timedelta(milliseconds=1)
            end_ms = int(end_dt.timestamp() * 1000)
        else:
            end_ms = int(time.time() * 1000)

        span_days = (end_ms - start_ms) / 86_400_000
        if span_days > 7.0:
            logger.error(
                "Date range exceeds 7-day maximum (%.1f days). Aborting.",
                span_days,
            )
            sys.exit(1)
        if span_days <= 0:
            logger.error("Invalid date range: start must be before end.")
            sys.exit(1)

        logger.info(
            "Fetching %.1f days of %s tick data from Binance Futures...",
            span_days, symbol,
        )
    else:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (args.hours * 3600 * 1000)
        logger.info(
            "Fetching %dh of %s tick data from Binance Futures...",
            args.hours, symbol,
        )

    logger.info(
        "Window: %s to %s",
        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
    )

    # Derive output filenames.
    base = symbol.replace("USDT", "").replace("USD", "").lower()
    if args.start_date:
        start_str = args.start_date.replace("-", "")
        if args.end_date:
            end_str = args.end_date.replace("-", "")
        else:
            end_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        csv_file = DATA_DIR / f"{base}_ticks_{start_str}_{end_str}.csv"
        gz_file = DATA_DIR / f"{base}_ticks_{start_str}_{end_str}.csv.gz"
    else:
        csv_file = DATA_DIR / f"{base}_ticks.csv"
        gz_file = DATA_DIR / f"{base}_ticks.csv.gz"

    # --- Auto-resume: check for existing checkpoint (Task 2.14b-2) ---
    effective_start_ms = start_ms
    resume_ts = get_resume_timestamp(csv_file)
    if resume_ts is not None:
        effective_start_ms = resume_ts + 1  # +1 to avoid re-fetching last trade.
        pct_done = (resume_ts - start_ms) / max(1, end_ms - start_ms) * 100
        logger.info(
            "RESUMING from checkpoint: last trade at %s (%.1f%% done). "
            "Skipping to ts=%d.",
            datetime.fromtimestamp(resume_ts / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            pct_done,
            effective_start_ms,
        )
    else:
        logger.info("No checkpoint found. Starting fresh.")

    if effective_start_ms >= end_ms:
        logger.info(
            "Checkpoint already covers the full window. Nothing to fetch."
        )
        # If plain CSV exists but gz doesn't, compress it.
        if csv_file.exists() and not gz_file.exists():
            compress_csv_to_gz(csv_file, gz_file)
        elif gz_file.exists():
            logger.info("Output already exists: %s", gz_file)
        return

    # --- Stream fetch with incremental saves ---
    writer = IncrementalCSVWriter(csv_file)
    try:
        session_trades = fetch_and_stream(
            symbol, effective_start_ms, end_ms, writer,
        )
    finally:
        writer.close()

    total_trades = writer.rows_written
    if resume_ts is not None:
        # Count pre-existing rows (lines minus header).
        with open(csv_file, "r", encoding="utf-8") as f:
            pre_existing = sum(1 for _ in f) - 1  # minus header
        total_trades = pre_existing
        logger.info(
            "Session fetched %d new trades. Total in file: %d.",
            session_trades, total_trades,
        )
    else:
        logger.info("Fetched %d trades total.", total_trades)

    if total_trades == 0:
        logger.error("No trade data collected. Check symbol and API status.")
        sys.exit(1)

    # --- Compress to gzip for the backtester ---
    compress_csv_to_gz(csv_file, gz_file)

    # --- Summary ---
    # Read first and last lines of gz for stats.
    first_ts = None
    last_ts = None
    tick_count = 0
    buy_count = 0
    sell_count = 0
    min_price = float("inf")
    max_price = float("-inf")

    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header.
        for row in reader:
            ts = int(row[0])
            price = float(row[1])
            side = row[3]
            if first_ts is None:
                first_ts = ts
            last_ts = ts
            tick_count += 1
            if side == "B":
                buy_count += 1
            else:
                sell_count += 1
            if price < min_price:
                min_price = price
            if price > max_price:
                max_price = price

    if first_ts and last_ts:
        span_hours = (last_ts - first_ts) / 3_600_000
        span_minutes = (last_ts - first_ts) / 60_000
        logger.info(
            "Done. %d ticks spanning %.1f hours (%.0f minutes).",
            tick_count, span_hours, span_minutes,
        )
        logger.info(
            "Price range: $%.2f - $%.2f | Buys: %d | Sells: %d | "
            "Buy%%: %.1f%%",
            min_price, max_price, buy_count, sell_count,
            buy_count / tick_count * 100 if tick_count > 0 else 0,
        )


if __name__ == "__main__":
    main()

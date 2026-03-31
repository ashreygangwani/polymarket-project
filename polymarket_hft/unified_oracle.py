"""
UnifiedOracle — Phase 1 Alpha Engine.

Multi-venue price oracle combining Binance (@bookTicker) and Hyperliquid
(l2Book + trades) into a synthetic fair price, with CVD and Z-Score signals.

Architecture:
    - One Binance WS connection (multiplexed @bookTicker streams)
    - One Hyperliquid WS connection (multiplexed l2Book + trades subscriptions)
    - One background sampler task (100ms) for Z-Score deque normalization
    - All per-asset state stored in self.state[asset] dict
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from typing import Any

import websockets
import websockets.exceptions

from config import (
    ASSET_MAPPING,
    BINANCE_WS_BASE,
    BN_WEIGHT,
    BN_WS_PING_INTERVAL_S,
    BN_WS_PING_TIMEOUT_S,
    CVD_WINDOW_S,
    HL_APP_PING_INTERVAL_S,
    HL_WEIGHT,
    HYPERLIQUID_WS,
    TRACKED_ASSETS,
    WS_RECONNECT_BACKOFF_FACTOR,
    WS_RECONNECT_DELAY_S,
    WS_RECONNECT_MAX_DELAY_S,
    ZSCORE_MIN_SAMPLES,
    ZSCORE_SAMPLE_INTERVAL_S,
    ZSCORE_WINDOW_SIZE,
)

logger = logging.getLogger("oracle")


class UnifiedOracle:
    """
    Maintains real-time per-asset state derived from two venue WebSockets.

    Public interface:
        state[asset]  — dict with synthetic_fair_price, cvd_delta, z_score
        get_snapshot(asset) — returns a frozen copy of the asset state
        is_ready(asset) — True once both venues have reported data
    """

    def __init__(self, assets: list[str] | None = None) -> None:
        self.assets: list[str] = assets or list(TRACKED_ASSETS)
        self._validate_assets()

        # Per-asset state: thread-safe within single asyncio loop.
        self.state: dict[str, dict[str, Any]] = {}
        self._cvd_windows: dict[str, deque] = {}
        self._zscore_windows: dict[str, deque] = {}

        for asset in self.assets:
            self.state[asset] = {
                "binance_best_bid": 0.0,
                "binance_best_ask": 0.0,
                "binance_mid": 0.0,
                "hl_best_bid": 0.0,
                "hl_best_ask": 0.0,
                "hl_mid": 0.0,
                "synthetic_fair_price": 0.0,
                "cvd_delta": 0.0,
                "z_score": 0.0,
                "last_update_bn": 0.0,
                "last_update_hl": 0.0,
            }
            # CVD: variable-rate (timestamp, signed_volume), pruned by time.
            self._cvd_windows[asset] = deque()
            # Z-Score: fixed-rate 100ms sampling, exactly 60s.
            self._zscore_windows[asset] = deque(maxlen=ZSCORE_WINDOW_SIZE)

        # Pre-built reverse lookups for O(1) message routing.
        self._hl_coin_map: dict[str, str] = {
            ASSET_MAPPING[a]["hyperliquid"]: a for a in self.assets
        }
        self._bn_ticker_map: dict[str, str] = {
            ASSET_MAPPING[a]["binance"].upper(): a for a in self.assets
        }

        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_snapshot(self, asset: str) -> dict[str, Any]:
        """Return a frozen copy of the asset state."""
        return dict(self.state[asset])

    def is_ready(self, asset: str) -> bool:
        """True once both Binance and Hyperliquid have sent at least one update."""
        s = self.state[asset]
        return s["last_update_bn"] > 0 and s["last_update_hl"] > 0

    async def start(self) -> None:
        """Launch all background tasks."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._run_binance_ws(), name="binance_ws"),
            asyncio.create_task(self._run_hyperliquid_ws(), name="hyperliquid_ws"),
            asyncio.create_task(self._zscore_sampler(), name="zscore_sampler"),
        ]
        logger.info(
            "UnifiedOracle started — tracking %s",
            ", ".join(self.assets),
        )

    async def stop(self) -> None:
        """Gracefully cancel all tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("UnifiedOracle stopped.")

    # ------------------------------------------------------------------
    # Binance WebSocket — @bookTicker (multiplexed)
    # ------------------------------------------------------------------

    def _build_binance_url(self) -> str:
        """Build a combined Binance stream URL for all tracked assets."""
        streams = []
        for asset in self.assets:
            ticker = ASSET_MAPPING[asset]["binance"]
            streams.append(f"{ticker}@bookTicker")
        return BINANCE_WS_BASE + "/".join(streams)

    async def _run_binance_ws(self) -> None:
        """Connect to Binance and process @bookTicker updates with auto-reconnect."""
        delay = WS_RECONNECT_DELAY_S
        url = self._build_binance_url()

        while self._running:
            try:
                logger.info("Binance WS connecting: %s", url[:80] + "...")
                async with websockets.connect(
                    url,
                    ping_interval=BN_WS_PING_INTERVAL_S,
                    ping_timeout=BN_WS_PING_TIMEOUT_S,
                ) as ws:
                    delay = WS_RECONNECT_DELAY_S  # Reset on success.
                    logger.info("Binance WS connected.")

                    async for raw_msg in ws:
                        try:
                            msg = json.loads(raw_msg)
                            data = msg.get("data")
                            if data is None:
                                continue

                            symbol = data.get("s", "")  # e.g. "BTCUSDT"
                            asset = self._bn_ticker_map.get(symbol)
                            if asset is None:
                                continue

                            bid = float(data["b"])
                            ask = float(data["a"])
                            mid = (bid + ask) / 2.0

                            s = self.state[asset]
                            s["binance_best_bid"] = bid
                            s["binance_best_ask"] = ask
                            s["binance_mid"] = mid
                            s["last_update_bn"] = time.time()

                            self._recompute_fair_price(asset)

                        except (KeyError, ValueError, TypeError) as exc:
                            logger.warning("Binance parse error: %s", exc)

            except asyncio.CancelledError:
                logger.info("Binance WS task cancelled.")
                return
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError,
            ) as exc:
                logger.warning("Binance WS disconnected: %s", exc)

            if self._running:
                logger.info("Binance WS reconnecting in %.1fs...", delay)
                await asyncio.sleep(delay)
                delay = min(delay * WS_RECONNECT_BACKOFF_FACTOR, WS_RECONNECT_MAX_DELAY_S)

    # ------------------------------------------------------------------
    # Hyperliquid WebSocket — l2Book + trades (multiplexed)
    # ------------------------------------------------------------------

    async def _run_hyperliquid_ws(self) -> None:
        """Connect to Hyperliquid and subscribe to l2Book + trades per asset."""
        delay = WS_RECONNECT_DELAY_S

        while self._running:
            ping_task: asyncio.Task | None = None
            try:
                logger.info("Hyperliquid WS connecting...")
                # Disable protocol-level pings — Hyperliquid ignores them.
                # We use application-level {"method":"ping"} instead.
                async with websockets.connect(
                    HYPERLIQUID_WS,
                    ping_interval=None,
                ) as ws:
                    delay = WS_RECONNECT_DELAY_S
                    logger.info("Hyperliquid WS connected.")

                    # Subscribe to l2Book and trades for each asset on the
                    # same connection (multiplexing).
                    for asset in self.assets:
                        coin = ASSET_MAPPING[asset]["hyperliquid"]
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "l2Book", "coin": coin},
                        }))
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "trades", "coin": coin},
                        }))
                    logger.info(
                        "Hyperliquid subscribed to l2Book + trades for %s",
                        ", ".join(self.assets),
                    )

                    # Launch application-level ping keepalive.
                    ping_task = asyncio.create_task(
                        self._hl_ping_keepalive(ws), name="hl_oracle_ping"
                    )

                    async for raw_msg in ws:
                        try:
                            msg = json.loads(raw_msg)
                            channel = msg.get("channel")
                            data = msg.get("data")
                            if channel is None or data is None:
                                # Ignore pong responses and subscription acks.
                                continue

                            if channel == "l2Book":
                                self._handle_hl_l2book(data)
                            elif channel == "trades":
                                self._handle_hl_trades(data)

                        except (KeyError, ValueError, TypeError) as exc:
                            logger.warning("Hyperliquid parse error: %s", exc)

            except asyncio.CancelledError:
                logger.info("Hyperliquid WS task cancelled.")
                if ping_task:
                    ping_task.cancel()
                return
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError,
            ) as exc:
                logger.warning("Hyperliquid WS disconnected: %s", exc)
            finally:
                if ping_task and not ping_task.done():
                    ping_task.cancel()
                    await asyncio.gather(ping_task, return_exceptions=True)

            if self._running:
                logger.info("Hyperliquid WS reconnecting in %.1fs...", delay)
                await asyncio.sleep(delay)
                delay = min(delay * WS_RECONNECT_BACKOFF_FACTOR, WS_RECONNECT_MAX_DELAY_S)

    async def _hl_ping_keepalive(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Send {"method":"ping"} every HL_APP_PING_INTERVAL_S to keep alive."""
        try:
            while True:
                await asyncio.sleep(HL_APP_PING_INTERVAL_S)
                await ws.send(json.dumps({"method": "ping"}))
                logger.debug("Hyperliquid Oracle ping sent.")
        except asyncio.CancelledError:
            pass
        except (
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.WebSocketException,
        ):
            pass  # Connection dropped — outer loop handles reconnect.

    def _handle_hl_l2book(self, data: dict) -> None:
        """
        Process Hyperliquid l2Book snapshot/update.

        Expected format:
        {
            "coin": "BTC",
            "levels": [
                [{"px": "93200.0", "sz": "0.5", "n": 12}, ...],  # bids
                [{"px": "93201.0", "sz": "0.4", "n": 10}, ...]   # asks
            ]
        }
        """
        coin = data.get("coin", "")
        asset = self._hl_coin_to_asset(coin)
        if asset is None:
            return

        levels = data.get("levels", [])
        if len(levels) < 2 or not levels[0] or not levels[1]:
            return

        best_bid = float(levels[0][0]["px"])
        best_ask = float(levels[1][0]["px"])
        mid = (best_bid + best_ask) / 2.0

        s = self.state[asset]
        s["hl_best_bid"] = best_bid
        s["hl_best_ask"] = best_ask
        s["hl_mid"] = mid
        s["last_update_hl"] = time.time()

        self._recompute_fair_price(asset)

    def _handle_hl_trades(self, data: list | dict) -> None:
        """
        Process Hyperliquid trades for CVD calculation.

        Expected format (list of trades):
        [
            {
                "coin": "BTC",
                "side": "B",  # "B" = buy (hit ask), "A" = sell (hit bid)
                "px": "93200.0",
                "sz": "0.5",
                "time": 1704067200000
            },
            ...
        ]
        """
        # Hyperliquid sends trades as a list.
        trades = data if isinstance(data, list) else [data]
        now = time.time()

        for trade in trades:
            coin = trade.get("coin", "")
            asset = self._hl_coin_to_asset(coin)
            if asset is None:
                continue

            side = trade.get("side", "")
            px = float(trade.get("px", 0))
            sz = float(trade.get("sz", 0))
            notional = px * sz

            # CVD: buys (hitting ask) are positive, sells (hitting bid) negative.
            # Explicitly reject unknown side values to prevent CVD corruption.
            if side == "B":
                signed_volume = notional
            elif side == "A":
                signed_volume = -notional
            else:
                logger.warning("Unknown HL trade side '%s', skipping.", side)
                continue

            self._cvd_windows[asset].append((now, signed_volume))
            self._prune_cvd_window(asset, now)
            self._recompute_cvd(asset, now)

    # ------------------------------------------------------------------
    # Fair Price Computation
    # ------------------------------------------------------------------

    def _recompute_fair_price(self, asset: str) -> None:
        """
        synthetic_fair_price = (HL_mid * 0.65) + (BN_mid * 0.35)

        Only recalculates if both venues have reported at least once.
        """
        s = self.state[asset]
        hl_mid = s["hl_mid"]
        bn_mid = s["binance_mid"]

        if hl_mid <= 0 or bn_mid <= 0:
            return

        s["synthetic_fair_price"] = (hl_mid * HL_WEIGHT) + (bn_mid * BN_WEIGHT)

    # ------------------------------------------------------------------
    # CVD Engine
    # ------------------------------------------------------------------

    def _prune_cvd_window(self, asset: str, now: float) -> None:
        """Remove entries older than CVD_WINDOW_S from the deque."""
        window = self._cvd_windows[asset]
        cutoff = now - CVD_WINDOW_S
        while window and window[0][0] < cutoff:
            window.popleft()

    def _recompute_cvd(self, asset: str, now: float) -> None:
        """Sum signed volumes over the rolling 60s window."""
        total = 0.0
        for _, vol in self._cvd_windows[asset]:
            total += vol
        self.state[asset]["cvd_delta"] = total

    # ------------------------------------------------------------------
    # Z-Score Sampler (100ms fixed-rate)
    # ------------------------------------------------------------------

    async def _zscore_sampler(self) -> None:
        """
        Background task: every 100ms, sample the current synthetic_fair_price
        for each asset and append to the fixed-size deque. Then recalculate
        the Z-Score.

        This guarantees exactly 60s of normalized data regardless of
        variable tick rates from the WebSockets.
        """
        logger.info("Z-Score sampler started (%.0fms interval).", ZSCORE_SAMPLE_INTERVAL_S * 1000)
        try:
            while self._running:
                for asset in self.assets:
                    price = self.state[asset]["synthetic_fair_price"]
                    if price > 0:
                        self._zscore_windows[asset].append(price)
                        self._recompute_zscore(asset)

                await asyncio.sleep(ZSCORE_SAMPLE_INTERVAL_S)
        except asyncio.CancelledError:
            logger.info("Z-Score sampler cancelled.")

    def _recompute_zscore(self, asset: str) -> None:
        """
        Z = (current_price - rolling_mean) / rolling_std

        Uses an O(n) pass over the deque. With n=600 and 64-bit floats,
        this completes in <10 microseconds — safe for the 100ms loop.
        """
        window = self._zscore_windows[asset]
        n = len(window)
        if n < ZSCORE_MIN_SAMPLES:
            self.state[asset]["z_score"] = 0.0
            return

        # Full-pass Welford recurrence (numerically stable).
        # Not incremental because deque(maxlen=) silently evicts old values
        # with no callback. At n=600 this completes in <10us.
        # Uses sample variance (n-1) for Bessel's correction.
        mean = 0.0
        m2 = 0.0
        for i, val in enumerate(window, 1):
            delta = val - mean
            mean += delta / i
            delta2 = val - mean
            m2 += delta * delta2

        variance = m2 / (n - 1)  # Bessel's correction (sample variance)
        std = math.sqrt(variance) if variance > 0 else 0.0

        current_price = window[-1]
        if std > 0:
            self.state[asset]["z_score"] = (current_price - mean) / std
        else:
            self.state[asset]["z_score"] = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hl_coin_to_asset(self, coin: str) -> str | None:
        """Reverse lookup: Hyperliquid coin symbol → base asset. O(1)."""
        return self._hl_coin_map.get(coin)

    def _validate_assets(self) -> None:
        """Ensure all tracked assets have valid mappings."""
        for asset in self.assets:
            if asset not in ASSET_MAPPING:
                raise ValueError(
                    f"Asset '{asset}' not found in ASSET_MAPPING. "
                    f"Available: {list(ASSET_MAPPING.keys())}"
                )

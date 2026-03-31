"""
RiskMonitor — Phase 1 Defensive Layer.

Aggregates liquidation data from two independent sources:
    1. Hyperliquid WebSocket (native liquidations subscription)
    2. Coinglass REST API (cross-venue: Binance, Bybit, OKX, etc.)

Sets TOXIC_FLOW_ACTIVE = True if aggregate liquidation volume exceeds
$1M within a rolling 5-second window from either or both sources.

Architecture:
    - Hyperliquid liquidations: piggybacked on the existing WS connection
      OR run as a standalone subscription (we use standalone for isolation).
    - Coinglass: async REST polling via aiohttp with a hard 500ms timeout.
    - Both sources feed into a single shared deque for aggregation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any

import aiohttp
import websockets
import websockets.exceptions

from config import (
    COINGLASS_API_KEY,
    COINGLASS_BASE_URL,
    COINGLASS_LIQUIDATION_ENDPOINT,
    COINGLASS_POLL_INTERVAL_S,
    COINGLASS_TIMEOUT_S,
    HL_APP_PING_INTERVAL_S,
    HYPERLIQUID_WS,
    TOXIC_FLOW_THRESHOLD_USD,
    TOXIC_FLOW_WINDOW_S,
    WS_RECONNECT_BACKOFF_FACTOR,
    WS_RECONNECT_DELAY_S,
    WS_RECONNECT_MAX_DELAY_S,
)

logger = logging.getLogger("risk_monitor")


class RiskMonitor:
    """
    Monitors cross-venue liquidation cascades and exposes a
    TOXIC_FLOW_ACTIVE flag for the execution engine.

    Public interface:
        toxic_flow_active   — bool property
        recent_liquidations — list of recent liquidation events
        start()             — launch background tasks
        stop()              — gracefully shut down
    """

    def __init__(self) -> None:
        # Rolling window: (timestamp, usd_value, source)
        self._liq_window: deque = deque()
        self._toxic_flow_active: bool = False
        self._running: bool = False
        self._tasks: list[asyncio.Task] = []

        # Dedup: track last-seen Coinglass liquidation timestamps
        # to avoid double-counting on overlapping poll windows.
        self._cg_last_seen_ts: float = 0.0

        # Metrics for logging.
        self._total_hl_liqs: int = 0
        self._total_cg_liqs: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def toxic_flow_active(self) -> bool:
        """True if aggregate liquidations exceed threshold in the rolling window."""
        return self._toxic_flow_active

    @property
    def recent_liquidations(self) -> list[dict[str, Any]]:
        """Return a snapshot of the current rolling window for inspection."""
        return [
            {"timestamp": ts, "usd_value": val, "source": src}
            for ts, val, src in self._liq_window
        ]

    def get_window_total_usd(self) -> float:
        """Current aggregate USD in the rolling window."""
        return sum(val for _, val, _ in self._liq_window)

    async def start(self) -> None:
        """Launch Hyperliquid WS and Coinglass poller as background tasks."""
        self._running = True
        self._tasks = [
            asyncio.create_task(
                self._run_hl_liquidation_ws(), name="hl_liquidation_ws"
            ),
            asyncio.create_task(
                self._run_coinglass_poller(), name="coinglass_poller"
            ),
        ]
        logger.info(
            "RiskMonitor started — toxic flow threshold: $%s in %.0fs window.",
            f"{TOXIC_FLOW_THRESHOLD_USD:,.0f}",
            TOXIC_FLOW_WINDOW_S,
        )

    async def stop(self) -> None:
        """Cancel all background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info(
            "RiskMonitor stopped. Lifetime liquidations ingested: "
            "HL=%d, Coinglass=%d.",
            self._total_hl_liqs,
            self._total_cg_liqs,
        )

    # ------------------------------------------------------------------
    # Hyperliquid Liquidation WebSocket
    # ------------------------------------------------------------------

    async def _run_hl_liquidation_ws(self) -> None:
        """
        Subscribe to Hyperliquid's 'liquidations' channel.

        Message format:
        {
            "channel": "liquidations",
            "data": {
                "liq": {
                    "coin": "BTC",
                    "side": "B",          # B=buy (short liq), A=sell (long liq)
                    "px": "93200.0",
                    "sz": "0.5"
                },
                "time": 1704067200000
            }
        }

        Some Hyperliquid versions send data as a list; we handle both.
        """
        delay = WS_RECONNECT_DELAY_S

        while self._running:
            ping_task: asyncio.Task | None = None
            try:
                logger.info("RiskMonitor: Hyperliquid liquidation WS connecting...")
                # Disable protocol-level pings — Hyperliquid ignores them.
                async with websockets.connect(
                    HYPERLIQUID_WS,
                    ping_interval=None,
                ) as ws:
                    delay = WS_RECONNECT_DELAY_S
                    # Subscribe to all liquidations (no coin filter = all coins).
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "liquidations"},
                    }))
                    logger.info("RiskMonitor: Subscribed to Hyperliquid liquidations.")

                    # Launch application-level ping keepalive.
                    ping_task = asyncio.create_task(
                        self._hl_ping_keepalive(ws), name="hl_risk_ping"
                    )

                    async for raw_msg in ws:
                        try:
                            msg = json.loads(raw_msg)
                            channel = msg.get("channel")
                            if channel != "liquidations":
                                continue

                            data = msg.get("data")
                            if data is None:
                                continue

                            self._process_hl_liquidation(data)

                        except (KeyError, ValueError, TypeError) as exc:
                            logger.warning(
                                "RiskMonitor: HL liquidation parse error: %s", exc
                            )

            except asyncio.CancelledError:
                logger.info("RiskMonitor: HL liquidation WS task cancelled.")
                if ping_task:
                    ping_task.cancel()
                return
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError,
            ) as exc:
                logger.warning(
                    "RiskMonitor: HL liquidation WS disconnected: %s", exc
                )
            finally:
                if ping_task and not ping_task.done():
                    ping_task.cancel()
                    await asyncio.gather(ping_task, return_exceptions=True)

            if self._running:
                logger.info(
                    "RiskMonitor: HL WS reconnecting in %.1fs...", delay
                )
                await asyncio.sleep(delay)
                delay = min(
                    delay * WS_RECONNECT_BACKOFF_FACTOR,
                    WS_RECONNECT_MAX_DELAY_S,
                )

    async def _hl_ping_keepalive(
        self, ws: websockets.WebSocketClientProtocol
    ) -> None:
        """Send {"method":"ping"} every HL_APP_PING_INTERVAL_S to keep alive."""
        try:
            while True:
                await asyncio.sleep(HL_APP_PING_INTERVAL_S)
                await ws.send(json.dumps({"method": "ping"}))
                logger.debug("RiskMonitor: HL ping sent.")
        except asyncio.CancelledError:
            pass
        except (
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.WebSocketException,
        ):
            pass  # Connection dropped — outer loop handles reconnect.

    def _process_hl_liquidation(self, data: dict | list) -> None:
        """Extract USD notional from Hyperliquid liquidation events."""
        now = time.time()
        events = data if isinstance(data, list) else [data]

        for event in events:
            liq = event.get("liq", event)
            px = float(liq.get("px", 0))
            sz = float(liq.get("sz", 0))
            notional = px * sz

            if notional > 0:
                self._ingest_liquidation(now, notional, "hyperliquid")
                self._total_hl_liqs += 1

                coin = liq.get("coin", "?")
                side = "SHORT_LIQ" if liq.get("side") == "B" else "LONG_LIQ"
                logger.debug(
                    "HL LIQ: %s %s $%s @ %s",
                    coin, side, f"{notional:,.0f}", px,
                )

    # ------------------------------------------------------------------
    # Coinglass REST Poller
    # ------------------------------------------------------------------

    async def _run_coinglass_poller(self) -> None:
        """
        Poll Coinglass liquidation endpoint every COINGLASS_POLL_INTERVAL_S.
        Hard timeout of 500ms per request to never block the event loop.
        """
        if not COINGLASS_API_KEY:
            logger.warning(
                "RiskMonitor: COINGLASS_API_KEY not set in .env — "
                "Coinglass poller disabled. Only Hyperliquid liquidations active."
            )
            return

        timeout = aiohttp.ClientTimeout(total=COINGLASS_TIMEOUT_S)
        headers = {
            "accept": "application/json",
            "CG-API-KEY": COINGLASS_API_KEY,
        }
        url = f"{COINGLASS_BASE_URL}{COINGLASS_LIQUIDATION_ENDPOINT}"

        logger.info(
            "RiskMonitor: Coinglass poller started (%.1fs interval, %.0fms timeout).",
            COINGLASS_POLL_INTERVAL_S,
            COINGLASS_TIMEOUT_S * 1000,
        )

        try:
            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers
            ) as session:
                while self._running:
                    try:
                        await self._poll_coinglass(session, url)
                    except asyncio.TimeoutError:
                        logger.debug("RiskMonitor: Coinglass request timed out (500ms).")
                    except aiohttp.ClientError as exc:
                        logger.warning("RiskMonitor: Coinglass HTTP error: %s", exc)
                    except (KeyError, ValueError, TypeError) as exc:
                        logger.warning("RiskMonitor: Coinglass parse error: %s", exc)

                    await asyncio.sleep(COINGLASS_POLL_INTERVAL_S)

        except asyncio.CancelledError:
            logger.info("RiskMonitor: Coinglass poller cancelled.")

    async def _poll_coinglass(
        self, session: aiohttp.ClientSession, url: str
    ) -> None:
        """
        Single Coinglass poll cycle.

        Coinglass v3 real-time liquidation response format:
        {
            "code": "0",
            "msg": "success",
            "data": [
                {
                    "symbol": "BTCUSDT",
                    "longVolUsd": 125000.0,
                    "shortVolUsd": 89000.0,
                    "longCount": 15,
                    "shortCount": 8,
                    "createTime": 1704067200000
                },
                ...
            ]
        }
        """
        async with session.get(url) as resp:
            if resp.status != 200:
                logger.warning(
                    "RiskMonitor: Coinglass returned HTTP %d.", resp.status
                )
                return

            body = await resp.json()

        if body.get("code") != "0":
            logger.warning(
                "RiskMonitor: Coinglass API error: %s", body.get("msg")
            )
            return

        data_list = body.get("data", [])
        if not data_list:
            return

        now = time.time()

        for entry in data_list:
            create_time_ms = entry.get("createTime", 0)
            create_time_s = create_time_ms / 1000.0

            # Dedup: skip entries we've already processed.
            if create_time_s <= self._cg_last_seen_ts:
                continue

            long_vol = float(entry.get("longVolUsd", 0))
            short_vol = float(entry.get("shortVolUsd", 0))
            total_vol = long_vol + short_vol

            if total_vol > 0:
                self._ingest_liquidation(now, total_vol, "coinglass")
                self._total_cg_liqs += 1

                symbol = entry.get("symbol", "?")
                logger.debug(
                    "CG LIQ: %s long=$%s short=$%s total=$%s",
                    symbol,
                    f"{long_vol:,.0f}",
                    f"{short_vol:,.0f}",
                    f"{total_vol:,.0f}",
                )

        # Advance dedup watermark.
        if data_list:
            max_ts = max(
                e.get("createTime", 0) / 1000.0 for e in data_list
            )
            if max_ts > self._cg_last_seen_ts:
                self._cg_last_seen_ts = max_ts

    # ------------------------------------------------------------------
    # Shared Liquidation Aggregation
    # ------------------------------------------------------------------

    def _ingest_liquidation(
        self, timestamp: float, usd_value: float, source: str
    ) -> None:
        """Add a liquidation event and recompute the toxic flow flag."""
        self._liq_window.append((timestamp, usd_value, source))
        self._prune_window(timestamp)
        self._evaluate_toxic_flow()

    def _prune_window(self, now: float) -> None:
        """Remove entries older than TOXIC_FLOW_WINDOW_S."""
        cutoff = now - TOXIC_FLOW_WINDOW_S
        while self._liq_window and self._liq_window[0][0] < cutoff:
            self._liq_window.popleft()

    def _evaluate_toxic_flow(self) -> None:
        """Check if aggregate liquidation volume exceeds threshold."""
        total = sum(val for _, val, _ in self._liq_window)
        was_active = self._toxic_flow_active
        self._toxic_flow_active = total >= TOXIC_FLOW_THRESHOLD_USD

        if self._toxic_flow_active and not was_active:
            logger.critical(
                "TOXIC FLOW ACTIVATED — $%s liquidations in %.0fs window "
                "(threshold: $%s). KILL SWITCH ARMED.",
                f"{total:,.0f}",
                TOXIC_FLOW_WINDOW_S,
                f"{TOXIC_FLOW_THRESHOLD_USD:,.0f}",
            )
        elif not self._toxic_flow_active and was_active:
            logger.info(
                "TOXIC FLOW CLEARED — $%s in window (below $%s threshold).",
                f"{total:,.0f}",
                f"{TOXIC_FLOW_THRESHOLD_USD:,.0f}",
            )

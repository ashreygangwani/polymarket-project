"""
Live Execution Engine — Phase 3.

Translates the backtested Run 5 logic into a live, asynchronous trading bot
using the official py-clob-client for Polymarket.

Features:
    - Dynamic Volatility Quoting (0.5c - 3.0c based on Z-Score) via POST_ONLY
    - Inventory Skew (asymmetric quoting)
    - Global Stop-Loss (8c, with FOK dump)
    - Post-Stop Cooldown (5-minute penalty box)
    - Macro Circuit Breaker ($25 loss -> halt + exit)
    - Gas Guard (pause if Polygon gas > 300 gwei)
    - Toxic Flow Kill Switch (from Phase 1 RiskMonitor)
    - Universal Market Scanner (find best BTC market by volume)

Usage:
    python live_execution.py --asset BTC --dry-run    # Paper trading
    python live_execution.py --asset BTC               # LIVE (real money)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from dotenv import load_dotenv

from config import STARTING_CAPITAL_USDC
from risk_monitor import RiskMonitor
from unified_oracle import UnifiedOracle

load_dotenv()

# Defer py-clob-client import for clean error message.
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        ApiCreds,
        AssetType,
        BalanceAllowanceParams,
        MarketOrderArgs,
        OrderArgs,
        OrderType,
    )
    from py_clob_client.constants import POLYGON
    from py_clob_client.order_builder.constants import BUY, SELL
except ImportError:
    print(
        "ERROR: py-clob-client not installed.\n"
        "  pip install py-clob-client\n"
        "See https://github.com/Polymarket/py-clob-client",
        file=sys.stderr,
    )
    sys.exit(1)

logger = logging.getLogger("live_exec")


# ---------------------------------------------------------------------------
# Run 5 Production Parameters (identical to backtest)
# ---------------------------------------------------------------------------

STRATEGY_EVAL_INTERVAL_S = 1.0      # Re-evaluate every 1 second
GAS_CHECK_INTERVAL_S = 30.0         # Check gas every 30 seconds
MAX_GAS_GWEI = 300                  # Pause orders if gas exceeds this

COOLDOWN_DURATION_S = 300.0         # 5-minute post-stop penalty box
MACRO_BREAKER_LIMIT = -25.0         # 5% of $500 starting capital
GLOBAL_STOP_DISTANCE = 0.08         # 8 cents in probability units

INVENTORY_LIMIT_USDC = 40.0         # Max maker inventory before bid-pause
MAX_SKEW_CENTS = 0.03               # Maximum skew at full inventory
MAX_EXPOSURE_USDC = 75.0            # Total position cap
MAKER_SIZE_USDC = 25.0              # USDC per side for maker orders
DIRECTIONAL_SIZE_USDC = 30.0        # USDC for directional entries

CVD_THRESHOLD = 3_000_000.0         # |CVD| > $3M for directional trigger
ZSCORE_THRESHOLD = 2.0              # Z > 2.0 for directional trigger

FILL_POLL_INTERVAL_S = 2.0          # Check for order fills every 2 seconds
BALANCE_RECONCILE_INTERVAL_S = 60.0 # Reconcile balances every 60 seconds

POLY_GAMMA_HOST = "https://gamma-api.polymarket.com"
POLYGON_RPC_URL = os.getenv(
    "POLYGON_RPC_URL", "https://polygon-rpc.com"
)


# ---------------------------------------------------------------------------
# Fee Curve — identical to simulator/clob_env.py
# ---------------------------------------------------------------------------

def taker_fee_rate(probability: float) -> float:
    """Polymarket 2026: fee = 2 * 0.0156 * min(p, 1-p)."""
    return 2.0 * 0.0156 * min(probability, 1.0 - probability)


def maker_rebate_rate(probability: float) -> float:
    """Maker rebate = 20% of the taker fee that was collected."""
    return 0.20 * taker_fee_rate(probability)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MarketInfo:
    """Active Polymarket market selected for trading."""
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    tick_size: str = "0.01"
    neg_risk: bool = False
    min_order_size: float = 1.0


@dataclass
class LivePosition:
    """Tracks an open position (mirrors simulator.clob_env.Position)."""
    entry_price: float      # Probability at entry
    entry_usdc: float       # USDC spent
    shares: float           # YES tokens held
    entry_time: float       # Unix timestamp
    scaled_out: bool = False
    runner_shares: float = 0.0


@dataclass
class TrackedOrder:
    """A resting order we placed on Polymarket."""
    order_id: str
    side: str               # BUY or SELL
    price: float
    size: float             # Shares
    size_usdc: float
    placed_at: float        # Unix timestamp


# ---------------------------------------------------------------------------
# Dynamic Spread — identical to backtest.py compute_dynamic_spread()
# ---------------------------------------------------------------------------

def compute_dynamic_spread(
    z_score: float, has_inventory: bool,
) -> float | None:
    """
    Adapt maker spread to volatility regime based on Z-Score.

    |Z| < 1.0:  0.005 (0.5c) — quiet, farm rebates
    1.0-2.0:    0.015 (1.5c) — normal
    |Z| > 2.0:  0.030 (3.0c) or None (pause if holding inventory)
    """
    abs_z = abs(z_score)
    if abs_z < 1.0:
        return 0.005
    elif abs_z <= 2.0:
        return 0.015
    else:
        if has_inventory:
            return None  # Pause quoting — avoid adverse selection.
        return 0.030


# ---------------------------------------------------------------------------
# Gas Guard (Task 3.1)
# ---------------------------------------------------------------------------

async def check_gas(
    session: aiohttp.ClientSession,
    rpc_url: str,
) -> tuple[bool, float]:
    """
    Query Polygon RPC for current gas price.

    Returns:
        (is_safe, gas_gwei) — is_safe is False if gas > MAX_GAS_GWEI.
    """
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_gasPrice",
        "params": [],
        "id": 1,
    }
    try:
        async with session.post(
            rpc_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=5.0),
        ) as resp:
            data = await resp.json()
            gas_wei = int(data["result"], 16)
            gas_gwei = gas_wei / 1e9
            return gas_gwei <= MAX_GAS_GWEI, gas_gwei
    except Exception as exc:
        logger.warning("Gas check failed: %s — assuming safe.", exc)
        return True, 0.0


# ---------------------------------------------------------------------------
# Universal Market Scanner (Task 3.4)
# ---------------------------------------------------------------------------

async def find_best_market(
    session: aiohttp.ClientSession,
    asset: str = "BTC",
) -> MarketInfo | None:
    """
    Search Polymarket Gamma API for the active crypto market with the
    highest 24h volume matching the given asset.

    Prioritizes 15-minute and 1-hour markets (10x volume boost).
    Falls back to any active market containing the asset keyword.
    """
    search_url = f"{POLY_GAMMA_HOST}/markets"
    params = {
        "closed": "false",
        "limit": "100",
        "order": "volume24hr",
        "ascending": "false",
    }

    try:
        async with session.get(
            search_url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10.0),
        ) as resp:
            if resp.status != 200:
                logger.error(
                    "Gamma API returned HTTP %d during market scan.", resp.status,
                )
                return None
            markets = await resp.json()
    except Exception as exc:
        logger.error("Market scan failed: %s", exc)
        return None

    if not isinstance(markets, list):
        logger.error("Unexpected Gamma API response type: %s", type(markets))
        return None

    # Keywords to match our target asset.
    keywords = [asset.upper(), asset.lower()]
    if asset.upper() == "BTC":
        keywords.extend(["Bitcoin", "bitcoin"])
    elif asset.upper() == "ETH":
        keywords.extend(["Ethereum", "ethereum", "Ether"])

    # Short-term market indicators (for volume boost).
    short_term_terms = [
        "15 min", "15-min", "15min",
        "1 hour", "1-hour", "1hr", "hourly",
    ]

    best: MarketInfo | None = None
    best_score = -1.0

    for mkt in markets:
        question = mkt.get("question", "")

        # Must mention our asset.
        if not any(kw in question for kw in keywords):
            continue

        volume = float(mkt.get("volume24hr", 0) or 0)
        tokens = mkt.get("tokens", [])
        condition_id = mkt.get("condition_id", "")

        if not tokens or not condition_id:
            continue

        # 10x boost for short-term markets.
        is_short_term = any(t in question.lower() for t in short_term_terms)
        score = volume * (10.0 if is_short_term else 1.0)

        if score > best_score:
            # Identify YES and NO tokens.
            yes_token = ""
            no_token = ""
            for tok in tokens:
                outcome = str(tok.get("outcome", "")).upper()
                token_id = tok.get("token_id", "")
                if outcome == "YES":
                    yes_token = token_id
                elif outcome == "NO":
                    no_token = token_id

            if yes_token and no_token:
                best_score = score
                best = MarketInfo(
                    condition_id=condition_id,
                    question=question,
                    yes_token_id=yes_token,
                    no_token_id=no_token,
                    min_order_size=float(mkt.get("minimum_order_size", 1) or 1),
                )

    if best:
        logger.info(
            "MARKET SELECTED: '%s' (condition=%s..., vol=$%.0f)",
            best.question, best.condition_id[:16], best_score,
        )
    else:
        logger.error("No active %s markets found on Polymarket.", asset)

    return best


# ---------------------------------------------------------------------------
# Client Initialization (Task 3.1)
# ---------------------------------------------------------------------------

def init_clob_client() -> ClobClient:
    """
    Initialize the py-clob-client using credentials from .env.

    Required .env variables:
        POLY_PK          — Polygon wallet private key (hex)
        POLY_HOST        — CLOB API host (default: https://clob.polymarket.com)
        CHAIN_ID         — Polygon chain ID (default: 137)
        CLOB_API_KEY     — API key (from derive_api_key())
        CLOB_SECRET      — API secret
        CLOB_PASS_PHRASE — API passphrase

    Optional:
        POLYGON_RPC_URL  — Dedicated RPC endpoint (Alchemy/QuickNode)
    """
    pk = os.getenv("POLY_PK")
    host = os.getenv("POLY_HOST", "https://clob.polymarket.com")
    chain_id = int(os.getenv("CHAIN_ID", str(POLYGON)))

    if not pk:
        logger.error("POLY_PK not set in .env. Cannot initialize client.")
        sys.exit(1)

    api_key = os.getenv("CLOB_API_KEY", "")
    api_secret = os.getenv("CLOB_SECRET", "")
    api_passphrase = os.getenv("CLOB_PASS_PHRASE", "")

    if api_key and api_secret and api_passphrase:
        # Full L2 client — can trade.
        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
        client = ClobClient(host, key=pk, chain_id=chain_id, creds=creds)
        logger.info(
            "CLOB client initialized (L2). Host=%s, Chain=%d.", host, chain_id,
        )
    else:
        # L1 client — derive API key automatically.
        client = ClobClient(host, key=pk, chain_id=chain_id)
        logger.warning("No API credentials in .env. Deriving API key...")
        try:
            creds_dict = client.derive_api_key()
            logger.info(
                "API key derived. Add these to .env for future runs:\n"
                "  CLOB_API_KEY=%s\n"
                "  CLOB_SECRET=%s\n"
                "  CLOB_PASS_PHRASE=%s",
                creds_dict.get("apiKey", ""),
                creds_dict.get("secret", ""),
                creds_dict.get("passphrase", ""),
            )
            creds = ApiCreds(
                api_key=creds_dict["apiKey"],
                api_secret=creds_dict["secret"],
                api_passphrase=creds_dict["passphrase"],
            )
            client = ClobClient(host, key=pk, chain_id=chain_id, creds=creds)
        except Exception as exc:
            logger.error("API key derivation failed: %s", exc)
            sys.exit(1)

    return client


# ---------------------------------------------------------------------------
# Live Execution Engine
# ---------------------------------------------------------------------------

class LiveExecutionEngine:
    """
    Production execution engine implementing Run 5 strategy.

    Translates the exact backtested logic into live Polymarket orders:
        - Dynamic Volatility Quoting via POST_ONLY maker orders
        - Inventory Skew (asymmetric quoting to reduce adverse selection)
        - Global Stop-Loss (8c breach triggers FOK dump)
        - Post-Stop Cooldown (5-minute bid-side lockout)
        - Macro Circuit Breaker ($25 realized loss halts all trading)
        - Toxic Flow Kill Switch (from Phase 1 RiskMonitor)
    """

    def __init__(
        self,
        client: ClobClient,
        oracle: UnifiedOracle,
        risk_monitor: RiskMonitor,
        market: MarketInfo,
        asset: str = "BTC",
        dry_run: bool = False,
    ) -> None:
        self.client = client
        self.oracle = oracle
        self.risk_monitor = risk_monitor
        self.market = market
        self.asset = asset
        self.dry_run = dry_run

        # Account state.
        self.startup_balance: float = 0.0
        self.usdc_balance: float = 0.0
        self.yes_shares: float = 0.0

        # Position tracking (FIFO, mirrors simulator).
        self.positions: list[LivePosition] = []

        # Order tracking (order_id -> TrackedOrder).
        self.tracked_orders: dict[str, TrackedOrder] = {}

        # Risk state.
        self.realized_pnl: float = 0.0
        self.last_stop_loss_time: float = 0.0
        self.macro_breaker_tripped: bool = False
        self.in_directional_mode: bool = False

        # Gas guard state.
        self._last_gas_check: float = 0.0
        self._gas_safe: bool = True
        self._gas_gwei: float = 0.0

        # Balance reconciliation.
        self._last_balance_reconcile: float = 0.0

        # Lifecycle.
        self._running: bool = False
        self._http_session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize state and fetch starting balances."""
        self._running = True
        self._http_session = aiohttp.ClientSession()

        # Fetch initial USDC balance (Task 3.3).
        self.startup_balance = await self._fetch_usdc_balance()
        self.usdc_balance = self.startup_balance

        # Fetch initial YES token balance.
        self.yes_shares = await self._fetch_yes_balance()

        logger.info(
            "LiveExecutionEngine started. Market: '%s'", self.market.question,
        )
        logger.info(
            "  Startup USDC: $%.2f | YES shares: %.4f | Dry run: %s",
            self.startup_balance, self.yes_shares, self.dry_run,
        )
        logger.info(
            "  Macro breaker at: $%.2f loss | Cooldown: %ds | "
            "Global stop: %.0fc | Inventory limit: $%.0f",
            abs(MACRO_BREAKER_LIMIT), int(COOLDOWN_DURATION_S),
            GLOBAL_STOP_DISTANCE * 100, INVENTORY_LIMIT_USDC,
        )

    async def stop(self) -> None:
        """Cancel all orders and shut down cleanly."""
        self._running = False

        # Cancel all resting orders on exchange.
        try:
            if not self.dry_run:
                await asyncio.to_thread(self.client.cancel_all)
            logger.info("All resting orders cancelled on shutdown.")
        except Exception as exc:
            logger.error("Error cancelling orders on shutdown: %s", exc)

        if self._http_session:
            await self._http_session.close()

        logger.info(
            "LiveExecutionEngine stopped. Realized PnL: $%.4f",
            self.realized_pnl,
        )

    # ------------------------------------------------------------------
    # Main Strategy Loop (Task 3.2)
    # ------------------------------------------------------------------

    async def run_strategy_loop(self) -> None:
        """
        Main loop: evaluates strategy every STRATEGY_EVAL_INTERVAL_S.

        Replicates the exact Run 5 tick-loop from backtest.py, adapted
        for live asynchronous execution.
        """
        logger.info(
            "Strategy loop started (%.1fs interval).", STRATEGY_EVAL_INTERVAL_S,
        )

        while self._running:
            try:
                await self._strategy_tick()
            except Exception as exc:
                logger.error(
                    "Strategy tick error: %s", exc, exc_info=True,
                )

            await asyncio.sleep(STRATEGY_EVAL_INTERVAL_S)

    async def _strategy_tick(self) -> None:
        """Single strategy evaluation cycle — the core of the bot."""
        now = time.time()

        # --- Gas Guard (Task 3.1) ---
        if now - self._last_gas_check > GAS_CHECK_INTERVAL_S:
            self._gas_safe, self._gas_gwei = await check_gas(
                self._http_session, POLYGON_RPC_URL,
            )
            self._last_gas_check = now
            if not self._gas_safe:
                logger.warning(
                    "GAS GUARD: %.1f gwei > %d limit. Orders paused.",
                    self._gas_gwei, MAX_GAS_GWEI,
                )

        if not self._gas_safe:
            return

        # --- Macro Breaker (Task 3.3) — already tripped? ---
        if self.macro_breaker_tripped:
            return

        # --- Oracle Readiness ---
        if not self.oracle.is_ready(self.asset):
            logger.debug("Oracle not ready for %s, skipping.", self.asset)
            return

        snapshot = self.oracle.get_snapshot(self.asset)
        z_score: float = snapshot["z_score"]
        cvd: float = snapshot["cvd_delta"]

        # --- Market Midpoint = Fair Probability ---
        fair_prob = await self._get_market_midpoint()
        if fair_prob is None or fair_prob <= 0.0:
            logger.debug("Could not fetch midpoint, skipping tick.")
            return

        # --- Toxic Flow Kill Switch ---
        if self.risk_monitor.toxic_flow_active:
            logger.warning("TOXIC FLOW ACTIVE — cancelling all, pausing.")
            await self._cancel_all_tracked()
            return

        # --- Global Portfolio Stop-Loss (8c) ---
        await self._check_global_stop_loss(fair_prob, now)

        # --- Macro Circuit Breaker Check (Task 3.3) ---
        if self.realized_pnl <= MACRO_BREAKER_LIMIT:
            await self._fire_macro_breaker()
            return

        if self.macro_breaker_tripped:
            return

        # --- Directional Entry (Z > 2.0 AND |CVD| > $3M) ---
        if (
            z_score > ZSCORE_THRESHOLD
            and abs(cvd) > CVD_THRESHOLD
            and not self.in_directional_mode
        ):
            await self._enter_directional(fair_prob, now)

        # Reset directional mode when flat.
        if self.in_directional_mode and not self.positions:
            self.in_directional_mode = False

        # --- Maker Quoting with Dynamic Spread + Inventory Skew ---
        if not self.in_directional_mode:
            has_inventory = self.yes_shares > 0
            spread = compute_dynamic_spread(z_score, has_inventory)

            if spread is None:
                # Volatile + inventory: pause quoting entirely.
                await self._cancel_all_tracked()
            else:
                await self._place_skewed_maker_orders(fair_prob, spread, now)

        # --- Periodic Balance Reconciliation ---
        if now - self._last_balance_reconcile > BALANCE_RECONCILE_INTERVAL_S:
            await self._reconcile_balances()
            self._last_balance_reconcile = now

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    async def _get_market_midpoint(self) -> float | None:
        """Fetch the YES token midpoint from Polymarket CLOB."""
        try:
            result = await asyncio.to_thread(
                self.client.get_midpoint, self.market.yes_token_id,
            )
            if result and "mid" in result:
                return float(result["mid"])
            return None
        except Exception as exc:
            logger.warning("Midpoint fetch failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Balance Management
    # ------------------------------------------------------------------

    async def _fetch_usdc_balance(self) -> float:
        """Fetch USDC (collateral) balance from Polymarket."""
        try:
            result = await asyncio.to_thread(
                self.client.get_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            )
            return float(result.get("balance", 0) or 0)
        except Exception as exc:
            logger.error("USDC balance fetch failed: %s", exc)
            return self.usdc_balance

    async def _fetch_yes_balance(self) -> float:
        """Fetch YES token balance from Polymarket."""
        try:
            result = await asyncio.to_thread(
                self.client.get_balance_allowance,
                BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=self.market.yes_token_id,
                ),
            )
            return float(result.get("balance", 0) or 0)
        except Exception as exc:
            logger.warning("YES balance fetch failed: %s", exc)
            return self.yes_shares

    async def _reconcile_balances(self) -> None:
        """Periodically sync local state with exchange balances."""
        remote_usdc = await self._fetch_usdc_balance()
        remote_yes = await self._fetch_yes_balance()

        drift_usdc = abs(remote_usdc - self.usdc_balance)
        drift_yes = abs(remote_yes - self.yes_shares)

        if drift_usdc > 0.01 or drift_yes > 0.01:
            logger.info(
                "BALANCE RECONCILE: USDC local=$%.2f remote=$%.2f | "
                "YES local=%.4f remote=%.4f",
                self.usdc_balance, remote_usdc,
                self.yes_shares, remote_yes,
            )
            self.usdc_balance = remote_usdc
            self.yes_shares = remote_yes

    # ------------------------------------------------------------------
    # Order Placement — Skewed Maker (POST_ONLY) — Task 3.2
    # ------------------------------------------------------------------

    async def _place_skewed_maker_orders(
        self,
        fair_prob: float,
        spread: float,
        now: float,
    ) -> None:
        """
        Place inventory-skewed POST_ONLY maker orders.

        Replicates CLOBSimulator.place_skewed_maker_orders() exactly:
          Skew = (Current_Exposure / Inventory_Limit) * Max_Skew_Cents
          Shift both quotes DOWN when long.
          Ask-only at inventory limit.
          Suppress bids during cooldown penalty box.
        """
        # Cancel existing resting orders.
        await self._cancel_all_tracked()

        current_exposure = sum(p.entry_usdc for p in self.positions)
        has_inventory = self.yes_shares > 0

        # Compute skew: linear penalty proportional to inventory fill.
        skew = 0.0
        if has_inventory and INVENTORY_LIMIT_USDC > 0:
            skew = (current_exposure / INVENTORY_LIMIT_USDC) * MAX_SKEW_CENTS

        # Apply skew: shift both quotes DOWN when long.
        bid_price = max(0.01, fair_prob - spread - skew)
        ask_price = min(0.99, fair_prob + spread - skew)

        # Round to market tick size.
        tick = float(self.market.tick_size)
        bid_price = round(bid_price / tick) * tick
        ask_price = round(ask_price / tick) * tick

        # Cooldown check: suppress bids during penalty box (Task 3.2).
        in_cooldown = (
            self.last_stop_loss_time > 0
            and now < self.last_stop_loss_time + COOLDOWN_DURATION_S
        )

        # At inventory limit: ASK ONLY to unwind.
        at_limit = current_exposure >= INVENTORY_LIMIT_USDC

        # --- Place BID (buy YES shares) ---
        if not in_cooldown and not at_limit:
            available = max(0, MAX_EXPOSURE_USDC - current_exposure)
            bid_size_usdc = min(
                MAKER_SIZE_USDC, available, self.usdc_balance,
            )
            if bid_size_usdc > self.market.min_order_size and bid_price > 0:
                bid_shares = bid_size_usdc / bid_price
                await self._post_maker_order(
                    side=BUY,
                    price=bid_price,
                    size=bid_shares,
                    size_usdc=bid_size_usdc,
                    label="BID",
                )

        # --- Place ASK (sell YES shares) ---
        if has_inventory and ask_price > 0:
            ask_shares = min(
                MAKER_SIZE_USDC / ask_price,
                self.yes_shares,
            )
            ask_usdc = ask_shares * ask_price
            if ask_shares > 0 and ask_usdc > self.market.min_order_size:
                await self._post_maker_order(
                    side=SELL,
                    price=ask_price,
                    size=ask_shares,
                    size_usdc=ask_usdc,
                    label="ASK",
                )

    async def _post_maker_order(
        self,
        side: str,
        price: float,
        size: float,
        size_usdc: float,
        label: str = "",
    ) -> str | None:
        """Post a single POST_ONLY GTC order and track it locally."""
        if self.dry_run:
            order_id = f"DRY_{label}_{price:.4f}_{int(time.time() * 1000)}"
            logger.info(
                "DRY RUN: POST_ONLY %s %.2f shares @ %.4f ($%.2f)",
                label, size, price, size_usdc,
            )
            self.tracked_orders[order_id] = TrackedOrder(
                order_id=order_id, side=side, price=price,
                size=size, size_usdc=size_usdc, placed_at=time.time(),
            )
            return order_id

        try:
            order_args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=self.market.yes_token_id,
            )
            signed = await asyncio.to_thread(
                self.client.create_order, order_args,
            )
            resp = await asyncio.to_thread(
                self.client.post_order, signed, OrderType.GTC, True,
            )

            order_id = (
                resp.get("orderID", "")
                or resp.get("id", "")
                or resp.get("order_id", "")
            )
            if order_id:
                self.tracked_orders[order_id] = TrackedOrder(
                    order_id=order_id, side=side, price=price,
                    size=size, size_usdc=size_usdc, placed_at=time.time(),
                )
                logger.debug(
                    "MAKER %s: %.2f shares @ %.4f (id=%s...)",
                    label, size, price, order_id[:16],
                )
            return order_id

        except Exception as exc:
            logger.error(
                "POST_ONLY %s failed (%.4f, %.2f shares): %s",
                label, price, size, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Global Stop-Loss — FOK Dump (Task 3.2)
    # ------------------------------------------------------------------

    async def _check_global_stop_loss(
        self, current_price: float, now: float,
    ) -> None:
        """
        If fair price drops 8c below any non-runner position's entry,
        dump that position via FOK (Fill-or-Kill) market sell.

        Records stop time for the 5-minute cooldown penalty box.
        """
        stops_fired = False

        for pos in list(self.positions):
            if pos.shares <= 0 or pos.scaled_out:
                continue

            if current_price < pos.entry_price - GLOBAL_STOP_DISTANCE:
                logger.warning(
                    "GLOBAL STOP: price %.4f < entry %.4f - %.2f. "
                    "Dumping %.2f shares via FOK.",
                    current_price, pos.entry_price,
                    GLOBAL_STOP_DISTANCE, pos.shares,
                )

                proceeds = await self._execute_fok_sell(
                    pos.shares, current_price,
                )

                if proceeds is not None:
                    fee = proceeds * taker_fee_rate(current_price)
                    net = proceeds - fee
                    pnl = net - pos.entry_usdc
                    self.realized_pnl += pnl
                    self.yes_shares = max(0, self.yes_shares - pos.shares)
                    self.usdc_balance += net

                    logger.warning(
                        "GLOBAL STOP EXECUTED: sold %.2f shares, "
                        "net=$%.2f, PnL=$%.4f, realized_total=$%.4f",
                        pos.shares, net, pnl, self.realized_pnl,
                    )

                    pos.shares = 0.0
                    pos.entry_usdc = 0.0
                    stops_fired = True

        # Clean up closed positions.
        self.positions = [p for p in self.positions if p.shares > 0]

        # Record stop time for cooldown penalty box (Task 3.2).
        if stops_fired:
            self.last_stop_loss_time = now

    async def _execute_fok_sell(
        self, shares: float, worst_price: float,
    ) -> float | None:
        """
        Execute a Fill-or-Kill sell to dump inventory.

        Returns gross proceeds on success, None on failure.
        """
        if self.dry_run:
            proceeds = shares * worst_price
            logger.info(
                "DRY RUN: FOK SELL %.2f shares @ worst %.4f = $%.2f",
                shares, worst_price, proceeds,
            )
            return proceeds

        try:
            order_args = MarketOrderArgs(
                token_id=self.market.yes_token_id,
                amount=shares,
                side=SELL,
                price=worst_price,
            )
            signed = await asyncio.to_thread(
                self.client.create_market_order, order_args,
            )
            resp = await asyncio.to_thread(
                self.client.post_order, signed, OrderType.FOK,
            )

            # Check for successful fill.
            status = resp.get("status", "")
            if status in ("matched", "filled") or resp.get("orderID"):
                logger.info("FOK SELL filled: %s", resp.get("orderID", "?"))
                return shares * worst_price

            # FOK not filled — try FAK (Fill-and-Kill, partial OK).
            logger.warning(
                "FOK sell not filled (status=%s). Retrying as FAK.", status,
            )
            order_args_fak = MarketOrderArgs(
                token_id=self.market.yes_token_id,
                amount=shares,
                side=SELL,
                price=worst_price,
            )
            signed_fak = await asyncio.to_thread(
                self.client.create_market_order, order_args_fak,
            )
            resp_fak = await asyncio.to_thread(
                self.client.post_order, signed_fak, OrderType.FAK,
            )
            if resp_fak.get("orderID"):
                return shares * worst_price

            logger.error("FAK sell also failed: %s", resp_fak)
            return None

        except Exception as exc:
            logger.error("FOK/FAK sell failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Directional Entry — Taker Buy (Task 3.2)
    # ------------------------------------------------------------------

    async def _enter_directional(
        self, fair_prob: float, now: float,
    ) -> None:
        """
        Aggressive taker buy for directional scalp.
        Triggered when Z > 2.0 AND |CVD| > $3M.
        """
        current_exposure = sum(p.entry_usdc for p in self.positions)
        available = max(0, MAX_EXPOSURE_USDC - current_exposure)
        size_usdc = min(DIRECTIONAL_SIZE_USDC, available, self.usdc_balance)

        if size_usdc < 1.0:
            return

        cvd_str = f"{self.oracle.state[self.asset]['cvd_delta']:+,.0f}"
        logger.info(
            "DIRECTIONAL ENTRY: Z=%.2f, CVD=$%s, prob=%.4f, size=$%.2f",
            self.oracle.state[self.asset]["z_score"],
            cvd_str, fair_prob, size_usdc,
        )

        shares = size_usdc / fair_prob if fair_prob > 0 else 0
        fee = size_usdc * taker_fee_rate(fair_prob)

        if self.dry_run:
            self.positions.append(LivePosition(
                entry_price=fair_prob, entry_usdc=size_usdc,
                shares=shares, entry_time=now,
            ))
            self.yes_shares += shares
            self.usdc_balance -= (size_usdc + fee)
            self.in_directional_mode = True
            logger.info(
                "DRY RUN: Directional filled %.2f shares @ %.4f", shares, fair_prob,
            )
            return

        try:
            order_args = MarketOrderArgs(
                token_id=self.market.yes_token_id,
                amount=size_usdc,
                side=BUY,
            )
            signed = await asyncio.to_thread(
                self.client.create_market_order, order_args,
            )
            resp = await asyncio.to_thread(
                self.client.post_order, signed, OrderType.FOK,
            )

            if resp.get("orderID"):
                self.positions.append(LivePosition(
                    entry_price=fair_prob, entry_usdc=size_usdc,
                    shares=shares, entry_time=now,
                ))
                self.yes_shares += shares
                self.usdc_balance -= (size_usdc + fee)
                self.in_directional_mode = True

                logger.info(
                    "DIRECTIONAL FILLED: %.2f shares @ %.4f, fee=$%.4f",
                    shares, fair_prob, fee,
                )
            else:
                logger.warning("Directional entry not filled: %s", resp)

        except Exception as exc:
            logger.error("Directional entry failed: %s", exc)

    # ------------------------------------------------------------------
    # Macro Circuit Breaker (Task 3.3)
    # ------------------------------------------------------------------

    async def _fire_macro_breaker(self) -> None:
        """
        MACRO BREAKER: If realized PnL <= -$25, cancel everything and exit.

        Per spec: fire client.cancel_all() and sys.exit().
        """
        self.macro_breaker_tripped = True

        logger.critical(
            "MACRO BREAKER TRIPPED: realized PnL=$%.2f "
            "(limit=$%.2f). CANCELLING ALL. SHUTTING DOWN.",
            self.realized_pnl, MACRO_BREAKER_LIMIT,
        )

        try:
            if not self.dry_run:
                await asyncio.to_thread(self.client.cancel_all)
            logger.info("cancel_all() executed by macro breaker.")
        except Exception as exc:
            logger.error("Macro breaker cancel_all failed: %s", exc)

        self._running = False
        sys.exit("MACRO BREAKER TRIPPED. SHUTTING DOWN.")

    # ------------------------------------------------------------------
    # Order Management — Cancel & Fill Polling
    # ------------------------------------------------------------------

    async def _cancel_all_tracked(self) -> None:
        """Cancel all locally tracked resting orders."""
        if not self.tracked_orders:
            return

        order_ids = list(self.tracked_orders.keys())

        if self.dry_run:
            logger.debug("DRY RUN: Cancelling %d orders.", len(order_ids))
            self.tracked_orders.clear()
            return

        try:
            await asyncio.to_thread(
                self.client.cancel_orders, order_ids,
            )
            logger.debug("Cancelled %d tracked orders.", len(order_ids))
        except Exception as exc:
            logger.warning(
                "Batch cancel failed (%s). Falling back to cancel_all.", exc,
            )
            try:
                await asyncio.to_thread(self.client.cancel_all)
            except Exception as exc2:
                logger.error("cancel_all fallback also failed: %s", exc2)

        self.tracked_orders.clear()

    async def run_fill_monitor(self) -> None:
        """
        Background task: poll tracked orders for fills.

        When a fill is detected, update local position/balance state.
        This bridges the gap between order placement and execution
        confirmation.
        """
        logger.info(
            "Fill monitor started (%.1fs interval).", FILL_POLL_INTERVAL_S,
        )
        try:
            while self._running:
                await asyncio.sleep(FILL_POLL_INTERVAL_S)

                if self.dry_run or not self.tracked_orders:
                    continue

                for order_id in list(self.tracked_orders.keys()):
                    try:
                        order_info = await asyncio.to_thread(
                            self.client.get_order, order_id,
                        )
                    except Exception as exc:
                        logger.debug(
                            "Fill poll for %s... failed: %s",
                            order_id[:12], exc,
                        )
                        continue

                    if order_info is None:
                        continue

                    # Parse fill status.
                    size_matched = float(
                        order_info.get("size_matched", 0)
                        or order_info.get("sizeMatched", 0)
                        or 0
                    )
                    original_size = float(
                        order_info.get("original_size", 0)
                        or order_info.get("originalSize", 0)
                        or 0
                    )

                    if size_matched <= 0:
                        continue

                    tracked = self.tracked_orders.get(order_id)
                    if tracked is None:
                        continue

                    # Process the fill.
                    self._process_fill(tracked, size_matched)

                    # Remove from tracking if fully filled.
                    fill_ratio = (
                        size_matched / original_size
                        if original_size > 0 else 1.0
                    )
                    if fill_ratio >= 0.99:
                        del self.tracked_orders[order_id]

        except asyncio.CancelledError:
            pass

    def _process_fill(
        self, order: TrackedOrder, shares_filled: float,
    ) -> None:
        """
        Process a detected maker fill — update positions and balances.

        BID fill: we acquired YES shares (new position).
        ASK fill: we sold YES shares (close oldest position, realize PnL).
        """
        price = order.price
        rebate = shares_filled * price * maker_rebate_rate(price)

        if order.side == BUY:
            # Maker BID filled — we bought YES shares.
            cost_usdc = shares_filled * price
            self.usdc_balance -= cost_usdc
            self.usdc_balance += rebate
            self.yes_shares += shares_filled

            self.positions.append(LivePosition(
                entry_price=price,
                entry_usdc=cost_usdc,
                shares=shares_filled,
                entry_time=time.time(),
            ))

            logger.info(
                "MAKER BID FILL: +%.2f shares @ %.4f, rebate=$%.4f",
                shares_filled, price, rebate,
            )

        elif order.side == SELL:
            # Maker ASK filled — we sold YES shares.
            proceeds = shares_filled * price
            self.usdc_balance += proceeds + rebate
            self.yes_shares = max(0, self.yes_shares - shares_filled)

            # FIFO PnL accounting.
            pnl = self._close_oldest_position(shares_filled, price)
            self.realized_pnl += pnl

            logger.info(
                "MAKER ASK FILL: -%.2f shares @ %.4f, PnL=$%.4f, "
                "rebate=$%.4f",
                shares_filled, price, pnl, rebate,
            )

    def _close_oldest_position(
        self, shares_sold: float, sell_price: float,
    ) -> float:
        """FIFO position close — identical to simulator logic."""
        remaining = shares_sold
        total_pnl = 0.0

        for pos in self.positions:
            if remaining <= 0:
                break
            if pos.shares <= 0:
                continue

            close_shares = min(remaining, pos.shares)
            entry_cost = close_shares * pos.entry_price
            exit_proceeds = close_shares * sell_price
            pnl = exit_proceeds - entry_cost
            total_pnl += pnl

            pos.shares -= close_shares
            remaining -= close_shares

        # Clean up fully closed positions.
        self.positions = [p for p in self.positions if p.shares > 0]
        return total_pnl

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def run_heartbeat(self, interval: float = 10.0) -> None:
        """Periodic status logging for observability."""
        try:
            while self._running:
                await asyncio.sleep(interval)

                snapshot = self.oracle.get_snapshot(self.asset)
                midpoint = await self._get_market_midpoint() or 0.50
                equity = self.usdc_balance + (self.yes_shares * midpoint)

                in_cooldown = (
                    self.last_stop_loss_time > 0
                    and time.time() < self.last_stop_loss_time + COOLDOWN_DURATION_S
                )

                logger.info(
                    "HEARTBEAT | equity=$%.2f | bal=$%.2f | shares=%.2f | "
                    "pos=%d | orders=%d | realized=$%.4f | "
                    "Z=%+.2f | CVD=$%s | gas=%.1f | "
                    "cooldown=%s | breaker=%s",
                    equity, self.usdc_balance, self.yes_shares,
                    len(self.positions), len(self.tracked_orders),
                    self.realized_pnl,
                    snapshot["z_score"],
                    f"{snapshot['cvd_delta']:+,.0f}",
                    self._gas_gwei,
                    "ACTIVE" if in_cooldown else "OFF",
                    "TRIPPED" if self.macro_breaker_tripped else "OK",
                )

        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    """Launch all subsystems and run the live execution engine."""

    # --- Task 3.1: Initialize CLOB Client ---
    client = init_clob_client()

    # --- Phase 1 Subsystems ---
    oracle = UnifiedOracle(assets=[args.asset])
    risk_monitor = RiskMonitor()

    await oracle.start()
    await risk_monitor.start()

    # Wait for oracle warmup (both venues must report).
    logger.info("Waiting for oracle warmup (max 60s)...")
    for _ in range(60):
        if oracle.is_ready(args.asset):
            break
        await asyncio.sleep(1)
    else:
        logger.error(
            "Oracle did not become ready for %s within 60s. Aborting.",
            args.asset,
        )
        await oracle.stop()
        await risk_monitor.stop()
        sys.exit(1)

    logger.info("Oracle ready. Scanning for best market...")

    # --- Task 3.4: Find Best Market ---
    async with aiohttp.ClientSession() as session:
        market = await find_best_market(session, asset=args.asset)

    if market is None:
        logger.error("No suitable market found. Aborting.")
        await oracle.stop()
        await risk_monitor.stop()
        sys.exit(1)

    # Fetch tick size and details from the CLOB order book.
    try:
        book = await asyncio.to_thread(
            client.get_order_book, market.yes_token_id,
        )
        if hasattr(book, "tick_size") and book.tick_size:
            market.tick_size = book.tick_size
        if hasattr(book, "neg_risk"):
            market.neg_risk = book.neg_risk
        if hasattr(book, "min_order_size") and book.min_order_size:
            market.min_order_size = float(book.min_order_size)
        logger.info(
            "Market details: tick=%s, neg_risk=%s, min_size=%s",
            market.tick_size, market.neg_risk, market.min_order_size,
        )
    except Exception as exc:
        logger.warning("Could not fetch order book details: %s", exc)

    # --- Initialize Execution Engine ---
    engine = LiveExecutionEngine(
        client=client,
        oracle=oracle,
        risk_monitor=risk_monitor,
        market=market,
        asset=args.asset,
        dry_run=args.dry_run,
    )
    await engine.start()

    # Launch concurrent tasks.
    strategy_task = asyncio.create_task(
        engine.run_strategy_loop(), name="strategy_loop",
    )
    fill_task = asyncio.create_task(
        engine.run_fill_monitor(), name="fill_monitor",
    )
    heartbeat_task = asyncio.create_task(
        engine.run_heartbeat(interval=10.0), name="exec_heartbeat",
    )

    logger.info(
        "Phase 3 LIVE. Asset=%s | Market='%s' | Dry=%s | Ctrl+C to stop.",
        args.asset, market.question, args.dry_run,
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

    # Orderly shutdown.
    strategy_task.cancel()
    fill_task.cancel()
    heartbeat_task.cancel()
    await asyncio.gather(
        strategy_task, fill_task, heartbeat_task,
        return_exceptions=True,
    )

    await engine.stop()
    await oracle.stop()
    await risk_monitor.stop()

    logger.info("All Phase 3 systems shut down cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket HFT Bot — Phase 3: Live Execution Engine",
    )
    parser.add_argument(
        "--asset", default="BTC",
        help="Base asset to trade (default: BTC).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Paper trading mode — log orders but don't submit to exchange.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
            "%(name)-14s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy third-party loggers.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("Starting Polymarket HFT Phase 3 — Live Execution...")

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    except SystemExit as exc:
        logger.critical("System exit: %s", exc)


if __name__ == "__main__":
    main()

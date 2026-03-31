"""
Live Market Maker Farmer — Phase 4: Mainnet Execution.

Ports the exact dynamic logic from maker_farmer_backtest.py into a live
asynchronous event loop using the official py-clob-client for Polymarket.

Supports auto-rolling market discovery: pass --asset SOL (or BTC/ETH/XRP)
and the bot will automatically find the current active 5-minute Up/Down
market, trade it, and rotate to the next window when it expires.

Architecture: DUAL-TOKEN BID-ONLY
    Places BID on YES at (fair - spread) AND BID on NO at (1-fair - spread).
    Captures spread from both directions without naked shorting.

Rules:
    Rule 1 -- 100% Maker: POST_ONLY BID entries only. Never cross to open.
    Rule 2 -- Toxicity Shield (dynamic): |CVD| > 15% of 10m vol OR |Z| > 2.0.
    Rule 3 -- TTE Killswitch: Flatten BOTH tokens via taker at 20% remaining.
    Rule 4 -- Per-Token Skew: $4 threshold → pause BID, place ASK to flatten.
    Rule 5 -- Grind Shield (dynamic): Pause side if CVD > 5% of 10m vol.
    Rule 6 -- Time-Decay: Slam limits at 25% window remaining.
    Rule 7 -- Zero-Edge Exit: Maker exit at fair_prob at 15% remaining.

Risk Controls:
    $5 Daily MTM Circuit Breaker (mark-to-market, resets at 00:00 UTC).

Usage:
    python live_farmer.py --asset SOL --dry-run          # Auto-rolling SOL 5m
    python live_farmer.py --asset BTC --dry-run          # Auto-rolling BTC 5m
    python live_farmer.py --token-id <TOKEN_ID> --dry-run  # Manual token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp
import websockets
import websockets.exceptions
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# Defer py-clob-client import for clean error message.
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        ApiCreds,
        AssetType,
        BalanceAllowanceParams,
        MarketOrderArgs,
        OpenOrderParams,
        OrderArgs,
        OrderType,
        TradeParams,
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

logger = logging.getLogger("live_farmer")


# ---------------------------------------------------------------------------
# Constants — identical to maker_farmer_backtest.py
# ---------------------------------------------------------------------------

STRATEGY_EVAL_INTERVAL_S = 1.0          # Re-evaluate quoting every 1 second.
TOXICITY_SHIELD_DURATION_S = 15.0       # 15-second pause after toxic flow.
INVENTORY_BREACH_COOLDOWN_S = 15.0      # 15s cooldown after breach flatten.
VOLUME_WINDOW_S = 600.0                 # 10-minute rolling window for volume.
CVD_WINDOW_S = 60.0                     # 60s rolling CVD window.

# Window-relative fractions.
TTE_KILLSWITCH_FRAC = 0.20              # Flatten at 20% remaining (60s for 5m).
TIME_DECAY_FRAC = 0.25                  # Time decay at 25% remaining (was 50%).
ZERO_EDGE_FRAC = 0.15                   # Zero-edge at 15% remaining (was 40%).

# Volume-relative thresholds.
TOXICITY_VOLUME_PCT = 0.15
GRIND_VOLUME_PCT = 0.05

# Polymarket WebSocket.
POLY_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_PING_INTERVAL_S = 10.0
WS_RECONNECT_DELAY_S = 1.0
WS_RECONNECT_MAX_DELAY_S = 30.0
WS_RECONNECT_BACKOFF_FACTOR = 2.0

# Gas guard.
GAS_CHECK_INTERVAL_S = 30.0
MAX_GAS_GWEI = 300

# On-chain USDC balance (Polygon).
# Native USDC on Polygon — this is where Polymarket redemptions settle.
USDC_POLYGON_ADDRESS = Web3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")
# Bridged USDC.e — some older deposits may use this.
USDCE_POLYGON_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
# Minimal ERC-20 ABI for balanceOf.
ERC20_BALANCE_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    }
]

# Balance reconciliation.
BALANCE_RECONCILE_INTERVAL_S = 60.0

POLYGON_RPC_URL = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
POLY_GAMMA_HOST = "https://gamma-api.polymarket.com"


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
class MarketInfo:
    """An active Polymarket 5-minute Up/Down market."""
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    slug: str = ""
    end_time: float = 0.0       # Unix timestamp when this window expires.


@dataclass
class TrackedOrder:
    """A resting POST_ONLY order on Polymarket."""
    order_id: str
    side: str           # "BUY" or "SELL"
    price: float
    size_shares: float
    size_usdc: float
    placed_at: float    # Unix timestamp


# ---------------------------------------------------------------------------
# Live State Manager
# ---------------------------------------------------------------------------

class LiveStateManager:
    """
    Maintains real-time market state from Polymarket WebSocket trades.

    Tracks:
        - Rolling 10-minute absolute volume (for dynamic thresholds)
        - Rolling 60-second CVD (signed cumulative volume delta)
        - Z-Score via Welford's online recurrence (100ms sampling)
        - Last trade price as synthetic fair price
    """

    def __init__(self) -> None:
        # Fair price from last trade.
        self.fair_price: float = 0.0

        # Rolling 10-minute volume tracker.
        self._vol_window: deque[tuple[float, float]] = deque()
        self._vol_running_sum: float = 0.0
        self.rolling_10m_volume: float = 0.0

        # Rolling 60-second CVD tracker.
        self._cvd_window: deque[tuple[float, float]] = deque()
        self._cvd_running_sum: float = 0.0
        self.cvd_delta: float = 0.0

        # Z-Score via Welford's recurrence (100ms sampling).
        self._zscore_window: deque[float] = deque(maxlen=600)  # 60s at 100ms
        self._last_sample_time: float = 0.0
        self.z_score: float = 0.0

        self.ready: bool = False
        self._trade_count: int = 0

    def process_trade(
        self, price: float, size: float, side: str, timestamp: float,
    ) -> None:
        """
        Process a single trade from the WebSocket.

        Args:
            price: Trade price (probability, 0-1).
            size: Trade size in shares.
            side: "buy" or "sell" (aggressor side).
            timestamp: Unix timestamp (seconds, float).
        """
        self.fair_price = price
        self._trade_count += 1
        notional = price * size

        # --- Rolling 10-minute volume ---
        self._vol_window.append((timestamp, notional))
        self._vol_running_sum += notional
        vol_cutoff = timestamp - VOLUME_WINDOW_S
        while self._vol_window and self._vol_window[0][0] < vol_cutoff:
            self._vol_running_sum -= self._vol_window.popleft()[1]
        self.rolling_10m_volume = self._vol_running_sum

        # --- Rolling 60-second CVD ---
        signed = notional if side == "buy" else -notional
        self._cvd_window.append((timestamp, signed))
        self._cvd_running_sum += signed
        cvd_cutoff = timestamp - CVD_WINDOW_S
        while self._cvd_window and self._cvd_window[0][0] < cvd_cutoff:
            self._cvd_running_sum -= self._cvd_window.popleft()[1]
        self.cvd_delta = self._cvd_running_sum

        # --- Z-Score sampling (100ms intervals) ---
        if timestamp - self._last_sample_time >= 0.1:
            self._zscore_window.append(price)
            self._last_sample_time = timestamp
            self._update_zscore()

        if self._trade_count >= 30:
            self.ready = True

    def _update_zscore(self) -> None:
        """Compute Z-Score using Welford's online recurrence."""
        n = len(self._zscore_window)
        if n < 30:
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
# Gas Guard
# ---------------------------------------------------------------------------

async def check_gas(
    session: aiohttp.ClientSession,
    rpc_url: str,
) -> tuple[bool, float]:
    """Query Polygon RPC for current gas price."""
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
# Auto-Rolling Market Discovery
# ---------------------------------------------------------------------------

# Asset keyword mapping for Gamma API search.
ASSET_KEYWORDS: dict[str, list[str]] = {
    "SOL": ["Solana", "SOL"],
    "BTC": ["Bitcoin", "BTC"],
    "ETH": ["Ethereum", "ETH", "Ether"],
    "XRP": ["XRP", "Ripple"],
}

# Slug prefix patterns for 5-minute Up/Down markets.
ASSET_SLUG_PREFIX: dict[str, str] = {
    "SOL": "sol-updown-5m",
    "BTC": "btc-updown-5m",
    "ETH": "eth-updown-5m",
    "XRP": "xrp-updown-5m",
}


async def find_active_5m_market(
    session: aiohttp.ClientSession,
    asset: str,
) -> MarketInfo | None:
    """
    Find the currently active 5-minute Up/Down market for the given asset.

    Uses the Gamma API **events** endpoint with predictable slug patterns:
        {asset}-updown-5m-{window_start_unix}

    Tries the current window, then the next (for boundary races).

    Args:
        session: aiohttp session for HTTP requests.
        asset: One of SOL, BTC, ETH, XRP.

    Returns:
        MarketInfo with YES token ID, or None if no active market found.
    """
    asset = asset.upper()
    slug_prefix = ASSET_SLUG_PREFIX.get(asset, f"{asset.lower()}-updown-5m")

    now = time.time()
    window_s = 300  # 5 minutes
    window_start = int(now - (now % window_s))

    # Try current window, then next window (boundary race).
    for offset in [0, window_s]:
        ts = window_start + offset
        slug = f"{slug_prefix}-{ts}"
        market = await _fetch_event_by_slug(session, slug)
        if market:
            return market

    logger.warning("No active 5-min %s market found (tried 2 windows).", asset)
    return None


async def _fetch_event_by_slug(
    session: aiohttp.ClientSession,
    slug: str,
) -> MarketInfo | None:
    """
    Fetch a 5-minute market from the Gamma API events endpoint by slug.

    The events endpoint returns the full market structure including
    clobTokenIds (YES/NO token IDs) and conditionId.
    """
    url = f"{POLY_GAMMA_HOST}/events"
    params = {"slug": slug}

    try:
        async with session.get(
            url, params=params, timeout=aiohttp.ClientTimeout(total=10.0),
        ) as resp:
            if resp.status != 200:
                logger.debug("Events API HTTP %d for slug %s", resp.status, slug)
                return None
            events = await resp.json()
    except Exception as exc:
        logger.debug("Events lookup failed for %s: %s", slug, exc)
        return None

    if not isinstance(events, list) or not events:
        return None

    event = events[0]

    # Check if the event is still active and not closed.
    if event.get("closed", True):
        logger.debug("Event %s is closed.", slug)
        return None

    markets = event.get("markets", [])
    if not markets:
        return None

    mkt = markets[0]
    question = mkt.get("question", "")
    condition_id = mkt.get("conditionId", "")

    # Parse clobTokenIds — JSON-encoded string array.
    clob_ids_raw = mkt.get("clobTokenIds", "")
    try:
        if isinstance(clob_ids_raw, str):
            clob_ids = json.loads(clob_ids_raw)
        else:
            clob_ids = clob_ids_raw
    except (json.JSONDecodeError, TypeError):
        clob_ids = []

    if not clob_ids or len(clob_ids) < 2:
        logger.debug("No clobTokenIds for %s", slug)
        return None

    # outcomes: ["Up", "Down"] — Up is YES (index 0), Down is NO (index 1).
    outcomes = mkt.get("outcomes", [])
    yes_token = clob_ids[0]
    no_token = clob_ids[1] if len(clob_ids) > 1 else ""

    # If outcomes explicitly label them, respect that ordering.
    if len(outcomes) >= 2:
        for i, outcome in enumerate(outcomes):
            if outcome.lower() in ("up", "yes") and i < len(clob_ids):
                yes_token = clob_ids[i]
            elif outcome.lower() in ("down", "no") and i < len(clob_ids):
                no_token = clob_ids[i]

    if not yes_token:
        return None

    # Parse end_time from endDate (ISO format with timezone).
    end_time = 0.0
    end_date = mkt.get("endDate", "")
    if end_date:
        try:
            dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            end_time = dt.timestamp()
        except (ValueError, TypeError):
            pass

    # Fallback: compute from slug timestamp + 300s.
    if end_time <= 0:
        try:
            slug_ts = int(slug.rsplit("-", 1)[-1])
            end_time = slug_ts + 300
        except (ValueError, IndexError):
            end_time = time.time() + 300

    # Reject if already expired.
    if end_time < time.time():
        logger.debug("Market %s already expired (end=%.0f).", slug, end_time)
        return None

    info = MarketInfo(
        condition_id=condition_id,
        question=question,
        yes_token_id=yes_token,
        no_token_id=no_token,
        slug=slug,
        end_time=end_time,
    )
    logger.info(
        "MARKET FOUND: '%s' | YES=%s...%s | ends in %.0fs",
        question, yes_token[:12], yes_token[-6:],
        max(0, end_time - time.time()),
    )
    return info


# ---------------------------------------------------------------------------
# Client Initialization (reused from live_execution.py)
# ---------------------------------------------------------------------------

def init_clob_client() -> ClobClient:
    """
    Initialize the py-clob-client using credentials from .env.

    Required .env variables:
        POLY_PK          — Polygon wallet private key (hex)
        POLY_HOST        — CLOB API host (default: https://clob.polymarket.com)
        CHAIN_ID         — Polygon chain ID (default: 137)
        CLOB_API_KEY     — API key
        CLOB_SECRET      — API secret
        CLOB_PASS_PHRASE — API passphrase
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

    # signature_type: 0 = EOA, 1 = Poly Proxy, 2 = Poly Gnosis Safe (browser wallets)
    sig_type = int(os.getenv("POLY_SIG_TYPE", "2"))

    # For proxy/safe wallets (sig_type 1 or 2), funder must be the proxy/Safe
    # address (where funds live), NOT the EOA.  Set POLY_FUNDER in .env, or the
    # bot will auto-discover it from your CLOB trade history after first init.
    funder = os.getenv("POLY_FUNDER", "").strip() or None
    if funder:
        logger.info("Funder (proxy) address from .env: %s", funder)

    if api_key and api_secret and api_passphrase:
        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
        client = ClobClient(
            host, key=pk, chain_id=chain_id, creds=creds,
            signature_type=sig_type, funder=funder,
        )
        logger.info(
            "CLOB client initialized (L2). Host=%s, Chain=%d, SigType=%d.",
            host, chain_id, sig_type,
        )
    else:
        client = ClobClient(host, key=pk, chain_id=chain_id, signature_type=sig_type, funder=funder)
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
            client = ClobClient(
                host, key=pk, chain_id=chain_id, creds=creds,
                signature_type=sig_type, funder=funder,
            )
        except Exception as exc:
            logger.error("API key derivation failed: %s", exc)
            sys.exit(1)

    # Auto-discover funder (proxy/Safe address) if not set and sig_type != 0
    if sig_type in (1, 2) and funder is None:
        try:
            from py_clob_client.clob_types import TradeParams
            trades = client.get_trades(TradeParams())
            if isinstance(trades, list) and trades:
                funder = trades[0].get("maker_address")
                if funder:
                    logger.info(
                        "Auto-discovered proxy address from trade history: %s. "
                        "Add POLY_FUNDER=%s to .env to skip this lookup.",
                        funder, funder,
                    )
                    # Re-create client with correct funder
                    client = ClobClient(
                        host, key=pk, chain_id=chain_id, creds=client.creds,
                        signature_type=sig_type, funder=funder,
                    )
            if funder is None:
                logger.error(
                    "No trade history found — cannot auto-discover proxy address. "
                    "Set POLY_FUNDER in .env to your Polymarket proxy/Safe address. "
                    "Make one trade on polymarket.com first, then re-run."
                )
                sys.exit(1)
        except Exception as exc:
            logger.error("Failed to auto-discover proxy address: %s", exc)
            logger.error("Set POLY_FUNDER in .env manually.")
            sys.exit(1)

    return client


# ---------------------------------------------------------------------------
# Live Farmer Engine
# ---------------------------------------------------------------------------

class LiveFarmerEngine:
    """
    Production maker farmer implementing the Hot Potato Protocol.

    Exact port of maker_farmer_backtest.py dynamic logic:
        - Rule 1: 100% POST_ONLY maker entries
        - Rule 2: Dynamic Toxicity Shield (15% of 10m vol OR Z > 2.0)
        - Rule 3: TTE Killswitch (taker flatten at 20% window remaining)
        - Rule 4: Hyper-Skew ($5 skew, $50 hard limit)
        - Rule 5: Dynamic Grind Shield (5% of 10m vol)
        - Rule 6: Time-Decay (slam limits to $5 at 50% remaining)
        - Rule 7: Zero-Edge Exit (fair_prob maker exit at 40% remaining)
        - $5 Daily MTM Circuit Breaker
    """

    def __init__(
        self,
        client: ClobClient,
        token_id: str = "",
        asset: str = "",
        dry_run: bool = True,
        starting_capital: float = 500.0,
        maker_size_usdc: float = 3.0,
        half_spread: float = 0.005,
        zscore_toxicity_threshold: float = 2.0,
        inventory_skew_threshold: float = 4.0,
        inventory_hard_limit: float = 50.0,
        skew_exit_offset: float = 0.01,
        macro_breaker_limit: float = -5.0,
        time_decay_inventory_limit: float = 15.0,
        window_minutes: float = 5.0,
    ) -> None:
        self.client = client
        # In dual-token mode, token_id is the YES token; no_token_id is NO.
        self.token_id = token_id          # YES / Up token
        self.no_token_id: str = ""        # NO / Down token
        self.asset = asset.upper() if asset else ""
        self._auto_rotate = bool(self.asset and not token_id)
        self.dry_run = dry_run

        # Our wallet/proxy address — used to identify our fills in trade data.
        # The trade API returns maker_address / owner as the proxy (funder)
        # address, NOT the API key UUID.
        self._api_owner_id = (
            os.getenv("POLY_FUNDER", "").strip()
            or os.getenv("CLOB_API_KEY", "")
        )

        # Strategy parameters.
        self.starting_capital = starting_capital
        self.maker_size_usdc = maker_size_usdc
        self.half_spread = half_spread
        self.zscore_toxicity_threshold = zscore_toxicity_threshold
        self.inventory_skew_threshold = inventory_skew_threshold
        self.inventory_hard_limit = inventory_hard_limit
        self.skew_exit_offset = skew_exit_offset
        self.macro_breaker_limit = macro_breaker_limit
        self.time_decay_inventory_limit = time_decay_inventory_limit
        self.window_minutes = window_minutes

        # Derived TTE thresholds (in seconds).
        window_s = window_minutes * 60.0
        self.tte_killswitch_s = window_s * TTE_KILLSWITCH_FRAC
        self.time_decay_s = window_s * TIME_DECAY_FRAC
        self.zero_edge_s = window_s * ZERO_EDGE_FRAC

        # Live state manager (CVD, Z-Score, rolling volume).
        self.state = LiveStateManager()

        # Account state — USDC balance shared between both sides.
        self.balance: float = starting_capital

        # Dual-token inventory tracking.
        # YES (Up) token:
        self.yes_inventory_shares: float = 0.0
        self.yes_cost_basis: float = 0.0
        # NO (Down) token:
        self.no_inventory_shares: float = 0.0
        self.no_cost_basis: float = 0.0

        # Capital locked in unredeemed expired-market shares.
        # When a 5-min window expires and the bot holds shares that it
        # hasn't sold, those shares need manual redemption.  We track
        # each pending item individually so we can re-check resolution
        # and clear losses instead of inflating equity forever.
        # Each entry: {"slug": str, "cost_basis": float, "shares": float,
        #              "token": "yes"|"no", "status": "pending"|"won",
        #              "added_ts": float}
        self._unredeemed_items: list[dict] = []

        # Dual-token resting orders: one BID per token, ASK only for flatten.
        self.yes_bid_order: TrackedOrder | None = None
        self.no_bid_order: TrackedOrder | None = None
        self.yes_ask_order: TrackedOrder | None = None   # Only used for skew flatten
        self.no_ask_order: TrackedOrder | None = None     # Only used for skew flatten

        # Risk state.
        self.macro_breaker_tripped: bool = False
        self.day_start_equity: float = starting_capital
        self._current_day: int = 0

        # Toxicity shield state.
        self.shield_active: bool = False
        self.shield_end_time: float = 0.0

        # TTE state.
        self.tte_halted: bool = False
        self.current_window_start: float = 0.0

        # Inventory breach cooldown.
        self.inventory_cooldown_end: float = 0.0

        # Counters for logging.
        self.total_fills: int = 0
        self.total_maker_rebates: float = 0.0
        self.total_taker_fees: float = 0.0
        self.total_shield_triggers: int = 0
        self.total_tte_flattens: int = 0
        self.total_grind_pauses: int = 0
        self.total_zero_edge_quotes: int = 0

        # Trade log: list of dicts recording every fill for session summary.
        self.trade_log: list[dict] = []

        # Fill detection: track which trade IDs we've already processed.
        self._processed_trade_ids: set[str] = set()
        self._last_fill_check: float = 0.0

        # Gas guard.
        self._last_gas_check: float = 0.0
        self._gas_safe: bool = True

        # Web3 on-chain balance (Polygon) — source of truth for USDC.
        rpc_url = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
        self._w3 = Web3(Web3.HTTPProvider(rpc_url))
        self._proxy_wallet = os.getenv("POLY_FUNDER", "").strip()
        if self._proxy_wallet:
            self._proxy_wallet = Web3.to_checksum_address(self._proxy_wallet)

        # Lifecycle.
        self._running: bool = False
        self._http_session: aiohttp.ClientSession | None = None
        self._tasks: list[asyncio.Task] = []

        # Auto-rotation state.
        self._current_market: MarketInfo | None = None
        self._rotation_pending: bool = False
        self._ws_task: asyncio.Task | None = None
        self._strategy_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._total_rotations: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize and launch all background tasks."""
        self._running = True
        self._http_session = aiohttp.ClientSession()
        now = time.time()

        # Set current day for breaker resets.
        self._current_day = int(now // 86400)

        # Auto-discover initial market if using --asset mode.
        if self._auto_rotate:
            market = await find_active_5m_market(self._http_session, self.asset)
            if not market:
                logger.error(
                    "No active 5-min %s market found. Retrying in 10s...",
                    self.asset,
                )
                await asyncio.sleep(10.0)
                market = await find_active_5m_market(self._http_session, self.asset)
            if not market:
                logger.error("Cannot find active market. Shutting down.")
                self._running = False
                return
            self.token_id = market.yes_token_id
            self.no_token_id = market.no_token_id
            self._current_market = market

        # Align first TTE window.
        window_s = self.window_minutes * 60.0
        self.current_window_start = (now // window_s) * window_s

        # Fetch initial balances if not dry run.
        # On-chain balance is the source of truth (CLOB API caches/lags).
        if not self.dry_run:
            onchain_bal = await self._fetch_onchain_usdc_balance()
            if onchain_bal >= 0:
                self.balance = onchain_bal
                logger.info("On-chain USDC balance: $%.2f (source of truth)", onchain_bal)
            else:
                # Fallback to CLOB API if on-chain fetch fails.
                self.balance = await self._fetch_usdc_balance()
                logger.info("CLOB API USDC balance: $%.2f (on-chain unavailable)", self.balance)
            self.starting_capital = self.balance
            self.day_start_equity = self.balance

            # Fetch existing YES and NO token balances.
            self.yes_inventory_shares = await self._fetch_token_balance(self.token_id)
            if self.yes_inventory_shares > 0:
                mid = await self._get_midpoint()
                if mid and mid > 0:
                    self.yes_cost_basis = self.yes_inventory_shares * mid

            if self.no_token_id:
                self.no_inventory_shares = await self._fetch_token_balance(self.no_token_id)
                if self.no_inventory_shares > 0:
                    no_mid = await self._get_no_midpoint()
                    if no_mid and no_mid > 0:
                        self.no_cost_basis = self.no_inventory_shares * no_mid

            # Pre-seed processed trade IDs so we don't retroactively recount
            # fills from before this session started.
            await self._seed_existing_trades()

        logger.info("=" * 62)
        logger.info("  LIVE MARKET MAKER FARMER — HOT POTATO PROTOCOL")
        logger.info("=" * 62)
        if self._auto_rotate:
            logger.info("  Asset:        %s (auto-rolling 5m)", self.asset)
        logger.info("  Mode:         DUAL-TOKEN BID-ONLY")
        logger.info("  YES Token:    %s...%s", self.token_id[:16], self.token_id[-8:])
        if self.no_token_id:
            logger.info("  NO Token:     %s...%s", self.no_token_id[:16], self.no_token_id[-8:])
        logger.info("  Dry Run:      %s", self.dry_run)
        logger.info("  Capital:      $%.2f", self.balance)
        logger.info("  Maker Size:   $%.2f", self.maker_size_usdc)
        logger.info("  Half Spread:  %.1fc", self.half_spread * 100)
        logger.info("  Window:       %.0f min", self.window_minutes)
        logger.info("  Breaker:      $%.2f daily MTM loss", abs(self.macro_breaker_limit))
        logger.info("  Skew:         $%.0f threshold", self.inventory_skew_threshold)
        logger.info("  Hard Limit:   $%.0f", self.inventory_hard_limit)
        logger.info("  TTE Kill:     %.0f%% (%.0fs)",
                     TTE_KILLSWITCH_FRAC * 100, self.tte_killswitch_s)
        logger.info("  Time-Decay:   %.0f%% (%.0fs)",
                     TIME_DECAY_FRAC * 100, self.time_decay_s)
        logger.info("  Zero-Edge:    %.0f%% (%.0fs)",
                     ZERO_EDGE_FRAC * 100, self.zero_edge_s)
        if self._current_market:
            logger.info("  Market:       %s", self._current_market.question)
            logger.info("  Expires in:   %.0fs",
                         max(0, self._current_market.end_time - time.time()))
        logger.info("=" * 62)

        # Seed fair price from CLOB REST midpoint (don't wait for WS trades).
        mid = await self._get_midpoint()
        if mid and mid > 0:
            self.state.fair_price = mid
            self.state.ready = True
            logger.info("Fair price seeded from CLOB midpoint: %.4f", mid)
        else:
            logger.info("No midpoint available yet. Waiting for WS trades.")

        # Launch background tasks.
        self._ws_task = asyncio.create_task(
            self._ws_trade_listener(), name="ws_trades",
        )
        self._strategy_task = asyncio.create_task(
            self._strategy_loop(), name="strategy_loop",
        )
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="heartbeat",
        )
        self._tasks = [self._ws_task, self._strategy_task, self._heartbeat_task]

        # Launch rotation watcher if auto-rotating.
        if self._auto_rotate:
            rotation_task = asyncio.create_task(
                self._rotation_watcher(), name="rotation",
            )
            self._tasks.append(rotation_task)

    async def stop(self) -> None:
        """Cancel all orders and shut down cleanly."""
        self._running = False

        # Cancel resting orders.
        await self._cancel_all_orders()

        # Cancel background tasks.
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._http_session:
            await self._http_session.close()

        # Sync final balances — on-chain is the source of truth.
        if not self.dry_run:
            real_bal = await self._fetch_onchain_usdc_balance()
            if real_bal < 0:
                real_bal = await self._fetch_usdc_balance()
            if real_bal > 0:
                self.balance = real_bal
            real_yes = await self._fetch_token_balance(self.token_id)
            if real_yes >= 0:
                self.yes_inventory_shares = real_yes
            if self.no_token_id:
                real_no = await self._fetch_token_balance(self.no_token_id)
                if real_no >= 0:
                    self.no_inventory_shares = real_no

        equity = self._compute_equity()
        net_pnl = equity - self.starting_capital
        logger.info("=" * 62)
        logger.info("  SHUTDOWN SUMMARY (DUAL-TOKEN)")
        logger.info("=" * 62)
        logger.info("  Starting Capital: $%.2f", self.starting_capital)
        logger.info("  Final Equity:    $%.2f", equity)
        logger.info("  Net PnL:         $%+.4f", net_pnl)
        logger.info("  YES Inventory:   %.2f shares ($%.2f)",
                     self.yes_inventory_shares,
                     self.yes_inventory_shares * self.state.fair_price)
        logger.info("  NO Inventory:    %.2f shares ($%.2f)",
                     self.no_inventory_shares,
                     self.no_inventory_shares * (1.0 - self.state.fair_price))
        logger.info("  Total Fills:     %d", self.total_fills)
        logger.info("  Maker Rebates:   $%+.4f", self.total_maker_rebates)
        logger.info("  Taker Fees:      $%.4f", self.total_taker_fees)
        logger.info("  Unredeemed:      $%.2f (pending manual redemption)",
                     self.capital_in_unredeemed)
        logger.info("  Shield Triggers: %d", self.total_shield_triggers)
        logger.info("  TTE Flattens:    %d", self.total_tte_flattens)
        logger.info("  Grind Pauses:    %d", self.total_grind_pauses)
        if self._auto_rotate:
            logger.info("  Rotations:       %d", self._total_rotations)
        logger.info("=" * 62)

        # Print detailed trade log.
        if self.trade_log:
            logger.info("")
            logger.info("  TRADE LOG (%d fills)", len(self.trade_log))
            logger.info(
                "  %-8s %-4s %8s %7s %8s %9s %9s %8s",
                "Time", "Side", "Shares", "Price", "USDC",
                "PnL", "Rebate", "Balance",
            )
            logger.info("  " + "-" * 70)
            cumulative_pnl = 0.0
            for t in self.trade_log:
                cumulative_pnl += t["pnl"]
                logger.info(
                    "  %-8s %-4s %8.2f %7.4f %8.2f %+9.4f %9.4f %8.2f",
                    t["time"], t["side"], t["shares"], t["price"],
                    t["usdc"], t["pnl"], t["rebate"], t["bal_after"],
                )
            logger.info("  " + "-" * 70)
            logger.info(
                "  Cumulative Realized PnL: $%+.4f  |  "
                "Unrealized: $%+.4f  |  Net: $%+.4f",
                cumulative_pnl,
                net_pnl - cumulative_pnl,
                net_pnl,
            )
            logger.info("=" * 62)

    # ------------------------------------------------------------------
    # WebSocket Trade Listener
    # ------------------------------------------------------------------

    async def _ws_trade_listener(self) -> None:
        """
        Connect to Polymarket WebSocket and stream live trades.

        Feeds each trade into LiveStateManager for CVD, Z-Score, and
        rolling volume computation.
        """
        delay = WS_RECONNECT_DELAY_S

        while self._running:
            try:
                async with websockets.connect(
                    POLY_WS_URL,
                    ping_interval=None,  # We handle pings manually.
                ) as ws:
                    # Subscribe to YES token ONLY for fair price / CVD / Z-score.
                    # NO fair price is derived as (1.0 - YES fair price).
                    # Subscribing to both would pollute fair_price with NO prices.
                    sub_msg = json.dumps({
                        "assets_ids": [self.token_id],
                        "type": "market",
                    })
                    await ws.send(sub_msg)
                    logger.info("WS subscribed to YES=%s... (NO derived as 1-YES)",
                                self.token_id[:16])

                    delay = WS_RECONNECT_DELAY_S  # Reset on success.

                    # Launch ping keepalive.
                    ping_task = asyncio.create_task(
                        self._ws_ping_keepalive(ws), name="ws_ping",
                    )

                    try:
                        async for raw_msg in ws:
                            self._handle_ws_message(raw_msg)
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass

            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError,
                asyncio.CancelledError,
            ) as exc:
                if isinstance(exc, asyncio.CancelledError):
                    return
                logger.warning("WS disconnected: %s", exc)

            if self._running:
                logger.info("WS reconnecting in %.0fs...", delay)
                await asyncio.sleep(delay)
                delay = min(delay * WS_RECONNECT_BACKOFF_FACTOR, WS_RECONNECT_MAX_DELAY_S)

    async def _ws_ping_keepalive(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Send PING every 10s to keep the connection alive."""
        while True:
            try:
                await asyncio.sleep(WS_PING_INTERVAL_S)
                await ws.send("PING")
            except asyncio.CancelledError:
                return
            except Exception:
                return

    def _handle_ws_message(self, raw: str) -> None:
        """Parse a Polymarket WebSocket message and feed trades to state."""
        if raw == "PONG":
            return

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Polymarket WS may send single objects or arrays of events.
        if isinstance(data, list):
            for item in data:
                self._process_ws_event(item)
        elif isinstance(data, dict):
            self._process_ws_event(data)

    def _process_ws_event(self, data: dict) -> None:
        """Process a single WebSocket event dict."""
        # Debug: log first few messages.
        if self.state._trade_count < 5:
            logger.debug("WS EVENT: %s", json.dumps(data)[:500])

        event_type = data.get("event_type", "")

        if event_type == "last_trade_price":
            # CRITICAL: Only update fair_price from YES token trades.
            # The WS subscribes to both YES and NO tokens, but fair_price
            # must reflect YES only. NO prices (~0.95 when YES is ~0.05)
            # would invert our quoting if they pollute fair_price.
            asset_id = data.get("asset_id", "")
            if asset_id and asset_id != self.token_id:
                return  # Skip NO token trades for fair price.

            price_str = data.get("price", "")
            size_str = data.get("size", "0")
            side = data.get("side", "buy").lower()
            ts_str = data.get("timestamp", "")

            try:
                price = float(price_str)
                size = float(size_str) if size_str else 1.0
                ts = float(ts_str) if ts_str else time.time()
            except (ValueError, TypeError):
                return

            if 0.0 < price <= 1.0:
                self.state.process_trade(price, size, side, ts)

        elif event_type == "price_change":
            # Price change events carry per-asset BBO updates.
            for pc in data.get("price_changes", []):
                if pc.get("asset_id") != self.token_id:
                    continue
                try:
                    best_bid = float(pc.get("best_bid", 0))
                    best_ask = float(pc.get("best_ask", 0))
                    if best_bid > 0 and best_ask > 0:
                        mid = (best_bid + best_ask) / 2.0
                        self.state.fair_price = mid
                        logger.debug("Fair price updated from price_change: %.4f (bid=%.2f ask=%.2f)",
                                     mid, best_bid, best_ask)
                except (ValueError, TypeError):
                    pass

        elif event_type == "tick_size_change":
            pass  # Informational only.

        elif event_type == "book":
            # Full orderbook snapshot — only use YES token book for fair price.
            asset_id = data.get("asset_id", "")
            if asset_id and asset_id != self.token_id:
                return  # Skip NO token book.

            bids = data.get("bids", [])
            asks = data.get("asks", [])
            if bids and asks:
                try:
                    best_bid = float(bids[0].get("price", 0))
                    best_ask = float(asks[0].get("price", 0))
                    if best_bid > 0 and best_ask > 0:
                        mid = (best_bid + best_ask) / 2.0
                        if 0.0 < mid <= 1.0:
                            self.state.fair_price = mid
                            if not self.state.ready:
                                logger.info(
                                    "Fair price initialized from book: %.4f "
                                    "(bid=%.4f ask=%.4f)",
                                    mid, best_bid, best_ask,
                                )
                except (ValueError, TypeError, IndexError):
                    pass

    # ------------------------------------------------------------------
    # Strategy Loop — exact port of maker_farmer_backtest.py
    # ------------------------------------------------------------------

    async def _strategy_loop(self) -> None:
        """Main strategy eval loop, runs every 1 second."""
        logger.info("Strategy loop started (%.1fs interval).", STRATEGY_EVAL_INTERVAL_S)

        # Wait for state readiness.
        while self._running and not self.state.ready:
            logger.debug("Waiting for trade data...")
            await asyncio.sleep(1.0)

        logger.info("State ready. Beginning strategy evaluation.")

        while self._running:
            try:
                await self._strategy_tick()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Strategy tick error: %s", exc, exc_info=True)

            await asyncio.sleep(STRATEGY_EVAL_INTERVAL_S)

    async def _strategy_tick(self) -> None:
        """Single strategy evaluation cycle — DUAL-TOKEN BID-ONLY architecture.

        Places BID on YES at (fair_yes - spread) AND BID on NO at (fair_no - spread).
        fair_yes + fair_no = 1.00 (binary market constraint).
        This captures spread from BOTH directions without naked shorting.
        """
        now = time.time()
        fair_prob = self.state.fair_price

        if fair_prob <= 0.0 or fair_prob >= 1.0:
            return

        # Clamp to tradeable range.
        fair_prob = max(0.02, min(0.98, fair_prob))
        no_fair = 1.0 - fair_prob  # Binary complement.

        # --- Fill Detection & Order State Sync ---
        await self._check_fills(fair_prob)
        await self._sync_order_state()

        # --- Gas Guard ---
        if now - self._last_gas_check > GAS_CHECK_INTERVAL_S:
            if self._http_session:
                self._gas_safe, gas_gwei = await check_gas(
                    self._http_session, POLYGON_RPC_URL,
                )
                self._last_gas_check = now
                if not self._gas_safe:
                    logger.warning(
                        "GAS GUARD: %.1f gwei > %d limit. Pausing.",
                        gas_gwei, MAX_GAS_GWEI,
                    )
        if not self._gas_safe:
            return

        # --- Day Boundary: reset macro breaker ---
        current_day = int(now // 86400)
        if current_day > self._current_day:
            equity = self._compute_equity()
            logger.info(
                "=== DAY BOUNDARY | equity=$%.2f | pnl=$%+.2f ===",
                equity, equity - self.day_start_equity,
            )
            self.day_start_equity = equity
            self.macro_breaker_tripped = False
            self._current_day = current_day

        # --- MTM Macro Breaker (every tick) ---
        equity = self._compute_equity()
        daily_mtm_pnl = equity - self.day_start_equity

        if daily_mtm_pnl <= self.macro_breaker_limit and not self.macro_breaker_tripped:
            self.macro_breaker_tripped = True
            await self._cancel_all_orders()
            if self.yes_inventory_shares > 0 or self.no_inventory_shares > 0:
                await self._flatten_all_inventory("BREAKER", fair_prob)
            logger.warning(
                "MTM BREAKER: mtm_pnl=$%.2f. Halting until midnight.",
                daily_mtm_pnl,
            )

        if self.macro_breaker_tripped:
            return

        # --- TTE Killswitch (DUAL) ---
        window_s = self.window_minutes * 60.0
        tick_window_start = (now // window_s) * window_s

        if tick_window_start > self.current_window_start:
            self.current_window_start = tick_window_start
            self.tte_halted = False

        time_remaining_s = self.current_window_start + window_s - now

        if not self.tte_halted and time_remaining_s < self.tte_killswitch_s:
            self.tte_halted = True
            await self._cancel_all_orders()
            if self.yes_inventory_shares > 0 or self.no_inventory_shares > 0:
                await self._flatten_all_inventory("TTE", fair_prob)
                self.total_tte_flattens += 1
                logger.info("TTE FLATTEN (DUAL) at %.0fs remaining.", time_remaining_s)
            return

        if self.tte_halted:
            return

        # --- Toxicity Shield (dynamic) ---
        cvd = self.state.cvd_delta
        z = self.state.z_score
        toxicity_trigger = self.state.rolling_10m_volume * TOXICITY_VOLUME_PCT

        if abs(cvd) > toxicity_trigger or abs(z) > self.zscore_toxicity_threshold:
            if not self.shield_active:
                self.shield_active = True
                self.shield_end_time = now + TOXICITY_SHIELD_DURATION_S
                self.total_shield_triggers += 1
                await self._cancel_all_orders()
                logger.debug(
                    "TOXICITY SHIELD: CVD=$%+,.0f, Z=%+.2f", cvd, z,
                )
            else:
                self.shield_end_time = now + TOXICITY_SHIELD_DURATION_S

        if self.shield_active and now >= self.shield_end_time:
            self.shield_active = False

        if self.shield_active:
            return

        # --- Inventory Breach Cooldown ---
        if now < self.inventory_cooldown_end:
            return

        # --- Balance check ---
        # Need USDC for at least one BID, or inventory to sell.
        can_afford_bids = self.balance >= self.maker_size_usdc
        has_inventory = self.yes_inventory_shares > 0 or self.no_inventory_shares > 0
        if not can_afford_bids and not has_inventory:
            return

        # --- Per-token inventory in USDC ---
        yes_inv_usdc = self.yes_inventory_shares * fair_prob
        no_inv_usdc = self.no_inventory_shares * no_fair

        # --- Inventory Hard Limit Breach (either token) ---
        if yes_inv_usdc > self.inventory_hard_limit:
            await self._cancel_all_orders()
            await self._flatten_token_inventory("yes", "INVENTORY_BREACH", fair_prob)
            self.inventory_cooldown_end = now + INVENTORY_BREACH_COOLDOWN_S
            return
        if no_inv_usdc > self.inventory_hard_limit:
            await self._cancel_all_orders()
            await self._flatten_token_inventory("no", "INVENTORY_BREACH", fair_prob)
            self.inventory_cooldown_end = now + INVENTORY_BREACH_COOLDOWN_S
            return

        # --- Determine per-token skew state ---
        # If YES inventory > skew threshold: pause YES BID, place YES ASK.
        # If NO inventory > skew threshold: pause NO BID, place NO ASK.
        yes_skewed = yes_inv_usdc > self.inventory_skew_threshold
        no_skewed = no_inv_usdc > self.inventory_skew_threshold

        # --- Compute bid prices ---
        yes_bid_price = max(fair_prob - self.half_spread, 0.01)
        no_bid_price = max(no_fair - self.half_spread, 0.01)

        # --- Rule 5: Grind Shield — suppress bids on side with excess flow ---
        want_yes_bid = True
        want_no_bid = True
        grind_cvd = self.state.cvd_delta
        grind_trigger = self.state.rolling_10m_volume * GRIND_VOLUME_PCT

        if grind_cvd > grind_trigger:
            # Heavy buying flow → pause YES BID (we'd buy into momentum).
            want_yes_bid = False
            self.total_grind_pauses += 1
        elif grind_cvd < -grind_trigger:
            # Heavy selling flow → pause NO BID.
            want_no_bid = False
            self.total_grind_pauses += 1

        # --- Apply skew overrides ---
        if yes_skewed:
            want_yes_bid = False
            # Place ASK to flatten YES inventory.
            yes_ask_price = max(fair_prob, 0.01)  # At fair or better.
            await self._place_skew_ask("yes", yes_ask_price, self.yes_inventory_shares)
        else:
            # Cancel any lingering YES ASK if skew resolved.
            if self.yes_ask_order is not None:
                await self._cancel_order(self.yes_ask_order)
                self.yes_ask_order = None

        if no_skewed:
            want_no_bid = False
            no_ask_price = max(no_fair, 0.01)
            await self._place_skew_ask("no", no_ask_price, self.no_inventory_shares)
        else:
            if self.no_ask_order is not None:
                await self._cancel_order(self.no_ask_order)
                self.no_ask_order = None

        # --- Zero-Edge Exit (dual): sell at fair to unwind ---
        in_time_decay = time_remaining_s <= self.time_decay_s
        zero_edge_active = time_remaining_s < self.zero_edge_s

        if zero_edge_active:
            self.total_zero_edge_quotes += 1
            # Place ASKs at fair to unwind any inventory.
            if self.yes_inventory_shares > 0:
                await self._place_skew_ask("yes", max(fair_prob, 0.01), self.yes_inventory_shares)
                want_yes_bid = False
            if self.no_inventory_shares > 0:
                await self._place_skew_ask("no", max(no_fair, 0.01), self.no_inventory_shares)
                want_no_bid = False

        # --- Place dual BIDs ---
        final_yes = yes_bid_price if (want_yes_bid and can_afford_bids) else None
        final_no = no_bid_price if (want_no_bid and can_afford_bids) else None
        logger.debug(
            "DUAL QUOTE: fair=%.4f no_fair=%.4f | "
            "YES_BID=%.4f(%s) NO_BID=%.4f(%s) | "
            "bal=$%.2f yes_inv=%.1f no_inv=%.1f | "
            "tte=%.0fs skew_y=%s skew_n=%s",
            fair_prob, no_fair,
            yes_bid_price, "ON" if final_yes else "OFF",
            no_bid_price, "ON" if final_no else "OFF",
            self.balance, self.yes_inventory_shares, self.no_inventory_shares,
            time_remaining_s, yes_skewed, no_skewed,
        )
        await self._place_dual_bids(
            yes_bid_price=final_yes,
            no_bid_price=final_no,
        )

    # ------------------------------------------------------------------
    # Fill Detection
    # ------------------------------------------------------------------

    FILL_CHECK_INTERVAL_S = 1.0  # Poll every 1 second (was 3s).

    async def _seed_existing_trades(self) -> None:
        """
        Mark all existing trades as already processed so _check_fills
        only tracks NEW fills from this session forward. Seeds both tokens.
        """
        for tid in [self.token_id, self.no_token_id]:
            if not tid:
                continue
            try:
                params = TradeParams(asset_id=tid)
                trades = await asyncio.to_thread(
                    self.client.get_trades, params,
                )
                if trades:
                    for t in trades:
                        trade_id = t.get("id", "")
                        if trade_id:
                            self._processed_trade_ids.add(trade_id)
            except Exception as exc:
                logger.warning("Failed to seed existing trades for %s...: %s", tid[:16], exc)
        logger.info(
            "Seeded %d existing trades (will only track new fills).",
            len(self._processed_trade_ids),
        )

    async def _check_fills(self, fair_prob: float) -> None:
        """
        Poll get_trades() for BOTH tokens to detect new fills.

        IMPORTANT: This is LOG-ONLY. It does NOT modify balance or inventory.
        _sync_order_state() is the authoritative state-update path — it fetches
        real balances from the exchange. If _check_fills also modified state,
        cost basis and equity would be double-counted.
        """
        if self.dry_run:
            return

        now = time.time()
        if now - self._last_fill_check < self.FILL_CHECK_INTERVAL_S:
            return
        self._last_fill_check = now

        my_owner = self._api_owner_id.lower()

        for token_id, token_label in [
            (self.token_id, "YES"),
            (self.no_token_id, "NO"),
        ]:
            if not token_id:
                continue
            try:
                params = TradeParams(asset_id=token_id)
                trades = await asyncio.to_thread(
                    self.client.get_trades, params,
                )
            except Exception as exc:
                logger.debug("get_trades failed for %s: %s", token_label, exc)
                continue

            if not trades:
                continue

            for trade in trades:
                trade_id = trade.get("id", "")
                if not trade_id or trade_id in self._processed_trade_ids:
                    continue

                for maker_order in trade.get("maker_orders", []):
                    order_owner = (
                        maker_order.get("owner", "")
                        or maker_order.get("maker_address", "")
                        or maker_order.get("makerAddress", "")
                    ).lower()
                    if order_owner != my_owner:
                        continue

                    fill_shares = float(maker_order.get("matched_amount", 0) or 0)
                    fill_price = float(maker_order.get("price", 0) or 0)
                    mo_side = maker_order.get("side", "")

                    if fill_shares <= 0:
                        continue

                    fill_usdc = fill_shares * fill_price
                    rebate = fill_usdc * maker_rebate_rate(fill_price)
                    self.total_maker_rebates += rebate

                    match_time = trade.get("match_time", "")
                    self.trade_log.append({
                        "time": time.strftime(
                            "%H:%M:%S",
                            time.localtime(int(match_time) if match_time else now),
                        ),
                        "side": f"{token_label}_{mo_side}",
                        "shares": fill_shares,
                        "price": fill_price,
                        "usdc": fill_usdc,
                        "rebate": rebate,
                        "pnl": 0.0,  # Computed at session end from exchange state.
                        "inv_after": 0.0,
                        "bal_after": self.balance,
                        "market": trade.get("market", "")[:16],
                        "trade_id": trade_id[:16],
                    })

                    logger.info(
                        "FILL DETECTED %s_%s: %.2f @ %.4f ($%.2f) rebate=$%.4f",
                        token_label, mo_side,
                        fill_shares, fill_price, fill_usdc, rebate,
                    )

                self._processed_trade_ids.add(trade_id)

    ORDER_SYNC_INTERVAL_S = 2.0   # Check open orders every 2 seconds.
    _last_order_sync: float = 0.0

    async def _sync_order_state(self) -> None:
        """
        PRIMARY fill-detection mechanism (dual-token aware).

        Checks which of our tracked orders (YES bid, NO bid, YES ask, NO ask)
        are still resting on the exchange. When an order disappears (filled or
        cancelled), immediately syncs balance and both token inventories from
        the exchange — the source of truth.
        """
        if self.dry_run:
            return

        # Nothing to sync if we have no tracked orders.
        has_orders = any([
            self.yes_bid_order, self.no_bid_order,
            self.yes_ask_order, self.no_ask_order,
        ])
        if not has_orders:
            return

        now = time.time()
        if now - self._last_order_sync < self.ORDER_SYNC_INTERVAL_S:
            return
        self._last_order_sync = now

        # Fetch open orders for BOTH tokens.
        open_ids = set()
        for tid in [self.token_id, self.no_token_id]:
            if not tid:
                continue
            try:
                open_orders = await asyncio.to_thread(
                    self.client.get_orders,
                    OpenOrderParams(asset_id=tid),
                )
                if isinstance(open_orders, list):
                    for o in open_orders:
                        oid = o.get("id", "") or o.get("order_id", "")
                        if oid:
                            open_ids.add(oid)
            except Exception as exc:
                logger.debug("get_orders sync failed for %s...: %s", tid[:16], exc)

        # Detect which tracked orders disappeared.
        any_gone = False
        for attr, label in [
            ("yes_bid_order", "YES BID"),
            ("no_bid_order", "NO BID"),
            ("yes_ask_order", "YES ASK"),
            ("no_ask_order", "NO ASK"),
        ]:
            order = getattr(self, attr)
            if order and order.order_id not in open_ids:
                logger.info(
                    "%s %s filled/gone → syncing state.",
                    label, order.order_id[:16],
                )
                setattr(self, attr, None)
                any_gone = True

        # When any order disappears, sync balance & both inventories
        # from the exchange — this is the SOLE authoritative state update.
        if any_gone:
            old_bal = self.balance
            old_yes = self.yes_inventory_shares
            old_no = self.no_inventory_shares

            real_bal = await self._fetch_usdc_balance()
            if real_bal >= 0:
                self.balance = real_bal

            # Sync YES inventory.
            real_yes = await self._fetch_token_balance(self.token_id)
            if real_yes >= 0:
                delta_yes = real_yes - old_yes
                if delta_yes > 0.001:
                    # Bought YES shares. Cost basis = shares * fair_price.
                    self.yes_cost_basis += delta_yes * self.state.fair_price
                elif delta_yes < -0.001 and old_yes > 0:
                    # Sold YES shares. Reduce cost basis proportionally.
                    sold_frac = min(1.0, abs(delta_yes) / old_yes)
                    self.yes_cost_basis *= (1.0 - sold_frac)
                    self.yes_cost_basis = max(0.0, self.yes_cost_basis)
                self.yes_inventory_shares = real_yes

            # Sync NO inventory.
            if self.no_token_id:
                real_no = await self._fetch_token_balance(self.no_token_id)
                if real_no >= 0:
                    delta_no = real_no - old_no
                    no_fair = 1.0 - self.state.fair_price
                    if delta_no > 0.001:
                        # Bought NO shares. Cost basis = shares * no_fair.
                        self.no_cost_basis += delta_no * no_fair
                    elif delta_no < -0.001 and old_no > 0:
                        sold_frac = min(1.0, abs(delta_no) / old_no)
                        self.no_cost_basis *= (1.0 - sold_frac)
                        self.no_cost_basis = max(0.0, self.no_cost_basis)
                    self.no_inventory_shares = real_no

            self.total_fills += 1
            logger.info(
                "STATE SYNCED: bal $%.2f→$%.2f | "
                "YES %.2f→%.2f (cb=$%.2f) | NO %.2f→%.2f (cb=$%.2f) | "
                "equity=$%.2f",
                old_bal, self.balance,
                old_yes, self.yes_inventory_shares, self.yes_cost_basis,
                old_no, self.no_inventory_shares, self.no_cost_basis,
                self._compute_equity(),
            )

    # ------------------------------------------------------------------
    # Order Management
    # ------------------------------------------------------------------

    async def _place_dual_bids(
        self,
        yes_bid_price: float | None = None,
        no_bid_price: float | None = None,
    ) -> None:
        """
        Place POST_ONLY BID on YES and BID on NO simultaneously.

        This is the core of the Dual-Token Bid-Only architecture:
        instead of Bid+Ask on one token, we Bid on BOTH tokens.
        fair_yes + fair_no = 1.00, so we capture spread from both sides.
        """
        # --- YES BID management ---
        if self.yes_bid_order is not None and (
            yes_bid_price is None
            or abs(self.yes_bid_order.price - yes_bid_price) > 0.0001
        ):
            await self._cancel_order(self.yes_bid_order)
            self.yes_bid_order = None

        if yes_bid_price is not None and self.yes_bid_order is None:
            size_usdc = min(self.maker_size_usdc, self.balance * 0.45)
            if size_usdc >= 0.50:
                shares = size_usdc / yes_bid_price
                self.yes_bid_order = await self._post_maker_order(
                    BUY, yes_bid_price, shares, size_usdc, "YES_BID",
                    token_id=self.token_id,
                )

        # --- NO BID management ---
        if self.no_bid_order is not None and (
            no_bid_price is None
            or abs(self.no_bid_order.price - no_bid_price) > 0.0001
        ):
            await self._cancel_order(self.no_bid_order)
            self.no_bid_order = None

        if no_bid_price is not None and self.no_bid_order is None and self.no_token_id:
            size_usdc = min(self.maker_size_usdc, self.balance * 0.45)
            if size_usdc >= 0.50:
                shares = size_usdc / no_bid_price
                self.no_bid_order = await self._post_maker_order(
                    BUY, no_bid_price, shares, size_usdc, "NO_BID",
                    token_id=self.no_token_id,
                )

    async def _place_skew_ask(
        self,
        token: str,
        ask_price: float,
        inventory_shares: float,
    ) -> None:
        """Place an ASK to flatten inventory on a specific token (skew management)."""
        attr = "yes_ask_order" if token == "yes" else "no_ask_order"
        token_id = self.token_id if token == "yes" else self.no_token_id
        label = f"{token.upper()}_ASK"

        existing = getattr(self, attr)
        if existing is not None and abs(existing.price - ask_price) > 0.0001:
            await self._cancel_order(existing)
            setattr(self, attr, None)
            existing = None

        if existing is None and inventory_shares > 0.01 and token_id:
            size_usdc = min(self.maker_size_usdc, inventory_shares * ask_price)
            shares = min(size_usdc / ask_price, inventory_shares)
            size_usdc = shares * ask_price
            if shares > 0.01:
                order = await self._post_maker_order(
                    SELL, ask_price, shares, size_usdc, label,
                    token_id=token_id,
                )
                setattr(self, attr, order)

    async def _post_maker_order(
        self,
        side: str,
        price: float,
        size: float,
        size_usdc: float,
        label: str = "",
        token_id: str = "",
    ) -> TrackedOrder | None:
        """Post a single POST_ONLY GTC order on the specified token."""
        target_token = token_id or self.token_id

        if self.dry_run:
            order_id = f"DRY_{label}_{price:.4f}_{int(time.time() * 1000)}"
            logger.info(
                "DRY RUN: WOULD PLACE POST_ONLY %s %.2f shares @ %.4f ($%.2f)",
                label, size, price, size_usdc,
            )
            return TrackedOrder(
                order_id=order_id, side=side, price=price,
                size_shares=size, size_usdc=size_usdc, placed_at=time.time(),
            )

        try:
            order_args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=target_token,
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
                logger.debug(
                    "MAKER %s: %.2f shares @ %.4f (id=%s...)",
                    label, size, price, order_id[:16],
                )
                return TrackedOrder(
                    order_id=order_id, side=side, price=price,
                    size_shares=size, size_usdc=size_usdc,
                    placed_at=time.time(),
                )
            return None

        except Exception as exc:
            logger.error(
                "POST_ONLY %s failed (%.4f, %.2f shares): %s",
                label, price, size, exc,
            )
            return None

    async def _cancel_order(self, order: TrackedOrder) -> None:
        """Cancel a single resting order."""
        if self.dry_run:
            logger.debug("DRY RUN: WOULD CANCEL %s", order.order_id[:24])
            return

        try:
            await asyncio.to_thread(self.client.cancel, order.order_id)
            logger.debug("CANCELLED %s", order.order_id[:16])
        except Exception as exc:
            logger.warning("Cancel failed for %s: %s", order.order_id[:16], exc)

    async def _cancel_all_orders(self) -> None:
        """Cancel all resting maker orders (both YES and NO sides)."""
        tasks = []
        for order_attr in ("yes_bid_order", "no_bid_order", "yes_ask_order", "no_ask_order"):
            order = getattr(self, order_attr)
            if order is not None:
                tasks.append(self._cancel_order(order))
                setattr(self, order_attr, None)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Belt-and-suspenders: cancel all on exchange.
        if not self.dry_run:
            try:
                await asyncio.to_thread(self.client.cancel_all)
            except Exception as exc:
                logger.warning("cancel_all failed: %s", exc)

    # ------------------------------------------------------------------
    # Inventory Flatten (Taker Market Order)
    # ------------------------------------------------------------------

    async def _flatten_all_inventory(self, reason: str, fair_prob: float) -> None:
        """Flatten BOTH YES and NO inventory via aggressive GTC limit orders."""
        if self.yes_inventory_shares > 0:
            await self._flatten_token_inventory("yes", reason, fair_prob)
        if self.no_inventory_shares > 0:
            await self._flatten_token_inventory("no", reason, fair_prob)

    async def _flatten_token_inventory(
        self, token: str, reason: str, fair_prob: float,
    ) -> None:
        """Flatten inventory for a single token (YES or NO) via aggressive GTC cross.

        Places a GTC limit SELL order priced 5c below best bid to guarantee fill.
        Falls back to FOK market order if GTC is rejected.
        """
        if token == "yes":
            shares = self.yes_inventory_shares
            token_id = self.token_id
            cost_basis = self.yes_cost_basis
            token_fair = fair_prob
        else:
            shares = self.no_inventory_shares
            token_id = self.no_token_id
            cost_basis = self.no_cost_basis
            token_fair = 1.0 - fair_prob

        if shares <= 0 or not token_id:
            return

        # --- Fetch current BBO for aggressive pricing ---
        best_bid = 0.0
        try:
            book = await asyncio.to_thread(
                self.client.get_order_book, token_id,
            )
            bids = book.get("bids", [])
            if bids:
                best_bid = float(bids[0].get("price", 0))
        except Exception as exc:
            logger.warning("FLATTEN %s: book fetch failed (%s)", token.upper(), exc)

        CROSS_OFFSET = 0.05
        if best_bid > 0:
            price = max(best_bid - CROSS_OFFSET, 0.01)
        else:
            price = max(token_fair - CROSS_OFFSET, 0.01)

        price = round(max(0.01, min(0.99, price)), 2)

        notional = shares * price
        fee = notional * taker_fee_rate(price)
        bal_change = notional - fee
        pnl = bal_change - cost_basis

        if self.dry_run:
            self.balance += bal_change
            self.total_taker_fees += fee
            logger.info(
                "DRY RUN: %s FLATTEN %s %.2f shares @ %.4f | "
                "pnl=$%+.4f fee=$%.4f",
                reason, token.upper(), shares, price, pnl, fee,
            )
            if token == "yes":
                self.yes_inventory_shares = 0.0
                self.yes_cost_basis = 0.0
            else:
                self.no_inventory_shares = 0.0
                self.no_cost_basis = 0.0
            return

        # --- Live: aggressive GTC SELL ---
        try:
            order_args = OrderArgs(
                price=price,
                size=shares,
                side=SELL,
                token_id=token_id,
            )
            signed = await asyncio.to_thread(
                self.client.create_order, order_args,
            )
            resp = await asyncio.to_thread(
                self.client.post_order, signed, OrderType.GTC,
            )

            order_id = (
                resp.get("orderID", "")
                or resp.get("id", "")
                or resp.get("order_id", "")
            )
            if order_id:
                self.balance += bal_change
                self.total_taker_fees += fee
                if token == "yes":
                    self.yes_inventory_shares = 0.0
                    self.yes_cost_basis = 0.0
                else:
                    self.no_inventory_shares = 0.0
                    self.no_cost_basis = 0.0
                logger.info(
                    "%s FLATTEN %s: SELL %.2f @ %.4f pnl=$%+.4f fee=$%.4f [GTC cross]",
                    reason, token.upper(), shares, price, pnl, fee,
                )
            else:
                # Fallback: FOK market order.
                logger.warning(
                    "%s FLATTEN %s GTC rejected → FOK fallback.",
                    reason, token.upper(),
                )
                mkt_args = MarketOrderArgs(
                    token_id=token_id,
                    amount=shares,
                    side=SELL,
                    price=token_fair,
                )
                signed_fok = await asyncio.to_thread(
                    self.client.create_market_order, mkt_args,
                )
                resp_fok = await asyncio.to_thread(
                    self.client.post_order, signed_fok, OrderType.FOK,
                )
                if resp_fok.get("orderID"):
                    fok_notional = shares * token_fair
                    fok_fee = fok_notional * taker_fee_rate(token_fair)
                    fok_bal = fok_notional - fok_fee
                    fok_pnl = fok_bal - cost_basis
                    self.balance += fok_bal
                    self.total_taker_fees += fok_fee
                    if token == "yes":
                        self.yes_inventory_shares = 0.0
                        self.yes_cost_basis = 0.0
                    else:
                        self.no_inventory_shares = 0.0
                        self.no_cost_basis = 0.0
                    logger.info(
                        "%s FLATTEN %s (FOK): pnl=$%+.4f fee=$%.4f",
                        reason, token.upper(), fok_pnl, fok_fee,
                    )
                else:
                    logger.error(
                        "%s FLATTEN %s FAILED: held to expiry.",
                        reason, token.upper(),
                    )

        except Exception as exc:
            logger.error("%s FLATTEN %s error: %s", reason, token.upper(), exc)

    # ------------------------------------------------------------------
    # Equity & Helpers
    # ------------------------------------------------------------------

    @property
    def capital_in_unredeemed(self) -> float:
        """Sum of all pending/won unredeemed items (backward compat)."""
        return sum(item["cost_basis"] for item in self._unredeemed_items)

    def _compute_equity(self) -> float:
        """Balance + mark-to-market BOTH inventories + unredeemed capital.

        Dual-token: YES valued at fair_price, NO valued at (1 - fair_price).
        """
        fair = self.state.fair_price
        no_fair = 1.0 - fair

        # YES inventory MTM.
        yes_mtm = self.yes_inventory_shares * fair if self.yes_inventory_shares > 0 else 0.0
        # NO inventory MTM.
        no_mtm = self.no_inventory_shares * no_fair if self.no_inventory_shares > 0 else 0.0

        unredeemed_value = 0.0
        for item in self._unredeemed_items:
            if item["status"] == "won":
                unredeemed_value += item["shares"]
            else:
                unredeemed_value += item["cost_basis"]
        return self.balance + yes_mtm + no_mtm + unredeemed_value

    async def _fetch_usdc_balance(self) -> float:
        """Fetch USDC balance from Polymarket."""
        try:
            result = await asyncio.to_thread(
                self.client.get_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            )
            raw = float(result.get("balance", 0) or 0)
            # API returns raw units (6 decimals for USDC) — convert to dollars
            return raw / 1e6 if raw > 1000 else raw
        except Exception as exc:
            logger.error("USDC balance fetch failed: %s", exc)
            return self.balance

    async def _fetch_token_balance(self, token_id: str) -> float:
        """Fetch conditional token balance from Polymarket for any token ID."""
        if not token_id:
            return 0.0
        try:
            result = await asyncio.to_thread(
                self.client.get_balance_allowance,
                BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=token_id,
                ),
            )
            raw = float(result.get("balance", 0) or 0)
            return raw / 1e6 if raw > 1000 else raw
        except Exception as exc:
            logger.warning("Token balance fetch failed for %s...: %s", token_id[:16], exc)
            return 0.0

    async def _get_no_midpoint(self) -> float | None:
        """Fetch NO token midpoint from Polymarket CLOB."""
        if not self.no_token_id:
            return None
        try:
            result = await asyncio.to_thread(
                self.client.get_midpoint, self.no_token_id,
            )
            if result and "mid" in result:
                return float(result["mid"])
            return None
        except Exception as exc:
            logger.warning("NO midpoint fetch failed: %s", exc)
            return None

    async def _fetch_onchain_usdc_balance(self) -> float:
        """Fetch true USDC balance directly from Polygon blockchain.

        The CLOB API balance endpoint caches/lags after on-chain redemptions.
        This reads the actual ERC-20 balanceOf our Polymarket proxy wallet,
        giving us the ground-truth USDC balance.

        Checks both native USDC and bridged USDC.e contracts.
        """
        if not self._proxy_wallet:
            logger.debug("No proxy wallet set — cannot fetch on-chain balance.")
            return -1.0

        try:
            def _read_balance() -> float:
                usdc_contract = self._w3.eth.contract(
                    address=USDC_POLYGON_ADDRESS, abi=ERC20_BALANCE_ABI,
                )
                usdce_contract = self._w3.eth.contract(
                    address=USDCE_POLYGON_ADDRESS, abi=ERC20_BALANCE_ABI,
                )
                raw_usdc = usdc_contract.functions.balanceOf(
                    self._proxy_wallet
                ).call()
                raw_usdce = usdce_contract.functions.balanceOf(
                    self._proxy_wallet
                ).call()
                # Both USDC and USDC.e use 6 decimals on Polygon.
                return (raw_usdc + raw_usdce) / 1e6

            balance = await asyncio.to_thread(_read_balance)
            return balance
        except Exception as exc:
            logger.warning("On-chain USDC balance fetch failed: %s", exc)
            return -1.0

    async def _get_midpoint(self) -> float | None:
        """Fetch YES token midpoint from Polymarket CLOB."""
        try:
            result = await asyncio.to_thread(
                self.client.get_midpoint, self.token_id,
            )
            if result and "mid" in result:
                return float(result["mid"])
            return None
        except Exception as exc:
            logger.warning("Midpoint fetch failed: %s", exc)
            return None

    async def _check_market_resolution(self, slug: str) -> float | None:
        """
        Check if an expired market's YES token resolved to $1 or $0.

        Returns:
            1.0 if YES won, 0.0 if YES lost, None if resolution unknown.
        """
        if not slug or not self._http_session:
            return None
        try:
            url = "https://gamma-api.polymarket.com/events"
            params = {"slug": slug}
            async with self._http_session.get(url, params=params, timeout=5) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            if not data:
                return None

            markets = data[0].get("markets", [])
            if not markets:
                return None

            mkt = markets[0]
            outcome_prices_raw = mkt.get("outcomePrices", "")
            if not outcome_prices_raw:
                return None

            prices = json.loads(outcome_prices_raw)
            # prices[0] = YES/Up outcome price after resolution.
            # "1" means YES won, "0" means YES lost.
            if prices and len(prices) >= 1:
                yes_price = float(prices[0])
                if yes_price >= 0.99:
                    return 1.0
                elif yes_price <= 0.01:
                    return 0.0
            return None  # Not clearly resolved yet.
        except Exception as exc:
            logger.debug("Resolution check failed for %s: %s", slug, exc)
            return None

    # ------------------------------------------------------------------
    # Heartbeat & Monitoring
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Log state every 30 seconds. Shows real exchange balances."""
        while self._running:
            try:
                await asyncio.sleep(30.0)

                # Fetch real balances — on-chain is the source of truth.
                # The CLOB API caches/lags after on-chain redemptions.
                if not self.dry_run:
                    real_bal = await self._fetch_onchain_usdc_balance()
                    if real_bal < 0:
                        real_bal = await self._fetch_usdc_balance()
                    if real_bal > 0:
                        if self.capital_in_unredeemed > 0:
                            expected_bal = self.balance
                            surplus = real_bal - expected_bal
                            if surplus > 0.50:
                                redeemed = min(surplus, self.capital_in_unredeemed)
                                remaining = redeemed
                                new_items = []
                                for item in self._unredeemed_items:
                                    if remaining <= 0:
                                        new_items.append(item)
                                    elif item["cost_basis"] <= remaining:
                                        remaining -= item["cost_basis"]
                                    else:
                                        item["cost_basis"] -= remaining
                                        remaining = 0
                                        new_items.append(item)
                                self._unredeemed_items = new_items
                                logger.info(
                                    "REDEMPTION DETECTED: $%.2f redeemed, "
                                    "$%.2f still unredeemed.",
                                    redeemed, self.capital_in_unredeemed,
                                )
                        self.balance = real_bal

                    # Sync YES inventory.
                    real_yes = await self._fetch_token_balance(self.token_id)
                    if real_yes >= 0:
                        self.yes_inventory_shares = real_yes
                        if real_yes > 0 and self.yes_cost_basis == 0:
                            self.yes_cost_basis = real_yes * self.state.fair_price

                    # Sync NO inventory.
                    if self.no_token_id:
                        real_no = await self._fetch_token_balance(self.no_token_id)
                        if real_no >= 0:
                            self.no_inventory_shares = real_no
                            if real_no > 0 and self.no_cost_basis == 0:
                                self.no_cost_basis = real_no * (1.0 - self.state.fair_price)

                # Re-check pending unredeemed items for resolution.
                # Items confirmed LOST are removed so they stop inflating equity.
                if self._unredeemed_items:
                    updated_items = []
                    for item in self._unredeemed_items:
                        if item["status"] == "pending":
                            resolved = await self._check_market_resolution(
                                item.get("slug", ""),
                            )
                            if resolved == 0.0:
                                logger.info(
                                    "UNREDEEMED RESOLVED LOSS: '%s' cost=$%.2f "
                                    "→ removed (worthless).",
                                    item.get("slug", "?"), item["cost_basis"],
                                )
                                continue  # Drop this item — it's a confirmed loss.
                            elif resolved == 1.0:
                                item["status"] = "won"
                                logger.info(
                                    "UNREDEEMED RESOLVED WIN: '%s' shares=%.2f "
                                    "→ worth $%.2f (awaiting redemption).",
                                    item.get("slug", "?"), item["shares"],
                                    item["shares"],
                                )
                        updated_items.append(item)
                    self._unredeemed_items = updated_items

                equity = self._compute_equity()
                pnl = equity - self.starting_capital
                daily_pnl = equity - self.day_start_equity
                fair = self.state.fair_price
                yes_usdc = self.yes_inventory_shares * fair
                no_usdc = self.no_inventory_shares * (1.0 - fair)
                unrd = self.capital_in_unredeemed
                logger.info(
                    "HEARTBEAT | eq=$%.2f (bal=$%.2f + yes_mtm=$%.2f + no_mtm=$%.2f + unrd=$%.2f) | "
                    "pnl=$%+.2f day=$%+.2f | "
                    "YES=%.1f shares (cb=$%.2f) NO=%.1f shares (cb=$%.2f) | "
                    "fair=%.4f | fills=%d",
                    equity, self.balance, yes_usdc, no_usdc, unrd,
                    pnl, daily_pnl,
                    self.yes_inventory_shares, self.yes_cost_basis,
                    self.no_inventory_shares, self.no_cost_basis,
                    fair, self.total_fills,
                )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Heartbeat error: %s", exc)

    # ------------------------------------------------------------------
    # Auto-Rolling Market Rotation
    # ------------------------------------------------------------------

    async def _rotation_watcher(self) -> None:
        """
        Monitor market expiry and rotate to the next 5-minute window.

        Checks every second. When the current market expires (or is about
        to expire within the TTE killswitch window), triggers rotation:
            1. Cancel all orders and flatten inventory.
            2. Stop WS and strategy tasks.
            3. Wait for the next window to open.
            4. Discover the new market via Gamma API.
            5. Restart WS and strategy with the new token.
        """
        logger.info("Rotation watcher started for %s (auto-rolling).", self.asset)

        while self._running:
            try:
                await asyncio.sleep(1.0)

                if not self._current_market or not self._auto_rotate:
                    continue

                now = time.time()
                time_to_expiry = self._current_market.end_time - now

                # Rotate when we're past expiry (TTE killswitch already
                # handled the flatten, we just need to swap tokens).
                if time_to_expiry > self.tte_killswitch_s:
                    continue

                # Already rotating?
                if self._rotation_pending:
                    continue

                self._rotation_pending = True
                logger.info(
                    "ROTATION: Market '%s' expiring in %.0fs. "
                    "Preparing rotation...",
                    self._current_market.question, max(0, time_to_expiry),
                )

                # Wait until the window actually expires + small buffer.
                if time_to_expiry > 0:
                    await asyncio.sleep(max(0, time_to_expiry + 2.0))

                await self._rotate_to_next_market()

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Rotation watcher error: %s", exc)
                self._rotation_pending = False
                await asyncio.sleep(5.0)

    async def _rotate_to_next_market(self) -> None:
        """
        Stop current market tasks, discover next market, and restart.

        Preserves cumulative PnL, fill counters, and daily breaker state
        across rotations. Only resets per-window state (CVD, volume,
        Z-Score, fair price, TTE halt).
        """
        logger.info("=" * 62)
        logger.info("  ROTATING TO NEXT %s 5-MIN WINDOW", self.asset)
        logger.info("=" * 62)

        # 1. Cancel all orders (should already be flat from TTE killswitch).
        await self._cancel_all_orders()

        # 2. Stop WS and strategy tasks (keep heartbeat running).
        for task in [self._ws_task, self._strategy_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # 3. Discover the next active market (retry up to 30s).
        new_market: MarketInfo | None = None
        for attempt in range(6):
            if not self._running:
                return
            new_market = await find_active_5m_market(
                self._http_session, self.asset,
            )
            if new_market and new_market.yes_token_id != self.token_id:
                break
            # Same market or none found — the next window might not be
            # open yet. Wait and retry.
            logger.info(
                "Waiting for next %s market (attempt %d/6)...",
                self.asset, attempt + 1,
            )
            await asyncio.sleep(5.0)

        if not new_market:
            logger.error(
                "Failed to find next %s market after 30s. "
                "Stopping rotation.",
                self.asset,
            )
            self._rotation_pending = False
            return

        # 4. Update token and market info.
        old_token = self.token_id[:12]
        old_market_slug = self._current_market.slug if self._current_market else ""
        self.token_id = new_market.yes_token_id
        self.no_token_id = new_market.no_token_id
        self._current_market = new_market
        self._total_rotations += 1

        # 5. Reset per-window state (preserve cumulative PnL/counters).
        #    Check resolution for BOTH YES and NO held shares.
        resolved_value = await self._check_market_resolution(old_market_slug)

        for token_label, inv_shares, cost_basis in [
            ("YES", self.yes_inventory_shares, self.yes_cost_basis),
            ("NO", self.no_inventory_shares, self.no_cost_basis),
        ]:
            if inv_shares > 0:
                if resolved_value == 1.0 and token_label == "YES":
                    redemption_value = inv_shares * 1.0
                    profit = redemption_value - cost_basis
                    self._unredeemed_items.append({
                        "slug": old_market_slug,
                        "cost_basis": redemption_value,
                        "shares": inv_shares,
                        "token": token_label.lower(),
                        "status": "won",
                        "added_ts": time.time(),
                    })
                    logger.info(
                        "ROTATION %s WIN: %.2f shares → $%.2f (profit $%+.2f) → unredeemed.",
                        token_label, inv_shares, redemption_value, profit,
                    )
                elif resolved_value == 0.0 and token_label == "YES":
                    logger.info(
                        "ROTATION %s LOSS: %.2f shares (cost $%.2f) → worthless.",
                        token_label, inv_shares, cost_basis,
                    )
                elif resolved_value == 0.0 and token_label == "NO":
                    # YES lost means NO won.
                    redemption_value = inv_shares * 1.0
                    profit = redemption_value - cost_basis
                    self._unredeemed_items.append({
                        "slug": old_market_slug,
                        "cost_basis": redemption_value,
                        "shares": inv_shares,
                        "token": "no",
                        "status": "won",
                        "added_ts": time.time(),
                    })
                    logger.info(
                        "ROTATION NO WIN: %.2f shares → $%.2f (profit $%+.2f) → unredeemed.",
                        inv_shares, redemption_value, profit,
                    )
                elif resolved_value == 1.0 and token_label == "NO":
                    # YES won means NO lost.
                    logger.info(
                        "ROTATION NO LOSS: %.2f shares (cost $%.2f) → worthless.",
                        inv_shares, cost_basis,
                    )
                else:
                    # Resolution unknown — conservatively add cost basis.
                    self._unredeemed_items.append({
                        "slug": old_market_slug,
                        "cost_basis": cost_basis,
                        "shares": inv_shares,
                        "token": token_label.lower(),
                        "status": "pending",
                        "added_ts": time.time(),
                    })
                    logger.info(
                        "ROTATION %s: %.2f shares (cost $%.2f) pending → unredeemed.",
                        token_label, inv_shares, cost_basis,
                    )

        self.yes_inventory_shares = 0.0
        self.yes_cost_basis = 0.0
        self.no_inventory_shares = 0.0
        self.no_cost_basis = 0.0

        self.state = LiveStateManager()
        self.tte_halted = False
        self.shield_active = False
        self.shield_end_time = 0.0
        # Note: do NOT clear _processed_trade_ids — fills from prior windows
        # are still valid and should not be double-counted.
        self.current_window_start = (time.time() // (self.window_minutes * 60.0)) * (self.window_minutes * 60.0)

        logger.info(
            "ROTATED: %s -> %s...%s | '%s' | rotation #%d",
            old_token, self.token_id[:12], self.token_id[-6:],
            new_market.question, self._total_rotations,
        )

        # 6. Seed fair price.
        mid = await self._get_midpoint()
        if mid and mid > 0:
            self.state.fair_price = mid
            self.state.ready = True
            logger.info("Fair price seeded: %.4f", mid)

        # 7. Restart WS and strategy tasks.
        self._ws_task = asyncio.create_task(
            self._ws_trade_listener(), name="ws_trades",
        )
        self._strategy_task = asyncio.create_task(
            self._strategy_loop(), name="strategy_loop",
        )

        # Update task list for clean shutdown.
        self._tasks = [
            t for t in self._tasks
            if t and not t.done()
        ]
        self._tasks.extend([self._ws_task, self._strategy_task])

        self._rotation_pending = False
        logger.info("Rotation complete. Trading on new market.")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Market Maker Farmer — Hot Potato Protocol",
    )

    # Token source: either --asset (auto-rolling) or --token-id (manual).
    token_group = parser.add_mutually_exclusive_group(required=True)
    token_group.add_argument(
        "--asset",
        choices=["SOL", "BTC", "ETH", "XRP", "sol", "btc", "eth", "xrp"],
        help="Asset for auto-rolling 5-min market discovery (SOL/BTC/ETH/XRP).",
    )
    token_group.add_argument(
        "--token-id",
        help="Manual Polymarket YES token ID (no auto-rotation).",
    )

    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Paper trading mode (default: True). Use --no-dry-run for live.",
    )
    parser.add_argument(
        "--no-dry-run", dest="dry_run", action="store_false",
        help="LIVE MODE: real money, real orders.",
    )
    parser.add_argument(
        "--capital", type=float, default=500.0,
        help="Starting capital in USDC (default: 500, overridden by exchange in live).",
    )
    parser.add_argument(
        "--maker-size", type=float, default=3.0,
        help="USDC per maker order side (default: 3).",
    )
    parser.add_argument(
        "--window-minutes", type=float, default=5.0,
        help="Market cycle window in minutes (default: 5).",
    )
    parser.add_argument(
        "--breaker", type=float, default=-5.0,
        help="Daily MTM loss limit in USDC (default: -5.0).",
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

    # Initialize CLOB client.
    if args.dry_run:
        logger.info("DRY RUN MODE — no real orders will be placed.")
        host = os.getenv("POLY_HOST", "https://clob.polymarket.com")
        client = ClobClient(host, chain_id=137)
    else:
        client = init_clob_client()

    engine = LiveFarmerEngine(
        client=client,
        token_id=args.token_id or "",
        asset=args.asset or "",
        dry_run=args.dry_run,
        starting_capital=args.capital,
        maker_size_usdc=args.maker_size,
        window_minutes=args.window_minutes,
        macro_breaker_limit=args.breaker,
    )

    # --- Async event loop with signal handling ---
    stop_event = asyncio.Event()

    async def run() -> None:
        loop = asyncio.get_running_loop()

        def _signal_handler() -> None:
            logger.info("Signal received. Shutting down...")
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

        await engine.start()
        if not engine._running:
            return
        await stop_event.wait()
        await engine.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()

"""
Configuration and constants for the Polymarket HFT bot.
Phase 1: UnifiedOracle + RiskMonitor parameters.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
COINGLASS_API_KEY: str = os.getenv("COINGLASS_API_KEY", "")

# ---------------------------------------------------------------------------
# Asset Configuration
# ---------------------------------------------------------------------------
# Base assets the oracle will track. Add/remove as needed.
TRACKED_ASSETS: list[str] = ["BTC", "ETH"]

# Venue-specific ticker mappings.
# Binance uses @bookTicker stream (lowercase required).
# Hyperliquid WS uses the base symbol directly.
ASSET_MAPPING: dict[str, dict[str, str]] = {
    "BTC": {"binance": "btcusdt", "hyperliquid": "BTC"},
    "ETH": {"binance": "ethusdt", "hyperliquid": "ETH"},
    "SOL": {"binance": "solusdt", "hyperliquid": "SOL"},
    "DOGE": {"binance": "dogeusdt", "hyperliquid": "DOGE"},
    "MATIC": {"binance": "maticusdt", "hyperliquid": "MATIC"},
    "AVAX": {"binance": "avaxusdt", "hyperliquid": "AVAX"},
    "LINK": {"binance": "linkusdt", "hyperliquid": "LINK"},
    "ARB": {"binance": "arbusdt", "hyperliquid": "ARB"},
}

# ---------------------------------------------------------------------------
# WebSocket Endpoints
# ---------------------------------------------------------------------------
BINANCE_WS_BASE: str = "wss://stream.binance.com:9443/stream?streams="
HYPERLIQUID_WS: str = "wss://api.hyperliquid.xyz/ws"

# ---------------------------------------------------------------------------
# Coinglass REST
# ---------------------------------------------------------------------------
COINGLASS_BASE_URL: str = "https://open-api-v3.coinglass.com"
COINGLASS_LIQUIDATION_ENDPOINT: str = "/api/futures/liquidation/v2/real-time"
COINGLASS_POLL_INTERVAL_S: float = 2.0
COINGLASS_TIMEOUT_S: float = 0.5  # Hard 500ms cap per user mandate

# ---------------------------------------------------------------------------
# Fair Price Weights
# ---------------------------------------------------------------------------
HL_WEIGHT: float = 0.65
BN_WEIGHT: float = 0.35

# ---------------------------------------------------------------------------
# Z-Score Parameters
# ---------------------------------------------------------------------------
ZSCORE_SAMPLE_INTERVAL_S: float = 0.1   # 100ms sampling
ZSCORE_WINDOW_SIZE: int = 600           # 600 samples = 60 seconds exactly
ZSCORE_MIN_SAMPLES: int = 30            # Minimum samples before Z valid

# ---------------------------------------------------------------------------
# CVD Parameters
# ---------------------------------------------------------------------------
CVD_WINDOW_S: float = 60.0  # Rolling 60-second CVD delta

# ---------------------------------------------------------------------------
# RiskMonitor / Toxic Flow Parameters
# ---------------------------------------------------------------------------
TOXIC_FLOW_WINDOW_S: float = 5.0       # Rolling 5-second window
TOXIC_FLOW_THRESHOLD_USD: float = 1_000_000.0  # $1M aggregate trigger

# ---------------------------------------------------------------------------
# WebSocket Reconnect Parameters
# ---------------------------------------------------------------------------
WS_RECONNECT_DELAY_S: float = 1.0
WS_RECONNECT_MAX_DELAY_S: float = 30.0
WS_RECONNECT_BACKOFF_FACTOR: float = 2.0

# Binance: uses standard WebSocket protocol pings (opcode 0x9) — works fine.
BN_WS_PING_INTERVAL_S: float = 20.0
BN_WS_PING_TIMEOUT_S: float = 10.0

# Hyperliquid: ignores protocol pings. Requires application-level
# {"method": "ping"} JSON payloads to keep the connection alive.
HL_APP_PING_INTERVAL_S: float = 40.0

# ---------------------------------------------------------------------------
# Capital Constraints (Phase 3 reference, defined here for visibility)
# ---------------------------------------------------------------------------
STARTING_CAPITAL_USDC: float = 500.0
MAX_SINGLE_MARKET_EXPOSURE_USDC: float = 75.0

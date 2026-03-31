# Phase 1 Design: UnifiedOracle + RiskMonitor

## Overview

Phase 1 establishes a sub-second, multi-venue source of truth for the Polymarket HFT bot using tick-level order flow from Binance and Hyperliquid, plus cross-venue liquidation monitoring from Hyperliquid and Coinglass.

## Architecture

Single-process asyncio with three independent WebSocket/REST tasks and one background sampler, orchestrated from `main.py`.

```
┌─────────────────── Single asyncio Process ───────────────────┐
│  Binance WS (@bookTicker, multiplexed) ──┐                   │
│                                           ├─► UnifiedOracle   │
│  Hyperliquid WS (l2Book + trades, mux) ──┘    per-asset      │
│                                                state dict     │
│  Z-Score Sampler (100ms tick) ──────────────►  z_score field  │
│                                                               │
│  Hyperliquid WS (liquidations) ──┐                            │
│                                   ├─► RiskMonitor             │
│  Coinglass REST (2s poll, 500ms) ┘    TOXIC_FLOW_ACTIVE flag │
│                                                               │
│  Heartbeat (5s) ─────────────────────► structured log output  │
└───────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### config.py
- All constants, thresholds, and WebSocket URLs
- Asset mapping dictionary (base asset → venue-specific tickers)
- `.env` loading for COINGLASS_API_KEY

### unified_oracle.py (UnifiedOracle class)
- **Binance WS**: Connects to combined `@bookTicker` stream. Extracts `best_bid`, `best_ask`, computes `mid = (bid + ask) / 2`. Single connection for all assets via Binance multi-stream URL.
- **Hyperliquid WS**: Single connection, subscribes to `l2Book` + `trades` for each asset. L2Book provides HL midpoint. Trades provide aggressor side (`B`/`A`) for CVD.
- **Fair Price**: `synthetic_fair_price = (HL_mid * 0.65) + (BN_mid * 0.35)`. Recalculated on every venue update.
- **CVD Engine**: `deque` of `(timestamp, signed_volume)`. Buys (side=B, hitting ask) are positive, sells negative. Pruned to 60s window on every trade. `cvd_delta` = sum of signed volumes.
- **Z-Score Sampler**: Background task samples `synthetic_fair_price` every 100ms into a `deque(maxlen=600)`. Guarantees exactly 60s of normalized data. Z-Score computed via Welford's online algorithm (single-pass, numerically stable). Only valid after 30 samples (3 seconds).

### risk_monitor.py (RiskMonitor class)
- **Hyperliquid liquidation WS**: Dedicated connection subscribing to `liquidations` channel. Computes `notional = px * sz` per event.
- **Coinglass REST poller**: `aiohttp` with `ClientTimeout(total=0.5)` (500ms hard cap). Polls every 2s. Deduplicates via `createTime` watermark. Sums `longVolUsd + shortVolUsd` per symbol.
- **Aggregation**: Both sources feed into a shared `deque` of `(timestamp, usd_value, source)`. Pruned to 5s window. `TOXIC_FLOW_ACTIVE = True` if `sum(window) >= $1M`.
- **State transitions**: Logged at CRITICAL level on activation, INFO on clear.

### main.py
- Parses `--assets` and `--log-level` CLI args
- Launches oracle, risk monitor, and heartbeat as `asyncio.create_task`
- Heartbeat logs per-asset state every 5s (BN_mid, HL_mid, SFP, CVD, Z-Score, toxic flow)
- Graceful shutdown via SIGINT/SIGTERM → stops all subsystems in order

## Key Decisions

1. **@bookTicker over @trade**: True mathematical midpoint `(bid+ask)/2` rather than last traded price, for accurate fair price weighting.
2. **Fixed-rate Z-Score sampling**: 100ms sampling into `deque(maxlen=600)` avoids variable tick rate skewing the 60s window.
3. **Welford's algorithm**: Single-pass mean+variance, numerically stable, O(n) with n=600 at worst.
4. **Separate WS connections for Oracle vs RiskMonitor**: Hyperliquid l2Book/trades on one connection, liquidations on another. Isolation prevents a subscription issue on one from affecting the other.
5. **Coinglass dedup via createTime watermark**: Prevents double-counting when poll windows overlap.
6. **OR-logic for toxic flow**: Either venue exceeding threshold triggers the flag.

## Files

```
polymarket_hft/
├── .env.example
├── .gitignore
├── config.py
├── unified_oracle.py
├── risk_monitor.py
├── main.py
└── requirements.txt
```

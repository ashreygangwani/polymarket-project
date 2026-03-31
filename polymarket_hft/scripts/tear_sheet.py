"""
Polymarket Trade Tear Sheet — prints all transactions with PnL analysis.

Usage:
    python scripts/tear_sheet.py
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from collections import defaultdict
from web3 import Web3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import dotenv_values

env = dotenv_values(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "polymarket_hft", ".env")
)
# Try project-level .env if nested path didn't work
if not env.get("POLY_PK"):
    env = dotenv_values(".env")

for k, v in env.items():
    os.environ[k] = v

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds, TradeParams, BalanceAllowanceParams, AssetType,
)


def main():
    pk = os.environ["POLY_PK"]
    host = os.environ.get("POLY_HOST", "https://clob.polymarket.com")
    creds = ApiCreds(
        api_key=os.environ["CLOB_API_KEY"],
        api_secret=os.environ["CLOB_SECRET"],
        api_passphrase=os.environ["CLOB_PASS_PHRASE"],
    )
    sig = int(os.environ.get("POLY_SIG_TYPE", "0"))
    funder = os.environ.get("POLY_FUNDER", "")
    client = ClobClient(
        host, key=pk, chain_id=137, creds=creds,
        signature_type=sig, funder=funder or None,
    )
    my_owner = os.environ["CLOB_API_KEY"]

    # ------------------------------------------------------------------
    # Fetch all trades
    # ------------------------------------------------------------------
    trades = client.get_trades(TradeParams())
    if not trades:
        print("No trades found.")
        return

    # ------------------------------------------------------------------
    # Parse fills
    # ------------------------------------------------------------------
    all_fills = []
    seen = set()

    for t in trades:
        trade_id = t.get("id", "")
        match_time = int(t.get("match_time", 0) or 0)
        status = t.get("status", "")
        tx_hash = (t.get("transaction_hash") or "")[:16]
        trader_side = t.get("trader_side", "")

        # Our MAKER fills
        for mo in t.get("maker_orders", []):
            if mo.get("owner") != my_owner:
                continue
            key = (trade_id, mo.get("order_id", ""))
            if key in seen:
                continue
            seen.add(key)
            shares = float(mo.get("matched_amount", 0) or 0)
            price = float(mo.get("price", 0) or 0)
            all_fills.append({
                "trade_id": trade_id,
                "time": match_time,
                "role": "MAKER",
                "side": mo.get("side", ""),
                "price": price,
                "shares": shares,
                "usdc": shares * price,
                "outcome": mo.get("outcome", ""),
                "asset": mo.get("asset_id", "")[:16],
                "status": status,
                "tx": tx_hash,
            })

        # Our TAKER fills
        if t.get("owner") == my_owner and trader_side == "TAKER":
            key = (trade_id, "taker")
            if key in seen:
                continue
            seen.add(key)
            shares = float(t.get("size", 0) or 0)
            price = float(t.get("price", 0) or 0)
            all_fills.append({
                "trade_id": trade_id,
                "time": match_time,
                "role": "TAKER",
                "side": t.get("side", ""),
                "price": price,
                "shares": shares,
                "usdc": shares * price,
                "outcome": t.get("outcome", ""),
                "asset": t.get("asset_id", "")[:16],
                "status": status,
                "tx": tx_hash,
            })

    all_fills.sort(key=lambda x: x["time"])

    # ------------------------------------------------------------------
    # Check resolutions for each unique asset
    # ------------------------------------------------------------------
    asset_markets = {}  # asset_prefix -> list of fills
    for f in all_fills:
        asset_markets.setdefault(f["asset"], []).append(f)

    # ------------------------------------------------------------------
    # Compute per-market PnL (FIFO cost basis)
    # ------------------------------------------------------------------
    # Group fills by market (asset_id prefix)
    market_groups = defaultdict(list)
    for f in all_fills:
        market_groups[f["asset"]].append(f)

    # ------------------------------------------------------------------
    # Print tear sheet
    # ------------------------------------------------------------------
    W = 110
    print("=" * W)
    print("  POLYMARKET TRADE TEAR SHEET")
    print("=" * W)
    t0 = datetime.fromtimestamp(all_fills[0]["time"]).strftime("%Y-%m-%d %H:%M")
    t1 = datetime.fromtimestamp(all_fills[-1]["time"]).strftime("%Y-%m-%d %H:%M")
    print(f"  Period:  {t0}  to  {t1}")
    print(f"  Trades:  {len(all_fills)}")
    print()

    # Header
    hdr = (
        f"  {'#':>3}  {'Time':>19}  {'Role':>6}  {'Side':>4}  "
        f"{'Outcome':>7}  {'Shares':>8}  {'Price':>7}  "
        f"{'USDC':>10}  {'Status':>10}"
    )
    print(hdr)
    print("  " + "-" * (W - 4))

    total_buy_usdc = 0.0
    total_sell_usdc = 0.0
    total_buy_shares = 0.0
    total_sell_shares = 0.0
    maker_n = 0
    taker_n = 0

    for i, f in enumerate(all_fills, 1):
        ts = datetime.fromtimestamp(f["time"]).strftime("%Y-%m-%d %H:%M:%S")
        sign = "+" if f["side"] == "BUY" else "-"
        usdc_str = f"{sign}${f['usdc']:>7.2f}"

        row = (
            f"  {i:>3}  {ts:>19}  {f['role']:>6}  {f['side']:>4}  "
            f"{f['outcome']:>7}  {f['shares']:>8.2f}  {f['price']:>7.4f}  "
            f"{usdc_str:>10}  {f['status']:>10}"
        )
        print(row)

        if f["side"] == "BUY":
            total_buy_usdc += f["usdc"]
            total_buy_shares += f["shares"]
        else:
            total_sell_usdc += f["usdc"]
            total_sell_shares += f["shares"]
        if f["role"] == "MAKER":
            maker_n += 1
        else:
            taker_n += 1

    print("  " + "-" * (W - 4))
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    net_shares = total_buy_shares - total_sell_shares
    net_usdc = total_sell_usdc - total_buy_usdc  # Positive = net cash in

    # Fetch current USDC balance — CLOB API (often cached/stale).
    try:
        bal_res = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        raw = float(bal_res.get("balance", 0) or 0)
        clob_usdc_bal = raw / 1e6 if raw > 1000 else raw
    except Exception:
        clob_usdc_bal = -1

    # Fetch on-chain USDC balance — source of truth (Polygon).
    onchain_usdc_bal = -1.0
    proxy_wallet = os.environ.get("POLY_FUNDER", "").strip()
    rpc_url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
    if proxy_wallet:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            proxy_addr = Web3.to_checksum_address(proxy_wallet)
            erc20_abi = [
                {
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function",
                }
            ]
            # Native USDC on Polygon.
            usdc_addr = Web3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")
            usdc_contract = w3.eth.contract(address=usdc_addr, abi=erc20_abi)
            raw_usdc = usdc_contract.functions.balanceOf(proxy_addr).call()
            # Bridged USDC.e on Polygon.
            usdce_addr = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
            usdce_contract = w3.eth.contract(address=usdce_addr, abi=erc20_abi)
            raw_usdce = usdce_contract.functions.balanceOf(proxy_addr).call()
            onchain_usdc_bal = (raw_usdc + raw_usdce) / 1e6
        except Exception as e:
            print(f"  [WARN] On-chain balance fetch failed: {e}")

    print("  SUMMARY")
    print("  " + "=" * 55)
    print(f"  Total Buys:         {total_buy_shares:>8.2f} shares   ${total_buy_usdc:>8.2f}")
    print(f"  Total Sells:        {total_sell_shares:>8.2f} shares   ${total_sell_usdc:>8.2f}")
    print(f"  Net Shares Held:    {net_shares:>+8.2f} (pending redemption)")
    print(f"  Net USDC Flow:     ${net_usdc:>+8.2f} (sells - buys)")
    print(f"  Maker Fills:        {maker_n}")
    print(f"  Taker Fills:        {taker_n}")
    print()
    if onchain_usdc_bal >= 0:
        print(f"  On-Chain USDC Bal: ${onchain_usdc_bal:>8.2f}  (Polygon — source of truth)")
    if clob_usdc_bal >= 0:
        print(f"  CLOB API USDC Bal: ${clob_usdc_bal:>8.2f}  (may be cached/stale)")
    if onchain_usdc_bal >= 0 and clob_usdc_bal >= 0:
        diff = onchain_usdc_bal - clob_usdc_bal
        if abs(diff) > 0.50:
            print(f"  ⚠ Discrepancy:    ${diff:>+8.2f}  (on-chain vs CLOB)")
    print("  " + "=" * 55)

    # ------------------------------------------------------------------
    # Per-market breakdown
    # ------------------------------------------------------------------
    print()
    print("  PER-MARKET BREAKDOWN")
    print("  " + "=" * 55)

    for asset, fills in sorted(market_groups.items(), key=lambda x: x[1][0]["time"]):
        buys = [f for f in fills if f["side"] == "BUY"]
        sells = [f for f in fills if f["side"] == "SELL"]
        buy_usdc = sum(f["usdc"] for f in buys)
        sell_usdc = sum(f["usdc"] for f in sells)
        buy_shares = sum(f["shares"] for f in buys)
        sell_shares = sum(f["shares"] for f in sells)
        net_sh = buy_shares - sell_shares
        outcome = fills[0]["outcome"]
        ts = datetime.fromtimestamp(fills[0]["time"]).strftime("%m-%d %H:%M")

        # Simple PnL: sells - buys. Unredeemed shares not counted.
        realized = sell_usdc - buy_usdc
        status = "OPEN" if net_sh > 0.01 else "CLOSED"

        print(
            f"  {ts}  {outcome:>4}  "
            f"B:{buy_shares:>6.1f}@${buy_usdc:>5.1f}  "
            f"S:{sell_shares:>6.1f}@${sell_usdc:>5.1f}  "
            f"net={net_sh:>+6.1f}sh  "
            f"pnl=${realized:>+6.2f}  [{status}]"
        )

    print("  " + "=" * 55)


if __name__ == "__main__":
    main()

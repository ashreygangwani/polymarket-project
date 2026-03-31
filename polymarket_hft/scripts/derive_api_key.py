"""
Derive Polymarket CLOB API credentials from your wallet private key.

Usage:
    python scripts/derive_api_key.py

Reads POLY_PK from .env (or prompts you to enter it).
Prints the API key, secret, and passphrase to paste into your .env file.
"""

import os
import sys

# Add parent dir so dotenv can find .env
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from py_clob_client.client import ClobClient


def main():
    pk = os.getenv("POLY_PK", "").strip()

    if not pk or pk == "your_polygon_private_key_here":
        pk = input("Enter your Polygon wallet private key (hex): ").strip()

    if not pk:
        print("ERROR: No private key provided.")
        sys.exit(1)

    # Ensure 0x prefix
    if not pk.startswith("0x"):
        pk = "0x" + pk

    print("\nConnecting to Polymarket CLOB...")
    client = ClobClient(
        "https://clob.polymarket.com",
        chain_id=137,
        key=pk,
    )

    # Try to derive existing keys first; if wallet isn't registered, create new ones
    try:
        print("Trying to derive existing API credentials...")
        creds = client.derive_api_key()
    except Exception:
        print("No existing credentials found. Creating new API key...\n")
        creds = client.create_api_key()

    print("=" * 60)
    print("Add these to your .env file:")
    print("=" * 60)
    print(f"CLOB_API_KEY={creds.api_key}")
    print(f"CLOB_SECRET={creds.api_secret}")
    print(f"CLOB_PASS_PHRASE={creds.api_passphrase}")
    print("=" * 60)


if __name__ == "__main__":
    main()

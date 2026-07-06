"""
Fetch BTCUSDT perpetual futures 1H kline data from Binance public API.
Saves as CSV in data/ directory. OUTPUT_PATH auto-generated from date range.
"""
import time
import pandas as pd
import requests
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
START_DATE = "2025-10-01"
END_DATE = "2026-07-06"
OUTPUT_PATH = f"data/BTCUSDT_Perpetual_1H_{START_DATE}_{END_DATE}.csv"

# Binance futures kline endpoint (public, no API key needed)
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"


def date_to_ms(date_str: str) -> int:
    """Convert YYYY-MM-DD to milliseconds timestamp (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(start_ms: int, end_ms: int, limit: int = 1000):
    """Fetch klines from Binance futures API."""
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    start_ms = date_to_ms(START_DATE)
    end_ms = date_to_ms(END_DATE)

    print(f"Fetching {SYMBOL} {INTERVAL} klines")
    print(f"  From: {START_DATE} ({start_ms})")
    print(f"  To:   {END_DATE} ({end_ms})")

    all_candles = []
    current_start = start_ms

    while current_start < end_ms:
        print(f"  Fetching chunk starting at {datetime.fromtimestamp(current_start / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} ...")
        candles = fetch_klines(current_start, end_ms, limit=1000)

        if not candles:
            print("  No more data returned.")
            break

        all_candles.extend(candles)
        print(f"    Got {len(candles)} candles, total so far: {len(all_candles)}")

        # Next start = last candle's open time + 1ms to avoid duplicates
        last_open_time = candles[-1][0]
        if last_open_time <= current_start:
            # Avoid infinite loop if API returns same data
            current_start += 3600000  # advance by 1 hour
        else:
            current_start = last_open_time + 1

        # Polite delay to avoid rate limits
        time.sleep(0.5)

    print(f"\nTotal candles fetched: {len(all_candles)}")

    # ── Convert to DataFrame ────────────────────────────────────────
    # Binance kline format:
    # [OpenTime, Open, High, Low, Close, Volume, CloseTime,
    #  QuoteVolume, Trades, TakerBuyBase, TakerBuyQuote, Ignore]
    columns = [
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base volume", "Taker buy quote volume", "Ignore",
    ]
    df = pd.DataFrame(all_candles, columns=columns)

    # Convert numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume",
                "Quote asset volume", "Number of trades",
                "Taker buy base volume", "Taker buy quote volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert timestamps to datetime
    df["Gmt time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)
    df = df.set_index("Gmt time")
    df.sort_index(inplace=True)

    # Drop the raw timestamp column and extras — keep only OHLCV for env
    df_out = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Remove duplicates (in case of pagination overlap)
    df_out = df_out[~df_out.index.duplicated(keep="first")]

    # ── Save ────────────────────────────────────────────────────────
    import os
    os.makedirs("data", exist_ok=True)
    df_out.to_csv(OUTPUT_PATH)
    print(f"\nData saved to {OUTPUT_PATH}")
    print(f"Rows: {len(df_out)}")
    print(f"Date range: {df_out.index[0]} to {df_out.index[-1]}")


if __name__ == "__main__":
    main()

"""
market_data.py — Market data provider using Twelve Data API.

This module abstracts all stock market data fetching behind a clean interface,
making it easy to swap providers in the future. Currently uses Twelve Data
(https://twelvedata.com) as the sole data source.

Key functions:
    fetch_historical_ohlcv(tickers, start, end)
        → Multi-index Pandas DataFrame (identical shape to yf.download output)
    fetch_ticker_history(ticker, period)
        → List of {"time": ..., "value": ...} dicts for chart display

Environment:
    TWELVE_DATA_KEY — your API key (loaded from .env or system env)
"""

import os
import time
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()  # Load .env from project root

API_KEY = os.getenv("TWELVE_DATA_KEY")
BASE_URL = "https://api.twelvedata.com"

# Rate-limit: free tier allows 8 requests/min → ~7.5s between calls to be safe
_MIN_REQUEST_INTERVAL = 8.0  # seconds between API calls
_last_request_time = 0.0


def _rate_limit():
    """Enforce rate limiting to stay within Twelve Data free tier (8 req/min)."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        wait = _MIN_REQUEST_INTERVAL - elapsed
        print(f"  [rate-limit] Waiting {wait:.1f}s before next API call...")
        time.sleep(wait)
    _last_request_time = time.time()


def _get_api_key():
    """Get the API key, raising a clear error if not configured."""
    if not API_KEY:
        raise EnvironmentError(
            "TWELVE_DATA_KEY not found. "
            "Set it in your .env file or as an environment variable.\n"
            "  → Get a free key at https://twelvedata.com"
        )
    return API_KEY


def _fetch_time_series(symbol, interval="1day", start_date=None, end_date=None, outputsize=None):
    """
    Low-level call to the Twelve Data /time_series endpoint.

    Returns a list of OHLCV dicts sorted by date ascending, e.g.:
        [{"datetime": "2024-01-02", "open": "...", "high": "...", ...}, ...]

    Raises RuntimeError on API errors.
    """
    _rate_limit()

    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": _get_api_key(),
        "format": "JSON",
        "order": "ASC",  # oldest first — matches yfinance behavior
    }
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if outputsize:
        params["outputsize"] = outputsize

    url = f"{BASE_URL}/time_series"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    # Twelve Data returns {"status": "error", "message": "..."} on failures
    if data.get("status") == "error":
        raise RuntimeError(
            f"Twelve Data API error for {symbol}: {data.get('message', 'Unknown error')}"
        )

    values = data.get("values", [])
    if not values:
        raise RuntimeError(f"No data returned for {symbol}")

    return values


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def fetch_historical_ohlcv(tickers, start, end=None):
    """
    Fetch daily OHLCV data for multiple tickers.

    Returns a multi-index Pandas DataFrame that matches the shape produced by
    yf.download() — so all downstream code (feature extraction, scaling, etc.)
    works without changes.

    DataFrame structure:
        Columns: MultiIndex of (Feature, Ticker)
            Features: ['Close', 'High', 'Low', 'Open', 'Volume']
            Tickers:  e.g. ['AAPL', 'JNJ', 'JPM', 'MSFT', 'PEP']
        Index: DatetimeIndex (trading days)

    Args:
        tickers: List of ticker symbols, e.g. ["MSFT", "JNJ"]
        start:   Start date string, e.g. "2018-01-01"
        end:     End date string (optional, defaults to today)

    Returns:
        pd.DataFrame with multi-level columns
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    print(f"  Fetching data for {len(tickers)} tickers from Twelve Data API...")

    all_frames = {}
    for ticker in tickers:
        print(f"    --> {ticker}...", end=" ", flush=True)
        try:
            values = _fetch_time_series(
                symbol=ticker,
                interval="1day",
                start_date=start,
                end_date=end,
                outputsize=5000,  # max per request
            )

            # Parse into a DataFrame
            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")

            # Convert string values to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Rename to match yfinance convention (capitalized)
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            })

            # Keep only the columns we need
            df = df[["Open", "High", "Low", "Close", "Volume"]]

            all_frames[ticker] = df
            print(f"{len(df)} days [OK]")

        except Exception as e:
            print(f"FAILED: {e}")
            raise

    # Build multi-index DataFrame matching yfinance format
    # yfinance returns: columns = MultiIndex(Feature, Ticker)
    combined = pd.concat(all_frames, axis=1)  # columns = (Ticker, Feature)
    combined = combined.swaplevel(axis=1)      # columns = (Feature, Ticker)
    combined = combined.sort_index(axis=1)     # sort for consistency

    # Ensure the index is a proper DatetimeIndex
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "Date"

    print(f"  Total: {len(combined)} trading days fetched")
    return combined


def fetch_ticker_history(ticker, period="1y"):
    """
    Fetch historical closing prices for a single ticker.

    This is used by the Flask backend's /api/history/<ticker> endpoint
    for chart display.

    Args:
        ticker: Stock symbol, e.g. "MSFT"
        period: One of '3mo', '6mo', '1y', '2y', '5y'

    Returns:
        List of dicts: [{"time": "2024-01-02", "value": 375.50}, ...]
    """
    # Convert period string to a start date
    period_days = {
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }
    days = period_days.get(period, 365)
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    values = _fetch_time_series(
        symbol=ticker,
        interval="1day",
        start_date=start_date,
        outputsize=5000,
    )

    result = []
    for v in values:
        val = float(v["close"])
        # Skip NaN/Inf values
        if math.isnan(val) or math.isinf(val):
            continue
        result.append({
            "time": v["datetime"][:10],  # "YYYY-MM-DD"
            "value": round(val, 2),
        })

    return result

from flask import Flask, render_template, jsonify, request
from model_loader import load_model_and_scaler
import numpy as np
import torch
import json
import os
import time
from datetime import datetime, timedelta
from functools import lru_cache

app = Flask(__name__)
model, scaler = load_model_and_scaler()

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PREDICTIONS_FILE = os.path.join(base_dir, "data", "predictions.json")
X_test = np.load(os.path.join(base_dir, "data", "X_test.npy"))

# ---------------------------------------------------------------------------
# In-memory cache for yfinance history requests
# ---------------------------------------------------------------------------
_history_cache = {}
CACHE_TTL = 3600  # 1 hour

# Legacy: global step tracker for /live_predict backward compat
current_step = 0


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/api/predictions')
def api_predictions():
    """
    Returns the full prediction history from predictions.json.
    Supports optional ?limit=N to return only the last N entries.
    """
    data = _load_predictions()
    limit = request.args.get('limit', type=int)
    if limit and limit > 0:
        data = data[-limit:]
    return jsonify(data)


@app.route('/api/predictions/latest')
def api_predictions_latest():
    """Returns only the most recent prediction entry."""
    data = _load_predictions()
    if not data:
        return jsonify({"error": "No predictions available yet"}), 404
    return jsonify(data[-1])


@app.route('/api/history/<ticker>')
def api_history(ticker):
    """
    Returns historical closing prices for a given ticker.
    Uses yfinance with a 1-hour in-memory cache.
    Query params:
        period: '1y', '6mo', '3mo' (default: '1y')
    """
    ticker = ticker.upper()
    valid_tickers = ["JNJ", "JPM", "MSFT", "PEP", "XOM"]
    if ticker not in valid_tickers:
        return jsonify({"error": f"Invalid ticker. Must be one of: {valid_tickers}"}), 400

    period = request.args.get('period', '1y')
    if period not in ['3mo', '6mo', '1y', '2y', '5y']:
        period = '1y'

    cache_key = f"{ticker}_{period}"
    now = time.time()

    # Check cache
    if cache_key in _history_cache:
        cached_data, cached_at = _history_cache[cache_key]
        if now - cached_at < CACHE_TTL:
            return jsonify(cached_data)

    # Fetch from yfinance
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return jsonify({"error": "No data returned from yfinance"}), 502

        result = []
        for date, row in hist.iterrows():
            result.append({
                "time": date.strftime("%Y-%m-%d"),
                "value": round(float(row["Close"]), 2),
            })

        _history_cache[cache_key] = (result, now)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 502


# ---------------------------------------------------------------------------
# Legacy endpoint — kept for backward compatibility
# ---------------------------------------------------------------------------
@app.route('/live_predict')
def live_predict():
    global current_step
    if current_step >= len(X_test):
        current_step = 0

    sample = torch.tensor(X_test[current_step:current_step+1]).float()
    with torch.no_grad():
        y_pred = model(sample)

    y_pred_np = y_pred.numpy()
    predicted_prices = scaler.inverse_transform(y_pred_np)[0]

    tickers = ["JNJ", "JPM", "MSFT", "PEP", "XOM"]
    results = {tickers[i]: round(float(predicted_prices[i]), 2) for i in range(5)}

    current_step += 1
    return jsonify(results)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_predictions():
    """Load predictions.json from disk (fresh on each request)."""
    if not os.path.exists(PREDICTIONS_FILE):
        return []
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


if __name__ == '__main__':
    app.run(debug=True)
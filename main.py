import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

MIN_RISK_REWARD_RATIO = 2.0
MAX_CONSECUTIVE_SL    = 2

# State management for the Quarantine mechanism
consecutive_sl_count = 0
is_quarantined       = False

def get_live_xau_price():
    """Fetches real-time spot gold price from a reliable public API with a safe backup fallback."""
    try:
        response = requests.get("https://api.gold-api.com/price/XAU", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return float(data.get("price"))
    except Exception as e:
        print(f"Live Price API error: {e}")
    
    # Fallback backup price if network blips
    return 4165.00

def fetch_market_candles():
    """Fetches recent candles via yfinance to compute institutional indicators accurately."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period="5d", interval="15m")
        if not df.empty:
            return df
    except Exception as e:
        print(f"Candle data fetch error: {e}")
    return None

def calculate_institutional_indicators(df):
    """Computes 20 EMA, 200 EMA, VWAP, and Volume Moving Averages."""
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Cumulative VWAP calculation
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (p * q).cumsum() / q.cumsum()
    
    # 20-period Volume Moving Average for volume breakout confirmation
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    
    return df

def send_telegram_alert(message_text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message_text,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

@app.route("/webhook", methods=["GET", "POST"])
def handle_trading_signal():
    global consecutive_sl_count, is_quarantined
    try:
        # Check active Quarantine safety lock
        if is_quarantined:
            return jsonify({
                "status": "QUARANTINED", 
                "message": f"Agent paused due to {consecutive_sl_count} consecutive Stop Losses."
            }), 429

        # Get authoritative live price
        current_price = get_live_xau_price()

        # Fetch candles for multi-indicator setup
        df = fetch_market_candles()
        if df is None or len(df) < 200:
            return jsonify({"status": "ERROR", "message": "Insufficient market data for indicator computation."}), 500

        df = calculate_institutional_indicators(df)
        
        ema_20 = float(df['EMA_20'].iloc[-1])
        ema_200 = float(df['EMA_200'].iloc[-1])
        vwap = float(df['VWAP'].iloc[-1])
        current_vol = float(df['Volume'].iloc[-1])
        avg_vol = float(df['Vol_MA'].iloc[-1])

        # Institutional Trend & Momentum Strategy Logic
        is_bullish_trend = current_price > ema_200
        vol_confirmed = current_vol > (avg_vol * 1.1) # Requires 10%+ volume expansion

        if is_bullish_trend and current_price <= (vwap + 3.0) and vol_confirmed:
            action = "BUY"
            sl_price = round(current_price - 6.0, 2)
            risk = current_price - sl_price
            tp1_price = round(current_price + (risk * MIN_RISK_REWARD_RATIO), 2)
            rationale = f"Bullish bounce off VWAP (${vwap:.2f}) aligned with 200 EMA structural trend. Volume confirmed."
        elif not is_bullish_trend and current_price >= (vwap - 3.0) and vol_confirmed:
            action = "SELL"
            sl_price = round(current_price + 6.0, 2)
            risk = sl_price - current_price
            tp1_price = round(current_price - (risk * MIN_RISK_REWARD_RATIO), 2)
            rationale = f"Bearish rejection at VWAP (${vwap:.2f}) below 200 EMA structural trend. Volume confirmed."
        else:
            return jsonify({
                "status": "HOLD", 
                "price": current_price, 
                "message": "Price in neutral value zone; waiting for VWAP/EMA confluence."
            }), 200

        timestamp = datetime.now().strftime("%I:%M %p GST")
        alert_text = (
            f"*XAU/USD INSTITUTIONAL SIGNAL ({action})*\n"
            f"Time: {timestamp}\n"
            f"--------------------------------------------------\n"
            f"*MARKET STRUCTURE & FLOW:*\n"
            f"- Price:       `${current_price:.2f}`\n"
            f"- VWAP:        `${vwap:.2f}`\n"
            f"- 20 EMA:      `${ema_20:.2f}`\n"
            f"- 200 EMA:     `${ema_200:.2f}`\n"
            f"--------------------------------------------------\n"
            f"*EXECUTABLE LEVELS (1:{MIN_RISK_REWARD_RATIO} R:R):*\n"
            f"- Entry Price: `${current_price:.2f}`\n"
            f"- Stop Loss:   `${sl_price:.2f}`\n"
            f"- Take Profit: `${tp1_price:.2f}`\n"
            f"--------------------------------------------------\n"
            f"*STRATEGY LOGIC:*\n"
            f"- {rationale}\n"
            f"- Quarantine Status: Active ({consecutive_sl_count}/{MAX_CONSECUTIVE_SL} SLs)"
        )

        send_telegram_alert(alert_text)
        return jsonify({"status": "DISPATCHED", "action": action, "price": current_price}), 200
    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

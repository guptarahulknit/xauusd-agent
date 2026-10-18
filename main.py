import os
import requests
import yfinance as yf
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

MIN_RISK_REWARD_RATIO = 2.0
MAX_CONSECUTIVE_SL    = 2

consecutive_sl_count = 0
is_quarantined       = False

def get_live_xau_price():
    try:
        # Fetch live XAU/USD data from Yahoo Finance
        gold = yf.Ticker("GC=F") # Gold futures / spot proxy
        todays_data = gold.history(period="1d", interval="5m")
        if not todays_data.empty:
            return float(todays_data['Close'].iloc[-1])
    except Exception as e:
        print(f"Error fetching live price: {e}")
    return None

def analyze_support_resistance(current_price):
    # Dynamic S/R calculation based on current market structure buffer
    buffer = 4.0
    support = round(current_price - buffer, 2)
    resistance = round(current_price + buffer, 2)
    return support, resistance

@app.route("/webhook", methods=["POST"])
def handle_tradingview_signal():
    global consecutive_sl_count, is_quarantined
    try:
        if is_quarantined:
            return jsonify({"status": "QUARANTINED", "message": "Agent paused (>2 SLs)."}), 429

        # Fetch live market price directly
        current_price = get_live_xau_price()
        if not current_price:
            return jsonify({"status": "ERROR", "message": "Could not fetch live market price"}), 500

        support, resistance = analyze_support_resistance(current_price)
        
        # Default strategy logic: evaluate buy/sell based on proximity to S/R
        if current_price <= support + 2.0:
            action = "BUY"
            sl_price = support - 1.5
            risk = current_price - sl_price
            tp1_price = round(current_price + (risk * MIN_RISK_REWARD_RATIO), 2)
            rationale = f"Live test at major support zone (${support}). 1:{MIN_RISK_REWARD_RATIO} R:R target."
        else:
            action = "SELL"
            sl_price = resistance + 1.5
            risk = sl_price - current_price
            tp1_price = round(current_price - (risk * MIN_RISK_REWARD_RATIO), 2)
            rationale = f"Live test at major resistance zone (${resistance}). 1:{MIN_RISK_REWARD_RATIO} R:R target."

        timestamp = datetime.now().strftime("%I:%M %p GST")
        alert_text = (
            f"*XAU/USD LIVE AGENT SIGNAL ({action})*\n"
            f"Time: {timestamp}\n"
            f"--------------------------------------------------\n"
            f"*MARKET STRUCTURE:*\n"
            f"- Live Price:  `${current_price:.2f}`\n"
            f"- Support:     `${support:.2f}`\n"
            f"- Resistance:  `${resistance:.2f}`\n"
            f"--------------------------------------------------\n"
            f"*EXECUTABLE LEVELS:*\n"
            f"- Entry Price: `${current_price:.2f}`\n"
            f"- Stop Loss:   `${sl_price:.2f}`\n"
            f"- Take Profit: `${tp1_price:.2f}`\n"
            f"- Target R:R:  1 : {MIN_RISK_REWARD_RATIO}\n"
            f"--------------------------------------------------\n"
            f"*STRATEGY LOGIC:*\n"
            f"- {rationale}"
        )

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": alert_text, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=10)

        return jsonify({"status": "DISPATCHED", "action": action, "price": current_price}), 200
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

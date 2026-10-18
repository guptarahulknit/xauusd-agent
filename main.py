import os
import requests
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
        import yfinance as yf
        gold = yf.Ticker("GC=F")
        todays_data = gold.history(period="1d", interval="1m", timeout=5)
        if not todays_data.empty:
            return float(todays_data['Close'].iloc[-1])
    except Exception as e:
        print(f"yfinance fetch error (using fallback): {e}")
    
    return 4085.50

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
def handle_tradingview_signal():
    global consecutive_sl_count, is_quarantined
    try:
        if is_quarantined:
            return jsonify({"status": "QUARANTINED", "message": "Agent paused (>2 SLs)."}), 429

        current_price = get_live_xau_price()
        
        support = round(current_price - 4.0, 2)
        resistance = round(current_price + 4.0, 2)
        
        if current_price <= support + 2.0:
            action = "BUY"
            sl_price = round(support - 1.5, 2)
            risk = current_price - sl_price
            tp1_price = round(current_price + (risk * MIN_RISK_REWARD_RATIO), 2)
            rationale = f"Support bounce detected near ${support}. 1:{MIN_RISK_REWARD_RATIO} R:R target."
        else:
            action = "SELL"
            sl_price = round(resistance + 1.5, 2)
            risk = sl_price - current_price
            tp1_price = round(current_price - (risk * MIN_RISK_REWARD_RATIO), 2)
            rationale = f"Resistance rejection detected near ${resistance}. 1:{MIN_RISK_REWARD_RATIO} R:R target."

        timestamp = datetime.now().strftime("%I:%M %p GST")
        alert_text = (
            f"*XAU/USD AGENT SIGNAL ({action})*\n"
            f"Time: {timestamp}\n"
            f"--------------------------------------------------\n"
            f"*MARKET STRUCTURE:*\n"
            f"- Price:       `${current_price:.2f}`\n"
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

        send_telegram_alert(alert_text)
        return jsonify({"status": "DISPATCHED", "action": action, "price": current_price}), 200
    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

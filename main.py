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

def send_telegram_alert(message_text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message_text,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending telegram alert: {e}")
        return False

@app.route("/webhook", methods=["POST"])
def handle_tradingview_signal():
    global consecutive_sl_count, is_quarantined
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Invalid payload"}), 400

        if is_quarantined:
            return jsonify({
                "status": "QUARANTINED",
                "message": "Agent in Quarantine mode (>2 SLs). Signals paused."
            }), 429

        if "trade_result" in payload:
            result = payload["trade_result"].upper()
            if result == "SL":
                consecutive_sl_count += 1
                if consecutive_sl_count >= MAX_CONSECUTIVE_SL:
                    is_quarantined = True
                    send_telegram_alert("*AGENT QUARANTINE TRIGGERED*\n2 consecutive SL hits. Dispatches paused.")
            elif result == "TP":
                consecutive_sl_count = 0
            return jsonify({"status": "SUCCESS", "consecutive_sl": consecutive_sl_count}), 200

        action      = payload.get("action", "BUY").upper()
        entry_price = float(payload.get("price", 0.0))
        sl_price    = float(payload.get("sl", 0.0))
        tp1_price   = float(payload.get("tp1", 0.0))
        timeframe   = payload.get("timeframe", "5m")
        rationale   = payload.get("rationale", "Setup Triggered")

        timestamp = datetime.now().strftime("%I:%M %p GST")
        alert_text = (
            f"*XAU/USD TRADE ALERT ({action})*\n"
            f"Time: {timestamp} | Timeframe: {timeframe}\n"
            f"--------------------------------------------------\n"
            f"*EXECUTABLE LEVELS:*\n"
            f"- Entry Price: `${entry_price:.2f}`\n"
            f"- Stop Loss:   `${sl_price:.2f}`\n"
            f"- Take Profit:  `${tp1_price:.2f}`\n"
            f"- Target R:R:   1 : {MIN_RISK_REWARD_RATIO}\n"
            f"--------------------------------------------------\n"
            f"*STRATEGY LOGIC:*\n"
            f"- {rationale}"
        )

        success = send_telegram_alert(alert_text)
        if not success:
            return jsonify({"status": "ERROR", "message": "Failed to send Telegram message"}), 500

        return jsonify({"status": "DISPATCHED", "action": action}), 200
    except Exception as e:
        print(f"Webhook exception: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset_quarantine():
    global consecutive_sl_count, is_quarantined
    consecutive_sl_count = 0
    is_quarantined = False
    send_telegram_alert("AGENT UNPAUSED: Live XAU/USD monitoring resumed.")
    return jsonify({"status": "RESUMED"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

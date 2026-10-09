import os
from datetime import datetime
from flask import Flask, request, jsonify
from twilio.rest import Client

app = Flask(__name__)

TWILIO_ACCOUNT_SID   = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
USER_WHATSAPP_TO     = os.getenv("USER_WHATSAPP_TO")

MIN_RISK_REWARD_RATIO = 2.0
MAX_CONSECUTIVE_SL    = 2

consecutive_sl_count = 0
is_quarantined       = False

def send_whatsapp_alert(message_text):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            body=message_text,
            to=USER_WHATSAPP_TO
        )
        print(f"[{datetime.now().strftime('%H:%M:%S GST')}] WhatsApp Alert Sent! SID: {msg.sid}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S GST')}] Error sending alert: {e}")

@app.route("/webhook", methods=["POST"])
def handle_tradingview_signal():
    global consecutive_sl_count, is_quarantined

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
                quarantine_msg = (
                    f"AGENT QUARANTINE TRIGGERED\n\n"
                    f"- Reason: {MAX_CONSECUTIVE_SL} consecutive Stop Loss hits.\n"
                    f"- Status: Live dispatches PAUSED.\n\n"
                    f"Initiating strategy re-calibration..."
                )
                send_whatsapp_alert(quarantine_msg)
        elif result == "TP":
            consecutive_sl_count = 0

        return jsonify({"status": "SUCCESS", "consecutive_sl": consecutive_sl_count}), 200

    if datetime.now().hour >= 20:
        return jsonify({"status": "SKIPPED", "reason": "Cutoff time reached (0 overnight position rule)"}), 200

    action      = payload.get("action", "BUY").upper()
    entry_price = float(payload.get("price", 0.0))
    sl_price    = float(payload.get("sl", 0.0))
    tp1_price   = float(payload.get("tp1", 0.0))
    timeframe   = payload.get("timeframe", "5m")
    rationale   = payload.get("rationale", "15m S/R Sweep + VWAP Band Rejection")

    timestamp = datetime.now().strftime("%I:%M %p GST")

    alert_text = (
        f"XAU/USD TRADE ALERT ({action})\n"
        f"Time: {timestamp} | Timeframe: {timeframe}\n"
        f"--------------------------------------------------\n"
        f"EXECUTABLE LEVELS:\n"
        f"- Entry Price: ${entry_price:.2f}\n"
        f"- Stop Loss:   ${sl_price:.2f}\n"
        f"- Take Profit:  ${tp1_price:.2f}\n"
        f"- Target R:R:   1 : {MIN_RISK_REWARD_RATIO}\n"
        f"--------------------------------------------------\n"
        f"STRATEGY LOGIC:\n"
        f"- {rationale}\n"
        f"--------------------------------------------------\n"
        f"RULE: Day-trade setup. Close all open positions before 11:30 PM GST."
    )

    send_whatsapp_alert(alert_text)
    return jsonify({"status": "DISPATCHED", "action": action}), 200

@app.route("/reset", methods=["POST"])
def reset_quarantine():
    global consecutive_sl_count, is_quarantined
    consecutive_sl_count = 0
    is_quarantined = False
    send_whatsapp_alert("AGENT UNPAUSED: Live XAU/USD monitoring resumed.")
    return jsonify({"status": "RESUMED"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

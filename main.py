
import os
import json
import math
import sqlite3
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify

# ============================================================
# PROFESSIONAL XAU/GOLD INTRADAY TRADING AGENT
# ============================================================
#
# Design:
#   1H  = market regime
#   15m = setup
#   5m  = entry trigger
#
# Default instrument:
#   GC=F (COMEX Gold futures) for ALL market data.
#
# Execution:
#   ALERT  = Telegram only (default)
#   PAPER   = simulated execution + journal
#   LIVE    = requires implementation of BrokerAdapter
#
# Risk defaults:
#   0.50% risk/trade
#   1.50% max daily loss
#   2 consecutive SLs => quarantine
#   max 3 trades/day
#
# IMPORTANT:
#   This is a systematic trading framework, not a profit guarantee.
#   Backtest with realistic slippage/fees before live deployment.
# ============================================================

app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------
SYMBOL = os.getenv("SYMBOL", "GC=F")
TIMEZONE_NAME = os.getenv("TIMEZONE_NAME", "UTC")

EXECUTION_MODE = os.getenv("EXECUTION_MODE", "ALERT").upper()  # ALERT/PAPER/LIVE
DB_PATH = os.getenv("DB_PATH", "trading_agent.db")

ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "100000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.005"))      # 0.50%
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "0.015"))      # 1.50%
MAX_CONSECUTIVE_SL = int(os.getenv("MAX_CONSECUTIVE_SL", "2"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "3"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "30"))

MIN_RR = float(os.getenv("MIN_RR", "2.0"))
TP1_R = float(os.getenv("TP1_R", "1.0"))
TP2_R = float(os.getenv("TP2_R", "2.0"))
RUNNER_R = float(os.getenv("RUNNER_R", "3.0"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_STOP_BUFFER = float(os.getenv("ATR_STOP_BUFFER", "0.20"))
SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "8"))
VWAP_PROX_ATR = float(os.getenv("VWAP_PROX_ATR", "0.60"))
MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.0005"))
MAX_ATR_PCT = float(os.getenv("MAX_ATR_PCT", "0.0120"))

SCORE_TO_TRADE = int(os.getenv("SCORE_TO_TRADE", "8"))

# Session windows are UTC by default.
# London active: 07:00-10:30
# New York active: 13:30-17:00
SESSION_WINDOWS = [
    ("LONDON", 7 * 60, 10 * 60 + 30),
    ("NEW_YORK", 13 * 60 + 30, 17 * 60),
]

# Manually configure high-impact news blackout windows.
# Format:
# NEWS_BLACKOUT_WINDOWS=2026-10-18T13:20:00/2026-10-18T13:45:00,...
NEWS_BLACKOUT_WINDOWS = os.getenv("NEWS_BLACKOUT_WINDOWS", "")

# Optional CSV files for historical backtesting.
# Expected 5m CSV columns: Datetime, Open, High, Low, Close, Volume
BACKTEST_CSV = os.getenv("BACKTEST_CSV", "")
DXY_CSV = os.getenv("DXY_CSV", "")
TNX_CSV = os.getenv("TNX_CSV", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("gold-agent")

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Signal:
    symbol: str
    action: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    runner: float
    risk_per_unit: float
    score: int
    setup: str
    session: str
    rationale: str
    timestamp: str
    atr_5m: float
    atr_15m: float
    ema200_1h: float
    ema20_15m: float
    vwap_15m: float
    dxy: Optional[float] = None
    tnx: Optional[float] = None


# ============================================================
# Database / persistence
# ============================================================
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id TEXT UNIQUE,
        symbol TEXT,
        action TEXT,
        entry REAL,
        stop REAL,
        tp1 REAL,
        tp2 REAL,
        runner REAL,
        risk_per_unit REAL,
        quantity REAL,
        score INTEGER,
        setup TEXT,
        session TEXT,
        timestamp TEXT,
        status TEXT,
        exit_price REAL,
        pnl REAL,
        r_multiple REAL,
        exit_reason TEXT,
        mae REAL,
        mfe REAL,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS daily_state (
        trade_date TEXT PRIMARY KEY,
        realized_pnl REAL DEFAULT 0,
        trades_count INTEGER DEFAULT 0,
        consecutive_sl INTEGER DEFAULT 0,
        quarantined INTEGER DEFAULT 0,
        updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_time TEXT,
        event_type TEXT,
        payload TEXT
    );
    """)
    conn.commit()
    conn.close()


init_db()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def today_key() -> str:
    return utc_now().strftime("%Y-%m-%d")


def get_daily_state() -> Dict[str, Any]:
    key = today_key()
    conn = db()
    row = conn.execute(
        "SELECT * FROM daily_state WHERE trade_date=?", (key,)
    ).fetchone()
    if row is None:
        conn.execute(
            """INSERT INTO daily_state
               (trade_date, realized_pnl, trades_count, consecutive_sl, quarantined, updated_at)
               VALUES (?,0,0,0,0,?)""",
            (key, utc_now().isoformat())
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM daily_state WHERE trade_date=?", (key,)
        ).fetchone()
    conn.close()
    return dict(row)


def update_daily_state(**kwargs):
    state = get_daily_state()
    fields = []
    values = []
    for k, v in kwargs.items():
        fields.append(f"{k}=?")
        values.append(v)
    fields.append("updated_at=?")
    values.append(utc_now().isoformat())
    values.append(state["trade_date"])

    conn = db()
    conn.execute(
        f"UPDATE daily_state SET {', '.join(fields)} WHERE trade_date=?",
        values
    )
    conn.commit()
    conn.close()


# ============================================================
# Telegram
# ============================================================
def send_telegram_alert(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not configured.")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown"
        }
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as exc:
        log.exception("Telegram error: %s", exc)
        return False


# ============================================================
# Market data
# ============================================================
def load_yfinance(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=True
        )
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.title)
        return df
    except Exception as exc:
        log.exception("yfinance error for %s: %s", symbol, exc)
        return None


def fetch_market_data():
    """
    One instrument is used throughout the strategy to avoid
    spot/futures basis mismatch.
    """
    df_5m = load_yfinance(SYMBOL, "30d", "5m")
    df_15m = load_yfinance(SYMBOL, "60d", "15m")
    df_1h = load_yfinance(SYMBOL, "730d", "1h")

    dxy = load_yfinance("DX-Y.NYB", "5d", "15m")
    tnx = load_yfinance("^TNX", "5d", "15m")

    if any(x is None for x in (df_5m, df_15m, df_1h)):
        return None

    return {
        "5m": df_5m,
        "15m": df_15m,
        "1h": df_1h,
        "dxy": dxy,
        "tnx": tnx
    }


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    required = ["Open", "High", "Low", "Close"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Volume" not in df:
        df["Volume"] = 0.0

    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    return df.dropna(subset=required)


# ============================================================
# Indicators
# ============================================================
def add_indicators(df: pd.DataFrame, session_vwap: bool = False) -> pd.DataFrame:
    df = clean_ohlcv(df)

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["TR"] = tr
    df["ATR"] = tr.rolling(ATR_PERIOD).mean()

    # Rolling mean volume; use only as a futures-volume context filter.
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    df["RelVolume"] = df["Volume"] / df["VolMA20"].replace(0, np.nan)

    if session_vwap:
        # Reset VWAP each UTC day. For a futures implementation,
        # consider using the exchange/session timezone you actually trade.
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        pv = typical * df["Volume"]
        session = df.index.date
        df["VWAP"] = pv.groupby(session).cumsum() / df["Volume"].groupby(session).cumsum()
    else:
        df["VWAP"] = np.nan

    # Candle anatomy
    df["Range"] = df["High"] - df["Low"]
    df["Body"] = (df["Close"] - df["Open"]).abs()
    df["UpperWick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["LowerWick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

    return df


# ============================================================
# Session / news controls
# ============================================================
def active_session(ts: Optional[datetime] = None) -> Tuple[bool, str]:
    ts = ts or utc_now()
    minutes = ts.hour * 60 + ts.minute

    for name, start, end in SESSION_WINDOWS:
        if start <= minutes < end:
            return True, name
    return False, "OUTSIDE"


def parse_news_windows() -> List[Tuple[datetime, datetime]]:
    windows = []
    if not NEWS_BLACKOUT_WINDOWS.strip():
        return windows

    for item in NEWS_BLACKOUT_WINDOWS.split(","):
        try:
            start_s, end_s = item.strip().split("/")
            start = datetime.fromisoformat(start_s.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_s.replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            windows.append((start, end))
        except Exception:
            log.warning("Invalid news blackout: %s", item)
    return windows


def is_news_blackout(ts: Optional[datetime] = None) -> bool:
    ts = ts or utc_now()
    for start, end in parse_news_windows():
        if start <= ts <= end:
            return True
    return False


# ============================================================
# Regime / setup / trigger
# ============================================================
def market_regime(df_1h: pd.DataFrame) -> str:
    df = add_indicators(df_1h)
    if len(df) < 210:
        return "UNKNOWN"

    last = df.iloc[-1]
    prev = df.iloc[-4]

    ema200_rising = last["EMA200"] > prev["EMA200"]
    ema200_falling = last["EMA200"] < prev["EMA200"]

    if last["Close"] > last["EMA200"] and ema200_rising:
        return "BULL"
    if last["Close"] < last["EMA200"] and ema200_falling:
        return "BEAR"
    return "RANGE"


def detect_setup(df_15m: pd.DataFrame, regime: str) -> Dict[str, Any]:
    df = add_indicators(df_15m, session_vwap=True)
    if len(df) < 210:
        return {"valid": False, "reason": "insufficient 15m data"}

    x = df.iloc[-1]
    recent = df.iloc[-8:]

    if regime == "BULL":
        # Pullback toward EMA20/VWAP, while price remains above EMA200.
        distance_vwap = abs(x["Close"] - x["VWAP"])
        near_vwap = distance_vwap <= max(x["ATR"] * VWAP_PROX_ATR, 0.5)
        above_ema20 = x["Close"] >= x["EMA20"]
        structure_ok = x["Low"] > recent["Low"].min() - x["ATR"] * 1.0

        if near_vwap or (x["Low"] <= x["EMA20"] and x["Close"] > x["EMA20"]):
            return {
                "valid": True,
                "side": "BUY",
                "df": df,
                "atr": float(x["ATR"]),
                "vwap": float(x["VWAP"]),
                "ema20": float(x["EMA20"]),
                "structure_ok": bool(structure_ok),
                "above_ema20": bool(above_ema20),
            }

    if regime == "BEAR":
        distance_vwap = abs(x["Close"] - x["VWAP"])
        near_vwap = distance_vwap <= max(x["ATR"] * VWAP_PROX_ATR, 0.5)
        below_ema20 = x["Close"] <= x["EMA20"]
        structure_ok = x["High"] < recent["High"].max() + x["ATR"] * 1.0

        if near_vwap or (x["High"] >= x["EMA20"] and x["Close"] < x["EMA20"]):
            return {
                "valid": True,
                "side": "SELL",
                "df": df,
                "atr": float(x["ATR"]),
                "vwap": float(x["VWAP"]),
                "ema20": float(x["EMA20"]),
                "structure_ok": bool(structure_ok),
                "below_ema20": bool(below_ema20),
            }

    return {"valid": False, "reason": "no 15m pullback setup", "df": df}


def detect_5m_trigger(df_5m: pd.DataFrame, side: str) -> Dict[str, Any]:
    df = add_indicators(df_5m, session_vwap=True)
    if len(df) < 50:
        return {"valid": False, "reason": "insufficient 5m data"}

    # We deliberately use only completed candles:
    #   rejection = -3
    #   trigger   = -2
    #   -1 may still be forming and is ignored for the decision.
    rejection = df.iloc[-3]
    trigger = df.iloc[-2]

    atr = float(rejection["ATR"])
    vwap = float(rejection["VWAP"])

    if not np.isfinite(atr) or atr <= 0:
        return {"valid": False, "reason": "invalid ATR"}

    candle_range = max(float(rejection["Range"]), 1e-9)
    lower_wick_ratio = float(rejection["LowerWick"]) / candle_range
    upper_wick_ratio = float(rejection["UpperWick"]) / candle_range

    if side == "BUY":
        touched_vwap = rejection["Low"] <= vwap + atr * 0.25
        rejected = rejection["Close"] > vwap and lower_wick_ratio >= 0.25
        bullish_close = rejection["Close"] > rejection["Open"]

        # The NEXT completed candle must actually break and CLOSE above
        # the rejection candle high. This is materially stronger than
        # simply checking that two candle highs are different.
        confirmed_break = (
            trigger["High"] > rejection["High"]
            and trigger["Close"] > rejection["High"]
        )

        return {
            "valid": bool(
                touched_vwap and rejected and bullish_close and confirmed_break
            ),
            "trigger_price": float(trigger["Close"]),
            "signal_candle_low": float(rejection["Low"]),
            "atr": atr,
            "vwap": vwap,
            "reason": "VWAP rejection + bullish close + confirmed breakout"
        }

    if side == "SELL":
        touched_vwap = rejection["High"] >= vwap - atr * 0.25
        rejected = rejection["Close"] < vwap and upper_wick_ratio >= 0.25
        bearish_close = rejection["Close"] < rejection["Open"]

        confirmed_break = (
            trigger["Low"] < rejection["Low"]
            and trigger["Close"] < rejection["Low"]
        )

        return {
            "valid": bool(
                touched_vwap and rejected and bearish_close and confirmed_break
            ),
            "trigger_price": float(trigger["Close"]),
            "signal_candle_high": float(rejection["High"]),
            "atr": atr,
            "vwap": vwap,
            "reason": "VWAP rejection + bearish close + confirmed breakdown"
        }

    return {"valid": False, "reason": "invalid side"}


# ============================================================
# Quality score
# ============================================================
def quality_score(
    side: str,
    regime: str,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    dxy_df: Optional[pd.DataFrame],
    tnx_df: Optional[pd.DataFrame]
) -> Tuple[int, List[str], List[str]]:
    score = 0
    positives = []
    negatives = []

    h = add_indicators(df_1h).iloc[-1]
    m = add_indicators(df_15m, session_vwap=True).iloc[-1]
    f = add_indicators(df_5m, session_vwap=True).iloc[-2]

    # 1H regime
    if side == "BUY" and regime == "BULL":
        score += 2
        positives.append("1H bullish regime")
    elif side == "SELL" and regime == "BEAR":
        score += 2
        positives.append("1H bearish regime")

    # 15m trend
    if side == "BUY" and m["Close"] > m["EMA20"]:
        score += 2
        positives.append("15m above EMA20")
    elif side == "SELL" and m["Close"] < m["EMA20"]:
        score += 2
        positives.append("15m below EMA20")
    else:
        negatives.append("15m trend not aligned")

    # VWAP
    if side == "BUY" and f["Close"] > f["VWAP"]:
        score += 2
        positives.append("5m reclaimed VWAP")
    elif side == "SELL" and f["Close"] < f["VWAP"]:
        score += 2
        positives.append("5m rejected below VWAP")

    # Volume: meaningful only for futures.
    relvol = f.get("RelVolume", np.nan)
    if np.isfinite(relvol) and relvol >= 1.10:
        score += 1
        positives.append("relative volume expansion")
    else:
        negatives.append("no volume expansion")

    # Volatility regime
    atr_pct = float(f["ATR"] / f["Close"])
    if MIN_ATR_PCT <= atr_pct <= MAX_ATR_PCT:
        score += 1
        positives.append("healthy volatility")
    else:
        negatives.append("unfavorable volatility regime")

    # DXY confirmation: supportive but not mandatory.
    if dxy_df is not None and not dxy_df.empty:
        d = clean_ohlcv(dxy_df)
        if len(d) >= 5:
            d_change = float(d["Close"].iloc[-1] - d["Close"].iloc[-5])
            if side == "BUY" and d_change < 0:
                score += 1
                positives.append("DXY falling")
            elif side == "SELL" and d_change > 0:
                score += 1
                positives.append("DXY rising")
            else:
                negatives.append("DXY not confirming")

    # TNX (^TNX) is approximately the 10Y yield in percent.
    if tnx_df is not None and not tnx_df.empty:
        t = clean_ohlcv(tnx_df)
        if len(t) >= 5:
            t_change = float(t["Close"].iloc[-1] - t["Close"].iloc[-5])
            if side == "BUY" and t_change <= 0:
                score += 1
                positives.append("10Y yield stable/falling")
            elif side == "SELL" and t_change >= 0:
                score += 1
                positives.append("10Y yield stable/rising")
            else:
                negatives.append("10Y not confirming")

    return score, positives, negatives


# ============================================================
# Stops / targets / sizing
# ============================================================
def recent_structure(df_5m: pd.DataFrame, side: str) -> float:
    df = add_indicators(df_5m, session_vwap=True)
    completed = df.iloc[:-1]
    recent = completed.tail(SWING_LOOKBACK)

    if side == "BUY":
        return float(recent["Low"].min())
    return float(recent["High"].max())


def build_trade_levels(
    side: str,
    entry: float,
    df_5m: pd.DataFrame
) -> Optional[Dict[str, float]]:
    df = add_indicators(df_5m, session_vwap=True)
    atr = float(df["ATR"].iloc[-2])

    if not np.isfinite(atr) or atr <= 0:
        return None

    swing = recent_structure(df_5m, side)

    if side == "BUY":
        stop = swing - atr * ATR_STOP_BUFFER
        risk = entry - stop
        if risk <= 0:
            return None
        tp1 = entry + risk * TP1_R
        tp2 = entry + risk * TP2_R
        runner = entry + risk * RUNNER_R
    else:
        stop = swing + atr * ATR_STOP_BUFFER
        risk = stop - entry
        if risk <= 0:
            return None
        tp1 = entry - risk * TP1_R
        tp2 = entry - risk * TP2_R
        runner = entry - risk * RUNNER_R

    # Reject absurdly tight/large stops.
    if risk < atr * 0.5 or risk > atr * 3.5:
        return None

    return {
        "stop": round(stop, 2),
        "risk": round(risk, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "runner": round(runner, 2),
        "atr": round(atr, 2)
    }


def point_value() -> float:
    """
    GC=F is normally $100 per $1 move per contract.
    MGC is normally $10 per $1 move.
    Override with POINT_VALUE for your broker/instrument.
    """
    return float(os.getenv("POINT_VALUE", "100"))


def calculate_position_size(risk_per_unit: float) -> float:
    if risk_per_unit <= 0:
        return 0.0

    max_dollar_risk = ACCOUNT_EQUITY * RISK_PER_TRADE
    qty = max_dollar_risk / (risk_per_unit * point_value())

    # Default to whole contracts for futures.
    return max(0.0, math.floor(qty))


# ============================================================
# Risk engine
# ============================================================
def risk_locked() -> Tuple[bool, str]:
    state = get_daily_state()

    if state["quarantined"]:
        return True, "quarantined"

    if state["trades_count"] >= MAX_TRADES_PER_DAY:
        return True, "max trades/day reached"

    max_loss = -ACCOUNT_EQUITY * MAX_DAILY_LOSS
    if state["realized_pnl"] <= max_loss:
        return True, "daily loss limit reached"

    if state["consecutive_sl"] >= MAX_CONSECUTIVE_SL:
        return True, "consecutive loss limit reached"

    # Cooldown based on most recent trade.
    conn = db()
    row = conn.execute(
        """SELECT timestamp FROM trades
           WHERE DATE(timestamp)=DATE(?) ORDER BY id DESC LIMIT 1""",
        (utc_now().isoformat(),)
    ).fetchone()
    conn.close()

    if row:
        try:
            last = datetime.fromisoformat(row["timestamp"])
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if utc_now() - last < timedelta(minutes=COOLDOWN_MINUTES):
                return True, "cooldown active"
        except Exception:
            pass

    # Only one open position at a time.
    conn = db()
    open_count = conn.execute(
        "SELECT COUNT(*) AS n FROM trades WHERE status='OPEN'"
    ).fetchone()["n"]
    conn.close()

    if open_count > 0:
        return True, "existing open position"

    return False, ""


# ============================================================
# Broker adapter
# ============================================================
class BrokerAdapter:
    """
    Replace this adapter with your broker's authenticated API.

    The agent intentionally refuses to pretend that a Telegram alert
    is a broker execution. LIVE mode must be explicitly implemented.
    """

    def place_order(self, signal: Signal, quantity: float) -> Dict[str, Any]:
        raise NotImplementedError(
            "LIVE mode requires implementation of BrokerAdapter.place_order()."
        )

    def close_order(self, trade_id: str, exit_price: float, reason: str):
        raise NotImplementedError


class PaperBroker(BrokerAdapter):
    def place_order(self, signal: Signal, quantity: float) -> Dict[str, Any]:
        return {
            "trade_id": f"PAPER-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            "status": "OPEN",
            "quantity": quantity,
            "entry": signal.entry
        }


broker = PaperBroker()


# ============================================================
# Trade journal
# ============================================================
def record_trade(signal: Signal, quantity: float, trade_id: str, status="OPEN"):
    conn = db()
    conn.execute("""
        INSERT INTO trades (
            trade_id, symbol, action, entry, stop, tp1, tp2, runner,
            risk_per_unit, quantity, score, setup, session, timestamp,
            status, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_id, signal.symbol, signal.action, signal.entry, signal.stop,
        signal.tp1, signal.tp2, signal.runner, signal.risk_per_unit,
        quantity, signal.score, signal.setup, signal.session,
        signal.timestamp, status, signal.rationale
    ))
    conn.commit()
    conn.close()

    state = get_daily_state()
    update_daily_state(trades_count=state["trades_count"] + 1)


def close_trade(trade_id: str, exit_price: float, reason: str,
                mae: Optional[float] = None, mfe: Optional[float] = None):
    conn = db()
    row = conn.execute(
        "SELECT * FROM trades WHERE trade_id=?", (trade_id,)
    ).fetchone()

    if row is None:
        conn.close()
        raise ValueError("Trade not found")

    direction = row["action"]
    entry = float(row["entry"])
    risk = float(row["risk_per_unit"])
    qty = float(row["quantity"])

    if direction == "BUY":
        pnl = (exit_price - entry) * qty * point_value()
    else:
        pnl = (entry - exit_price) * qty * point_value()

    r_multiple = pnl / max(risk * qty * point_value(), 1e-9)

    conn.execute("""
        UPDATE trades
        SET status='CLOSED', exit_price=?, pnl=?, r_multiple=?,
            exit_reason=?, mae=?, mfe=?
        WHERE trade_id=?
    """, (
        exit_price, pnl, r_multiple, reason, mae, mfe, trade_id
    ))
    conn.commit()
    conn.close()

    state = get_daily_state()
    consecutive = state["consecutive_sl"]

    if reason.upper() == "SL":
        consecutive += 1
    elif pnl > 0:
        consecutive = 0

    quarantined = int(consecutive >= MAX_CONSECUTIVE_SL)

    update_daily_state(
        realized_pnl=state["realized_pnl"] + pnl,
        consecutive_sl=consecutive,
        quarantined=quarantined
    )

    return {
        "trade_id": trade_id,
        "pnl": pnl,
        "r_multiple": r_multiple,
        "consecutive_sl": consecutive,
        "quarantined": bool(quarantined)
    }


# ============================================================
# Signal generation
# ============================================================
def generate_signal() -> Tuple[Optional[Signal], Dict[str, Any]]:
    market = fetch_market_data()
    if market is None:
        return None, {"status": "ERROR", "message": "Insufficient market data"}

    df5 = clean_ohlcv(market["5m"])
    df15 = clean_ohlcv(market["15m"])
    df1h = clean_ohlcv(market["1h"])

    # IMPORTANT: completed 5m candle, not current forming candle.
    current_price = float(df5["Close"].iloc[-1])
    ts = df5.index[-1].to_pydatetime()

    active, session = active_session(ts)
    if not active:
        return None, {"status": "SKIPPED", "message": "Outside active session"}

    if is_news_blackout(ts):
        return None, {"status": "SKIPPED", "message": "News blackout"}

    regime = market_regime(df1h)
    if regime not in ("BULL", "BEAR"):
        return None, {"status": "HOLD", "message": f"Regime={regime}"}

    setup = detect_setup(df15, regime)
    if not setup.get("valid"):
        return None, {"status": "HOLD", "message": setup.get("reason")}

    side = setup["side"]

    trigger = detect_5m_trigger(df5, side)
    if not trigger.get("valid"):
        return None, {"status": "HOLD", "message": trigger.get("reason")}

    score, positives, negatives = quality_score(
        side, regime, df1h, df15, df5, market["dxy"], market["tnx"]
    )

    if score < SCORE_TO_TRADE:
        return None, {
            "status": "HOLD",
            "message": f"Score {score} < {SCORE_TO_TRADE}",
            "positives": positives,
            "negatives": negatives
        }

    # Enter only on trigger break, using current price.
    entry = current_price

    levels = build_trade_levels(side, entry, df5)
    if levels is None:
        return None, {"status": "HOLD", "message": "Invalid structure/ATR stop"}

    dxy_val = None
    tnx_val = None

    if market["dxy"] is not None and not market["dxy"].empty:
        dxy_val = float(market["dxy"]["Close"].iloc[-1])

    if market["tnx"] is not None and not market["tnx"].empty:
        tnx_val = float(market["tnx"]["Close"].iloc[-1])

    rationale = (
        f"{regime} 1H regime; {side} 15m pullback; "
        f"5m VWAP rejection trigger; score={score}. "
        f"Positive: {', '.join(positives)}. "
        f"Negative: {', '.join(negatives) if negatives else 'none'}."
    )

    signal = Signal(
        symbol=SYMBOL,
        action=side,
        entry=round(entry, 2),
        stop=levels["stop"],
        tp1=levels["tp1"],
        tp2=levels["tp2"],
        runner=levels["runner"],
        risk_per_unit=levels["risk"],
        score=score,
        setup="1H regime + 15m pullback + 5m VWAP rejection",
        session=session,
        rationale=rationale,
        timestamp=utc_now().isoformat(),
        atr_5m=levels["atr"],
        atr_15m=float(setup["atr"]),
        ema200_1h=float(add_indicators(df1h)["EMA200"].iloc[-1]),
        ema20_15m=float(setup["ema20"]),
        vwap_15m=float(setup["vwap"]),
        dxy=dxy_val,
        tnx=tnx_val
    )

    return signal, {
        "status": "SIGNAL",
        "signal": asdict(signal)
    }


# ============================================================
# Alerts / execution
# ============================================================
def format_signal(signal: Signal, quantity: float) -> str:
    return (
        f"*XAU/GOLD SYSTEMATIC SIGNAL â€” {signal.action}*\n"
        f"Symbol: `{signal.symbol}`\n"
        f"Session: `{signal.session}`\n"
        f"Score: `{signal.score}`\n"
        f"Entry: `{signal.entry:.2f}`\n"
        f"SL: `{signal.stop:.2f}`\n"
        f"TP1: `{signal.tp1:.2f}` (1R)\n"
        f"TP2: `{signal.tp2:.2f}` (2R)\n"
        f"Runner: `{signal.runner:.2f}` (3R)\n"
        f"Risk/unit: `{signal.risk_per_unit:.2f}`\n"
        f"Suggested qty: `{quantity}`\n"
        f"1H EMA200: `{signal.ema200_1h:.2f}`\n"
        f"15m EMA20: `{signal.ema20_15m:.2f}`\n"
        f"15m VWAP: `{signal.vwap_15m:.2f}`\n"
        f"ATR 5m: `{signal.atr_5m:.2f}`\n"
        f"ATR 15m: `{signal.atr_15m:.2f}`\n"
        f"DXY: `{signal.dxy if signal.dxy is not None else 'N/A'}`\n"
        f"10Y: `{signal.tnx if signal.tnx is not None else 'N/A'}`\n\n"
        f"*Logic:*\n{signal.rationale}"
    )


def execute_signal(signal: Signal) -> Dict[str, Any]:
    locked, reason = risk_locked()
    if locked:
        return {"status": "BLOCKED", "reason": reason}

    quantity = calculate_position_size(signal.risk_per_unit)
    if quantity < 1:
        return {
            "status": "BLOCKED",
            "reason": "Calculated quantity < 1 contract; reduce risk or use MGC."
        }

    alert = format_signal(signal, quantity)
    send_telegram_alert(alert)

    if EXECUTION_MODE == "ALERT":
        return {"status": "ALERTED", "quantity": quantity}

    if EXECUTION_MODE == "PAPER":
        result = broker.place_order(signal, quantity)
        record_trade(signal, quantity, result["trade_id"], "OPEN")
        return {
            "status": "PAPER_OPEN",
            "trade_id": result["trade_id"],
            "quantity": quantity
        }

    if EXECUTION_MODE == "LIVE":
        # This intentionally refuses to place a fake "live" order.
        # Implement BrokerAdapter with your actual broker's authenticated API.
        try:
            result = broker.place_order(signal, quantity)
            record_trade(signal, quantity, result["trade_id"], "OPEN")
            return {
                "status": "LIVE_OPEN",
                "trade_id": result["trade_id"],
                "quantity": quantity
            }
        except NotImplementedError as exc:
            return {"status": "LIVE_NOT_CONFIGURED", "reason": str(exc)}

    return {"status": "ERROR", "reason": f"Unknown EXECUTION_MODE={EXECUTION_MODE}"}


# ============================================================
# Backtest engine
# ============================================================
def load_backtest_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
    df = df.set_index(dt_col).sort_index()
    return clean_ohlcv(df)


def backtest(csv_path: str) -> Dict[str, Any]:
    """
    Event-driven simplified backtest.

    It uses the same basic signal architecture but evaluates completed
    candles sequentially. For institutional validation, run a second
    independent backtester and include broker-specific fills/slippage.
    """
    df5 = load_backtest_csv(csv_path)

    # Build higher timeframes from 5m data.
    df15 = df5.resample("15min").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()

    df1h = df5.resample("1h").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()

    trades = []
    equity_r = 0.0
    peak_r = 0.0
    max_dd = 0.0

    start_index = max(220, 300)

    for i in range(start_index, len(df5) - 2):
        current_ts = df5.index[i]
        active, session = active_session(current_ts.to_pydatetime())
        if not active:
            continue
        if is_news_blackout(current_ts.to_pydatetime()):
            continue

        f5 = df5.iloc[:i + 1]
        f15 = df15[df15.index <= current_ts]
        f1h = df1h[df1h.index <= current_ts]

        if len(f15) < 210 or len(f1h) < 210:
            continue

        regime = market_regime(f1h)
        if regime not in ("BULL", "BEAR"):
            continue

        setup = detect_setup(f15, regime)
        if not setup.get("valid"):
            continue

        side = setup["side"]
        trigger = detect_5m_trigger(f5, side)
        if not trigger.get("valid"):
            continue

        levels = build_trade_levels(
            side,
            float(f5["Close"].iloc[-1]),
            f5
        )
        if not levels:
            continue

        entry = float(f5["Close"].iloc[-1])
        stop = levels["stop"]
        tp2 = levels["tp2"]

        # Simple forward simulation until SL/2R.
        exit_price = None
        reason = None

        for j in range(i + 1, min(i + 100, len(df5))):
            bar = df5.iloc[j]

            if side == "BUY":
                if bar["Low"] <= stop:
                    exit_price = stop
                    reason = "SL"
                    break
                if bar["High"] >= tp2:
                    exit_price = tp2
                    reason = "TP2"
                    break
            else:
                if bar["High"] >= stop:
                    exit_price = stop
                    reason = "SL"
                    break
                if bar["Low"] <= tp2:
                    exit_price = tp2
                    reason = "TP2"
                    break

        if exit_price is None:
            continue

        if side == "BUY":
            r = (exit_price - entry) / levels["risk"]
        else:
            r = (entry - exit_price) / levels["risk"]

        equity_r += r
        peak_r = max(peak_r, equity_r)
        max_dd = max(max_dd, peak_r - equity_r)

        trades.append({
            "time": current_ts.isoformat(),
            "side": side,
            "entry": entry,
            "stop": stop,
            "exit": exit_price,
            "reason": reason,
            "R": r
        })

    if not trades:
        return {"status": "NO_TRADES", "trades": 0}

    arr = np.array([t["R"] for t in trades], dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    win_rate = len(wins) / len(arr)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    profit_factor = (
        float(wins.sum() / abs(losses.sum()))
        if len(losses) and losses.sum() != 0 else np.inf
    )

    return {
        "status": "OK",
        "trades": len(trades),
        "win_rate": round(win_rate, 4),
        "avg_win_R": round(avg_win, 3),
        "avg_loss_R": round(avg_loss, 3),
        "expectancy_R": round(expectancy, 3),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else "inf",
        "net_R": round(float(arr.sum()), 3),
        "max_drawdown_R": round(float(max_dd), 3),
        "trades_detail": trades
    }


# ============================================================
# Flask endpoints
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "symbol": SYMBOL,
        "execution_mode": EXECUTION_MODE,
        "time": utc_now().isoformat(),
        "daily_state": get_daily_state()
    })


@app.route("/signal", methods=["GET", "POST"])
def signal_endpoint():
    try:
        locked, reason = risk_locked()
        if locked:
            return jsonify({
                "status": "BLOCKED",
                "reason": reason,
                "daily_state": get_daily_state()
            }), 200

        signal, details = generate_signal()

        if signal is None:
            return jsonify(details), 200

        execution = execute_signal(signal)

        return jsonify({
            "status": "SIGNAL_PROCESSED",
            "signal": asdict(signal),
            "execution": execution
        }), 200

    except Exception as exc:
        log.exception("Signal endpoint error")
        return jsonify({
            "status": "ERROR",
            "message": str(exc)
        }), 500


@app.route("/trade/close", methods=["POST"])
def close_trade_endpoint():
    """
    Broker/webhook should call this when a position actually closes.
    JSON:
      {"trade_id":"...", "exit_price":1234.5, "reason":"SL"}
    """
    try:
        payload = request.get_json(force=True)
        result = close_trade(
            trade_id=payload["trade_id"],
            exit_price=float(payload["exit_price"]),
            reason=payload.get("reason", "MANUAL"),
            mae=payload.get("mae"),
            mfe=payload.get("mfe")
        )

        send_telegram_alert(
            f"*TRADE CLOSED*\n"
            f"ID: `{result['trade_id']}`\n"
            f"PnL: `${result['pnl']:.2f}`\n"
            f"R: `{result['r_multiple']:.2f}`\n"
            f"Consecutive SL: `{result['consecutive_sl']}`\n"
            f"Quarantined: `{result['quarantined']}`"
        )

        return jsonify(result), 200
    except Exception as exc:
        log.exception("Close trade error")
        return jsonify({"status": "ERROR", "message": str(exc)}), 400


@app.route("/backtest", methods=["POST"])
def backtest_endpoint():
    try:
        payload = request.get_json(silent=True) or {}
        path = payload.get("csv_path") or BACKTEST_CSV

        if not path:
            return jsonify({
                "status": "ERROR",
                "message": "Provide BACKTEST_CSV or csv_path."
            }), 400

        result = backtest(path)
        return jsonify(result), 200
    except Exception as exc:
        log.exception("Backtest error")
        return jsonify({"status": "ERROR", "message": str(exc)}), 500


@app.route("/risk/reset", methods=["POST"])
def reset_quarantine():
    """
    Manual reset. Protect this endpoint behind authentication in production.
    """
    update_daily_state(quarantined=0, consecutive_sl=0)
    return jsonify({
        "status": "RESET",
        "daily_state": get_daily_state()
    })


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

import os, json, math, sqlite3, logging, hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify

# ============================================================
# PROFESSIONAL GOLD INTRADAY AGENT
# 1H regime -> 15m setup -> 5m trigger -> macro confirmation
# -> score -> structure/ATR risk -> sizing -> lifecycle -> journal
#
# ALERT/PAPER are supported. LIVE deliberately fails closed.
# Validate with backtest + out-of-sample + paper trading first.
# ============================================================

app = Flask(__name__)

SYMBOL = os.getenv("SYMBOL", "GC=F")
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "ALERT").upper()
DB_PATH = os.getenv("DB_PATH", "trading_agent.db")

ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "100000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.005"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "0.015"))
MAX_CONSECUTIVE_SL = int(os.getenv("MAX_CONSECUTIVE_SL", "2"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "3"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "30"))

# GC: $100 for a $1.00 move, $0.10 tick.
# MGC: $10 for a $1.00 move, $0.10 tick.
CONTRACT_MULTIPLIER = float(os.getenv("CONTRACT_MULTIPLIER", "100"))
TICK_SIZE = float(os.getenv("TICK_SIZE", "0.10"))
COMMISSION_PER_CONTRACT = float(os.getenv("COMMISSION_PER_CONTRACT", "0"))
SLIPPAGE_TICKS = float(os.getenv("SLIPPAGE_TICKS", "1"))

MIN_RR = float(os.getenv("MIN_RR", "2.0"))
TP1_R = float(os.getenv("TP1_R", "1.0"))
TP2_R = float(os.getenv("TP2_R", "2.0"))
RUNNER_R = float(os.getenv("RUNNER_R", "3.0"))
TP1_PCT = float(os.getenv("TP1_PCT", "0.50"))
TP2_PCT = float(os.getenv("TP2_PCT", "0.25"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_STOP_BUFFER = float(os.getenv("ATR_STOP_BUFFER", "0.20"))
SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "8"))
VWAP_PROX_ATR = float(os.getenv("VWAP_PROX_ATR", "0.60"))
MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.0005"))
MAX_ATR_PCT = float(os.getenv("MAX_ATR_PCT", "0.012"))
SCORE_TO_TRADE = int(os.getenv("SCORE_TO_TRADE", "9"))

SESSION_WINDOWS = [
    ("LONDON", 7 * 60, 10 * 60 + 30),
    ("NEW_YORK", 13 * 60 + 30, 17 * 60),
]
NEWS_BLACKOUT_WINDOWS = os.getenv("NEWS_BLACKOUT_WINDOWS", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gold-agent")


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
    trigger_bar_timestamp: str
    atr_5m: float
    atr_15m: float
    ema200_1h: float
    ema20_15m: float
    vwap_15m: float
    dxy: Optional[float] = None
    tnx: Optional[float] = None
    signal_id: Optional[str] = None


# ------------------------- database --------------------------

def db():
    c = sqlite3.connect(DB_PATH, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=15000")
    return c


def init_db():
    c = db()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS trades(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      trade_id TEXT UNIQUE NOT NULL, signal_id TEXT,
      symbol TEXT, action TEXT, entry REAL, stop REAL,
      tp1 REAL, tp2 REAL, runner REAL, risk_per_unit REAL,
      quantity REAL, remaining_quantity REAL, score INTEGER,
      setup TEXT, session TEXT, signal_timestamp TEXT,
      opened_at TEXT, status TEXT, exit_price REAL,
      pnl REAL DEFAULT 0, r_multiple REAL DEFAULT 0,
      exit_reason TEXT, tp1_hit INTEGER DEFAULT 0,
      tp2_hit INTEGER DEFAULT 0, runner_hit INTEGER DEFAULT 0,
      stop_moved_to_be INTEGER DEFAULT 0, notes TEXT
    );
    CREATE TABLE IF NOT EXISTS signals(
      signal_id TEXT PRIMARY KEY, symbol TEXT, action TEXT,
      trigger_bar_timestamp TEXT, created_at TEXT, score INTEGER,
      status TEXT, payload TEXT
    );
    CREATE TABLE IF NOT EXISTS daily_state(
      trade_date TEXT PRIMARY KEY, realized_pnl REAL DEFAULT 0,
      trades_count INTEGER DEFAULT 0, signals_count INTEGER DEFAULT 0,
      consecutive_sl INTEGER DEFAULT 0, quarantined INTEGER DEFAULT 0,
      updated_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
    """)
    c.commit()
    c.close()


init_db()


def now():
    return datetime.now(timezone.utc)


def day_key():
    return now().strftime("%Y-%m-%d")


def finite(x):
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def tick(p):
    return round(round(float(p) / TICK_SIZE) * TICK_SIZE, 2)


def get_state():
    key = day_key()
    c = db()
    r = c.execute("SELECT * FROM daily_state WHERE trade_date=?", (key,)).fetchone()
    if not r:
        c.execute("""INSERT INTO daily_state
          (trade_date,updated_at) VALUES (?,?)""", (key, now().isoformat()))
        c.commit()
        r = c.execute("SELECT * FROM daily_state WHERE trade_date=?", (key,)).fetchone()
    c.close()
    return dict(r)


def update_state(**kw):
    allowed = {"realized_pnl","trades_count","signals_count",
               "consecutive_sl","quarantined"}
    if not kw:
        return
    if any(k not in allowed for k in kw):
        raise ValueError("Invalid daily state field")
    parts = [f"{k}=?" for k in kw]
    vals = list(kw.values()) + [now().isoformat(), day_key()]
    c = db()
    c.execute(f"""UPDATE daily_state SET {','.join(parts)},
                 updated_at=? WHERE trade_date=?""", vals)
    c.commit()
    c.close()


def open_trade():
    c = db()
    r = c.execute("""SELECT * FROM trades
                     WHERE status IN ('OPEN','TP1_HIT','TP2_HIT')
                     ORDER BY id DESC LIMIT 1""").fetchone()
    c.close()
    return dict(r) if r else None


# ------------------------- telegram --------------------------

def telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text,
                  "parse_mode": "Markdown"},
            timeout=10)
        return r.status_code == 200
    except Exception as e:
        log.exception("Telegram: %s", e)
        return False


# ------------------------- market data -----------------------

def load_yf(symbol, period, interval):
    try:
        import yfinance as yf
        d = yf.Ticker(symbol).history(
            period=period, interval=interval,
            auto_adjust=False, prepost=True)
        if d is None or d.empty:
            return None
        d = d.rename(columns=str.title)
        return clean(d)
    except Exception as e:
        log.exception("yfinance %s: %s", symbol, e)
        return None


def clean(d):
    d = d.copy()
    d.index = pd.to_datetime(d.index, utc=True)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    for x in ["Open","High","Low","Close"]:
        d[x] = pd.to_numeric(d[x], errors="coerce")
    if "Volume" not in d:
        d["Volume"] = 0.0
    d["Volume"] = pd.to_numeric(d["Volume"], errors="coerce").fillna(0)
    return d.dropna(subset=["Open","High","Low","Close"])


def market():
    a = load_yf(SYMBOL, "10d", "5m")
    b = load_yf(SYMBOL, "30d", "15m")
    h = load_yf(SYMBOL, "60d", "1h")
    dxy = load_yf("DX-Y.NYB", "5d", "15m")
    tnx = load_yf("^TNX", "5d", "15m")
    if any(x is None or x.empty for x in [a,b,h]):
        return None
    return {"5m":a, "15m":b, "1h":h, "dxy":dxy, "tnx":tnx}


# ------------------------- indicators ------------------------

def indicators(d, vwap=False):
    d = d.copy()
    d["EMA20"] = d.Close.ewm(span=20, adjust=False).mean()
    d["EMA50"] = d.Close.ewm(span=50, adjust=False).mean()
    d["EMA200"] = d.Close.ewm(span=200, adjust=False).mean()
    pc = d.Close.shift(1)
    d["TR"] = pd.concat([
        d.High-d.Low, (d.High-pc).abs(), (d.Low-pc).abs()
    ], axis=1).max(axis=1)
    d["ATR"] = d.TR.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    d["VolMA20"] = d.Volume.rolling(20, min_periods=20).mean()
    d["RelVolume"] = d.Volume / d.VolMA20.replace(0, np.nan)

    if vwap:
        typical = (d.High+d.Low+d.Close)/3
        key = pd.Series(d.index.date, index=d.index)
        d["VWAP"] = (typical*d.Volume).groupby(key).cumsum() /                     d.Volume.groupby(key).cumsum().replace(0,np.nan)
    else:
        d["VWAP"] = np.nan

    d["Range"] = d.High-d.Low
    d["Body"] = (d.Close-d.Open).abs()
    d["BodyPct"] = d.Body/d.Range.replace(0,np.nan)
    d["UpperWick"] = d.High-d[["Open","Close"]].max(axis=1)
    d["LowerWick"] = d[["Open","Close"]].min(axis=1)-d.Low
    return d


# ------------------------- controls --------------------------

def active_session(ts):
    ts = ts.astimezone(timezone.utc)
    m = ts.hour*60+ts.minute
    for name, start, end in SESSION_WINDOWS:
        if start <= m < end:
            return True, name
    return False, "OUTSIDE"


def news_blackout(ts):
    if not NEWS_BLACKOUT_WINDOWS.strip():
        return False
    for item in NEWS_BLACKOUT_WINDOWS.split(","):
        try:
            a,b = item.strip().split("/")
            a = datetime.fromisoformat(a.replace("Z","+00:00"))
            b = datetime.fromisoformat(b.replace("Z","+00:00"))
            if a <= ts <= b:
                return True
        except Exception:
            log.warning("Bad news window: %s", item)
    return False


def regime(h):
    d = indicators(h)
    if len(d) < 210:
        return "UNKNOWN"
    x, p = d.iloc[-2], d.iloc[-5]  # closed candles only
    if not all(finite(x[k]) for k in ["Close","EMA200"]):
        return "UNKNOWN"
    if x.Close > x.EMA200 and x.EMA200 > p.EMA200:
        return "BULL"
    if x.Close < x.EMA200 and x.EMA200 < p.EMA200:
        return "BEAR"
    return "RANGE"


# ------------------------- setup/trigger ---------------------

def setup(m, reg):
    d = indicators(m, True)
    if len(d) < 210:
        return {"valid":False, "reason":"insufficient 15m data"}
    x, p = d.iloc[-2], d.iloc[-10:-2]
    if not all(finite(x[k]) for k in ["Close","Low","High","EMA20","ATR","VWAP"]):
        return {"valid":False, "reason":"invalid 15m indicators"}

    near = abs(x.Close-x.VWAP) <= max(x.ATR*VWAP_PROX_ATR,0.5)

    if reg == "BULL":
        pull = x.Low <= x.EMA20 and x.Close > x.EMA20
        structure = x.Low >= p.Low.min()-x.ATR*0.5
        if (near or pull) and structure:
            return {"valid":True,"side":"BUY","atr":float(x.ATR),
                    "vwap":float(x.VWAP),"ema20":float(x.EMA20)}
    if reg == "BEAR":
        pull = x.High >= x.EMA20 and x.Close < x.EMA20
        structure = x.High <= p.High.max()+x.ATR*0.5
        if (near or pull) and structure:
            return {"valid":True,"side":"SELL","atr":float(x.ATR),
                    "vwap":float(x.VWAP),"ema20":float(x.EMA20)}
    return {"valid":False,"reason":"no valid 15m pullback"}


def trigger(f, side):
    d = indicators(f, True)
    if len(d) < 60:
        return {"valid":False,"reason":"insufficient 5m data"}
    r, t = d.iloc[-3], d.iloc[-2]  # completed candles only
    if not all(finite(r[k]) for k in ["ATR","VWAP","Range","Open","Close","High","Low"]):
        return {"valid":False,"reason":"invalid trigger indicators"}

    atr = float(r.ATR)
    rng = max(float(r.Range),1e-9)
    lw = float(r.LowerWick)/rng
    uw = float(r.UpperWick)/rng
    body = float(r.Body)/rng

    if side == "BUY":
        ok = (r.Low <= r.VWAP+atr*.25 and r.Close > r.VWAP and
              lw >= .25 and body >= .20 and r.Close > r.Open and
              t.High > r.High and t.Close > r.High)
        reason = "VWAP rejection + bullish candle + breakout"
    else:
        ok = (r.High >= r.VWAP-atr*.25 and r.Close < r.VWAP and
              uw >= .25 and body >= .20 and r.Close < r.Open and
              t.Low < r.Low and t.Close < r.Low)
        reason = "VWAP rejection + bearish candle + breakdown"

    return {"valid":bool(ok),"atr":atr,"vwap":float(r.VWAP),
            "trigger_price":float(t.Close),
            "bar":d.index[-2].isoformat(),"reason":reason}


# ------------------------- macro/score ----------------------

def latest(d):
    if d is None or d.empty: return None
    i = -2 if len(d) >= 2 else -1
    x = float(d.Close.iloc[i])
    return x if finite(x) else None


def macro(side,dxy,tnx):
    score, pos, neg = 0, [], []
    dv, tv = latest(dxy), latest(tnx)

    for d, name in [(dxy,"DXY"),(tnx,"US10Y")]:
        if d is None or len(d) < 5: continue
        x, p = float(d.Close.iloc[-2]), float(d.Close.iloc[-5])
        rising = x > p
        favorable = (side=="BUY" and not rising) or (side=="SELL" and rising)
        adverse = (side=="BUY" and rising) or (side=="SELL" and not rising)
        if favorable:
            score += 1; pos.append(name+" favorable")
        elif adverse:
            neg.append(name+" adverse")
    return score,pos,neg,dv,tv


def score(side, reg, h, m, f, dxy, tnx):
    H, M, F = indicators(h).iloc[-2], indicators(m,True).iloc[-2], indicators(f,True).iloc[-2]
    s, pos, neg = 0, [], []

    if (side=="BUY" and reg=="BULL") or (side=="SELL" and reg=="BEAR"):
        s += 2; pos.append("1H regime aligned")
    if (side=="BUY" and M.Close>M.EMA20) or (side=="SELL" and M.Close<M.EMA20):
        s += 2; pos.append("15m EMA20 aligned")
    if (side=="BUY" and F.Close>F.VWAP) or (side=="SELL" and F.Close<F.VWAP):
        s += 2; pos.append("5m VWAP aligned")

    if finite(F.RelVolume) and F.RelVolume >= 1.10:
        s += 1; pos.append("volume expansion")
    else:
        neg.append("weak volume")

    atrpct = float(F.ATR/F.Close)
    if MIN_ATR_PCT <= atrpct <= MAX_ATR_PCT:
        s += 1; pos.append("healthy volatility")
    else:
        neg.append("poor volatility")

    ms, mp, mn, dv, tv = macro(side,dxy,tnx)
    s += ms; pos += mp; neg += mn
    return s,pos,neg,dv,tv


# ------------------------- trade geometry --------------------

def structure_room(side, entry, m):
    d = indicators(m).iloc[:-1].tail(20)
    if d.empty: return None
    if side=="BUY":
        x = d.High[d.High>entry]
        return None if x.empty else float(x.min()-entry)
    x = d.Low[d.Low<entry]
    return None if x.empty else float(entry-x.max())


def levels(side, entry, f, m):
    d = indicators(f,True)
    atr = float(d.ATR.iloc[-2])
    if not finite(atr) or atr <= 0: return None
    r = d.iloc[:-1].tail(SWING_LOOKBACK)
    if side=="BUY":
        stop = float(r.Low.min())-atr*ATR_STOP_BUFFER
        risk = entry-stop
    else:
        stop = float(r.High.max())+atr*ATR_STOP_BUFFER
        risk = stop-entry

    if risk <= 0 or risk < atr*.5 or risk > atr*3.5:
        return None

    room = structure_room(side,entry,m)
    if room is not None and room < risk*TP2_R*.90:
        return None

    if side=="BUY":
        a,b,c = entry+risk*TP1_R,entry+risk*TP2_R,entry+risk*RUNNER_R
    else:
        a,b,c = entry-risk*TP1_R,entry-risk*TP2_R,entry-risk*RUNNER_R

    stop,a,b,c = map(tick,[stop,a,b,c])
    risk = abs(entry-stop)
    rr = abs(b-entry)/risk if risk else 0
    if rr < MIN_RR: return None
    return {"stop":stop,"risk":risk,"tp1":a,"tp2":b,"runner":c,
            "atr":atr,"rr":rr}


def quantity(risk):
    if risk <= 0: return 0
    return max(0,math.floor((ACCOUNT_EQUITY*RISK_PER_TRADE) /
                            (risk*CONTRACT_MULTIPLIER)))


# ------------------------- risk -------------------------------

def risk_lock():
    s = get_state()
    if s["quarantined"]: return True,"quarantined"
    if s["trades_count"] >= MAX_TRADES_PER_DAY: return True,"max trades/day"
    if s["realized_pnl"] <= -ACCOUNT_EQUITY*MAX_DAILY_LOSS:
        return True,"daily loss limit"
    if s["consecutive_sl"] >= MAX_CONSECUTIVE_SL:
        return True,"consecutive SL limit"
    if open_trade(): return True,"open position exists"
    return False,""


def cooldown_lock():
    c = db()
    r = c.execute("""SELECT opened_at FROM trades
                     WHERE status IN ('CLOSED','OPEN','TP1_HIT','TP2_HIT')
                     ORDER BY id DESC LIMIT 1""").fetchone()
    c.close()
    if not r or not r["opened_at"]: return False,""
    try:
        t = datetime.fromisoformat(r["opened_at"])
        mins = (now()-t).total_seconds()/60
        if mins < COOLDOWN_MINUTES:
            return True,f"cooldown {mins:.1f}/{COOLDOWN_MINUTES} min"
    except Exception: pass
    return False,""


# ------------------------- signal -----------------------------

def make_signal():
    mk = market()
    if mk is None: return None,{"status":"ERROR","message":"market data unavailable"}

    f,m,h = mk["5m"],mk["15m"],mk["1h"]
    if len(f)<60: return None,{"status":"ERROR","message":"5m data insufficient"}

    bar_ts = f.index[-2]
    entry = tick(float(f.Close.iloc[-2]))
    active,sess = active_session(bar_ts.to_pydatetime())
    if not active: return None,{"status":"SKIPPED","message":"outside session"}
    if news_blackout(bar_ts.to_pydatetime()):
        return None,{"status":"SKIPPED","message":"news blackout"}

    reg = regime(h)
    if reg not in ("BULL","BEAR"):
        return None,{"status":"HOLD","message":"regime="+reg}

    st = setup(m,reg)
    if not st["valid"]: return None,{"status":"HOLD","message":st["reason"]}

    tr = trigger(f,st["side"])
    if not tr["valid"]: return None,{"status":"HOLD","message":tr["reason"]}

    sc,pos,neg,dv,tv = score(st["side"],reg,h,m,f,mk["dxy"],mk["tnx"])
    if sc < SCORE_TO_TRADE:
        return None,{"status":"HOLD","message":f"score {sc} < {SCORE_TO_TRADE}",
                     "positives":pos,"negatives":neg}

    lv = levels(st["side"],entry,f,m)
    if not lv: return None,{"status":"HOLD","message":"invalid risk geometry"}

    H = indicators(h).iloc[-2]
    raw = f"{SYMBOL}|{st['side']}|{tr['bar']}|{entry}|{sc}"
    sid = hashlib.sha256(raw.encode()).hexdigest()[:24]
    sig = Signal(
        SYMBOL,st["side"],entry,lv["stop"],lv["tp1"],lv["tp2"],lv["runner"],
        lv["risk"],sc,"1H regime + 15m pullback + 5m VWAP rejection",
        sess,"; ".join(pos),now().isoformat(),tr["bar"],lv["atr"],st["atr"],
        float(H.EMA200),st["ema20"],st["vwap"],dv,tv,sid)
    return sig,{"status":"SIGNAL","signal":asdict(sig),
                "rr":lv["rr"],"positives":pos,"negatives":neg}


# ------------------------- persistence/execution --------------

def insert_signal(sig):
    c = db()
    try:
        c.execute("""INSERT INTO signals
          (signal_id,symbol,action,trigger_bar_timestamp,created_at,score,status,payload)
          VALUES(?,?,?,?,?,?,?,?)""",
          (sig.signal_id,sig.symbol,sig.action,sig.trigger_bar_timestamp,
           now().isoformat(),sig.score,"GENERATED",
           json.dumps(asdict(sig),sort_keys=True)))
        c.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        c.close()


def alert_existing(sig):
    msg = (f"*GOLD {sig.action} SIGNAL*\n"
           f"Score `{sig.score}` | {sig.session}\n"
           f"Entry `{sig.entry:.2f}` | SL `{sig.stop:.2f}`\n"
           f"TP1 `{sig.tp1:.2f}` | TP2 `{sig.tp2:.2f}` | Runner `{sig.runner:.2f}`\n"
           f"Risk/unit `{sig.risk_per_unit:.2f}`\n"
           f"DXY `{sig.dxy}` | US10Y `{sig.tnx}`\n"
           f"{sig.rationale}")
    ok = telegram(msg)
    c=db(); c.execute("UPDATE signals SET status=? WHERE signal_id=?",
                      ("ALERTED",sig.signal_id)); c.commit(); c.close()
    s=get_state(); update_state(signals_count=s["signals_count"]+1)
    return {"status":"ALERTED","quantity":quantity(sig.risk_per_unit),
            "telegram_sent":ok}


def paper_open(sig,q):
    fill = tick(sig.entry + (SLIPPAGE_TICKS*TICK_SIZE if sig.action=="BUY"
                             else -SLIPPAGE_TICKS*TICK_SIZE))
    tid = "PAPER-"+sig.signal_id
    c=db()
    try:
        c.execute("""INSERT INTO trades
          (trade_id,signal_id,symbol,action,entry,stop,tp1,tp2,runner,
           risk_per_unit,quantity,remaining_quantity,score,setup,session,
           signal_timestamp,opened_at,status)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (tid,sig.signal_id,sig.symbol,sig.action,fill,sig.stop,sig.tp1,sig.tp2,
           sig.runner,sig.risk_per_unit,q,q,sig.score,sig.setup,sig.session,
           sig.timestamp,now().isoformat(),"OPEN"))
        c.commit()
    except sqlite3.IntegrityError:
        c.close(); return {"status":"DUPLICATE","trade_id":tid}
    c.close()
    s=get_state(); update_state(trades_count=s["trades_count"]+1,
                                signals_count=s["signals_count"]+1)
    c=db(); c.execute("UPDATE signals SET status=? WHERE signal_id=?",
                      ("PAPER_OPEN",sig.signal_id)); c.commit(); c.close()
    telegram(f"*PAPER GOLD {sig.action} OPEN*\nEntry `{fill:.2f}` | "
             f"SL `{sig.stop:.2f}` | TP1 `{sig.tp1:.2f}` | TP2 `{sig.tp2:.2f}` | "
             f"Runner `{sig.runner:.2f}` | Qty `{q}`")
    return {"status":"PAPER_OPEN","trade_id":tid,"quantity":q,"fill":fill}


def execute(sig):
    locked,why=risk_lock()
    if locked: return {"status":"BLOCKED","reason":why}
    locked,why=cooldown_lock()
    if locked: return {"status":"BLOCKED","reason":why}

    q=quantity(sig.risk_per_unit)
    if q<1: return {"status":"BLOCKED","reason":"quantity < 1; use MGC or lower risk"}

    # Atomic unique signal ID prevents repeated cron/webhook calls.
    if not insert_signal(sig):
        return {"status":"DUPLICATE_SIGNAL","signal_id":sig.signal_id}

    if EXECUTION_MODE=="ALERT":
        return alert_existing(sig)
    if EXECUTION_MODE=="PAPER":
        return paper_open(sig,q)
    if EXECUTION_MODE=="LIVE":
        c=db(); c.execute("UPDATE signals SET status=? WHERE signal_id=?",
                          ("LIVE_BLOCKED",sig.signal_id)); c.commit(); c.close()
        return {"status":"BLOCKED",
                "reason":"LIVE disabled until a broker adapter is implemented"}
    return {"status":"BLOCKED","reason":"invalid EXECUTION_MODE"}


# ------------------------- paper lifecycle -------------------

def realize(trade, price, qty, reason):
    qty=min(float(qty),float(trade["remaining_quantity"]))
    if qty<=0: return
    if trade["action"]=="BUY":
        gross=(price-trade["entry"])*qty*CONTRACT_MULTIPLIER
    else:
        gross=(trade["entry"]-price)*qty*CONTRACT_MULTIPLIER
    pnl=gross-qty*COMMISSION_PER_CONTRACT
    risk=float(trade["risk_per_unit"])*qty*CONTRACT_MULTIPLIER
    r=pnl/risk if risk else 0
    rem=float(trade["remaining_quantity"])-qty
    status="CLOSED" if rem<=0 else ("TP1_HIT" if reason=="TP1" else "TP2_HIT")
    c=db()
    c.execute("""UPDATE trades SET remaining_quantity=?,status=?,
                 exit_price=?,pnl=COALESCE(pnl,0)+?,
                 r_multiple=COALESCE(r_multiple,0)+?,exit_reason=?
                 WHERE trade_id=?""",
              (max(0,rem),status,price,pnl,r,reason,trade["trade_id"]))
    c.commit(); c.close()
    s=get_state(); newp=s["realized_pnl"]+pnl
    cs=s["consecutive_sl"]
    if rem<=0:
        if reason=="STOP": cs+=1
        elif pnl>0: cs=0
    q=int(cs>=MAX_CONSECUTIVE_SL or newp<=-ACCOUNT_EQUITY*MAX_DAILY_LOSS)
    update_state(realized_pnl=newp,consecutive_sl=cs,quarantined=q)


def manage_paper():
    t=open_trade()
    if not t: return {"status":"NO_OPEN_POSITION"}
    mk=market()
    if mk is None: return {"status":"ERROR","message":"market data unavailable"}
    b=mk["5m"].iloc[-2]
    hi,lo=float(b.High),float(b.Low)

    # Conservative same-bar assumption: stop first if both stop and target hit.
    if t["action"]=="BUY":
        if lo<=t["stop"]:
            realize(t,t["stop"],t["remaining_quantity"],"STOP")
            return {"status":"CLOSED","reason":"STOP","price":t["stop"]}
        if not t["tp1_hit"] and hi>=t["tp1"]:
            q=max(0,math.floor(t["quantity"]*TP1_PCT))
            if q>=1:
                realize(t,t["tp1"],q,"TP1")
                c=db(); c.execute("""UPDATE trades SET tp1_hit=1,stop=? 
                                    WHERE trade_id=?""",
                                  (tick(t["entry"]),t["trade_id"])); c.commit(); c.close()
                return {"status":"TP1_HIT","price":t["tp1"],"quantity":q}
        t=open_trade() or t
        if not t["tp2_hit"] and hi>=t["tp2"]:
            q=max(0,math.floor(t["quantity"]*TP2_PCT))
            if q>=1:
                realize(t,t["tp2"],q,"TP2")
                c=db(); c.execute("UPDATE trades SET tp2_hit=1 WHERE trade_id=?",
                                  (t["trade_id"],)); c.commit(); c.close()
                return {"status":"TP2_HIT","price":t["tp2"],"quantity":q}
        t=open_trade() or t
        if t and float(t["remaining_quantity"])>0 and hi>=t["runner"]:
            realize(t,t["runner"],t["remaining_quantity"],"RUNNER")
            return {"status":"CLOSED","reason":"RUNNER","price":t["runner"]}
    else:
        if hi>=t["stop"]:
            realize(t,t["stop"],t["remaining_quantity"],"STOP")
            return {"status":"CLOSED","reason":"STOP","price":t["stop"]}
        if not t["tp1_hit"] and lo<=t["tp1"]:
            q=max(0,math.floor(t["quantity"]*TP1_PCT))
            if q>=1:
                realize(t,t["tp1"],q,"TP1")
                c=db(); c.execute("""UPDATE trades SET tp1_hit=1,stop=?
                                    WHERE trade_id=?""",
                                  (tick(t["entry"]),t["trade_id"])); c.commit(); c.close()
                return {"status":"TP1_HIT","price":t["tp1"],"quantity":q}
        t=open_trade() or t
        if not t["tp2_hit"] and lo<=t["tp2"]:
            q=max(0,math.floor(t["quantity"]*TP2_PCT))
            if q>=1:
                realize(t,t["tp2"],q,"TP2")
                c=db(); c.execute("UPDATE trades SET tp2_hit=1 WHERE trade_id=?",
                                  (t["trade_id"],)); c.commit(); c.close()
                return {"status":"TP2_HIT","price":t["tp2"],"quantity":q}
        t=open_trade() or t
        if t and float(t["remaining_quantity"])>0 and lo<=t["runner"]:
            realize(t,t["runner"],t["remaining_quantity"],"RUNNER")
            return {"status":"CLOSED","reason":"RUNNER","price":t["runner"]}
    return {"status":"MONITORING","trade_id":t["trade_id"],
            "remaining_quantity":t["remaining_quantity"]}


# ------------------------- metrics ---------------------------

def metrics():
    c=db()
    rows=c.execute("SELECT * FROM trades WHERE status='CLOSED' ORDER BY id").fetchall()
    c.close()
    if not rows: return {"closed_trades":0,"message":"No completed trades"}
    p=np.array([float(r["pnl"] or 0) for r in rows])
    rr=np.array([float(r["r_multiple"] or 0) for r in rows])
    wins=p[p>0]; losses=p[p<0]
    eq=np.cumsum(p); peak=np.maximum.accumulate(np.insert(eq,0,0))[1:]
    dd=eq-peak
    s=get_state()
    return {
        "closed_trades":len(p),
        "win_rate":round(float((p>0).mean()),4),
        "avg_pnl":round(float(p.mean()),2),
        "avg_R":round(float(rr.mean()),3),
        "profit_factor":round(float(wins.sum()/abs(losses.sum())),3) if len(losses) else None,
        "max_drawdown":round(abs(float(dd.min())),2),
        "best_trade":round(float(p.max()),2),
        "worst_trade":round(float(p.min()),2),
        "today_realized_pnl":round(float(s["realized_pnl"]),2),
        "today_trades":s["trades_count"],
        "today_signals":s["signals_count"],
        "consecutive_sl":s["consecutive_sl"],
        "quarantined":bool(s["quarantined"])
    }


# ------------------------- Flask ------------------------------

def authorized():
    if not WEBHOOK_SECRET: return True
    supplied=request.headers.get("X-Webhook-Key") or request.args.get("key")
    return supplied==WEBHOOK_SECRET


@app.route("/webhook",methods=["GET","POST"])
def webhook():
    if not authorized(): return jsonify({"status":"UNAUTHORIZED"}),401
    try:
        # Manage an existing paper position before considering a new signal.
        if EXECUTION_MODE=="PAPER":
            managed=manage_paper()
            if managed["status"] in {"MONITORING","TP1_HIT","TP2_HIT"}:
                return jsonify({"status":"MANAGING","management":managed}),200

        sig,info=make_signal()
        if sig is None: return jsonify(info),200
        return jsonify({"status":"SUCCESS","signal":asdict(sig),
                        "execution":execute(sig)}),200
    except Exception as e:
        log.exception("webhook error")
        return jsonify({"status":"ERROR","message":str(e)}),500


@app.route("/manage",methods=["GET","POST"])
def manage():
    if not authorized(): return jsonify({"status":"UNAUTHORIZED"}),401
    if EXECUTION_MODE!="PAPER":
        return jsonify({"status":"SKIPPED","message":"PAPER mode only"}),200
    try: return jsonify(manage_paper()),200
    except Exception as e:
        log.exception("manage error")
        return jsonify({"status":"ERROR","message":str(e)}),500


@app.route("/health")
def health():
    return jsonify({"status":"OK","symbol":SYMBOL,
                    "execution_mode":EXECUTION_MODE,
                    "time":now().isoformat()})


@app.route("/state")
def state_endpoint():
    if not authorized(): return jsonify({"status":"UNAUTHORIZED"}),401
    return jsonify({"daily_state":get_state(),"open_trade":open_trade()})


@app.route("/metrics")
def metrics_endpoint():
    if not authorized(): return jsonify({"status":"UNAUTHORIZED"}),401
    return jsonify(metrics())


if __name__=="__main__":
    port=int(os.getenv("PORT","5000"))
    log.info("Starting gold agent: %s / %s",SYMBOL,EXECUTION_MODE)
    if EXECUTION_MODE=="LIVE":
        log.critical("LIVE requested but blocked: broker adapter not implemented.")
    app.run(host="0.0.0.0",port=port,threaded=True)

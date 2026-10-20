#!/usr/bin/env python3
"""
GOLD INTRADAY AGENT  V4.2
1H regime -> 15m EMA20 pullback -> 5m VWAP rejection -> STOP-ENTRY order
-> swing/ATR stop + 15m room check -> equity-based sizing
-> bar-by-bar lifecycle (stop-first, gap-aware) -> journal -> kill switches

ONE code path for live, paper and backtest (no live/backtest drift):
  python gold_agent_v4_2.py                                       # = serve (default; Railway-friendly)
  python gold_agent_v4_2.py serve [--csv bars.csv]                # service, self-scheduled every RUN_INTERVAL_SEC
  python gold_agent_v4_2.py backtest --csv bars.csv [--start 2023-01-01 --end 2025-12-31] [--out trades.csv]
  gunicorn gold_agent_v4_2:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT   # also works

Railway + cron-job.org (v4.2 fixes "404 Not Found -> cronjob disabled automatically"):
  * Start command:  python gold_agent_v4_2.py      (no subcommand needed any more)
  * Cron URL:       https://<app>.up.railway.app/webhook   (GET or POST; trailing slash OK)
  * Auth:           header X-Webhook-Key: <WEBHOOK_SECRET>  (or ?key=<secret> if the cron tool can't send headers)
  * /webhook answers 200 instantly and runs the cycle in the background -> cron never times out.
  * The agent self-schedules every RUN_INTERVAL_SEC; cron is a keep-alive + backup trigger.
  * Use ONE worker/replica and a Railway Volume, e.g. DB_PATH=/data/gold_agent.db

CSV columns: timestamp,open,high,low,close,volume  (UTC; 1m or 5m bars, continuous back-adjusted contract)
Every parameter is an environment variable (see Config) so sweeps / walk-forward need no code changes.
"""
import os
import sys
import json
import math
import hmac
import html
import sqlite3
import logging
import hashlib
import threading
import time
import argparse
import dataclasses
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import time as dtime
from functools import wraps
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")
LDN = ZoneInfo("Europe/London")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gold-agent-v4")


# ============================================================
# CONFIG
# ============================================================

def _f(k, d): return float(os.getenv(k, d))
def _i(k, d): return int(os.getenv(k, d))
def _b(k, d): return os.getenv(k, d).strip().lower() in ("1", "true", "yes", "y")


def _t(k, d):
    hh, mm = os.getenv(k, d).split(":")
    return dtime(int(hh), int(mm))


@dataclass(frozen=True)
class Config:
    symbol: str = os.getenv("SYMBOL", "MGC=F")                 # micro gold: sizing works at 100k
    mode: str = os.getenv("EXECUTION_MODE", "ALERT").upper()     # ALERT | PAPER | LIVE(blocked)
    db_path: str = os.getenv("DB_PATH", "gold_agent_v4.db")
    data_provider: str = os.getenv("DATA_PROVIDER", "yfinance")  # yfinance (prototype only) | csv
    data_csv: str = os.getenv("DATA_CSV", "")
    run_interval_sec: int = _i("RUN_INTERVAL_SEC", "60")
    max_data_lag_min: float = _f("MAX_DATA_LAG_MIN", "7")      # last CLOSED 5m bar must be this fresh

    # --- account & risk ---
    start_equity: float = _f("ACCOUNT_EQUITY", "100000")
    risk_per_trade: float = _f("RISK_PER_TRADE", "0.005")
    max_daily_loss: float = _f("MAX_DAILY_LOSS", "0.015")
    max_weekly_loss: float = _f("MAX_WEEKLY_LOSS", "0.03")
    max_dd_halt: float = _f("MAX_DRAWDOWN_HALT", "0.08")
    max_consec_losses: int = _i("MAX_CONSECUTIVE_LOSSES", "2")
    max_trades_day: int = _i("MAX_TRADES_PER_DAY", "3")
    cooldown_min: int = _i("COOLDOWN_MINUTES", "30")
    max_contracts: int = _i("MAX_CONTRACTS", "10")
    expected_r: float = _f("EXPECTED_R", "0.15")          # set from YOUR out-of-sample backtest
    drift_tolerance_r: float = _f("DRIFT_TOLERANCE_R", "0.30")
    drift_window: int = _i("DRIFT_WINDOW", "30")

    # --- contract & costs ---
    multiplier: float = _f("CONTRACT_MULTIPLIER", "10")   # MGC=10, GC=100
    tick: float = _f("TICK_SIZE", "0.10")
    commission_side: float = _f("COMMISSION_PER_SIDE", "1.00")  # set to your broker's all-in fee
    entry_slip_ticks: float = _f("ENTRY_SLIPPAGE_TICKS", "2")
    stop_slip_ticks: float = _f("STOP_SLIPPAGE_TICKS", "2")

    # --- sessions (local exchange time, DST-safe) & news ---
    london_start: dtime = _t("LONDON_START", "08:00")     # Europe/London
    london_end: dtime = _t("LONDON_END", "11:30")
    ny_start: dtime = _t("NY_START", "09:30")             # America/New_York (after 08:30 data)
    ny_end: dtime = _t("NY_END", "12:30")
    flatten_et: dtime = _t("FLATTEN_ET", "16:30")
    recurring_blackouts_et: str = os.getenv("RECURRING_BLACKOUTS_ET", "08:20-08:50,09:55-10:20,13:55-14:45")
    news_file: str = os.getenv("NEWS_EVENTS_FILE", "")   # JSON: [{"time":"2026-11-06T13:30:00Z","name":"NFP"}]
    news_before_min: int = _i("NEWS_BEFORE_MIN", "15")
    news_after_min: int = _i("NEWS_AFTER_MIN", "30")

    # --- strategy ---
    atr_n: int = _i("ATR_PERIOD", "14")
    regime_slope_bars: int = _i("REGIME_SLOPE_BARS", "20")
    regime_buffer_atr: float = _f("REGIME_BUFFER_ATR", "0.30")
    setup_valid_bars: int = _i("SETUP_VALID_BARS", "2")
    pullback_min_atr: float = _f("PULLBACK_MIN_ATR", "0.75")
    swing_lookback: int = _i("SWING_LOOKBACK", "8")
    stop_buffer_atr: float = _f("ATR_STOP_BUFFER", "0.20")
    min_risk_atr: float = _f("MIN_RISK_ATR", "0.60")
    max_risk_atr: float = _f("MAX_RISK_ATR", "3.00")
    min_atr_pct: float = _f("MIN_ATR_PCT", "0.0005")
    max_atr_pct: float = _f("MAX_ATR_PCT", "0.012")
    vol_shock_mult: float = _f("VOL_SHOCK_MULT", "2.2")
    use_volume_filter: bool = _b("USE_VOLUME_FILTER", "false")  # enable only with real exchange volume
    min_rel_volume: float = _f("MIN_REL_VOLUME", "1.10")
    entry_valid_bars: int = _i("ENTRY_VALID_BARS", "2")
    room_min_r: float = _f("ROOM_MIN_R", "1.80")

    # --- exits (default: single 2R target + profit lock; partials optional) ---
    target_r: float = _f("TARGET_R", "2.0")
    tp1_r: float = _f("TP1_R", "1.0")
    tp1_pct: float = _f("TP1_PCT", "0.0")                 # 0 = no partial
    lock_trigger_r: float = _f("LOCK_TRIGGER_R", "1.5")
    lock_r: float = _f("LOCK_R", "0.5")
    time_stop_bars: int = _i("TIME_STOP_BARS", "36")      # 3 hours on 5m

    # --- service ---
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "")
    allow_insecure: bool = _b("ALLOW_INSECURE", "false")
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat: str = os.getenv("TELEGRAM_CHAT_ID", "")


# ============================================================
# HELPERS
# ============================================================

def finite(*xs) -> bool:
    try:
        return all(np.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def to_tick(p: float, tick: float, mode: str = "nearest") -> float:
    q = float(p) / tick
    if mode == "down":
        q = math.floor(q + 1e-9)
    elif mode == "up":
        q = math.ceil(q - 1e-9)
    else:
        q = round(q)
    return round(q * tick, 6)


def ts_utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize(UTC) if t.tzinfo is None else t.tz_convert(UTC)


def trading_day(ts) -> str:
    """CME trading day: the session that opens 18:00 ET belongs to the next date."""
    return (ts_utc(ts).tz_convert(ET) + pd.Timedelta(hours=6)).date().isoformat()


def week_key(day_iso: str) -> Tuple[int, int]:
    y, w, _ = pd.Timestamp(day_iso).isocalendar()
    return int(y), int(w)


def session_of(ts, cfg: Config) -> Optional[str]:
    ts = ts_utc(ts)
    for name, tz, a, b in (("LONDON", LDN, cfg.london_start, cfg.london_end),
                           ("NEW_YORK", ET, cfg.ny_start, cfg.ny_end)):
        loc = ts.tz_convert(tz)
        if loc.weekday() < 5 and a <= loc.time() < b:
            return name
    return None


def parse_windows(spec: str) -> List[Tuple[dtime, dtime]]:
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        a, b = item.split("-")
        out.append((dtime(*map(int, a.split(":"))), dtime(*map(int, b.split(":")))))
    return out


def load_events(path: str) -> List[pd.Timestamp]:
    if not path or not os.path.exists(path):
        if path:
            log.warning("NEWS_EVENTS_FILE not found: %s", path)
        return []
    with open(path) as fh:
        return sorted(ts_utc(e["time"]) for e in json.load(fh))


def in_blackout(start: pd.Timestamp, end: pd.Timestamp, cfg: Config,
                events: List[pd.Timestamp]) -> Tuple[bool, str]:
    windows = parse_windows(cfg.recurring_blackouts_et)
    t = start
    while t <= end:
        loc = t.tz_convert(ET)
        for a, b in windows:
            if loc.weekday() < 5 and a <= loc.time() < b:
                return True, f"recurring US data window {a:%H:%M}-{b:%H:%M} ET"
        t += pd.Timedelta(minutes=5)
    pre, post = pd.Timedelta(minutes=cfg.news_before_min), pd.Timedelta(minutes=cfg.news_after_min)
    for ev in events:
        if ev - pre <= end and ev + post >= start:
            return True, f"news event {ev.isoformat()}"
    return False, ""


# ============================================================
# DATA
# ============================================================

def clean(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d.columns = [str(c).strip().title() for c in d.columns]
    d.index = pd.to_datetime(d.index, utc=True)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    if "Volume" not in d:
        d["Volume"] = 0.0
    for x in ("Open", "High", "Low", "Close", "Volume"):
        d[x] = pd.to_numeric(d[x], errors="coerce")
    d["Volume"] = d["Volume"].fillna(0.0)
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    bad = ((d.High < d[["Open", "Close"]].max(axis=1)) |
           (d.Low > d[["Open", "Close"]].min(axis=1)) | (d.Close <= 0))
    if bad.any():
        log.warning("Dropped %d malformed bars", int(bad.sum()))
    return d.loc[~bad, ["Open", "High", "Low", "Close", "Volume"]]


def resample(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = d.resample(rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def load_csv(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    tcol = next(c for c in d.columns if c.lower() in ("timestamp", "time", "datetime", "date", "ts_event"))
    d = d.set_index(pd.to_datetime(d[tcol], utc=True)).drop(columns=[tcol])
    d = clean(d)
    step = d.index.to_series().diff().median()
    return resample(d, "5min") if step < pd.Timedelta(minutes=5) else d


def load_yf_5m(symbol: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """Prototype feed only: delayed, patched across rolls, ~60 days of 5m history."""
    import yfinance as yf
    delay = 2
    for attempt in range(1, retries + 1):
        try:
            d = yf.Ticker(symbol).history(period="59d", interval="5m", auto_adjust=False, prepost=True)
            if d is not None and not d.empty:
                return clean(d)
        except Exception as e:
            log.warning("yfinance attempt %d/%d failed: %s", attempt, retries, e)
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    log.error("yfinance failed for %s", symbol)
    return None


# ============================================================
# FEATURES  (all causal: each row uses only data up to that row)
# ============================================================

def indicators(d: pd.DataFrame, atr_n: int, vwap: bool) -> pd.DataFrame:
    d = d.copy()
    c = d.Close
    d["EMA20"] = c.ewm(span=20, adjust=False).mean()
    d["EMA200"] = c.ewm(span=200, adjust=False).mean()
    pc = c.shift(1)
    tr = pd.concat([d.High - d.Low, (d.High - pc).abs(), (d.Low - pc).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / atr_n, adjust=False, min_periods=atr_n).mean()   # Wilder
    d["ATRMed"] = d.ATR.rolling(100, min_periods=50).median()
    d["RelVol"] = d.Volume / d.Volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
    if vwap:  # anchored to the CME session open (18:00 ET), not UTC midnight
        key = pd.Series((d.index.tz_convert(ET) + pd.Timedelta(hours=6)).date, index=d.index)
        tp = (d.High + d.Low + d.Close) / 3
        d["VWAP"] = (tp * d.Volume).groupby(key).cumsum() / \
            d.Volume.groupby(key).cumsum().replace(0, np.nan)
    d["Range"] = d.High - d.Low
    d["Body"] = (d.Close - d.Open).abs()
    d["UpperWick"] = d.High - d[["Open", "Close"]].max(axis=1)
    d["LowerWick"] = d[["Open", "Close"]].min(axis=1) - d.Low
    return d


def build_features(base5: pd.DataFrame, cfg: Config):
    f5 = indicators(base5, cfg.atr_n, vwap=True)
    f15 = indicators(resample(base5, "15min"), cfg.atr_n, vwap=True)
    f60 = indicators(resample(base5, "1h"), cfg.atr_n, vwap=False)
    return f5, f15, f60


def closed(df: pd.DataFrame, minutes: int, asof: pd.Timestamp) -> pd.DataFrame:
    """Bars whose END time <= asof. Index = bar open time. Never trusts iloc[-2]."""
    k = df.index.searchsorted(asof - pd.Timedelta(minutes=minutes), side="right")
    return df.iloc[:k]


# ============================================================
# STRATEGY  (pure functions; identical in live and backtest)
# ============================================================

def regime(h: pd.DataFrame, cfg: Config) -> str:
    n = cfg.regime_slope_bars
    if len(h) < max(400, n + 2):          # EMA200 warm-up
        return "UNKNOWN"
    x, p = h.iloc[-1], h.iloc[-1 - n]
    if not finite(x.Close, x.EMA200, x.ATR, p.EMA200):
        return "UNKNOWN"
    buf = cfg.regime_buffer_atr * x.ATR
    if x.Close > x.EMA200 + buf and x.EMA200 > p.EMA200:
        return "BULL"
    if x.Close < x.EMA200 - buf and x.EMA200 < p.EMA200:
        return "BEAR"
    return "RANGE"


def setup(m: pd.DataFrame, side: str, cfg: Config) -> Tuple[bool, str]:
    """Real pullback: touched & held EMA20, minimum depth, structure (higher low / lower high) intact."""
    if len(m) < 60:
        return False, "insufficient 15m data"
    for j in range(1, cfg.setup_valid_bars + 1):
        x, prior = m.iloc[-j], m.iloc[-j - 20:-j]
        if len(prior) < 20 or not finite(x.Close, x.Low, x.High, x.EMA20, x.ATR):
            continue
        a = x.ATR
        if side == "BUY":
            ok = (x.Low <= x.EMA20 + 0.1 * a and x.Close > x.EMA20 and
                  prior.High.max() - x.Low >= cfg.pullback_min_atr * a and
                  x.Low > prior.Low.min())
        else:
            ok = (x.High >= x.EMA20 - 0.1 * a and x.Close < x.EMA20 and
                  x.High - prior.Low.min() >= cfg.pullback_min_atr * a and
                  x.High < prior.High.max())
        if ok:
            return True, f"15m EMA20 pullback ({j} bar(s) ago)"
    return False, "no qualified 15m pullback"


def trigger(f: pd.DataFrame, side: str, cfg: Config) -> Tuple[Optional[float], str]:
    """5m VWAP rejection candle -> stop-entry 1 tick beyond it (no chasing the breakout close)."""
    if len(f) < 120:
        return None, "insufficient 5m data"
    r = f.iloc[-1]
    if not finite(r.ATR, r.VWAP, r.Range) or r.Range <= 0:
        return None, "invalid 5m bar/indicators (check volume/VWAP)"
    a, rng = r.ATR, r.Range
    loc = (r.Close - r.Low) / rng
    if side == "BUY":
        ok = (r.Low <= r.VWAP + .25 * a and r.Close > r.VWAP and r.LowerWick / rng >= .25 and
              r.Body / rng >= .20 and r.Close > r.Open and loc >= .55)
        entry = to_tick(r.High + cfg.tick, cfg.tick, "up")
    else:
        ok = (r.High >= r.VWAP - .25 * a and r.Close < r.VWAP and r.UpperWick / rng >= .25 and
              r.Body / rng >= .20 and r.Close < r.Open and loc <= .45)
        entry = to_tick(r.Low - cfg.tick, cfg.tick, "down")
    return (entry, "5m VWAP rejection") if ok else (None, "no 5m VWAP rejection")


def pivots(m: pd.DataFrame, kind: str, left: int = 3, right: int = 3, lookback: int = 80) -> List[float]:
    v = (m.High if kind == "high" else m.Low).values[-lookback:]
    out = []
    for i in range(left, len(v) - right):
        w = v[i - left:i + right + 1]
        if (kind == "high" and v[i] >= w.max()) or (kind == "low" and v[i] <= w.min()):
            out.append(float(v[i]))
    return out


def levels(side: str, entry: float, f: pd.DataFrame, m: pd.DataFrame, cfg: Config):
    atr = float(f.ATR.iloc[-1])
    sw = f.iloc[-cfg.swing_lookback:]
    s = 1 if side == "BUY" else -1
    if side == "BUY":
        stop = to_tick(sw.Low.min() - cfg.stop_buffer_atr * atr, cfg.tick, "down")
        risk = entry - stop
        obst = [p for p in pivots(m, "high") if p > entry]
        room = (min(obst) - entry) if obst else None
    else:
        stop = to_tick(sw.High.max() + cfg.stop_buffer_atr * atr, cfg.tick, "up")
        risk = stop - entry
        obst = [p for p in pivots(m, "low") if p < entry]
        room = (entry - max(obst)) if obst else None
    if not (cfg.min_risk_atr * atr <= risk <= cfg.max_risk_atr * atr):
        return None, f"risk {risk:.2f} outside {cfg.min_risk_atr}-{cfg.max_risk_atr} x ATR"
    if room is not None and room < cfg.room_min_r * risk:
        return None, f"15m swing only {room / risk:.2f}R away (< {cfg.room_min_r}R)"
    target = to_tick(entry + s * cfg.target_r * risk, cfg.tick)
    tp1 = to_tick(entry + s * cfg.tp1_r * risk, cfg.tick) if cfg.tp1_pct > 0 else None
    return {"stop": stop, "target": target, "tp1": tp1, "risk": round(risk, 6),
            "atr": atr, "room_r": None if room is None else round(room / risk, 2)}, "ok"


def evaluate_entry(F5, F15, F60, asof: pd.Timestamp, cfg: Config, events, check_lag: bool = True):
    f = closed(F5, 5, asof)
    if len(f) < 120:
        return None, {"status": "HOLD", "reason": "insufficient 5m data"}
    bar_ts = f.index[-1]
    bar_close = bar_ts + pd.Timedelta(minutes=5)
    info = {"bar": bar_ts.isoformat()}
    lag = (asof - bar_close).total_seconds() / 60
    if check_lag and lag > cfg.max_data_lag_min:
        return None, {**info, "status": "HOLD", "reason": f"stale data: last bar closed {lag:.1f} min ago"}
    sess = session_of(bar_close, cfg)
    if not sess:
        return None, {**info, "status": "SKIPPED", "reason": "outside session"}

    h, m = closed(F60, 60, asof), closed(F15, 15, asof)
    reg = regime(h, cfg)
    if reg not in ("BULL", "BEAR"):
        return None, {**info, "status": "HOLD", "reason": f"regime={reg}"}
    side = "BUY" if reg == "BULL" else "SELL"

    ok, why = setup(m, side, cfg)
    if not ok:
        return None, {**info, "status": "HOLD", "reason": why}
    entry, why = trigger(f, side, cfg)
    if entry is None:
        return None, {**info, "status": "HOLD", "reason": why}

    r = f.iloc[-1]
    atr_pct = float(r.ATR / r.Close)
    if not (cfg.min_atr_pct <= atr_pct <= cfg.max_atr_pct):
        return None, {**info, "status": "HOLD", "reason": f"volatility {atr_pct:.4%} out of band"}
    if finite(r.ATRMed) and (r.ATR > cfg.vol_shock_mult * r.ATRMed or r.Range > 3 * r.ATR):
        return None, {**info, "status": "HOLD", "reason": "volatility shock"}
    if cfg.use_volume_filter and not (finite(r.RelVol) and r.RelVol >= cfg.min_rel_volume):
        return None, {**info, "status": "HOLD", "reason": "weak relative volume"}

    expires = bar_close + pd.Timedelta(minutes=5 * cfg.entry_valid_bars)
    bo, why = in_blackout(bar_close, expires, cfg, events)
    if bo:
        return None, {**info, "status": "SKIPPED", "reason": why}

    lv, why = levels(side, entry, f, m, cfg)
    if lv is None:
        return None, {**info, "status": "HOLD", "reason": why}

    sid = hashlib.sha256(f"{cfg.symbol}|{side}|{bar_ts.isoformat()}".encode()).hexdigest()[:20]
    order = {
        "signal_id": sid, "symbol": cfg.symbol, "side": side, "entry": entry,
        "stop": lv["stop"], "target": lv["target"], "tp1": lv["tp1"], "risk": lv["risk"],
        "trigger_bar": bar_ts.isoformat(), "created": asof.isoformat(), "expires": expires.isoformat(),
        "session": sess,
        "features": {  # logged for post-trade analysis; never used to size or filter
            "regime": reg, "atr_5m": round(float(r.ATR), 3), "atr_pct": round(atr_pct, 5),
            "rel_vol": round(float(r.RelVol), 2) if finite(r.RelVol) else None,
            "dist_vwap_atr": round(float((r.Close - r.VWAP) / r.ATR), 2),
            "dist_ema200_1h_atr": round(float((h.Close.iloc[-1] - h.EMA200.iloc[-1]) / h.ATR.iloc[-1]), 2),
            "room_r": lv["room_r"], "hour_et": bar_close.tz_convert(ET).hour,
            "weekday": bar_close.tz_convert(ET).weekday()},
    }
    return order, {**info, "status": "SIGNAL", "reason": f"{reg} | 15m pullback | 5m VWAP rejection"}


# ============================================================
# SIZING, RISK GATES, KILL SWITCHES
# ============================================================

def size(cfg: Config, equity: float, risk_per_unit: float) -> int:
    per_contract = (risk_per_unit * cfg.multiplier + 2 * cfg.commission_side +
                    (cfg.entry_slip_ticks + cfg.stop_slip_ticks) * cfg.tick * cfg.multiplier)
    if per_contract <= 0 or equity <= 0:
        return 0
    return int(min(cfg.max_contracts, math.floor(equity * cfg.risk_per_trade / per_contract)))


def risk_gate(cfg: Config, trades: List[dict], entries_today: int,
              last_exit: Optional[pd.Timestamp], asof: pd.Timestamp, halted: Optional[str]):
    equity = cfg.start_equity + sum(t["pnl"] for t in trades)
    if halted:
        return False, f"halted: {halted}", equity
    day = trading_day(asof)
    wk = week_key(day)
    day_pnl = sum(t["pnl"] for t in trades if t["day"] == day)
    week_pnl = sum(t["pnl"] for t in trades if week_key(t["day"]) == wk)
    consec = 0
    for t in reversed(trades):
        if t["day"] != day or t["r"] >= 0:
            break
        consec += 1
    if entries_today >= cfg.max_trades_day:
        return False, "max trades/day", equity
    if day_pnl <= -equity * cfg.max_daily_loss:
        return False, "daily loss limit", equity
    if week_pnl <= -equity * cfg.max_weekly_loss:
        return False, "weekly loss limit", equity
    if consec >= cfg.max_consec_losses:
        return False, "consecutive loss limit", equity
    if last_exit is not None and asof - last_exit < pd.Timedelta(minutes=cfg.cooldown_min):
        return False, "cooldown after exit", equity
    return True, "", equity


def halt_check(cfg: Config, trades: List[dict]) -> Optional[str]:
    if not trades:
        return None
    eq = cfg.start_equity + np.cumsum([t["pnl"] for t in trades])
    peak = np.maximum.accumulate(np.r_[cfg.start_equity, eq])[1:]
    dd = float(((peak - eq) / peak).max())
    if dd >= cfg.max_dd_halt:
        return f"drawdown {dd:.1%} >= {cfg.max_dd_halt:.0%}"
    if len(trades) >= cfg.drift_window:
        avg = float(np.mean([t["r"] for t in trades[-cfg.drift_window:]]))
        if avg < cfg.expected_r - cfg.drift_tolerance_r:
            return f"live drift: last {cfg.drift_window} trades avg {avg:+.2f}R vs expected {cfg.expected_r:+.2f}R"
    return None


# ============================================================
# POSITION LIFECYCLE  (one closed 5m bar at a time; same in live & backtest)
# ============================================================

def new_position(order: dict, qty: int, trade_id: str) -> dict:
    return {"trade_id": trade_id, "signal_id": order["signal_id"], "side": order["side"],
            "status": "PENDING", "session": order["session"], "trading_day": trading_day(order["created"]),
            "created_at": order["created"], "entry_order": order["entry"], "stop": order["stop"],
            "init_stop": order["stop"], "target": order["target"], "tp1": order["tp1"],
            "qty": qty, "remaining": qty, "fill": None, "risk_per_unit": None, "bars_held": 0,
            "expires": order["expires"], "pnl": 0.0, "r_multiple": None, "locked": False,
            "tp1_done": False, "opened_at": None, "closed_at": None, "exit_reason": None,
            "exit_price": None, "last_bar": order["trigger_bar"], "features": order.get("features", {})}


def _close(pos, px, qty, reason, ts, cfg) -> dict:
    s = 1 if pos["side"] == "BUY" else -1
    qty = min(qty, pos["remaining"])
    pos["pnl"] = round(pos["pnl"] + s * (px - pos["fill"]) * qty * cfg.multiplier
                       - 2 * cfg.commission_side * qty, 2)
    pos["remaining"] -= qty
    if pos["remaining"] <= 0:
        pos.update(status="CLOSED", closed_at=ts.isoformat(), exit_reason=reason, exit_price=px)
        # R on the WHOLE position's initial risk (partials are weighted correctly)
        pos["r_multiple"] = round(pos["pnl"] / (pos["risk_per_unit"] * pos["qty"] * cfg.multiplier), 4)
    return {"type": reason, "price": px, "qty": qty, "ts": ts.isoformat()}


def step(pos: dict, ts: pd.Timestamp, o: float, h: float, l: float, c: float, cfg: Config) -> List[dict]:
    """Conservative intrabar assumptions: stop before targets; gaps fill at the open; no targets on fill bar."""
    ev: List[dict] = []
    s = 1 if pos["side"] == "BUY" else -1
    slip_e, slip_s = cfg.entry_slip_ticks * cfg.tick, cfg.stop_slip_ticks * cfg.tick
    bar_end = ts + pd.Timedelta(minutes=5)
    pos["last_bar"] = ts.isoformat()

    if pos["status"] == "PENDING":
        if ts >= pd.Timestamp(pos["expires"]):
            pos.update(status="CANCELLED", closed_at=ts.isoformat(), exit_reason="EXPIRED")
            return [{"type": "EXPIRED", "ts": ts.isoformat()}]
        e = pos["entry_order"]
        if not (h >= e if s > 0 else l <= e):
            return ev
        base = max(e, o) if s > 0 else min(e, o)
        fill = to_tick(base + s * slip_e, cfg.tick)
        if s * (fill - pos["stop"]) <= 0:
            pos.update(status="CANCELLED", closed_at=ts.isoformat(), exit_reason="GAP_THROUGH_STOP")
            return [{"type": "GAP_THROUGH_STOP", "ts": ts.isoformat()}]
        pos.update(status="OPEN", fill=fill, risk_per_unit=abs(fill - pos["stop"]), opened_at=ts.isoformat())
        ev.append({"type": "FILLED", "price": fill, "qty": pos["qty"], "ts": ts.isoformat()})
        if s * ((l if s > 0 else h) - pos["stop"]) <= 0:      # fill and stop in same bar -> assume stopped
            ev.append(_close(pos, to_tick(pos["stop"] - s * slip_s, cfg.tick), pos["remaining"], "STOP", bar_end, cfg))
        return ev

    if pos["status"] != "OPEN":
        return ev
    pos["bars_held"] += 1
    fill, R = pos["fill"], pos["risk_per_unit"]
    adverse, favour = (l, h) if s > 0 else (h, l)
    stop_reason = "LOCK_STOP" if pos["locked"] else "STOP"

    if s * (o - pos["stop"]) <= 0:
        ev.append(_close(pos, to_tick(o - s * slip_s, cfg.tick), pos["remaining"], stop_reason, bar_end, cfg))
        return ev
    if s * (adverse - pos["stop"]) <= 0:
        ev.append(_close(pos, to_tick(pos["stop"] - s * slip_s, cfg.tick), pos["remaining"], stop_reason, bar_end, cfg))
        return ev
    if pos["tp1"] is not None and not pos["tp1_done"] and s * (favour - pos["tp1"]) >= 0:
        q = math.floor(pos["qty"] * cfg.tp1_pct)
        if 1 <= q < pos["remaining"]:
            ev.append(_close(pos, pos["tp1"], q, "TP1", bar_end, cfg))
        pos["tp1_done"] = True
    if s * (favour - pos["target"]) >= 0:
        ev.append(_close(pos, pos["target"], pos["remaining"], "TARGET", bar_end, cfg))
        return ev
    if not pos["locked"] and cfg.lock_trigger_r > 0 and s * (favour - fill) >= cfg.lock_trigger_r * R:
        new = to_tick(fill + s * cfg.lock_r * R, cfg.tick)
        if s * (new - pos["stop"]) > 0:
            pos["stop"] = new
        pos["locked"] = True
        ev.append({"type": "STOP_LOCKED", "price": pos["stop"], "ts": bar_end.isoformat()})
    if pos["bars_held"] >= cfg.time_stop_bars:
        ev.append(_close(pos, to_tick(c - s * slip_s, cfg.tick), pos["remaining"], "TIME_STOP", bar_end, cfg))
        return ev
    et = bar_end.tz_convert(ET).time()
    if cfg.flatten_et <= et < dtime(18, 0):
        ev.append(_close(pos, to_tick(c - s * slip_s, cfg.tick), pos["remaining"], "FLATTEN", bar_end, cfg))
    return ev


# ============================================================
# PERFORMANCE
# ============================================================

def performance(trades: List[dict], mc_runs: int = 2000, seed: int = 7) -> Dict[str, Any]:
    if not trades:
        return {"trades": 0}
    r = np.array([t["r"] for t in trades], dtype=float)
    p = np.array([t["pnl"] for t in trades], dtype=float)

    def max_dd(x):
        eq = np.cumsum(x)
        return float((np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq).max())

    streak = worst = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        worst = max(worst, streak)
    rng = np.random.default_rng(seed)
    mc = [max_dd(rng.permutation(r)) for _ in range(mc_runs)] if len(r) > 1 else [max_dd(r)]
    gains, losses = p[p > 0].sum(), -p[p < 0].sum()
    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["closed_at"], utc=True).dt.year

    def by(col):
        g = df.groupby(col)["r"]
        return {str(k): {"n": int(v.count()), "avg_R": round(float(v.mean()), 3), "sum_R": round(float(v.sum()), 2)}
                for k, v in g}

    return {
        "trades": int(len(r)), "win_rate": round(float((r > 0).mean()), 3),
        "expectancy_R": round(float(r.mean()), 3), "total_R": round(float(r.sum()), 2),
        "profit_factor": round(float(gains / losses), 3) if losses > 0 else None,
        "net_pnl": round(float(p.sum()), 2), "max_dd_R": round(max_dd(r), 2),
        "mc_dd95_R": round(float(np.percentile(mc, 95)), 2), "longest_losing_streak": int(worst),
        "exit_reasons": dict(Counter(df["exit_reason"])),
        "by_session": by("session"), "by_side": by("side"), "by_year": by("year"),
    }


# ============================================================
# PERSISTENCE (single RLock -> no self-deadlock; one transaction per operation)
# ============================================================

class Store:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        c = sqlite3.connect(self.path, timeout=30)
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript("""
            CREATE TABLE IF NOT EXISTS positions(
              trade_id TEXT PRIMARY KEY, signal_id TEXT, side TEXT, status TEXT, trading_day TEXT,
              session TEXT, created_at TEXT, opened_at TEXT, closed_at TEXT, qty REAL, fill REAL,
              init_stop REAL, target REAL, pnl REAL, r_multiple REAL, exit_reason TEXT, state TEXT);
            CREATE TABLE IF NOT EXISTS signals(signal_id TEXT PRIMARY KEY, created_at TEXT, side TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS evaluations(bar_ts TEXT PRIMARY KEY, evaluated_at TEXT, status TEXT, reason TEXT);
            CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
            CREATE INDEX IF NOT EXISTS ix_pos_status ON positions(status);
            """)
            c.commit()
        finally:
            c.close()

    @contextmanager
    def tx(self):
        with self.lock:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            c.row_factory = sqlite3.Row
            try:
                c.execute("PRAGMA busy_timeout=30000")
                c.execute("BEGIN IMMEDIATE")
                yield c
                c.execute("COMMIT")
            except Exception:
                try:
                    c.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                c.close()

    def get(self, k: str) -> Optional[str]:
        with self.tx() as c:
            r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
            return r["v"] if r else None

    def set(self, k: str, v: Optional[str]):
        with self.tx() as c:
            if v is None:
                c.execute("DELETE FROM kv WHERE k=?", (k,))
            else:
                c.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))

    def save_position(self, p: dict):
        with self.tx() as c:
            c.execute("""INSERT INTO positions(trade_id,signal_id,side,status,trading_day,session,created_at,
                         opened_at,closed_at,qty,fill,init_stop,target,pnl,r_multiple,exit_reason,state)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                         ON CONFLICT(trade_id) DO UPDATE SET status=excluded.status, opened_at=excluded.opened_at,
                         closed_at=excluded.closed_at, qty=excluded.qty, fill=excluded.fill, pnl=excluded.pnl,
                         r_multiple=excluded.r_multiple, exit_reason=excluded.exit_reason, state=excluded.state""",
                      (p["trade_id"], p["signal_id"], p["side"], p["status"], p["trading_day"], p["session"],
                       p["created_at"], p["opened_at"], p["closed_at"], p["qty"], p["fill"], p["init_stop"],
                       p["target"], p["pnl"], p["r_multiple"], p["exit_reason"], json.dumps(p)))

    def insert_signal(self, order: dict) -> bool:
        with self.tx() as c:
            try:
                c.execute("INSERT INTO signals VALUES(?,?,?,?)",
                          (order["signal_id"], order["created"], order["side"], json.dumps(order)))
                return True
            except sqlite3.IntegrityError:
                return False

    def load_trades(self, status: Optional[str] = None) -> List[dict]:
        """Full saved position states in a fixed order (reporting/audit).
        Risk gates and kill switches must keep using closed_trades()."""
        q = "SELECT state FROM positions" + (" WHERE status=?" if status else "") + \
            " ORDER BY created_at, trade_id"
        with self.tx() as c:
            rows = c.execute(q, (status,) if status else ()).fetchall()
        return [json.loads(r["state"]) for r in rows]

    def log_evaluation(self, bar_ts: str, status: str, reason: str):
        with self.tx() as c:
            c.execute("""INSERT INTO evaluations(bar_ts,evaluated_at,status,reason)
                         VALUES(?,?,?,?) ON CONFLICT(bar_ts) DO UPDATE SET
                         evaluated_at=excluded.evaluated_at, status=excluded.status, reason=excluded.reason""",
                      (bar_ts, pd.Timestamp.now(tz=UTC).isoformat(), status, reason))

    log_eval = log_evaluation   # backward-compatible alias used by Engine

    def active_position(self) -> Optional[dict]:
        with self.tx() as c:
            r = c.execute("SELECT state FROM positions WHERE status IN ('PENDING','OPEN') "
                          "ORDER BY created_at DESC LIMIT 1").fetchone()
            return json.loads(r["state"]) if r else None

    def closed_trades(self) -> List[dict]:
        with self.tx() as c:
            rows = c.execute("SELECT * FROM positions WHERE status='CLOSED' ORDER BY closed_at").fetchall()
        return [{"closed_at": ts_utc(r["closed_at"]), "day": trading_day(r["closed_at"]), "pnl": r["pnl"],
                 "r": r["r_multiple"], "side": r["side"], "session": r["session"],
                 "exit_reason": r["exit_reason"]} for r in rows]

    def entries_on(self, day: str) -> int:
        with self.tx() as c:
            return c.execute("SELECT COUNT(*) n FROM positions WHERE trading_day=? AND status!='CANCELLED'",
                             (day,)).fetchone()["n"]

    def funnel(self, limit: int = 2000) -> Dict[str, int]:
        with self.tx() as c:
            rows = c.execute("SELECT reason FROM evaluations ORDER BY bar_ts DESC LIMIT ?", (limit,)).fetchall()
        return dict(Counter(re.sub(r"-?\d+\.\d+", "#", r["reason"]) for r in rows).most_common(15))


# ============================================================
# NOTIFICATIONS (HTML mode: underscores like NEW_YORK no longer break messages)
# ============================================================

class Notifier:
    def __init__(self, cfg: Config):
        self.token, self.chat = cfg.telegram_token, cfg.telegram_chat

    def __call__(self, text: str) -> bool:
        log.info("NOTIFY | %s", text.replace("\n", " | "))
        if not self.token or not self.chat:
            return False
        try:
            r = requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                              json={"chat_id": self.chat, "text": text, "parse_mode": "HTML",
                                    "disable_web_page_preview": True}, timeout=10)
            if r.status_code != 200:
                log.warning("Telegram %s: %s", r.status_code, r.text[:200])
            return r.status_code == 200
        except requests.RequestException as e:
            log.warning("Telegram error: %s", e)
            return False


def fmt_order(o: dict, qty: int, mode: str) -> str:
    e = html.escape
    tp1 = f" | TP1 <code>{o['tp1']:.2f}</code>" if o["tp1"] else ""
    return (f"<b>{e(mode)} GOLD {e(o['side'])} STOP-ENTRY</b> ({e(o['session'])})\n"
            f"Entry <code>{o['entry']:.2f}</code> | SL <code>{o['stop']:.2f}</code>{tp1} | "
            f"Target <code>{o['target']:.2f}</code>\n"
            f"Qty <code>{qty}</code> | Risk/unit <code>{o['risk']:.2f}</code>\n"
            f"Valid until {e(pd.Timestamp(o['expires']).tz_convert(ET).strftime('%H:%M ET'))}")


# ============================================================
# ENGINE (live / paper)
# ============================================================

class Engine:
    def __init__(self, cfg: Config, store: Store, notify: Notifier):
        self.cfg, self.store, self.notify = cfg, store, notify
        self.run_lock = threading.Lock()
        self.events = load_events(cfg.news_file)
        self._cache: Tuple[float, Any] = (0.0, None)

    def features(self):
        t, F = self._cache
        if F is not None and time.time() - t < 45:
            return F
        base = load_csv(self.cfg.data_csv) if self.cfg.data_provider == "csv" else load_yf_5m(self.cfg.symbol)
        if base is None or len(base) < 6000:
            return None
        F = build_features(base, self.cfg)
        self._cache = (time.time(), F)
        return F

    def run_once(self, asof=None) -> Dict[str, Any]:
        if not self.run_lock.acquire(blocking=False):
            return {"status": "BUSY"}
        try:
            return self._run(ts_utc(asof) if asof is not None else pd.Timestamp.now(tz=UTC))
        finally:
            self.run_lock.release()

    def _run(self, asof: pd.Timestamp) -> Dict[str, Any]:
        cfg, st = self.cfg, self.store
        F = self.features()
        if F is None:
            return {"status": "ERROR", "reason": "market data unavailable"}
        F5, F15, F60 = F
        out: Dict[str, Any] = {"asof": asof.isoformat(), "mode": cfg.mode}

        # 1) manage: replay EVERY closed bar since last processed (no skipped / double-counted bars)
        pos = st.active_position()
        if pos:
            bars = closed(F5, 5, asof)
            bars = bars[bars.index > pd.Timestamp(pos["last_bar"])]
            for ts, b in bars.iterrows():
                for ev in step(pos, ts, b.Open, b.High, b.Low, b.Close, cfg):
                    self._announce(pos, ev)
                if pos["status"] in ("CLOSED", "CANCELLED"):
                    break
            st.save_position(pos)
            out["position"] = {k: pos[k] for k in ("trade_id", "side", "status", "fill", "stop",
                                                   "target", "remaining", "pnl", "r_multiple")}
            if pos["status"] in ("PENDING", "OPEN"):
                return {**out, "status": "MANAGING"}

        # 2) kill switches (persist until manual /resume)
        trades = st.closed_trades()
        halted = st.get("halted")
        if not halted:
            halted = halt_check(cfg, trades)
            if halted:
                st.set("halted", halted)
                self.notify(f"<b>GOLD AGENT HALTED</b>\n{html.escape(halted)}")

        # 3) evaluate once per closed 5m bar
        f = closed(F5, 5, asof)
        if f.empty:
            return {**out, "status": "ERROR", "reason": "no closed bars"}
        bar = f.index[-1].isoformat()
        if st.get("last_eval_bar") == bar:
            return {**out, "status": "WAITING_NEXT_BAR", "bar": bar}
        st.set("last_eval_bar", bar)

        order, info = evaluate_entry(F5, F15, F60, asof, cfg, self.events)
        if order is None:
            st.log_eval(bar, info["status"], info["reason"])
            return {**out, **info}

        ok, why, equity = risk_gate(cfg, trades, st.entries_on(trading_day(asof)),
                                    trades[-1]["closed_at"] if trades else None, asof, halted)
        if not ok:
            st.log_eval(bar, "BLOCKED", why)
            return {**out, **info, "status": "BLOCKED", "reason": why}
        qty = size(cfg, equity, order["risk"])
        if qty < 1:
            st.log_eval(bar, "BLOCKED", "qty<1")
            return {**out, "status": "BLOCKED", "reason": "qty < 1 (use MGC or reduce stop distance)"}
        if cfg.mode == "LIVE":
            st.log_eval(bar, "BLOCKED", "LIVE disabled")
            return {**out, "status": "BLOCKED", "reason": "LIVE disabled until a broker adapter exists"}
        if not st.insert_signal(order):
            return {**out, "status": "DUPLICATE_SIGNAL", "signal_id": order["signal_id"]}

        pos = new_position(order, qty, f"{cfg.mode}-{order['signal_id']}")
        st.save_position(pos)
        st.log_eval(bar, "SIGNAL", info["reason"])
        self.notify(fmt_order(order, qty, cfg.mode))
        return {**out, **info, "order": order, "qty": qty, "equity": round(equity, 2)}

    def _announce(self, pos: dict, ev: dict):
        e = html.escape
        msg = f"<b>{e(self.cfg.mode)} GOLD {e(pos['side'])}</b> {e(ev['type'])}"
        if "price" in ev:
            msg += f" @ <code>{ev['price']:.2f}</code>"
        if pos["status"] == "CLOSED":
            msg += f"\nPnL <code>{pos['pnl']:.2f}</code> | R <code>{pos['r_multiple']:+.2f}</code>"
        self.notify(msg)


def scheduler(engine: Engine, every: int):
    while True:
        try:
            res = engine.run_once()
            log.info("run: %s | %s", res.get("status"), res.get("reason", ""))
        except Exception:
            log.exception("scheduled run failed")
        time.sleep(max(10, every - (time.time() % every)))   # align to the minute


# ============================================================
# HTTP API
# ============================================================

def create_app(engine: Engine, cfg: Config):
    from flask import Flask, request, jsonify
    app = Flask(__name__)
    app.url_map.strict_slashes = False          # /webhook and /webhook/ both work
    last: Dict[str, Any] = {"result": None, "at": None}

    def supplied_key() -> str:
        k = request.headers.get("X-Webhook-Key") or request.args.get("key") or ""
        if not k:
            a = request.headers.get("Authorization", "")
            k = a[7:] if a.lower().startswith("bearer ") else ""
        return k

    def auth(fn):
        @wraps(fn)
        def inner(*a, **k):
            ok = (hmac.compare_digest(supplied_key().encode(), cfg.webhook_secret.encode())
                  if cfg.webhook_secret else cfg.allow_insecure)
            return fn(*a, **k) if ok else (jsonify({"status": "UNAUTHORIZED"}), 401)
        return inner

    def _cycle():
        try:
            res = engine.run_once()
        except Exception as e:                  # one bad cycle must never kill the service
            log.exception("webhook cycle failed")
            res = {"status": "ERROR", "reason": str(e)}
        last.update(result=res, at=pd.Timestamp.now(tz=UTC).isoformat())
        log.info("webhook run: %s | %s", res.get("status"), res.get("reason", ""))

    @app.get("/")
    def root():
        return jsonify({"service": "gold-agent", "version": "4.2", "status": "OK",
                        "endpoints": ["/health", "/webhook", "/state", "/metrics", "/halt", "/resume"]})

    @app.get("/health")
    def health():
        return jsonify({"status": "OK", "symbol": cfg.symbol, "mode": cfg.mode,
                        "halted": engine.store.get("halted"), "time": pd.Timestamp.now(tz=UTC).isoformat()})

    @app.route("/webhook", methods=["GET", "POST"])
    @app.route("/run", methods=["GET", "POST"])
    @auth
    def run():
        # ?wait=1 -> synchronous (manual testing). Default: fire-and-forget, instant 200 for cron.
        if request.args.get("wait") == "1":
            _cycle()
            return jsonify(last["result"]), 200
        if engine.run_lock.locked():
            return jsonify({"status": "BUSY", "last_result": last["result"], "last_run_at": last["at"]}), 200
        threading.Thread(target=_cycle, daemon=True).start()
        return jsonify({"status": "ACCEPTED", "last_result": last["result"], "last_run_at": last["at"]}), 200

    @app.get("/state")
    @auth
    def state():
        return jsonify({"active_position": engine.store.active_position(),
                        "halted": engine.store.get("halted"),
                        "last_webhook_result": last["result"], "last_webhook_at": last["at"],
                        "decision_funnel": engine.store.funnel()})

    @app.get("/metrics")
    @auth
    def metrics():
        return jsonify(performance(engine.store.closed_trades(), mc_runs=500))

    @app.route("/halt", methods=["GET", "POST"])
    @auth
    def halt():
        engine.store.set("halted", "manual halt")
        return jsonify({"status": "HALTED"})

    @app.route("/resume", methods=["GET", "POST"])
    @auth
    def resume():
        engine.store.set("halted", None)
        return jsonify({"status": "RESUMED"})

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"status": "NOT_FOUND", "path": request.path,
                        "hint": "use /webhook, /health, /state or /metrics"}), 404

    @app.errorhandler(405)
    def bad_method(_e):
        return jsonify({"status": "METHOD_NOT_ALLOWED", "path": request.path}), 405

    return app


# ------------------------------------------------------------
# SERVICE BOOTSTRAP (shared by `python ... serve` and `gunicorn module:app`)
# ------------------------------------------------------------

_service: Dict[str, Any] = {"app": None}
_service_lock = threading.Lock()


def validate_service_config(cfg: Config):
    if cfg.mode not in ("ALERT", "PAPER", "LIVE"):
        raise SystemExit(f"Invalid EXECUTION_MODE {cfg.mode}")
    if not cfg.webhook_secret and not cfg.allow_insecure:
        raise SystemExit("WEBHOOK_SECRET is required (or set ALLOW_INSECURE=1 for local testing)")
    if cfg.data_provider == "yfinance":
        log.warning("yfinance is a PROTOTYPE feed (delayed, roll-patched). Do not trade real money on it.")
    if cfg.mode == "LIVE":
        log.critical("LIVE requested but blocked: no broker adapter implemented.")
    os.makedirs(os.path.dirname(os.path.abspath(cfg.db_path)), exist_ok=True)


def build_service(cfg: Optional[Config] = None):
    """Build the Flask app and start the scheduler exactly once per process."""
    with _service_lock:
        if _service["app"] is None:
            cfg = cfg or Config()
            validate_service_config(cfg)
            engine = Engine(cfg, Store(cfg.db_path), Notifier(cfg))
            if cfg.run_interval_sec > 0:
                threading.Thread(target=scheduler, args=(engine, cfg.run_interval_sec),
                                 daemon=True, name="scheduler").start()
            _service["app"] = create_app(engine, cfg)
            log.info("Gold agent v4.2 ready | %s | %s | db=%s", cfg.symbol, cfg.mode, cfg.db_path)
        return _service["app"]


def app(environ, start_response):
    """WSGI entry: `gunicorn gold_agent_v4_2:app --workers 1`.
    Built lazily, so importing this module (e.g. for backtests) has no side effects."""
    return build_service()(environ, start_response)


# ============================================================
# BACKTEST (same evaluate_entry / risk_gate / size / step as live)
# ============================================================

def backtest(base5: pd.DataFrame, cfg: Config, events, start=None, end=None,
             out_csv: str = "backtest_trades.csv") -> Dict[str, Any]:
    F5, F15, F60 = build_features(base5, cfg)
    idx = F5.index
    O, H, L, C = (F5[x].to_numpy() for x in ("Open", "High", "Low", "Close"))
    start = ts_utc(start) if start else idx[0]
    end = ts_utc(end) if end else idx[-1] + pd.Timedelta(minutes=5)
    trades: List[dict] = []
    entries: Counter = Counter()
    funnel: Counter = Counter()
    pos = None

    for i in range(len(idx)):
        ts = idx[i]
        asof = ts + pd.Timedelta(minutes=5)
        if pos is not None:
            step(pos, ts, O[i], H[i], L[i], C[i], cfg)
            if pos["status"] == "CLOSED":
                trades.append({"closed_at": ts_utc(pos["closed_at"]), "day": trading_day(pos["closed_at"]),
                               "pnl": pos["pnl"], "r": pos["r_multiple"], "side": pos["side"],
                               "session": pos["session"], "exit_reason": pos["exit_reason"],
                               "opened_at": pos["opened_at"], "fill": pos["fill"], "qty": pos["qty"],
                               **{f"f_{k}": v for k, v in pos["features"].items()}})
                pos = None
            elif pos["status"] == "CANCELLED":
                entries[pos["trading_day"]] -= 1
                funnel["order expired/cancelled"] += 1
                pos = None
            else:
                continue
        if not (start <= asof <= end) or session_of(asof, cfg) is None:
            continue
        order, info = evaluate_entry(F5, F15, F60, asof, cfg, events, check_lag=False)
        if order is None:
            funnel[re.sub(r"-?\d+\.\d+", "#", info["reason"])] += 1
            continue
        day = trading_day(asof)
        ok, why, equity = risk_gate(cfg, trades, entries[day],
                                    trades[-1]["closed_at"] if trades else None, asof, None)
        if not ok:
            funnel["blocked: " + why] += 1
            continue
        qty = size(cfg, equity, order["risk"])
        if qty < 1:
            funnel["blocked: qty<1"] += 1
            continue
        funnel["ORDER PLACED"] += 1
        entries[day] += 1
        pos = new_position(order, qty, "BT-" + order["signal_id"])

    if trades:
        pd.DataFrame(trades).to_csv(out_csv, index=False)
    res = performance(trades)
    res["funnel"] = dict(funnel.most_common(20))
    res["window"] = [start.isoformat(), end.isoformat()]
    return res


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = [os.getenv("APP_COMMAND", "serve")] + argv     # Railway starts with no args -> serve
    ap = argparse.ArgumentParser(description="Gold intraday agent v4.2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve")
    sv.add_argument("--csv", help="serve from a CSV feed instead of yfinance (testing/replay)")
    b = sub.add_parser("backtest")
    b.add_argument("--csv", required=True)
    b.add_argument("--start")
    b.add_argument("--end")
    b.add_argument("--out", default="backtest_trades.csv")
    args = ap.parse_args(argv)

    cfg = Config()
    if getattr(args, "csv", None):
        cfg = dataclasses.replace(cfg, data_csv=args.csv, data_provider="csv")

    if args.cmd == "backtest":
        res = backtest(load_csv(cfg.data_csv), cfg, load_events(cfg.news_file), args.start, args.end, args.out)
        print(json.dumps(res, indent=2, default=str))
        return

    flask_app = build_service(cfg)
    port = int(os.getenv("PORT", "5000"))          # Railway injects PORT; must bind 0.0.0.0
    log.info("Listening on 0.0.0.0:%d", port)
    try:
        from waitress import serve        # production WSGI, single process (SQLite + in-process locks)
        serve(flask_app, host="0.0.0.0", port=port, threads=4)
    except ImportError:
        log.warning("waitress not installed; using Flask dev server")
        flask_app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBKR Real-Time Strategy Runner with ML Upgrades
-----------------------------------------------

Major Features
- Robust connection and auto-reconnect to IBKR via ib_insync
- Historical backfill of 1-minute bars, + real-time 5-second bars -> 1-minute aggregation
- Chan theory K-line combining (fixed & reusable)
- Double-top detection with neckline close-below confirmation
- Correct short accounting (credit at entry, debit at cover), CSV logs
- EOD flatten
- Optional exchange-side Bracket (OCA) orders

New ML Upgrades (3 chosen as best fit):
1) Meta-labeler (Logistic Regression with StandardScaler + probability calibration) to gate entries
2) Regime filter (HMM from hmmlearn) to restrict trading to favorable regimes
3) Volatility forecaster (GARCH via arch; ATR fallback) to set dynamic SL/TP and position size

Graceful Degradation:
- If sklearn/hmmlearn/arch are missing, code auto-falls back to rule-based behavior.

Prereqs:
- pip install ib_insync pandas pytz
- Optional (recommended): pip install scikit-learn joblib hmmlearn arch

Author: ChatGPT (GPT-5 Thinking)
Date: 2025-08-14
"""

from __future__ import annotations

# ============================== Imports ===============================
import os
import sys
import math
import time
import queue
import signal
import random
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import pytz

from enum import Enum

from ib_insync import (
    IB, util, Stock, Contract, BarData, RealTimeBar,
    MarketOrder, LimitOrder, StopOrder, Trade
)

# Optional ML libs (graceful fallback if unavailable)
SKLEARN_OK = True
HMM_OK = True
ARCH_OK = True
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    import joblib
except Exception:
    SKLEARN_OK = False
try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    HMM_OK = False
try:
    from arch import arch_model
except Exception:
    ARCH_OK = False

warnings.filterwarnings("ignore")

# ============================== Config ===============================

# ---- IB TWS/Gateway ----
HOST = "127.0.0.1"
PORT = 7497                 # Paper: 7497, Live: 7496
CLIENT_ID = 101

# ---- Trading Mode ----
LIVE_TRADING = False        # DRY-RUN by default; set True after testing.

# ---- Contract / Universe ----
SYMBOL = "DATS"
EXCHANGE = "SMART"
CURRENCY = "USD"

# ---- Capital & Risk ----
INITIAL_CAPITAL = 3000.0
RISK_PER_TRADE = 0.01       # 1% capital at risk (if risk sizing is enabled)
USE_RISK_SIZING = True      # else use NOTIONAL_PCT_PER_TRADE
NOTIONAL_PCT_PER_TRADE = 0.95

# ---- Strategy (Double Top) ----
PEAK_TOLERANCE = 0.03
BASE_STOP_LOSS_PCT = 0.025  # 2.5% baseline (scaled by vol forecaster if enabled)
BASE_TAKE_PROFIT_PCT = 0.075  # 7.5% baseline (scaled by vol forecaster if enabled)
REQUIRE_NECKLINE_CLOSE_BELOW = True

MAX_CONCURRENT_POSITIONS = 1
COOLDOWN_MINUTES_AFTER_TRADE = 5

# ---- Orders ----
USE_BRACKET_OCA = True      # use exchange-side SL/TP
SLIPPAGE_BPS = 5            # dry-run slippage estimate (bps)

# ---- Session (US/Eastern) ----
LOCAL_TZ = pytz.timezone("US/Eastern")
SESSION_OPEN = (9, 30, 0)
SESSION_CLOSE = (16, 0, 0)

# ---- Data ----
BACKFILL_MINUTES = 390
REFRESH_SECONDS = 1.0
STATUS_INTERVAL_SECONDS = 30.0

# ---- Persistence ----
DATA_DIR = "./ib_data"
MODELS_DIR = "./models"
BARS_CSV = os.path.join(DATA_DIR, f"{SYMBOL}_bars_1m.csv")
TRADES_CSV = os.path.join(DATA_DIR, f"{SYMBOL}_trades.csv")
LOG_CSV = os.path.join(DATA_DIR, "runner_log.csv")
META_MODEL_PATH = os.path.join(MODELS_DIR, f"{SYMBOL}_meta_labeler.pkl")

# ---- ML toggles ----
USE_META_LABELER = True         # logistic regression gate
USE_REGIME_FILTER = True        # HMM regime filter
USE_VOL_FORECASTER = True       # GARCH with ATR fallback
META_THRESHOLD = 0.58           # probability threshold to take trade
VOL_TP_MULT = 2.5               # multiplier on predicted sigma/ATR for TP distance (short)
VOL_SL_MULT = 1.0               # multiplier on predicted sigma/ATR for SL distance (short)
VOL_LOOKBACK = 60               # bars for vol features/ATR fallback

# ---- Debug ----
VERBOSE = True

# ============================== Utilities ===============================

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def now_eastern() -> datetime:
    return datetime.now(LOCAL_TZ)

def today_eastern_date() -> datetime.date:
    return now_eastern().date()

def session_times_for(date_: datetime.date) -> Tuple[datetime, datetime]:
    o = LOCAL_TZ.localize(datetime(date_.year, date_.month, date_.day, *SESSION_OPEN))
    c = LOCAL_TZ.localize(datetime(date_.year, date_.month, date_.day, *SESSION_CLOSE))
    return o, c

def is_rth_open(now_: datetime) -> bool:
    o, c = session_times_for(now_.date())
    return o <= now_ < c

def ts_to_str(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

def safe_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def write_csv_row(path: str, row: Dict[str, Any]):
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)

def _coerce_aware_dt(dt_like: Any) -> datetime:
    if isinstance(dt_like, datetime):
        if dt_like.tzinfo is None:
            return dt_like.replace(tzinfo=timezone.utc)
        return dt_like
    ts = pd.to_datetime(dt_like, utc=True)
    return ts.to_pydatetime()

# ============================== Chan Combine ===============================

class KLINE_DIR(Enum):
    UNKNOWN = 0
    UP = 1
    DOWN = -1
    COMBINE = 2
    INCLUDED = 3

class FX_TYPE(Enum):
    UNKNOWN = 0
    TOP = 1
    BOTTOM = -1

@dataclass
class CCombineItem:
    time_begin: datetime
    time_end: datetime
    high: float
    low: float

    @staticmethod
    def from_any(b: Any) -> "CCombineItem":
        if isinstance(b, BarData):
            dt = _coerce_aware_dt(b.date)
            return CCombineItem(dt, dt, float(b.high), float(b.low))
        elif isinstance(b, dict):
            dt = _coerce_aware_dt(b.get("date"))
            return CCombineItem(dt, dt, float(b["high"]), float(b["low"]))
        elif isinstance(b, pd.Series):
            dt = _coerce_aware_dt(b.get("date"))
            return CCombineItem(dt, dt, float(b["high"]), float(b["low"]))
        else:
            raise TypeError(f"Unsupported bar type: {type(b)}")

class CKLineCombiner:
    def __init__(self, first_bar: Any, direction: KLINE_DIR):
        item = CCombineItem.from_any(first_bar)
        self._time_begin = item.time_begin
        self._time_end = item.time_end
        self._high = item.high
        self._low = item.low
        self._dir = direction if direction in (KLINE_DIR.UP, KLINE_DIR.DOWN) else KLINE_DIR.UNKNOWN
        self._bars: List[Any] = [first_bar]

    @property
    def time_begin(self): return self._time_begin
    @property
    def time_end(self): return self._time_end
    @property
    def high(self): return self._high
    @property
    def low(self): return self._low
    @property
    def dir(self): return self._dir
    @property
    def bars(self): return self._bars

    def _test_combine(self, curr: CCombineItem, new: CCombineItem,
                      exclude_included=False, allow_top_equal=None) -> KLINE_DIR:
        hc, lc = curr.high, curr.low
        hn, ln = new.high, new.low
        if hc >= hn and lc <= ln:
            return KLINE_DIR.COMBINE
        if hc <= hn and lc >= ln:
            return KLINE_DIR.INCLUDED if exclude_included else KLINE_DIR.COMBINE
        if hc > hn and lc > ln:
            return KLINE_DIR.DOWN
        if hc < hn and lc < ln:
            return KLINE_DIR.UP
        raise Exception("Combine type unknown")

    def try_add(self, bar: Any, exclude_included=False, allow_top_equal=None) -> KLINE_DIR:
        curr = CCombineItem(self._time_begin, self._time_end, self._high, self._low)
        new = CCombineItem.from_any(bar)
        decision = self._test_combine(curr, new, exclude_included, allow_top_equal)
        if decision == KLINE_DIR.COMBINE:
            self._bars.append(bar)
            if self._dir == KLINE_DIR.UP:
                self._high = max(self._high, new.high)
                self._low = max(self._low, new.low)
            elif self._dir == KLINE_DIR.DOWN:
                self._high = min(self._high, new.high)
                self._low = min(self._low, new.low)
            else:
                self._high = max(self._high, new.high)
                self._low = min(self._low, new.low)
            self._time_end = new.time_end
        return decision

def combine_bars_chan(df: pd.DataFrame,
                      exclude_included=False, allow_top_equal=None) -> pd.DataFrame:
    assert {'date','open','high','low','close','volume'}.issubset(df.columns)
    combined: List[Dict[str, Any]] = []
    combiner: Optional[CKLineCombiner] = None
    for _, row in df.iterrows():
        bar = row.to_dict()
        if combiner is None:
            init_dir = KLINE_DIR.UP if bar['close'] >= bar['open'] else KLINE_DIR.DOWN
            combiner = CKLineCombiner(bar, init_dir)
            combined.append({
                'date': bar['date'], 'open': bar['open'], 'high': bar['high'],
                'low': bar['low'], 'close': bar['close'], 'volume': bar['volume'],
                'lst': [bar], 'dir': init_dir.name
            })
            continue
        decision = combiner.try_add(bar, exclude_included, allow_top_equal)
        if decision == KLINE_DIR.COMBINE:
            last = combined[-1]
            last['date'] = bar['date']
            last['high'] = max(last['high'], bar['high'])
            last['low'] = min(last['low'], bar['low'])
            last['close'] = bar['close']
            last['volume'] += bar['volume']
            last['lst'].append(bar)
        else:
            new_dir = KLINE_DIR.UP if bar['close'] >= bar['open'] else KLINE_DIR.DOWN
            combiner = CKLineCombiner(bar, new_dir)
            combined.append({
                'date': bar['date'], 'open': bar['open'], 'high': bar['high'],
                'low': bar['low'], 'close': bar['close'], 'volume': bar['volume'],
                'lst': [bar], 'dir': new_dir.name
            })
    return pd.DataFrame(combined)

# ============================== Resampling ===============================

def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    assert 'DateTime' in df.columns
    g = (df.set_index('DateTime')
           .resample(freq)
           .agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}))
    return g.dropna().reset_index()

# ============================== Double-Top Detector ===============================

@dataclass
class DoubleTop:
    peak1_idx: int
    peak2_idx: int
    neckline: float
    break_idx: int
    peak1_time: datetime
    peak2_time: datetime
    break_time: datetime

class DoubleTopDetector:
    def __init__(self, peak_tol: float = PEAK_TOLERANCE):
        self.peak_tol = peak_tol

    def _find_local_peaks(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        peaks: List[Tuple[int, float]] = []
        for i in range(1, len(df)-1):
            if df.loc[i, 'High'] > df.loc[i-1, 'High'] and df.loc[i, 'High'] >= df.loc[i+1, 'High']:
                peaks.append((i, df.loc[i, 'High']))
        return peaks

    def detect(self, df: pd.DataFrame) -> List[DoubleTop]:
        if len(df) < 3: return []
        peaks = self._find_local_peaks(df)
        out: List[DoubleTop] = []
        for i in range(len(peaks)-1):
            idx1, h1 = peaks[i]
            idx2, h2 = peaks[i+1]
            if abs(h2 - h1) / max(1e-9, h1) <= self.peak_tol:
                left, right = idx1, idx2
                local_min = float(df.loc[left:right, 'Low'].min())
                break_idx = idx2 + 1
                if break_idx < len(df):
                    out.append(DoubleTop(
                        peak1_idx=idx1, peak2_idx=idx2, neckline=local_min,
                        break_idx=break_idx,
                        peak1_time=df.loc[idx1, 'DateTime'],
                        peak2_time=df.loc[idx2, 'DateTime'],
                        break_time=df.loc[break_idx, 'DateTime']
                    ))
        out.sort(key=lambda x: x.break_idx)
        return out

# ============================== Real-time Aggregator ===============================

class MinuteBarAggregator:
    """
    Aggregates 5s RealTimeBars into 1-minute OHLCV bars with a precise 'Close'
    (tracks last traded price explicitly).
    """
    def __init__(self):
        self.current_minute: Optional[int] = None
        self.open_price: Optional[float] = None
        self.high_price: float = -math.inf
        self.low_price: float = math.inf
        self.volume: int = 0
        self.last_price: Optional[float] = None
        self.completed: Deque[Dict[str, Any]] = deque(maxlen=10000)

    def on_rtb(self, rtb: RealTimeBar):
        # IB gives naive UTC datetime — coerce to aware then to LOCAL_TZ
        ts = _coerce_aware_dt(rtb.time).astimezone(LOCAL_TZ)
        minute_key = int(ts.timestamp() // 60)
        price = float(rtb.close if rtb.close is not None else (rtb.wap or rtb.high or rtb.low))
        if price is None:
            return

        if self.current_minute is None:
            self._start_minute(minute_key, price, int(rtb.volume or 0), ts)
            return

        if minute_key != self.current_minute:
            # finalize last minute with last_price as Close
            self._finalize_minute()
            self._start_minute(minute_key, price, int(rtb.volume or 0), ts)
            return

        # Update current minute
        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.volume += int(rtb.volume or 0)
        self.last_price = price

    def _start_minute(self, minute_key: int, price: float, vol: int, ts: datetime):
        self.current_minute = minute_key
        self.open_price = price
        self.high_price = price
        self.low_price = price
        self.volume = int(vol or 0)
        self.last_price = price

    def _finalize_minute(self):
        if self.current_minute is None or self.open_price is None or self.last_price is None:
            return
        bar_end_ts = datetime.fromtimestamp(self.current_minute * 60, tz=timezone.utc).astimezone(LOCAL_TZ)
        row = {
            'DateTime': bar_end_ts,
            'Open': float(self.open_price),
            'High': float(self.high_price if self.high_price != -math.inf else self.open_price),
            'Low':  float(self.low_price if self.low_price != math.inf else self.open_price),
            'Close': float(self.last_price),
            'Volume': int(self.volume)
        }
        self.completed.append(row)
        # reset
        self.current_minute = None
        self.open_price = None
        self.high_price = -math.inf
        self.low_price = math.inf
        self.volume = 0
        self.last_price = None

    def drain_completed(self) -> List[Dict[str, Any]]:
        items = list(self.completed)
        self.completed.clear()
        return items

# ============================== Data Feed ===============================

class DataFeed:
    def __init__(self, ib: IB, contract: Contract):
        self.ib = ib
        self.contract = contract
        self.aggregator = MinuteBarAggregator()
        self.rtb_sub = None

    def backfill_1min_today(self) -> pd.DataFrame:
        now_et = now_eastern()
        o, _ = session_times_for(now_et.date())
        minutes_since_open = max(1, int((now_et - o).total_seconds() // 60))
        duration_minutes = min(minutes_since_open, BACKFILL_MINUTES)
        duration_str = f"{duration_minutes} M"
        try:
            bars = self.ib.reqHistoricalData(
                self.contract,
                endDateTime='',
                durationStr=duration_str,
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                keepUpToDate=False
            )
        except Exception as e:
            safe_print(f"[ERROR] backfill failed: {e}")
            return pd.DataFrame(columns=['DateTime','Open','High','Low','Close','Volume'])
        if not bars:
            return pd.DataFrame(columns=['DateTime','Open','High','Low','Close','Volume'])
        df = pd.DataFrame({
            'DateTime': [_coerce_aware_dt(b.date).astimezone(LOCAL_TZ) for b in bars],
            'Open': [b.open for b in bars],
            'High': [b.high for b in bars],
            'Low': [b.low for b in bars],
            'Close': [b.close for b in bars],
            'Volume': [b.volume for b in bars]
        }).sort_values('DateTime').reset_index(drop=True)
        return df

    def start_realtime(self):
        if self.rtb_sub is not None:
            try: self.ib.cancelRealTimeBars(self.contract)
            except Exception: pass
            self.rtb_sub = None
        self.rtb_sub = self.ib.reqRealTimeBars(
            self.contract, whatToShow='TRADES', useRTH=True, realTimeBarsOptions=[]
        )
        self.rtb_sub.updateEvent += self._on_rtb

    def stop_realtime(self):
        if self.rtb_sub is not None:
            try:
                self.rtb_sub.updateEvent -= self._on_rtb
            except Exception:
                pass
            try:
                self.ib.cancelRealTimeBars(self.contract)
            except Exception:
                pass
            self.rtb_sub = None

    def _on_rtb(self, rtb: RealTimeBar):
        try:
            self.aggregator.on_rtb(rtb)
        except Exception as e:
            safe_print(f"[ERROR] Aggregator: {e}")

# ============================== Risk & Orders ===============================

@dataclass
class Position:
    shares: int
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    order_ids: List[int] = field(default_factory=list)

class RiskManager:
    def __init__(self, initial_capital: float):
        self.capital = initial_capital
        self.realized_pnl = 0.0
        self.open_positions: Dict[str, Position] = {}
        self.last_trade_time: Dict[str, datetime] = {}

    def can_open_new(self, symbol: str) -> bool:
        if len(self.open_positions) >= MAX_CONCURRENT_POSITIONS:
            return False
        last = self.last_trade_time.get(symbol)
        if last and (now_eastern() - last) < timedelta(minutes=COOLDOWN_MINUTES_AFTER_TRADE):
            return False
        return True

    def size_by_risk(self, entry: float, stop: float) -> int:
        risk_per_share = max(1e-6, abs(entry - stop))
        budget = self.capital * RISK_PER_TRADE
        return max(0, int(budget // risk_per_share))

    def size_by_notional(self, entry: float) -> int:
        budget = self.capital * NOTIONAL_PCT_PER_TRADE
        return max(0, int(budget // entry))

    def on_short_open(self, symbol: str, shares: int, fill_price: float, stop: float, take: float, order_ids: List[int]):
        # CREDIT proceeds
        self.capital += fill_price * shares
        self.open_positions[symbol] = Position(shares, fill_price, stop, take, now_eastern(), order_ids)
        write_csv_row(LOG_CSV, {
            "time": ts_to_str(now_eastern()), "event": "OPEN_SHORT",
            "symbol": symbol, "shares": shares, "entry": fill_price,
            "stop": stop, "tp": take, "capital": self.capital
        })

    def on_cover(self, symbol: str, cover_price: float) -> float:
        pos = self.open_positions.pop(symbol, None)
        if pos is None: return 0.0
        # DEBIT to buy back
        self.capital -= cover_price * pos.shares
        pnl = (pos.entry_price - cover_price) * pos.shares
        self.realized_pnl += pnl
        self.last_trade_time[symbol] = now_eastern()
        write_csv_row(LOG_CSV, {
            "time": ts_to_str(now_eastern()), "event": "COVER",
            "symbol": symbol, "shares": pos.shares, "exit": cover_price,
            "pnl": pnl, "capital": self.capital
        })
        write_csv_row(TRADES_CSV, {
            "open_time": ts_to_str(pos.opened_at), "close_time": ts_to_str(now_eastern()),
            "symbol": symbol, "side": "SHORT", "shares": pos.shares,
            "entry": pos.entry_price, "exit": cover_price,
            "stop": pos.stop_loss, "tp": pos.take_profit, "pnl": pnl
        })
        return pnl

def make_bracket_orders(parent_action: str, qty: int,
                        limit_price: Optional[float],
                        tp_price: float, sl_price: float,
                        tif="DAY", oca_group: Optional[str]=None):
    parent = MarketOrder(parent_action, qty, tif=tif) if limit_price is None else \
             LimitOrder(parent_action, qty, limit_price, tif=tif)
    parent.transmit = False
    if oca_group:
        parent.ocaGroup = oca_group
        parent.ocaType = 1
    child_action = "BUY" if parent_action.upper() == "SELL" else "SELL"
    tp = LimitOrder(child_action, qty, tp_price, tif=tif); tp.parentId = 0; tp.transmit = False
    sl = StopOrder(child_action, qty, sl_price, tif=tif); sl.parentId = 0; sl.transmit = True
    if oca_group:
        tp.ocaGroup = oca_group; tp.ocaType = 1
        sl.ocaGroup = oca_group; sl.ocaType = 1
    return parent, tp, sl

# ============================== ML: Meta-labeler ===============================

def _rsi(series: pd.Series, window=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_sig
    return macd, macd_sig, macd_hist

def _atr(df: pd.DataFrame, window=14) -> pd.Series:
    hl = (df['High'] - df['Low']).abs()
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()

META_FEATURES = [
    'ret1','ret5','ret15','vol5','vol20','atr14',
    'rsi14','macd','macd_sig','macd_hist',
    'dt_peak_gap_bars','dt_peak_height_diff','dt_break_momentum'
]

def build_meta_features(ohlcv: pd.DataFrame, dtop: DoubleTop) -> pd.DataFrame:
    """
    ohlcv: DataFrame with DateTime, Open, High, Low, Close, Volume
    Returns single-row feature frame at break_idx time.
    """
    df = ohlcv.copy()
    df = df.set_index('DateTime')
    f = pd.DataFrame(index=df.index)
    f['ret1'] = df['Close'].pct_change()
    f['ret5'] = df['Close'].pct_change(5)
    f['ret15'] = df['Close'].pct_change(15)
    f['vol5'] = f['ret1'].rolling(5).std()
    f['vol20'] = f['ret1'].rolling(20).std()
    f['atr14'] = _atr(df.rename(columns={'Open':'Open','High':'High','Low':'Low','Close':'Close'}), 14)
    f['rsi14'] = _rsi(df['Close'], 14)
    macd, macd_sig, macd_hist = _macd(df['Close'])
    f['macd'] = macd; f['macd_sig'] = macd_sig; f['macd_hist'] = macd_hist

    # geometry at break
    b = dtop.break_idx
    p1, p2 = dtop.peak1_idx, dtop.peak2_idx
    if 0 <= p1 < len(df) and 0 <= p2 < len(df) and 0 <= b < len(df):
        f.iloc[b, f.columns.get_loc('dt_peak_gap_bars')] = (p2 - p1) if 'dt_peak_gap_bars' in f.columns else np.nan
    else:
        f['dt_peak_gap_bars'] = np.nan
    # ensure columns exist then set values
    if 'dt_peak_height_diff' not in f.columns: f['dt_peak_height_diff'] = np.nan
    if 'dt_break_momentum' not in f.columns: f['dt_break_momentum'] = np.nan
    # fill geometry values (guard bounds)
    if 0 <= p1 < len(df) and 0 <= p2 < len(df):
        peak_diff = df.iloc[p2]['High'] - df.iloc[p1]['High']
    else:
        peak_diff = np.nan
    if 0 <= b < len(df):
        break_mom = (df.iloc[b]['Close'] - dtop.neckline) / max(1e-9, df.iloc[b]['Close'])
    else:
        break_mom = np.nan
    f.iloc[min(b, len(f)-1), f.columns.get_loc('dt_peak_height_diff')] = peak_diff
    f.iloc[min(b, len(f)-1), f.columns.get_loc('dt_break_momentum')] = break_mom

    # prepare single row
    row = f.iloc[min(b, len(f)-1):min(b, len(f)-1)+1][META_FEATURES].copy()
    row = row.apply(lambda s: s.fillna(0.0))
    row.index.name = 'DateTime'
    return row

class MetaLabeler:
    def __init__(self, model_path: str = META_MODEL_PATH, threshold: float = META_THRESHOLD):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.available = SKLEARN_OK

    def load_or_none(self):
        if not self.available: return None
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                return self.model
        except Exception as e:
            safe_print(f"[META] load failed: {e}")
        return None

    def fit_and_save(self, ohlcv: pd.DataFrame, dtops: List[DoubleTop]):
        """Simple in-session training using triple-barrier labels (short-friendly)."""
        if not self.available:
            safe_print("[META] sklearn not available; skipping training.")
            return None
        # Build dataset over detected events in backfill
        Xs, ys = [], []
        for d in dtops:
            # label via TP/SL horizon consistent with base % (simple, no leakage)
            b = d.break_idx
            if b >= len(ohlcv)-1: continue
            entry = float(ohlcv.iloc[b]['Close'])
            pt = entry * (1 - BASE_TAKE_PROFIT_PCT)  # short profit (down)
            sl = entry * (1 + BASE_STOP_LOSS_PCT)    # short stop (up)
            horizon = min(len(ohlcv)-1, b+30)
            y = 0
            for i in range(b+1, horizon+1):
                lo = float(ohlcv.iloc[i]['Low'])
                hi = float(ohlcv.iloc[i]['High'])
                if lo <= pt: y = 1; break
                if hi >= sl: y = 0; break
            feat_row = build_meta_features(ohlcv, d)
            Xs.append(feat_row.values[0])
            ys.append(y)
        if not Xs:
            safe_print("[META] No training events available; skipping.")
            return None

        X = np.array(Xs); y = np.array(ys)
        base = LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced')
        clf = CalibratedClassifierCV(base, method='isotonic', cv=3)
        pipe = Pipeline([
            ('scale', StandardScaler(with_mean=True, with_std=True)),
            ('clf', clf)
        ])
        pipe.fit(X, y)
        self.model = pipe
        try:
            joblib.dump(pipe, self.model_path)
        except Exception as e:
            safe_print(f"[META] save failed: {e}")
        return pipe

    def predict_ok(self, ohlcv: pd.DataFrame, dtop: DoubleTop) -> bool:
        if not self.available: 
            return True  # no sklearn -> let rule-based run
        if self.model is None:
            self.load_or_none()
            if self.model is None:
                return True  # no model -> allow trade
        feat = build_meta_features(ohlcv, dtop)
        try:
            p = float(self.model.predict_proba(feat.values)[0,1])
            safe_print(f"[META] p_success={p:.3f}")
            return p >= self.threshold
        except Exception as e:
            safe_print(f"[META] inference failed: {e}")
            return True

# ============================== ML: Regime Filter ===============================

class RegimeFilterHMM:
    def __init__(self, n_states=2):
        self.n_states = n_states
        self.available = HMM_OK
        self.hmm = None
        self.state_desc: Dict[int, str] = {}

    def fit(self, df_1m: pd.DataFrame):
        if not self.available or len(df_1m) < 50:
            self.available = False
            return self
        X = self._make_X(df_1m)
        try:
            self.hmm = GaussianHMM(n_components=self.n_states, covariance_type='full', n_iter=200, random_state=42)
            self.hmm.fit(X)
            states = self.hmm.predict(X)
            # Describe states by mean return & vol
            for s in range(self.n_states):
                mask = (states==s)
                if mask.sum() == 0: 
                    self.state_desc[s] = "unknown"
                    continue
                mret = X[mask,0].mean()
                mvol = X[mask,1].mean()
                tag = "bear/highvol" if (mret<0 and mvol>=np.median(X[:,1])) else "bull/lowvol"
                self.state_desc[s] = tag
        except Exception as e:
            safe_print(f"[REGIME] fit failed: {e}")
            self.available = False
        return self

    def allow_short(self, df_1m: pd.DataFrame) -> bool:
        if not self.available or self.hmm is None or len(df_1m) < 50:
            return True  # no filter
        X = self._make_X(df_1m.tail(200))
        try:
            _, post = self.hmm.score_samples(X)
            s = int(post[-1].argmax())
            tag = self.state_desc.get(s, "unknown")
            safe_print(f"[REGIME] state={s} ({tag})")
            return "bear" in tag
        except Exception as e:
            safe_print(f"[REGIME] inference failed: {e}")
            return True

    def _make_X(self, df: pd.DataFrame) -> np.ndarray:
        ret = df['Close'].pct_change().fillna(0.0).values
        vol = pd.Series(ret).rolling(10).std().fillna(0.0).values
        return np.column_stack([ret, vol])

# ============================== ML: Volatility Forecaster ===============================

class VolatilityForecaster:
    """
    Predicts next-bar volatility. If 'arch' unavailable or too little data,
    falls back to ATR/rolling std.
    """
    def __init__(self, lookback=VOL_LOOKBACK):
        self.lookback = lookback
        self.available = ARCH_OK
        self.model = None

    def fit(self, df_1m: pd.DataFrame):
        if len(df_1m) < max(50, self.lookback+10):
            self.available = False
            return self
        if not self.available:
            return self
        try:
            # Fit simple GARCH(1,1) on returns
            r = df_1m['Close'].pct_change().dropna()[-self.lookback*5:]  # more data than lookback if available
            if len(r) < 30:
                self.available = False
                return self
            am = arch_model(100*r, p=1, q=1, mean='zero', vol='GARCH', dist='normal')  # scale to pct*100 to stabilize
            self.model = am.fit(disp='off')
        except Exception as e:
            safe_print(f"[VOL] GARCH fit failed: {e}")
            self.available = False
        return self

    def predict_sigma(self, df_1m: pd.DataFrame) -> float:
        """
        Returns predicted next-bar volatility (as price fraction, e.g., 0.005 = 0.5%).
        """
        if self.available and self.model is not None:
            try:
                # one-step ahead forecast of variance
                f = self.model.forecast(horizon=1, reindex=False)
                var = float(f.variance.values[-1,0])
                sigma_pct = math.sqrt(max(1e-12, var)) / 100.0  # undo scaling
                return sigma_pct
            except Exception as e:
                safe_print(f"[VOL] forecast failed: {e}")
        # Fallback: ATR or rolling std
        tail = df_1m.tail(self.lookback).copy()
        if len(tail) < 5:
            return 0.01  # default 1%
        atr = _atr(tail.rename(columns={'Open':'Open','High':'High','Low':'Low','Close':'Close'}), window=min(14, len(tail)))
        if not atr.empty and tail.iloc[-1]['Close'] != 0:
            sigma = float(atr.iloc[-1] / max(1e-9, tail.iloc[-1]['Close']))
            return max(1e-5, sigma)
        # or rolling std of returns
        ret = tail['Close'].pct_change().std()
        return max(1e-5, float(ret))

# ============================== Strategy ===============================

class DoubleTopShortMLStrategy:
    def __init__(self, ib: IB, contract: Contract, risk: RiskManager):
        self.ib = ib
        self.contract = contract
        self.symbol = contract.symbol
        self.risk = risk
        self.detector = DoubleTopDetector(PEAK_TOLERANCE)
        self.used_break_times: set = set()

        # ML components
        self.meta = MetaLabeler(META_MODEL_PATH, META_THRESHOLD) if USE_META_LABELER else None
        self.regime = RegimeFilterHMM(n_states=2) if USE_REGIME_FILTER else None
        self.vol = VolatilityForecaster(VOL_LOOKBACK) if USE_VOL_FORECASTER else None

    def fit_ml_daily(self, ohlcv: pd.DataFrame):
        # Regime fit
        if self.regime is not None:
            self.regime.fit(ohlcv)
        # Vol fit
        if self.vol is not None:
            self.vol.fit(ohlcv)
        # Meta model: train once if model missing and have enough events
        if self.meta is not None and self.meta.load_or_none() is None:
            dtops = self.detector.detect(ohlcv.rename(columns={'Open':'Open','High':'High','Low':'Low','Close':'Close'}))
            if len(dtops) >= 5:  # require a few events
                self.meta.fit_and_save(ohlcv, dtops)

    def maybe_open_short(self, df_1m: pd.DataFrame) -> Optional[str]:
        if not self.risk.can_open_new(self.symbol):
            return None

        det_df = df_1m.rename(columns={'Open':'Open','High':'High','Low':'Low','Close':'Close'})
        dtops = self.detector.detect(det_df)
        if not dtops: return None

        last_idx = len(det_df) - 1
        last_time = det_df.loc[last_idx, 'DateTime']

        # Candidates whose break is at the last bar
        candidates = [d for d in dtops if d.break_idx == last_idx]
        if not candidates: return None
        d = candidates[0]

        # Neckline confirmation
        if REQUIRE_NECKLINE_CLOSE_BELOW and det_df.loc[last_idx, 'Close'] >= d.neckline:
            return None

        # De-duplication
        if last_time in self.used_break_times:
            return None
        self.used_break_times.add(last_time)

        # Regime gate
        if self.regime is not None and not self.regime.allow_short(df_1m):
            safe_print("[GATE] Regime filter vetoed the trade.")
            return None

        entry_price = float(det_df.loc[last_idx, 'Close'])

        # Vol-based dynamic SL/TP (fallback to base)
        stop_price = entry_price * (1.0 + BASE_STOP_LOSS_PCT)
        take_price = entry_price * (1.0 - BASE_TAKE_PROFIT_PCT)
        if self.vol is not None:
            sigma = self.vol.predict_sigma(df_1m)
            # sigma is fraction of price; compute distances
            sl_dist = VOL_SL_MULT * sigma * entry_price
            tp_dist = VOL_TP_MULT * sigma * entry_price
            if sl_dist > 0 and tp_dist > 0:
                stop_price = entry_price + sl_dist
                take_price = max(0.01, entry_price - tp_dist)
                safe_print(f"[VOL] sigma~{sigma:.4f} -> SL={stop_price:.4f}, TP={take_price:.4f}")

        # Meta-labeler gate
        if self.meta is not None:
            try:
                ok = self.meta.predict_ok(df_1m.copy(), d)
                if not ok:
                    safe_print("[GATE] Meta-labeler vetoed the trade.")
                    return None
            except Exception as e:
                safe_print(f"[META] gate error: {e}")

        # Sizing
        shares = self.risk.size_by_risk(entry_price, stop_price) if USE_RISK_SIZING \
                 else self.risk.size_by_notional(entry_price)
        if shares < 1:
            safe_print("[WARN] Not enough capital for 1 share at risk parameters.")
            return None

        if LIVE_TRADING and USE_BRACKET_OCA:
            # Place bracket orders
            oca_group = f"OCA_{self.symbol}_{int(time.time())}"
            parent, tp, sl = make_bracket_orders(
                parent_action="SELL", qty=shares, limit_price=None,
                tp_price=take_price, sl_price=stop_price, tif="DAY", oca_group=oca_group
            )
            parent_trade: Trade = self.ib.placeOrder(self.contract, parent)

            # Wait for IB to assign orderId to parent
            t0 = time.time()
            parent_id = None
            while time.time() - t0 < 5.0:
                if parent_trade.order is not None and getattr(parent_trade.order, "orderId", 0):
                    parent_id = parent_trade.order.orderId
                    break
                self.ib.sleep(0.1)
            if parent_id is None:
                # fallback best effort
                parent_id = getattr(parent_trade.order, "orderId", 0)

            # Set parentId for children
            tp.parentId = parent_id
            sl.parentId = parent_id

            tp_trade = self.ib.placeOrder(self.contract, tp)
            sl_trade = self.ib.placeOrder(self.contract, sl)

            # Determine entry fill
            fill_price = entry_price
            t1 = time.time()
            while not parent_trade.isDone() and time.time() - t1 < 5.0:
                self.ib.sleep(0.1)
            if parent_trade.fills:
                fill_price = parent_trade.fills[-1].execution.avgPrice or entry_price

            self.risk.on_short_open(self.symbol, shares, fill_price, stop_price, take_price,
                                    order_ids=[parent_id, tp_trade.order.orderId, sl_trade.order.orderId])
            safe_print(f"[OPEN] Short {shares} @ {fill_price:.4f} (TP={take_price:.4f}, SL={stop_price:.4f})")
            return "OPENED"
        else:
            # Dry-run/manual mode
            slip = entry_price * (SLIPPAGE_BPS / 10000.0)
            fill_price = entry_price - slip  # selling short; conservative
            self.risk.on_short_open(self.symbol, shares, fill_price, stop_price, take_price, order_ids=[])
            safe_print(f"[OPEN-DRY] Short {shares} @ {fill_price:.4f} (TP={take_price:.4f}, SL={stop_price:.4f})")
            return "OPENED"

    def maybe_close_short_manual(self, last_price: float) -> Optional[str]:
        """Manual exit logic (for DRY-RUN / non-bracket mode)"""
        pos = self.risk.open_positions.get(self.symbol)
        if pos is None: return None
        if LIVE_TRADING and USE_BRACKET_OCA:
            return None  # exchange-managed
        if last_price >= pos.stop_loss:
            pnl = self.risk.on_cover(self.symbol, last_price)
            safe_print(f"[STOP] Cover @ {last_price:.4f}, PnL={pnl:.2f}, Cap={self.risk.capital:.2f}")
            return "STOP"
        if last_price <= pos.take_profit:
            pnl = self.risk.on_cover(self.symbol, last_price)
            safe_print(f"[TP] Cover @ {last_price:.4f}, PnL={pnl:.2f}, Cap={self.risk.capital:.2f}")
            return "TP"
        return None

    def flatten_eod(self, last_price: float):
        pos = self.risk.open_positions.get(self.symbol)
        if pos is None: return
        if LIVE_TRADING:
            cover = MarketOrder("BUY", pos.shares, tif="DAY")
            trade = self.ib.placeOrder(self.contract, cover)
            while not trade.isDone():
                self.ib.sleep(0.2)
            exit_px = last_price
            if trade.fills:
                exit_px = trade.fills[-1].execution.avgPrice or exit_px
            pnl = self.risk.on_cover(self.symbol, exit_px)
            safe_print(f"[EOD] Cover @ {exit_px:.4f}, PnL={pnl:.2f}, Cap={self.risk.capital:.2f}")
        else:
            pnl = self.risk.on_cover(self.symbol, last_price)
            safe_print(f"[EOD-DRY] Cover @ {last_price:.4f}, PnL={pnl:.2f}, Cap={self.risk.capital:.2f}")

# ============================== Engine ===============================

class StrategyEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ib = IB()
        self.contract = Stock(symbol, EXCHANGE, CURRENCY)
        self.feed = DataFeed(self.ib, self.contract)
        self.risk = RiskManager(INITIAL_CAPITAL)
        self.strategy = DoubleTopShortMLStrategy(self.ib, self.contract, self.risk)
        self.df_1m = pd.DataFrame(columns=['DateTime','Open','High','Low','Close','Volume'])
        self.last_status = time.time()
        self.running = True
        ensure_dirs()
        self.ib.errorEvent += self._on_error

    def _on_error(self, reqId, errorCode, errorString, contract):
        safe_print(f"[IB-ERROR] reqId={reqId} code={errorCode}: {errorString}")

    def safe_connect(self):
        while not self.ib.isConnected():
            try:
                safe_print("[INFO] Connecting to IBKR...")
                self.ib.connect(HOST, PORT, clientId=CLIENT_ID)
                safe_print("[INFO] Connected.")
            except Exception as e:
                safe_print(f"[ERROR] connect failed: {e}")
                time.sleep(1.0)
        try:
            self.ib.qualifyContracts(self.contract)
        except Exception as e:
            safe_print(f"[ERROR] qualifyContracts: {e}")
            raise

    def run(self):
        try:
            self.safe_connect()
            now_et = now_eastern()
            o, c = session_times_for(now_et.date())
            if now_et < o:
                safe_print("[INFO] Before market open. Exiting.")
                return
            if now_et >= c:
                safe_print("[INFO] After market close. Exiting.")
                return

            safe_print("[INFO] Backfilling 1-min bars...")
            self.df_1m = self.feed.backfill_1min_today()
            self._persist_bars()

            # Fit ML components on backfill snapshot
            self.strategy.fit_ml_daily(self.df_1m.copy())

            safe_print("[INFO] Starting real-time bars...")
            self.feed.start_realtime()

            while self.running and is_rth_open(now_eastern()):
                self.ib.waitOnUpdate(timeout=REFRESH_SECONDS)
                new_bars = self.feed.aggregator.drain_completed()
                if new_bars:
                    self._append_new_bars(new_bars)

                if time.time() - self.last_status >= STATUS_INTERVAL_SECONDS:
                    self.last_status = time.time()
                    last_px = float('nan')
                    if len(self.df_1m) > 0:
                        last_px = float(self.df_1m.iloc[-1]['Close'])
                        safe_print(f"[STATUS] {self.symbol} time={ts_to_str(now_eastern())} "
                                   f"bars={len(self.df_1m)} cap={self.risk.capital:.2f} "
                                   f"PnL={self.risk.realized_pnl:.2f} last={last_px:.4f}")

            # EOD flatten
            last_close = float(self.df_1m.iloc[-1]['Close']) if len(self.df_1m) else 0.0
            self.strategy.flatten_eod(last_close)

        except KeyboardInterrupt:
            safe_print("[INFO] Interrupted. Shutting down...")
        except Exception as e:
            safe_print(f"[FATAL] {e}")
            traceback.print_exc()
        finally:
            try: self.feed.stop_realtime()
            except Exception: pass
            if self.ib.isConnected():
                try: self.ib.disconnect()
                except Exception: pass
            safe_print(f"[SUMMARY] Final Cap={self.risk.capital:.2f} PnL={self.risk.realized_pnl:.2f}")

    def _append_new_bars(self, bars: List[Dict[str, Any]]):
        new_df = pd.DataFrame(bars).sort_values('DateTime').reset_index(drop=True)
        # Today-only
        tdate = today_eastern_date()
        new_df = new_df[new_df['DateTime'].dt.date == tdate]
        if new_df.empty: return

        self.df_1m = pd.concat([self.df_1m, new_df], ignore_index=True)
        self.df_1m = self.df_1m.drop_duplicates(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

        # Try open on last bar
        if len(self.df_1m) >= 3:
            opened = self.strategy.maybe_open_short(self.df_1m.copy())
            # Manual exits for non-bracket modes
            last_close = float(self.df_1m.iloc[-1]['Close'])
            self.strategy.maybe_close_short_manual(last_close)

        # Persist bars
        self._persist_bars()

        # Optional snapshots (resample + Chan) for debug
        if len(self.df_1m) >= 10:
            snap_5 = resample_ohlcv(self.df_1m, '5T')
            snap_30 = resample_ohlcv(self.df_1m, '30T')
            # Chan combine of recent window (non-critical)
            tail = self.df_1m.tail(50).copy()
            chan_src = tail.rename(columns={'DateTime':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
            combined = combine_bars_chan(chan_src)
            safe_print(f"[CHAN] last50->{len(combined)} combined segments")

    def _persist_bars(self):
        try:
            if not self.df_1m.empty:
                self.df_1m.to_csv(BARS_CSV, index=False)
        except Exception as e:
            safe_print(f"[WARN] persist bars failed: {e}")

# ============================== CLI ===============================

def main():
    engine = StrategyEngine(SYMBOL)

    def handle_signal(signum, frame):
        safe_print(f"[INFO] Signal {signum} received. Stopping...")
        engine.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    engine.run()

if __name__ == "__main__":
    main()

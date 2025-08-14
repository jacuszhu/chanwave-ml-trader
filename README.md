# chanwave-ml-trader
AI-powered trading bot with Chan Theory, Elliott Waves, and ML models, built for Interactive Brokers API.


---

# chanwave-ml-trader

**AI-assisted double-top short strategy for Interactive Brokers (IBKR)** — real-time 5-sec→1-min aggregation, Chan-theory bar combining, calibrated meta-labeler gate, HMM regime filter, and GARCH/ATR volatility-scaled SL/TP. Exchange-side bracket orders, robust reconnects, CSV logs.

> **Default mode:** DRY-RUN (no live orders). Flip a single flag to go live **after** testing in paper.

---

## Table of Contents

* [Features](#features)
* [Architecture](#architecture)
* [Requirements](#requirements)
* [Quickstart](#quickstart)
* [Configuration](#configuration)
* [Machine Learning Options](#machine-learning-options)
* [Data & Outputs](#data--outputs)
* [Usage Notes (IBKR)](#usage-notes-ibkr)
* [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)
* [Safety & Disclaimer](#safety--disclaimer)
* [License](#license)

---

## Features

* **Real-time engine**: IBKR `reqRealTimeBars` (5s) → clean 1-minute OHLCV aggregator.
* **Backfill**: pulls same-day 1-min bars (RTH) for context.
* **Chan theory combine**: directional K-line combiner for overlapping bars.
* **Double-top detection**: local peaks within tolerance + **neckline close-below** confirmation.
* **Correct short accounting**: cash is **credited** at short entry, **debited** at cover.
* **Risk & orders**:

  * One open position max (configurable), cool-down between trades.
  * **Bracket OCA** (parent SELL + child BUY TP & SL) on-exchange in live mode.
  * EOD auto-flatten.
* **ML add-ons (graceful fallbacks)**:

  1. **Meta-labeler** (Logistic Regression + calibration) gates entries by success probability.
  2. **Regime filter** (HMM) — trade only in “bear/highvol”-like states.
  3. **Volatility forecaster** (GARCH; ATR fallback) scales SL/TP & sizing.
* **Persistence & logs**: bars, trades, and events saved as CSV.

---

## Architecture

```
IB TWS/Gateway
      │
      ├── ib_insync (events)
      │      └── 5s RealTimeBars → 1m aggregator
      │
      ├── Backfill 1m (today)
      │
      ├── Strategy
      │     ├─ DoubleTopDetector (neckline confirmation)
      │     ├─ ML gates: MetaLabeler + Regime(HMM)
      │     ├─ Volatility forecaster (GARCH/ATR) → dynamic SL/TP
      │     └─ Orders: Bracket OCA OR manual (dry-run)
      │
      └── CSV: bars_1m, trades, event log
```

---

## Requirements

**Core**

```bash
pip install ib_insync pandas numpy pytz
```

**Optional ML (recommended)**

```bash
pip install scikit-learn joblib
pip install hmmlearn
pip install arch
```

> If an optional package is missing, the feature auto-disables and the bot still runs.

---

## Quickstart

1. Open IB **TWS** (or **Gateway**) and enable API connections.
2. Clone the repo and install deps:

   ```bash
   git clone https://github.com/jacuszhu/chanwave-ml-trader.git
   cd chanwave-ml-trader
   pip install -r requirements.txt   # if you create one; otherwise install as above
   ```
3. Open the main script and check **config** at the top (host/port, symbol, etc.).
4. Run in **dry-run** (default):

   ```bash
   python strategy_runner.py
   ```
5. Verify logs and CSV outputs. When confident, switch to live:

   * Set `LIVE_TRADING = True` (and ideally keep `USE_BRACKET_OCA = True`).

---

## Configuration

Open the script and review these flags:

* **Connection**: `HOST`, `PORT`, `CLIENT_ID`
* **Mode**: `LIVE_TRADING` (False = dry-run, True = live)
* **Instrument**: `SYMBOL`, `EXCHANGE`, `CURRENCY`
* **Risk**: `INITIAL_CAPITAL`, `RISK_PER_TRADE`, `USE_RISK_SIZING`, `NOTIONAL_PCT_PER_TRADE`
* **Double-top**: `PEAK_TOLERANCE`, `REQUIRE_NECKLINE_CLOSE_BELOW`
* **Stops/Targets (base)**: `BASE_STOP_LOSS_PCT`, `BASE_TAKE_PROFIT_PCT`
* **Orders**: `USE_BRACKET_OCA`, `SLIPPAGE_BPS` (dry-run)
* **Session**: US/Eastern RTH by default
* **Data dirs**: `./ib_data` (CSV), `./models` (saved meta-labeler)

---

## Machine Learning Options

All toggles are at the top of the script.

### 1) Meta-labeler gate

* `USE_META_LABELER = True`
* **Model**: Logistic Regression + StandardScaler + isotonic calibration.
* **Threshold**: `META_THRESHOLD = 0.58` (tune).
* **Model path**: `./models/{SYMBOL}_meta_labeler.pkl`
  If missing, it can train a first pass from backfilled events (requires several double-top events).

### 2) Regime filter (HMM)

* `USE_REGIME_FILTER = True`
* `n_states=2` by default; the filter allows shorts only in states tagged like “bear/highvol”.

### 3) Volatility forecaster

* `USE_VOL_FORECASTER = True`
* If `arch` is available, uses **GARCH(1,1)**; otherwise falls back to **ATR/rolling std**.
* SL/TP scaled by `VOL_SL_MULT`, `VOL_TP_MULT` × predicted σ.

> All ML features **fail gracefully** — if a lib is missing or data is insufficient, the script logs a note and runs rules-only.

---

## Data & Outputs

* `ib_data/{SYMBOL}_bars_1m.csv` – accumulated 1-minute OHLCV
* `ib_data/{SYMBOL}_trades.csv` – open/close records (dry-run or live)
* `ib_data/runner_log.csv` – event log (opens, covers, messages)
* `models/{SYMBOL}_meta_labeler.pkl` – saved meta-labeler (optional)

---

## Usage Notes (IBKR)

* Run **TWS**/**Gateway** first; enable API connections.
* Paper trading port is commonly **7497**, live **7496** (adjust `PORT`).
* Bracket orders (when live) place a **parent SELL** and child **BUY** TP/SL under an OCA group.

---

## Troubleshooting

* **No trades firing**:

  * Check gates in logs (`[REGIME]`, `[META]`, `[GATE] ... vetoed`).
  * Temporarily set `USE_META_LABELER=False` and/or `USE_REGIME_FILTER=False`.
* **Pacing**: Real-time stream is event-driven; backfill runs once at start.
* **Model missing**: The engine proceeds without it; you’ll see a log message.
* **Timezones**: Strategy uses **US/Eastern** RTH for session control; bars are stored with tz info.

---

## Roadmap

* Walk-forward meta training with purged/embargoed CV
* Execution classifier (market vs passive) to reduce slippage
* Multi-symbol supervisor & account reconciliation on restart
* Dashboard/metrics panel

---

## Safety & Disclaimer

This code is for **research and educational** purposes only. Markets carry risk. Test thoroughly in **paper trading** before live use. You are responsible for compliance, risk controls, and operational robustness.

---

## License

**MIT** — see [`LICENSE`](LICENSE).

---

### How to upload this `README.md` to GitHub

* **Yes**: you can **drag & drop** this file via **Add file → Upload files** on your repo page, then **Commit changes**.
* Or from your computer:

  ```bash
  # from the repo folder
  echo "<paste the README content into README.md>" > README.md
  git add README.md
  git commit -m "Add README"
  git push origin main
  ```

Want me to also generate a `requirements.txt` and a clean folder tree you can upload in one shot?

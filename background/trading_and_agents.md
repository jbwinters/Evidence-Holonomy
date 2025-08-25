Below is a **full trading‑bot blueprint in pseudocode** that uses your UEC/Arrow‑of‑Time (AoT) signal as a *regime detector*, plus **direction** and **sizing** modules to turn it into trades. I also describe how the same UEC signal can serve as an **AI agent’s “feeling of time.”** It’s deliberately detailed so your RA can implement/backtest it with your BTC candles (timestamp, open, close, volume).

> ⚠️ Research use only — not financial advice. Treat this as a hypothesis‑testing framework; stress test with transaction costs, slippage, and walk‑forward validation.

---

## 1) Signals you’ll compute

* **UEC irreversibility (regime):** sliding‑window KL‑holonomy on **joint (return × volume) symbols**.
* **Direction (edge):** combine multiple predictors:

  * *Price momentum:* EWM returns, MACD/Kalman trend, rolling Sharpe.
  * *Flow imbalance proxies:* volume z‑score, candle shape (up/down & wick), realized skew.
  * *Machine‑learned classifier:* logistic (or gradient boosted) probability of next bar up `p_up` from features.
* **Risk (sizing):** realized volatility (ATR/EMA vol), drawdown state, turnover constraints.

---

## 2) Feature engineering (BTC, 1‑min or daily)

**Per bar:**

```
r_t      = log(close_t / close_{t-1})            # log return
v_t      = volume_t                               # raw volume
atr_t    = ATR over L bars (use high,low,close if available; else use |r_t| EWMA)
wick_t   = (high_t - low_t) / close_t             # if you have H/L; else omit
body_t   = (close_t - open_t) / open_t
rv_t     = rolling variance of returns
skew_t   = rolling skew of returns
v_z_t    = zscore(volume_t over long window)
```

**Symbolization for UEC (joint bins):**

```
r_bin_t  = quantile_bin(r_t, k_r)                 # e.g., k_r = 12
v_bin_t  = quantile_bin(v_t, k_v)                 # e.g., k_v = 8
sym_t    = r_bin_t * k_v + v_bin_t               # joint symbol in [0 .. k_r*k_v-1]
```

---

## 3) UEC regime score (sliding window)

We compute **KL‑rate holonomy** on a moving window of symbols. Use your tested code: transition‑encode → reverse → decode‑second; then `KL(P||Q)` per step.

```
UEC_PARAMS:
    k_r = 12             # return bins (minute) / 8 (daily)
    k_v = 8              # volume bins (minute) / 6 (daily)
    k   = k_r * k_v
    R   = 3              # KT order
    W   = 256            # window length (minutes) ~ 4.3h; or 64 for faster reaction
    stride = 1           # recompute each new bar

function compute_uec_stream(symbols):
    for t in range(W, len(symbols)):
        seq_win = symbols[t-W+1 ... t]                   # length W
        uec[t]  = KL_rate_time_reversal(seq_win, k, R)   # bits per bar
    # normalize to make thresholds stable
    uec_z = zscore(ewm(uec, span=3*W))                   # fast z-score of a smoothed UEC
    return uec, uec_z
```

**Interpretation**

* `uec_z` high (e.g., > +1.5): market is **irreversible** right now → directional regime / structural stress.
* `uec_z` low or negative: **equilibrium‑like** → mean‑reversion regime.

---

## 4) Direction module (ensemble)

We need a **probability of up move** next bar `p_up_t` and an expected move scale.

```
DIR_PARAMS:
    mom_fast = EMA(r_t, span=12)
    mom_slow = EMA(r_t, span=48)
    macd     = mom_fast - mom_slow
    trend_kf = KalmanTrend(r_t)                      # optional

    features_t = [
        r_{t-1}, r_{t-2}, ..., r_{t-12},
        macd_t, trend_kf_t,
        v_z_t, body_t, wick_t, rv_t, skew_t,
        sign(body_t), 1{close_t > open_t},
        uec_z_{t-1}                                 # allow regime to influence direction
    ]

model_dir = LogisticRegression(or XGBoost)
model_dir.train_past(features, y_next_up)           # walk-forward training

function direction_score(t):
    p_up = model_dir.predict_proba(features_t)      # P(next return > 0)
    exp_move = ATR or EWM(|r|)                      # magnitude proxy for sizing
    return p_up, exp_move
```

**Directional decision:**
`edge_t = p_up - 0.5` → sign of trade if |edge\_t| is large enough.

---

## 5) Sizing module (risk‑aware, Kelly‑scaled)

```
SIZE_PARAMS:
    target_vol_day = 0.02 (2% daily) or per-minute equivalent
    kelly_cap      = 0.5  # half‑Kelly cap
    dd_max         = 0.2  # 20% max drawdown stop
    max_leverage   = 2.0

function position_size(edge, exp_move, realized_vol, uec_z):
    # Convert edge to expected return per bar E[r] (crude):
    mu = edge * exp_move
    var = realized_vol^2 + 1e-8

    f_kelly = clamp(mu/var, -1, 1)                   # classic Kelly
    f_kelly = f_kelly * kelly_cap

    # Volatility targeting:
    vol_scalar = target_vol_per_bar / max(realized_vol, 1e-6)

    # Regime amplification: if uec_z is high, allow larger size (but capped)
    regime_amp = 1 + 0.5 * tanh(uec_z / 2)           # ~1.0 to ~1.5

    size = f_kelly * vol_scalar * regime_amp
    size = clamp(size, -max_leverage, +max_leverage)
    return size
```

---

## 6) State machine: turning the signals into trades

```
THRESHOLDS:
    z_hi   = +1.5     # irreversible regime
    z_lo   =  0.0     # back to neutral or mean-revert
    edge_th = 0.03    # min |p_up - 0.5| to act (tune in backtest)

STOPS/TAKES:
    stop_k = 2.5 * ATR
    take_k = 3.0 * ATR
    time_stop = 4 * W_bars    # give trades time to work in regime

state ∈ {FLAT, TREND_LONG, TREND_SHORT, MEANREV}

on each new bar t:
    update features, uec, uec_z, p_up, exp_move, realized_vol
    size = position_size(edge=p_up-0.5, exp_move, realized_vol, uec_z)

    if state == FLAT:
        if uec_z > z_hi and |p_up-0.5| > edge_th:
            if p_up > 0.5: open LONG with size; state=TREND_LONG
            else:          open SHORT with size; state=TREND_SHORT
        elif uec_z <= z_lo:
            # optional: mean‑reversion scalps around VWAP/EMA, very small size
            state = MEANREV

    if state == TREND_LONG:
        # add or hold only while regime persists and direction intact
        if uec_z < z_lo or p_up < 0.5: reduce/exit to FLAT
        if drawdown_since_entry > stop_k: exit→FLAT
        if profit_since_entry > take_k:   partial take (trail remainder)
        # optional pyramiding: if uec_z rising and pnl>0, add 0.5x units

    if state == TREND_SHORT:
        symmetric to long

    if state == MEANREV:
        # small contrarian bets inside Bollinger bands when uec_z low
        if uec_z > z_hi: close MR and switch to trend logic
        # stop small and quick time_stop
```

**Execution:**
Prefer limit/iceberg around VWAP in calm regimes; in spikes, use marketable limits with slippage model. Avoid trading around exchange outages; block trading when `uec_z` is extreme *and* liquidity dries up (large spread or tiny volume).

---

## 7) Backtest & evaluation

* **Walk‑forward**: train direction model on the past N days, test on next M days; roll.
* **Transaction costs**: include spread, fee, and slippage; stress them up.
* **Metrics**: hit rate, avg trade P\&L, Sharpe, Sortino, max DD, Tail risk (5% VaR), turnover.
* **Ablations**:

  * Without UEC gate vs with UEC gate.
  * Direction = simple momentum vs ML classifier.
  * Joint (r×v) symbols vs returns‑only for UEC.
* **Lead‑lag sanity**: cross‑correlate `uec_z` vs realized volatility and vs trend change indicators; measure average **lead** (bars) for detection.

**Parameter suggestions**

| Frequency | W (bars) | bins (r×v) |  R |  edge\_th | z\_hi |
| --------- | -------: | ---------: | -: | --------: | ----: |
| 1‑minute  |  128–256 |       12×8 |  3 | 0.03–0.05 |  +1.5 |
| Daily     |    30–60 |        8×6 |  3 | 0.04–0.07 |  +1.2 |

---

## 8) Minimal backtest pseudocode (end‑to‑end)

```python
df = load_csv("btc.csv")  # timestamp, open, high?, low?, close, volume
df = df.dropna().sort_values("timestamp")

# === Features & symbols ===
r = log(df.close).diff()
v = df.volume
r_bins = quantile_bins(r, k_r)
v_bins = quantile_bins(v, k_v)
sym = bin_index(r, r_bins) * k_v + bin_index(v, v_bins)

# === UEC stream ===
uec, uec_z = compute_uec_stream(sym)           # from section 3

# === Direction model (walk-forward) ===
for split in rolling_splits(df, train=90d, test=30d):
    trainX, trainY = build_features_labels(split.train)
    testX          = build_features(split.test)
    model_dir.fit(trainX, trainY)
    p_up_test      = model_dir.predict_proba(testX)

    # === Trading loop ===
    pos=0; cash=0; pnl_series=[]
    for t in split.test.index:
        update risk, atr, realized_vol
        size = position_size(p_up_test[t]-0.5, exp_move_t, vol_t, uec_z[t])
        state, orders = state_machine(state, uec_z[t], p_up_test[t], atr_t, ...)
        fills = execute(orders, book/price, slippage_model)
        pnl_series.append(mark_to_market(pos, df.close[t], fills, fees))
    record_metrics(split, pnl_series)

report_summary(all_splits)
```

---

## 9) “Agent’s feeling of time” — how UEC plugs into an AI agent

Think of UEC as a **sensor** that tells an agent how *directional* time currently is in its stream of observations. That unlocks a few powerful behaviors:

1. **Adaptive internal clock**

   * When `UEC` is **low** (reversible): slow the agent’s internal tick; aggregate observations; plan longer‑horizon moves (the world is quiet).
   * When `UEC` is **high**: speed up tick; react faster; store more state (the world is changing in a directed way).

2. **Memory & checkpointing**

   * High `UEC` → **create checkpoints**; tag events as anchors (e.g., regime boundaries).
   * Low `UEC` → **compress** memory; decay older details; keep only summaries (saves compute).

3. **Planning horizon control**

   * Use `UEC` to choose model horizon: in high irreversibility, prefer **short, reactive** policies; in low, prefer **long, deliberative** ones.

4. **Intrinsic reward for exploration**

   * In reinforcement learning, add a curiosity bonus proportional to *change in UEC* (∆UEC). The agent explores states where the time arrow strengthens/weakens, which often coincide with meaningful dynamics and causal structure.

5. **Causal orientation**

   * Many causal discovery methods struggle with symmetry. A **nonzero UEC** is a direct empirical cue that cause→effect asymmetry is present; the agent can bias structure learning towards models that reproduce the measured UEC.

**Agent loop sketch:**

```python
for t in steps:
    o_t = observe()
    s_t = discretize(o_t)                     # symbols
    uec_t = update_uec_window(s_t)

    # time-dilation
    if uec_t > z_hi: dt_agent = fast_tick
    elif uec_t < z_lo: dt_agent = slow_tick

    # memory gating
    if uec_t rising fast:
        checkpoint(memory)
    if uec_t low:
        compress(memory)

    # planner choice
    policy = reactive if uec_t high else deliberative

    # intrinsic reward
    r_int = beta * max(0, uec_t - uec_{t-1})
    a_t = policy.act(o_t, r_int, memory)
    execute(a_t)
```

**Intuition:** UEC gives the agent a *felt* directionality and urgency. It’s a principled way to tie **compressibility/entropy** of experience to **resource allocation** and **planning**.

---

## 10) What to expect (and how to judge success)

* In **backtests**, the UEC‑gated strategy should:

  * Reduce false starts in mean‑revert regimes (by staying flat), and
  * Capture more of the big directional moves (by scaling up when UEC is high).
* **Ablation wins**: The “with UEC gate” version should show higher risk‑adjusted return than the same pipeline without the UEC filter.
* **Lead**: On average, `uec_z` should **lead** realized volatility or trend breakpoints by a fraction of the window length. Quantify via cross‑correlation.

---

If you’d like, I can turn this pseudocode into **actual Python** that plugs directly into your `uec_battery` code (using your KT estimator) and runs a walk‑forward backtest on your BTC CSV.


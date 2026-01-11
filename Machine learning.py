
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

# ----------------------------
# 0. CONFIG
# ----------------------------
START = "2018-01-01"
TICKERS = ["WMT", "SPY"]
ATR_LEN = 14
RSI_LEN = 14
VOL_WIN = 20
SPY_SMA = 50
N_SPLITS = 5         # walk-forward splits on train+validation portion
TRANSACTION_COST_BPS = 0.0  # set e.g., 2.0 for 2 bps per entry/exit
RANDOM_STATE = 42
ANNUALIZATION = 252

# Grid for optimization
C_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]
THRESH_GRID = [0.45, 0.50, 0.55]

# Optional animation
MAKE_ANIMATION = True
ANIMATION_FPS = 30
ANIMATION_FILENAME = "section6_equity.gif"  # requires pillow writer

# ----------------------------
# 1. FETCH & PREP DATA
# ----------------------------
df_raw = yf.download(TICKERS, start=START, progress=False)

# Consolidate to single-level columns for easier reference
# df_raw has multiindex columns: (price_field, ticker)
close = df_raw["Close"]
high = df_raw["High"]["WMT"]
low  = df_raw["Low"]["WMT"]
vol  = df_raw["Volume"]["WMT"]

df = pd.DataFrame({
    "WMT": close["WMT"],
    "SPY": close["SPY"],
    "High": high,
    "Low": low,
    "Volume": vol
}).dropna()

# ----------------------------
# 2. FEATURE ENGINEERING (Leakage-safe, uses only past info)
# ----------------------------
df["WMT_ret"] = df["WMT"].pct_change()
df["SPY_ret"] = df["SPY"].pct_change()

# True Range & ATR (Wilder's smoothing)
prev_close = df["WMT"].shift(1)
true_range = pd.concat([
    (df["High"] - df["Low"]),
    (df["High"] - prev_close).abs(),
    (df["Low"] - prev_close).abs()
], axis=1).max(axis=1)
df["ATR"] = true_range.ewm(alpha=1/ATR_LEN, adjust=False).mean()

# RSI (Wilder’s method)
delta = df["WMT"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/RSI_LEN, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/RSI_LEN, adjust=False).mean()
rs = avg_gain / (avg_loss.replace(0, np.nan))
df["RSI"] = 100 - (100 / (1 + rs))

# Safe z-score helper (avoids division by zero)
def zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    std = std.replace(0, np.nan)
    return (series - mean) / std

# Volume z-score (20-day)
df["Vol_Z"] = zscore(df["Volume"], VOL_WIN)

# SPY distance from 50-day SMA (z-scored)
spy_sma = df["SPY"].rolling(SPY_SMA).mean()
spy_dist = (df["SPY"] - spy_sma) / spy_sma
df["SPY_Dist_Z"] = zscore(spy_dist, VOL_WIN)

# Target: WMT goes up tomorrow
df["Target"] = (df["WMT_ret"].shift(-1) > 0).astype(int)

# Clean NaNs from rolling windows
df = df.dropna()

FEATURES = ["ATR", "RSI", "Vol_Z", "SPY_Dist_Z"]

# ----------------------------
# 3. CREATE SECTIONS (Chronological)
# ----------------------------
n = len(df)
fold = n // 6
trainval_df = df.iloc[: n - fold]   # first 5/6 for optimization
test_df     = df.iloc[n - fold :]   # last 1/6 for final test

X_trainval = trainval_df[FEATURES].copy()
y_trainval = trainval_df["Target"].copy()
ret_trainval = trainval_df["WMT_ret"].copy()

X_test = test_df[FEATURES].copy()
y_test = test_df["Target"].copy()
ret_test = test_df["WMT_ret"].copy()
spy_ret_test = test_df["SPY_ret"].copy()

# ----------------------------
# 4. WALK-FORWARD OPTIMIZATION (C & threshold => max Sharpe)
# ----------------------------
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

def sharpe_ratio(returns):
    daily_mean = np.nanmean(returns)
    daily_std = np.nanstd(returns)
    if daily_std == 0 or np.isnan(daily_std):
        return 0.0
    return (daily_mean * ANNUALIZATION) / (daily_std * np.sqrt(ANNUALIZATION))

def apply_transaction_costs(signal_series, cost_bps):
    """
    Apply a per-trade cost on position changes (entries/exits).
    cost_bps is in basis points (e.g., 2 bps = 0.0002).
    """
    if cost_bps <= 0:
        return signal_series * 0.0
    changes = signal_series.diff().abs().fillna(0)     # entries/exits
    cost_per_change = cost_bps / 1e4                   # convert bps to fraction
    return changes * cost_per_change

best_score = -np.inf
best_params = {"C": None, "threshold": None}

# store CV scores for heatmap
cv_results = []

for C in C_GRID:
    for thr in THRESH_GRID:
        scores = []
        for train_idx, val_idx in tscv.split(X_trainval):
            X_tr = X_trainval.iloc[train_idx]
            y_tr = y_trainval.iloc[train_idx]
            X_va = X_trainval.iloc[val_idx]
            ret_va = ret_trainval.iloc[val_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(
                    C=C, solver="lbfgs",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE
                ))
            ])
            pipe.fit(X_tr, y_tr)

            proba = pipe.predict_proba(X_va)[:, 1]
            signal = pd.Series((proba > thr).astype(int), index=X_va.index, name="signal")

            strat_ret = signal.shift(1).mul(ret_va, fill_value=0)
            costs = apply_transaction_costs(signal.shift(1).fillna(0), TRANSACTION_COST_BPS)
            strat_ret_net = strat_ret - costs

            scores.append(sharpe_ratio(strat_ret_net))

        mean_score = np.mean(scores) if len(scores) > 0 else -np.inf
        cv_results.append({"C": C, "threshold": thr, "cv_sharpe": mean_score})
        if mean_score > best_score:
            best_score = mean_score
            best_params = {"C": C, "threshold": thr}

print(f"Selected params -> C: {best_params['C']}, threshold: {best_params['threshold']:.2f}, CV Sharpe: {best_score:.3f}")

# ----------------------------
# 5. FIT FINAL MODEL & TEST ON SECTION 6
# ----------------------------
final_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        C=best_params["C"],
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])
final_pipe.fit(X_trainval, y_trainval)

proba_test = final_pipe.predict_proba(X_test)[:, 1]
signal_test = pd.Series((proba_test > best_params["threshold"]).astype(int),
                        index=X_test.index, name="signal")

# Strategy returns: use signal(t) to trade at t+1
strategy_returns = signal_test.shift(1).mul(ret_test, fill_value=0)

# Transaction costs on entries/exits
costs_test = apply_transaction_costs(signal_test.shift(1).fillna(0), TRANSACTION_COST_BPS)
strategy_returns_net = strategy_returns - costs_test

# ----------------------------
# 6. PERFORMANCE METRICS
# ----------------------------
def max_drawdown(equity_curve):
    cummax = equity_curve.cummax()
    drawdown = equity_curve / cummax - 1.0
    return drawdown.min()

def beta_to_spy(strat_ret, spy_ret):
    aligned = pd.concat([strat_ret, spy_ret], axis=1).dropna()
    if aligned.iloc[:, 1].var() == 0:
        return np.nan
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var_spy = aligned.iloc[:, 1].var()
    return cov / var_spy

equity_strategy = (1 + strategy_returns_net).cumprod()
equity_spy = (1 + spy_ret_test.fillna(0)).cumprod()

sharpe_test = sharpe_ratio(strategy_returns_net)
vol_annual = np.nanstd(strategy_returns_net) * np.sqrt(ANNUALIZATION)
beta = beta_to_spy(strategy_returns_net, spy_ret_test)
hit_rate = (strategy_returns_net > 0).mean()
days = len(strategy_returns_net)
cagr = equity_strategy.iloc[-1] ** (ANNUALIZATION / days) - 1 if days > 0 else np.nan
mdd = max_drawdown(equity_strategy)

print("\n--- Final Test (Section 6) ---")
print(f"Sharpe (net):       {sharpe_test:.2f}")
print(f"Volatility (ann):   {vol_annual:.2%}")
print(f"Beta to SPY:        {beta:.2f}")
print(f"Hit Rate:           {hit_rate:.2%}")
print(f"CAGR:               {cagr:.2%}")
print(f"Max Drawdown:       {mdd:.2%}")
print(f"Final Value Strategy: {equity_strategy.iloc[-1]:.2f}")
print(f"Final Value S&P 500: {equity_spy.iloc[-1]:.2f}")
print(f"Params used -> C={best_params['C']}, threshold={best_params['threshold']:.2f}, costs={TRANSACTION_COST_BPS} bps\n")

# ----------------------------
# 7. PLOTS
# ----------------------------

## (A) Existing equity curves (static)
plt.figure(figsize=(12, 6))
equity_strategy.plot(label="ML Strategy (Fold 6, net)", color="blue", lw=2)
equity_spy.plot(label="S&P 500 (Fold 6)", color="red", linestyle="--")
plt.title(f"Section 6 Performance | Sharpe: {sharpe_test:.2f} | Beta: {beta:.2f} | MDD: {mdd:.2%}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## (B) Feature importance (which parameters matter)
# Coefficients in standardized space (thanks to the scaler)
coef = final_pipe.named_steps["logreg"].coef_[0]
importance = pd.Series(np.abs(coef), index=FEATURES).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance.index, importance.values, color="teal")
plt.title("Feature Importance (|LogReg Coefficients| in standardized space)")
plt.xlabel("Absolute Coefficient")
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

## (C) Extra plot: WMT with signals + predicted probabilities
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
test_df["WMT"].plot(ax=ax1, color="orange", lw=1.5, label="WMT Close")
# mark days where signal==1 (note: trade occurs next day)
long_days = signal_test[signal_test == 1].index
ax1.scatter(long_days, test_df.loc[long_days, "WMT"], marker="^", color="green", label="Long Signal", zorder=3)
ax1.set_title("WMT Price with Long Signals (Final Section)")
ax1.legend()
ax1.grid(True, alpha=0.3)

pd.Series(proba_test, index=X_test.index).plot(ax=ax2, color="blue", lw=1.2, label="Predicted Prob (Up)")
ax2.axhline(best_params["threshold"], color="orange", linestyle="--", label=f"Threshold={best_params['threshold']:.2f}")
ax2.set_ylabel("Probability")
ax2.set_title("Predicted Probabilities vs Threshold (Final Section)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## (D) ML “convergence”-style plots

# (D1) Regularization path: coefficients vs log10(C)
C_path = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
coef_path = {f: [] for f in FEATURES}
for C_val in C_path:
    pipe_tmp = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=C_val, solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])
    pipe_tmp.fit(X_trainval, y_trainval)
    coefs = pipe_tmp.named_steps["logreg"].coef_[0]
    for f, c in zip(FEATURES, coefs):
        coef_path[f].append(c)

plt.figure(figsize=(10, 6))
for f in FEATURES:
    plt.plot(np.log10(C_path), coef_path[f], marker="o", label=f)
plt.axvline(np.log10(best_params["C"]), color="grey", linestyle="--", label=f"Selected C={best_params['C']}")
plt.title("Regularization Path: Logistic Coefficients vs log10(C)")
plt.xlabel("log10(C)  (left: stronger regularization)")
plt.ylabel("Coefficient (standardized space)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# (D2) Heatmap of CV Sharpe across (C, threshold)
cv_df = pd.DataFrame(cv_results)
pivot = cv_df.pivot(index="C", columns="threshold", values="cv_sharpe").sort_index()
plt.figure(figsize=(8, 6))
im = plt.imshow(pivot.values, cmap="viridis", aspect="auto", origin="lower")
plt.colorbar(im, label="CV Sharpe")
plt.xticks(range(len(pivot.columns)), [f"{c:.2f}" for c in pivot.columns])
plt.yticks(range(len(pivot.index)), [str(r) for r in pivot.index])
plt.title("Walk-forward CV Sharpe Heatmap (C vs Threshold)")
plt.xlabel("Threshold")
plt.ylabel("C")
plt.tight_layout()
plt.show()

# ----------------------------
# 8. OPTIONAL ANIMATION of equity curves
# ----------------------------
if MAKE_ANIMATION:
    # Animate the progressive growth of the equity curves over the test window
    fig_anim, ax_anim = plt.subplots(figsize=(12, 6))
    ax_anim.set_title("Section 6 Equity Curve Animation")
    ax_anim.set_xlabel("Date")
    ax_anim.set_ylabel("Equity")
    ax_anim.grid(True, alpha=0.3)

    x = equity_strategy.index
    y_strat = equity_strategy.values
    y_spy = equity_spy.values

    line_strat, = ax_anim.plot([], [], color="blue", lw=2, label="ML Strategy (net)")
    line_spy,   = ax_anim.plot([], [], color="red", lw=2, linestyle="--", label="S&P 500")
    ax_anim.legend()

    # set limits
    ax_anim.set_xlim(x.min(), x.max())
    y_min = min(np.min(y_strat), np.min(y_spy))
    y_max = max(np.max(y_strat), np.max(y_spy))
    ax_anim.set_ylim(y_min * 0.98, y_max * 1.02)

    def init():
        line_strat.set_data([], [])
        line_spy.set_data([], [])
        return line_strat, line_spy

    def update(frame):
        # draw up to current frame
        idx = frame
        line_strat.set_data(x[:idx], y_strat[:idx])
        line_spy.set_data(x[:idx], y_spy[:idx])
        return line_strat, line_spy

    ani = FuncAnimation(fig_anim, update, frames=len(x), init_func=init,
                        blit=True, interval=1000/ANIMATION_FPS)

    try:
        # Requires Pillow: pip install pillow
        ani.save(ANIMATION_FILENAME, writer="pillow", fps=ANIMATION_FPS)
        print(f"Animation saved to: {ANIMATION_FILENAME}")
    except Exception as e:
        print("Animation save failed. If Pillow is missing, install it via 'pip install pillow'.")
        print("Error:", e)

    plt.close(fig_anim)

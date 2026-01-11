
# Quant Finance — Machine Learning Strategy (WMT vs SPY)

A reproducible Python pipeline that builds a **leakage‑safe** machine‑learning strategy for **Walmart (WMT)** using **SPY** as market context. It fetches data via `yfinance`, engineers technical features (ATR, RSI, volume z‑score, SPY distance from SMA), optimizes a **Logistic Regression** classifier and decision threshold via **walk‑forward cross‑validation** to maximize **Sharpe ratio**, applies optional **transaction costs**, and evaluates on a held‑out final section with comprehensive plots and (optionally) a GIF animation of the equity curves.

---

## Key Features
- **Leakage‑safe feature engineering**: all inputs use only past information.
- **Walk‑forward cross‑validation** (`TimeSeriesSplit`) on the first 5/6 of the data.
- **Model pipeline**: `StandardScaler` + `LogisticRegression (lbfgs, class_weight="balanced")`.
- **Grid search** over regularization `C` and classification **threshold** to maximize CV **Sharpe**.
- **Transaction costs** modeled in **basis points** per entry/exit.
- **Final evaluation** (last 1/6 of data): Sharpe, annualized volatility, beta to SPY, hit rate, CAGR, max drawdown.
- **Visualizations**: equity curves, feature importance, signals vs probability, regularization path, and a CV‑Sharpe heatmap.
- **Optional animation**: saves a GIF of the strategy vs SPY equity evolution.

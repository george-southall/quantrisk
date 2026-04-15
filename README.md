# QuantRisk Analytics Platform

A professional-grade portfolio risk management system built in Python, demonstrating quantitative finance engineering across data infrastructure, risk modelling, factor analysis, and interactive visualisation.

---

## Features

| Module | Capability |
|---|---|
| **Transaction Import** | Trading 212 CSV export parser; multi-file merge with deduplication; automatic LSE ticker resolution (e.g. `VWRP` → `VWRP.L`) |
| **Portfolio Engine** | Cost-basis holdings from real transactions; average-cost P&L; static-weight analytics derived from actual positions |
| **Data Ingestion** | yfinance market data with SQLite caching & retry; FRED macro data (VIX, rates, spreads) |
| **Risk Metrics** | Sharpe, Sortino, Calmar, Beta, Jensen's Alpha, Treynor, Information Ratio, drawdown analysis |
| **VaR & CVaR** | Historical, Parametric (Normal + Student-t), Monte Carlo (Cholesky decomposition) |
| **Stress Testing** | 6 historical crisis scenarios (GFC, COVID, Dot-com, Black Monday, 2022 rate shock, EU debt); custom hypothetical shocks |
| **Factor Models** | Fama-French 3/5-factor OLS regression with ETF-proxy fallback; PCA statistical factor model; performance attribution |
| **Portfolio Optimisation** | Efficient frontier, max Sharpe, min variance, target-return portfolios (scipy SLSQP); Capital Market Line visualisation |
| **Backtesting** | Walk-forward engine with 6 strategies, transaction costs, slippage, and tearsheet evaluation |
| **Regime Detection** | Hidden Markov Model (2- or 3-state) fitted to portfolio returns; per-regime statistics and duration analysis |
| **Options & Greeks** | Black-Scholes pricing with all five Greeks (Δ, Γ, ν, Θ, ρ); P&L surface by spot/vol/time |
| **Dashboard** | 11-page Streamlit app with interactive Plotly charts and CSV/HTML export |

---

## Tech Stack

- **Python 3.11+**
- **pandas / numpy / scipy** — data and quantitative computation
- **statsmodels** — OLS factor regression
- **scikit-learn** — PCA factor model
- **hmmlearn** — Hidden Markov Model regime detection
- **yfinance / fredapi** — market and macro data
- **SQLAlchemy** — SQLite price cache
- **pydantic-settings** — configuration via `.env`
- **Streamlit + Plotly** — interactive dashboard
- **pytest + pytest-cov** — testing (>80% coverage)
- **GitHub Actions** — CI/CD (Python 3.11 & 3.12)

---

## Project Structure

```
quantrisk/
├── ingestion/
│   ├── market_data.py      # yfinance price fetcher with SQLite cache
│   ├── trading212.py       # Trading 212 CSV parser & LSE ticker resolution
│   └── macro_data.py       # FRED macro data fetcher
├── portfolio/
│   ├── portfolio.py        # Static-weight portfolio (analytics engine)
│   ├── transactions.py     # Transaction-based portfolio (cost-basis P&L)
│   ├── returns.py          # Return calculation utilities
│   └── optimizer.py        # Efficient frontier optimiser
├── risk/                   # VaR, CVaR, metrics, drawdown
├── stress_testing/         # Historical scenarios, hypothetical shocks, MC simulation
├── factor_models/          # Fama-French regression, PCA, performance attribution
├── backtesting/            # Walk-forward engine, strategies, tearsheet evaluator
├── regime/                 # Hidden Markov Model regime detection
├── derivatives/            # Black-Scholes pricing and Greeks
└── utils/                  # Logger, Plotly chart builders

dashboard/
├── app.py                  # Entry point (redirects to Portfolio Overview)
├── sidebar.py              # Data source selector and portfolio loader
├── data_source.py          # Bridge: TransactionPortfolio → Portfolio
├── export_utils.py         # CSV and HTML chart download helpers
└── pages/
    ├── 1_Portfolio_Overview.py
    ├── 2_Holdings.py
    ├── 3_Risk_Metrics.py
    ├── 4_VaR_Deep_Dive.py
    ├── 5_Stress_Testing.py
    ├── 6_Factor_Analysis.py
    ├── 7_Backtesting.py
    ├── 8_Portfolio_Optimisation.py
    ├── 9_Live_Risk_Monitor.py
    ├── 10_Regime_Detection.py
    └── 11_Options_Greeks.py

data/
└── demo_transactions.csv   # Synthetic 3-year, 16-asset portfolio (committed)

scripts/
└── generate_demo_data.py   # Regenerate demo data with configurable parameters

tests/                      # pytest test suite
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/george-southall/quantrisk.git
cd quantrisk
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
# Add your FRED API key for macro data (free at https://fred.stlouisfed.org/docs/api/api_key.html)
# Without it, Fama-French factors use ETF proxies automatically
```

### 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

The demo portfolio loads automatically — no setup required. It contains 3 years of realistic
transaction history across 16 assets generated from real historical prices.

### 4. Use your own data

Toggle **"Use demo portfolio"** off in the sidebar and upload a CSV export from Trading 212
(**History → Download CSV** in the app). Multiple files are accepted and duplicates are
removed automatically.

### 5. Run tests

```bash
pytest
```

---

## Dashboard Pages

| # | Page | Description |
|---|---|---|
| 1 | **Portfolio Overview** | Real-money summary (deposited, value, P&L); cumulative returns vs benchmark; weights pie; rolling Sharpe/vol/drawdown |
| 2 | **Holdings** | Current positions with avg cost, unrealised P&L, and weight; portfolio value history with buy/sell markers; full transaction log |
| 3 | **Risk Metrics** | Drawdown chart, return distribution, correlation/covariance heatmaps, rolling pairwise correlation, drawdown event table |
| 4 | **VaR Deep Dive** | Historical/Parametric/MC VaR and CVaR, Monte Carlo wealth path simulation |
| 5 | **Stress Testing** | Historical crisis scenario P&L, custom hypothetical shock builder |
| 6 | **Factor Analysis** | Fama-French 3/5-factor regression, PCA model, performance attribution waterfall |
| 7 | **Backtesting** | Walk-forward strategy comparison, annual/monthly return heatmaps, weight history |
| 8 | **Portfolio Optimisation** | Efficient frontier with CML, max Sharpe / min variance / target-return portfolio weights |
| 9 | **Live Risk Monitor** | Real-time VaR, drawdown, and volatility metrics with breach alerts and VaR term structure |
| 10 | **Regime Detection** | HMM-fitted market regimes (Bull/Bear/Volatile) with statistics, duration histogram, return distributions |
| 11 | **Options & Greeks** | Black-Scholes option pricing, all five Greeks, P&L surface by spot/vol/time, sensitivity table |

---

## Demo Portfolio

The repository ships with `data/demo_transactions.csv` — a synthetic but realistic 3-year
portfolio generated using real historical prices. It is the default data source when you
first open the app.

| Attribute | Value |
|---|---|
| Period | April 2023 – April 2026 |
| Assets | 16 (VWRP, VUAG, QQQ, AAPL, MSFT, NVDA, GOOGL, AMZN, META, JPM, BRK-B, V, COST, UNH, TSLA, PLTR) |
| Transactions | 762 (708 buys, 37 deposits, 17 sells) |
| Total deposited | ~£105,000 |
| Strategy | Monthly DCA with proportional deficit allocation; quarterly rebalancing with profit-taking |

To regenerate with different settings, edit `CONFIG` in `scripts/generate_demo_data.py` and run:

```bash
python scripts/generate_demo_data.py
```

---

## How the Data Pipeline Works

```
Trading 212 CSV (or demo file)
        │
        ▼
quantrisk/ingestion/trading212.py
  load_multiple_csvs() → list[Transaction]
        │
        ▼
quantrisk/portfolio/transactions.py
  TransactionPortfolio
  ├── holdings()          avg-cost positions
  ├── unrealised_pnl()    live P&L per ticker
  ├── value_history()     portfolio value over time
  └── transaction_df()    full trade log
        │
        ▼
dashboard/data_source.py
  tx_portfolio_to_portfolio()
  ├── resolves LSE tickers (VWRP → VWRP.L)
  ├── derives weights from cost basis
  └── sets start_date from first trade
        │
        ▼
quantrisk/portfolio/portfolio.py
  Portfolio.load()        fetches prices, computes returns
        │
        ▼
  All analytics pages (risk metrics, VaR, factor models, etc.)
```

---

## Backtesting Strategies

| Strategy | Description |
|---|---|
| `equal_weight` | 1/N across all assets |
| `inverse_volatility` | Weight proportional to 1/σ |
| `risk_parity` | Equal risk contribution (scipy SLSQP) |
| `minimum_variance` | Minimum portfolio variance (scipy SLSQP) |
| `maximum_sharpe` | Maximum Sharpe ratio (scipy SLSQP) |
| `momentum` | Long top-quintile 12-1 month momentum |

---

## Configuration

Settings are loaded from `.env` (never committed — copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `FRED_API_KEY` | — | FRED API key (optional; without it, FF factors use ETF proxies) |
| `DEFAULT_BENCHMARK` | `SPY` | Default benchmark ticker |
| `MC_NUM_SIMULATIONS` | `10000` | Monte Carlo simulation count |
| `TRANSACTION_COST_BPS` | `10` | One-way transaction cost in basis points |
| `SLIPPAGE_BPS` | `5` | Slippage in basis points |

> **Note:** `DEFAULT_START_DATE` is no longer configurable — the start date is derived
> automatically from the first trade in your transaction history.

---

## Privacy

Personal broker data is never committed to the repository:

```
*.csv, *.xlsx, *.xls   # ignored by .gitignore
uploads/               # ignored
```

The only exception is `data/demo_transactions.csv`, which is synthetic and intentionally public.

---

## CI/CD

GitHub Actions runs on every push (Python 3.11 + 3.12 matrix):

1. `ruff` linting
2. `pytest` with coverage — fails if below 80%

---

## License

MIT

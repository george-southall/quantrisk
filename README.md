# QuantRisk Analytics Platform

A professional-grade portfolio risk management system built in Python, demonstrating quantitative finance engineering across data infrastructure, risk modelling, factor analysis, and interactive visualisation.

---

## Features

| Module | Capability |
|---|---|
| **Data Ingestion** | yfinance market data with SQLite caching & retry; FRED macro data (VIX, rates, spreads) |
| **Portfolio Engine** | Multi-asset portfolio with buy-and-hold or periodic rebalancing, benchmark comparison |
| **Risk Metrics** | Sharpe, Sortino, Calmar, Beta, Jensen's Alpha, Treynor, Information Ratio, drawdown analysis |
| **VaR & CVaR** | Historical, Parametric (Normal + Student-t), Monte Carlo (Cholesky decomposition) |
| **Stress Testing** | 6 historical crisis scenarios (GFC, COVID, Dot-com, Black Monday, 2022 rate shock, EU debt); custom hypothetical shocks |
| **Factor Models** | Fama-French 3/5-factor OLS regression; PCA statistical factor model; performance attribution |
| **Backtesting** | Walk-forward engine with 6 strategies, transaction costs, slippage, and tearsheet evaluation |
| **Dashboard** | 6-page Streamlit app with interactive Plotly charts |

---

## Tech Stack

- **Python 3.11+**
- **pandas / numpy / scipy** — data and quantitative computation
- **statsmodels** — OLS factor regression
- **scikit-learn** — PCA factor model
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
├── ingestion/          # Market data (yfinance) and macro data (FRED)
├── portfolio/          # Portfolio class, return calculations
├── risk/               # VaR, CVaR, metrics, drawdown
├── stress_testing/     # Historical scenarios, hypothetical shocks, MC simulation
├── factor_models/      # Fama-French regression, PCA, performance attribution
├── backtesting/        # Walk-forward engine, strategies, tearsheet evaluator
└── utils/              # Logger, Plotly chart builders

dashboard/
├── app.py              # Portfolio Overview (home page)
├── sidebar.py          # Shared portfolio configuration and caching
└── pages/
    ├── 2_Risk_Metrics.py
    ├── 3_VaR_Deep_Dive.py
    ├── 4_Stress_Testing.py
    ├── 5_Factor_Analysis.py
    └── 6_Backtesting.py

tests/                  # pytest test suite
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

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
```

### 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

### 4. Run tests

```bash
pytest
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Portfolio Overview** | Cumulative returns vs benchmark, weights, rolling Sharpe/vol/drawdown |
| **Risk Metrics** | Drawdown chart, return distribution, correlation heatmap, drawdown event table |
| **VaR Deep Dive** | Historical/Parametric/MC VaR and CVaR, Monte Carlo wealth path simulation |
| **Stress Testing** | Historical crisis scenario P&L, custom hypothetical shock builder |
| **Factor Analysis** | Fama-French 3/5-factor regression, PCA model, performance attribution waterfall |
| **Backtesting** | Walk-forward strategy comparison, annual/monthly return heatmaps, weight history |

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

All settings are in `.env` (never committed):

| Variable | Default | Description |
|---|---|---|
| `FRED_API_KEY` | — | FRED API key (get one free at fred.stlouisfed.org) |
| `DEFAULT_START_DATE` | `2015-01-01` | Default data start date |
| `DEFAULT_BENCHMARK` | `SPY` | Default benchmark ticker |
| `MC_NUM_SIMULATIONS` | `10000` | Monte Carlo simulation count |
| `TRANSACTION_COST_BPS` | `10` | One-way transaction cost in basis points |
| `SLIPPAGE_BPS` | `5` | Slippage in basis points |

---

## CI/CD

GitHub Actions runs on every push (Python 3.11 + 3.12 matrix):

1. `ruff` linting
2. `pytest` with coverage — fails if below 80%

---

## License

MIT

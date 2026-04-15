"""Portfolio Overview — portfolio value, cumulative returns, weights, rolling stats."""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Portfolio Overview | QuantRisk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import yfinance as yf

from dashboard.sidebar import render_sidebar
from quantrisk.ingestion.trading212 import resolve_yf_ticker
from quantrisk.risk.metrics import compute_all_metrics
from quantrisk.utils.plotting import (
    plot_cumulative_returns,
    plot_rolling_stats,
    plot_weights_pie,
)

portfolio = render_sidebar()

st.title("Portfolio Overview")
st.markdown("---")

# ── Portfolio value summary (from transaction data) ────────────────────────────
tx_portfolio = st.session_state.get("tx_portfolio")
if tx_portfolio is not None:
    @st.cache_data(ttl=300, show_spinner=False)
    def _overview_prices(tickers: tuple[str, ...]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for ticker in tickers:
            try:
                hist = yf.download(
                    resolve_yf_ticker(ticker), period="5d",
                    auto_adjust=True, progress=False,
                )
                if not hist.empty:
                    close = hist["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    prices[ticker] = float(close.dropna().iloc[-1])
            except Exception:
                pass
        return prices

    _holdings = tx_portfolio.holdings()
    _prices = _overview_prices(tuple(sorted(_holdings.keys()))) if _holdings else {}

    _deposited = tx_portfolio.total_deposited()
    _value     = tx_portfolio.current_value(_prices)
    _cash      = tx_portfolio.cash_balance()
    _upnl      = sum(tx_portfolio.unrealised_pnl(_prices).values())
    _ret       = tx_portfolio.total_return(_prices)

    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
    vc1.metric("Total Deposited",  f"£{_deposited:,.0f}")
    vc2.metric(
        "Portfolio Value",
        f"£{_value:,.0f}",
        delta=f"£{_value - _deposited:+,.0f}",
    )
    vc3.metric("Cash Balance",     f"£{_cash:,.0f}")
    vc4.metric(
        "Unrealised P&L",
        f"£{_upnl:+,.0f}",
        delta=f"{_upnl / (_deposited or 1):.2%}",
        delta_color="normal",
    )
    vc5.metric("Total Return",     f"{_ret:.2%}", delta_color="normal")
    st.markdown("---")

# ── Key risk metrics ───────────────────────────────────────────────────────────
metrics = compute_all_metrics(portfolio.returns, portfolio.benchmark_returns)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Ann. Return",     f"{metrics['annualised_return']:.2%}")
col2.metric("Ann. Volatility", f"{metrics['annualised_volatility']:.2%}")
col3.metric("Sharpe Ratio",    f"{metrics['sharpe_ratio']:.2f}")
col4.metric("Max Drawdown",    f"{metrics['max_drawdown']:.2%}")
col5.metric("Sortino Ratio",   f"{metrics['sortino_ratio']:.2f}")

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    returns_dict = {portfolio.name: portfolio.returns}
    if portfolio.benchmark_returns is not None:
        returns_dict[portfolio.benchmark] = portfolio.benchmark_returns
    st.plotly_chart(
        plot_cumulative_returns(returns_dict, title="Cumulative Returns vs Benchmark"),
        width='stretch',
    )

with right:
    st.plotly_chart(
        plot_weights_pie(portfolio.weights, title="Portfolio Weights"),
        width='stretch',
    )

# ── Rolling stats ──────────────────────────────────────────────────────────────
rolling = portfolio.rolling_stats(window=252)
st.plotly_chart(
    plot_rolling_stats(rolling, title="Rolling Statistics (252-day window)"),
    width='stretch',
)

# ── Full metrics table ─────────────────────────────────────────────────────────
st.subheader("Full Metrics")

rows = {
    "Annualised Return":     f"{metrics['annualised_return']:.2%}",
    "Annualised Volatility": f"{metrics['annualised_volatility']:.2%}",
    "Sharpe Ratio":          f"{metrics['sharpe_ratio']:.2f}",
    "Sortino Ratio":         f"{metrics['sortino_ratio']:.2f}",
    "Calmar Ratio":          f"{metrics['calmar_ratio']:.2f}",
    "Max Drawdown":          f"{metrics['max_drawdown']:.2%}",
    "Max DD Duration":       f"{metrics['max_drawdown_duration_days']} days",
    "Skewness":              f"{metrics['skewness']:.3f}",
    "Excess Kurtosis":       f"{metrics['kurtosis']:.3f}",
}
if "beta" in metrics:
    rows["Beta"]               = f"{metrics['beta']:.2f}"
    rows["Jensen's Alpha"]     = f"{metrics['alpha']:.2%}"
    rows["Treynor Ratio"]      = f"{metrics['treynor_ratio']:.2f}"
    rows["Information Ratio"]  = f"{metrics['information_ratio']:.2f}"

st.dataframe(
    pd.DataFrame.from_dict(rows, orient="index", columns=["Value"]),
    width='stretch',
)

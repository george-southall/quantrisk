"""Portfolio Overview — cumulative returns, weights, rolling stats, key metrics."""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Portfolio Overview | QuantRisk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.sidebar import render_sidebar
from quantrisk.risk.metrics import compute_all_metrics
from quantrisk.utils.plotting import (
    plot_cumulative_returns,
    plot_rolling_stats,
    plot_weights_pie,
)

portfolio = render_sidebar()

st.title("Portfolio Overview")
st.markdown("---")

# ── Key metrics ────────────────────────────────────────────────────────────────
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
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        plot_weights_pie(portfolio.weights, title="Portfolio Weights"),
        use_container_width=True,
    )

# ── Rolling stats ──────────────────────────────────────────────────────────────
rolling = portfolio.rolling_stats(window=252)
st.plotly_chart(
    plot_rolling_stats(rolling, title="Rolling Statistics (252-day window)"),
    use_container_width=True,
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
    use_container_width=True,
)

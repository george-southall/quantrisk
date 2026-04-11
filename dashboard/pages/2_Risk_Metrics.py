"""Risk Metrics page — drawdown, distribution, correlation, full risk report."""

import streamlit as st

st.set_page_config(page_title="Risk Metrics | QuantRisk", page_icon="📉", layout="wide")

from dashboard.sidebar import render_sidebar
from quantrisk.risk.drawdown import drawdown_table
from quantrisk.risk.metrics import RiskReport
from quantrisk.utils.plotting import (
    plot_correlation_heatmap,
    plot_drawdown,
    plot_return_distribution,
)

portfolio = render_sidebar()

st.title("Risk Metrics")
st.markdown("---")

report = RiskReport(portfolio).compute()
m = report.metrics

# ── Top metrics ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Max Drawdown",   f"{m['max_drawdown']:.2%}")
c2.metric("Max DD Duration", f"{m['max_drawdown_duration_days']} days")
c3.metric("Skewness",       f"{m['skewness']:.3f}")
c4.metric("Excess Kurtosis", f"{m['kurtosis']:.3f}")

st.markdown("---")

# ── Drawdown chart ─────────────────────────────────────────────────────────────
st.plotly_chart(
    plot_drawdown(portfolio.returns, title="Drawdown from Peak"),
    use_container_width=True,
)

# ── Return distribution ────────────────────────────────────────────────────────
st.plotly_chart(
    plot_return_distribution(portfolio.returns, title="Daily Return Distribution"),
    use_container_width=True,
)

# ── Correlation heatmap ────────────────────────────────────────────────────────
st.subheader("Asset Correlation Matrix")
st.plotly_chart(
    plot_correlation_heatmap(portfolio.correlation_matrix, title="Asset Correlations"),
    use_container_width=True,
)

# ── Drawdown event table ───────────────────────────────────────────────────────
st.subheader("Top Drawdown Events")
dd_df = drawdown_table(portfolio.returns, top_n=10)
st.dataframe(dd_df, use_container_width=True)

"""Risk Metrics page — drawdown, distribution, correlation, full risk report."""

import streamlit as st

st.set_page_config(page_title="Risk Metrics | QuantRisk", page_icon="📉", layout="wide")

from dashboard.export_utils import chart_download_button, csv_download_button
from dashboard.sidebar import render_sidebar
from quantrisk.risk.drawdown import drawdown_table
from quantrisk.risk.metrics import RiskReport
from quantrisk.utils.plotting import (
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_drawdown,
    plot_return_distribution,
    plot_rolling_correlation,
)

portfolio = render_sidebar()

st.title("Risk Metrics")
st.markdown("---")

report = RiskReport(portfolio).compute()
m = report.metrics

# ── Top metrics ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Max Drawdown",    f"{m['max_drawdown']:.2%}")
c2.metric("Max DD Duration", f"{m['max_drawdown_duration_days']} days")
c3.metric("Skewness",        f"{m['skewness']:.3f}")
c4.metric("Excess Kurtosis", f"{m['kurtosis']:.3f}")

st.markdown("---")

# ── Drawdown chart ─────────────────────────────────────────────────────────────
dd_fig = plot_drawdown(portfolio.returns, title="Drawdown from Peak")
st.plotly_chart(dd_fig, use_container_width=True)
chart_download_button(dd_fig, "drawdown.html", "Download Drawdown Chart", key="dl_dd_chart")

# ── Return distribution ────────────────────────────────────────────────────────
dist_fig = plot_return_distribution(portfolio.returns, title="Daily Return Distribution")
st.plotly_chart(dist_fig, use_container_width=True)
chart_download_button(dist_fig, "return_distribution.html", "Download Distribution Chart", key="dl_dist_chart")

# ── Correlation / Covariance ───────────────────────────────────────────────────
st.subheader("Asset Relationships")

tab_corr, tab_cov, tab_rolling = st.tabs(
    ["Correlation Matrix", "Covariance Matrix", "Rolling Correlation"]
)

with tab_corr:
    corr_fig = plot_correlation_heatmap(portfolio.correlation_matrix, title="Asset Correlations")
    st.plotly_chart(corr_fig, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        csv_download_button(
            portfolio.correlation_matrix,
            "correlation_matrix.csv",
            "Download Correlation CSV",
            key="dl_corr_csv",
        )
    with col_b:
        chart_download_button(corr_fig, "correlation_heatmap.html", "Download Chart", key="dl_corr_chart")

with tab_cov:
    cov_matrix = portfolio.asset_returns.cov()
    cov_fig = plot_covariance_heatmap(cov_matrix, title="Asset Covariance Matrix (daily, ×100)")
    st.plotly_chart(cov_fig, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        csv_download_button(
            cov_matrix,
            "covariance_matrix.csv",
            "Download Covariance CSV",
            key="dl_cov_csv",
        )
    with col_b:
        chart_download_button(cov_fig, "covariance_heatmap.html", "Download Chart", key="dl_cov_chart")

with tab_rolling:
    tickers = list(portfolio.tickers)
    if len(tickers) >= 2:
        col1, col2, col3 = st.columns(3)
        asset1 = col1.selectbox("Asset 1", tickers, index=0, key="rc_a1")
        asset2 = col2.selectbox("Asset 2", tickers, index=min(1, len(tickers) - 1), key="rc_a2")
        window = col3.select_slider("Window (days)", options=[21, 30, 60, 90, 126, 252], value=60)

        if asset1 == asset2:
            st.info("Select two different assets.")
        else:
            rc_fig = plot_rolling_correlation(
                portfolio.asset_returns, asset1, asset2, window=window
            )
            st.plotly_chart(rc_fig, use_container_width=True)
            chart_download_button(rc_fig, "rolling_correlation.html", "Download Chart", key="dl_rc_chart")
    else:
        st.info("Add at least two assets to the portfolio to view pairwise correlations.")

st.markdown("---")

# ── Drawdown event table ───────────────────────────────────────────────────────
st.subheader("Top Drawdown Events")
dd_df = drawdown_table(portfolio.returns, top_n=10)
st.dataframe(dd_df, use_container_width=True)
csv_download_button(dd_df, "drawdown_events.csv", "Download Drawdown Events CSV", key="dl_dd_csv")

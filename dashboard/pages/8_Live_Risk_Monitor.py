"""Live Risk Monitor — current VaR, drawdown, and threshold breach indicators."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

st.set_page_config(
    page_title="Live Risk Monitor | QuantRisk",
    page_icon="🚨",
    layout="wide",
)

from dashboard.sidebar import render_sidebar
from quantrisk.portfolio.returns import (
    drawdown_series,
    rolling_annualised_volatility,
)
from quantrisk.risk.var import historical_var
from quantrisk.utils.plotting import plot_cumulative_returns, plot_drawdown

portfolio = render_sidebar()

st.title("Live Risk Monitor")

# ── Refresh controls ───────────────────────────────────────────────────────────
col_hdr, col_btn = st.columns([5, 1])
if "live_refreshed_at" not in st.session_state:
    st.session_state["live_refreshed_at"] = datetime.now()

with col_hdr:
    st.caption(f"Last refreshed: {st.session_state['live_refreshed_at'].strftime('%Y-%m-%d %H:%M:%S')}")

with col_btn:
    if st.button("Refresh Now", use_container_width=True):
        st.cache_resource.clear()
        st.session_state["live_refreshed_at"] = datetime.now()
        st.rerun()

st.markdown("---")

# ── Threshold configuration ────────────────────────────────────────────────────
with st.expander("Alert thresholds", expanded=True):
    t1, t2, t3 = st.columns(3)
    var_threshold = t1.number_input(
        "1-Day VaR threshold (%)",
        min_value=0.5, max_value=20.0, value=3.0, step=0.5,
        help="Raise alert when current 1-day historical VaR (95%) exceeds this.",
    ) / 100
    dd_threshold = t2.number_input(
        "Drawdown threshold (%)",
        min_value=1.0, max_value=50.0, value=10.0, step=1.0,
        help="Raise alert when current drawdown from peak exceeds this.",
    ) / 100
    vol_threshold = t3.number_input(
        "30-day Vol threshold (%)",
        min_value=1.0, max_value=100.0, value=20.0, step=1.0,
        help="Raise alert when trailing 30-day annualised vol exceeds this.",
    ) / 100

st.markdown("---")

# ── Compute live metrics ───────────────────────────────────────────────────────
returns = portfolio.returns.dropna()

current_var = historical_var(returns.tail(252), confidence=0.95, horizon=1)
current_dd = abs(float(drawdown_series(returns).iloc[-1]))
rolling_vol = float(rolling_annualised_volatility(returns, window=30).iloc[-1])
latest_return = float(returns.iloc[-1])
latest_date = returns.index[-1].date()

cum_val = float((1 + returns).prod())

# ── Metric cards ──────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Portfolio Value (× $1)",
    f"${cum_val:.4f}",
    delta=f"{latest_return:+.2%} ({latest_date})",
)

var_delta = current_var - var_threshold
m2.metric(
    "1-Day VaR (95%, hist.)",
    f"{current_var:.2%}",
    delta=f"{var_delta:+.2%} vs threshold",
    delta_color="inverse" if var_delta > 0 else "normal",
)

dd_delta = current_dd - dd_threshold
m3.metric(
    "Drawdown from Peak",
    f"{current_dd:.2%}",
    delta=f"{dd_delta:+.2%} vs threshold",
    delta_color="inverse" if dd_delta > 0 else "normal",
)

vol_delta = rolling_vol - vol_threshold
m4.metric(
    "30-Day Rolling Vol",
    f"{rolling_vol:.2%}",
    delta=f"{vol_delta:+.2%} vs threshold",
    delta_color="inverse" if vol_delta > 0 else "normal",
)

st.markdown("---")

# ── Breach summary banner ──────────────────────────────────────────────────────
breaches = []
if current_var > var_threshold:
    breaches.append(f"**VaR** {current_var:.2%} > threshold {var_threshold:.2%}")
if current_dd > dd_threshold:
    breaches.append(f"**Drawdown** {current_dd:.2%} > threshold {dd_threshold:.2%}")
if rolling_vol > vol_threshold:
    breaches.append(f"**Volatility** {rolling_vol:.2%} > threshold {vol_threshold:.2%}")

if breaches:
    st.error("🚨 **Risk Threshold Breached** — " + " · ".join(breaches))
else:
    st.success("✅ All risk metrics within configured thresholds.")

st.markdown("---")

# ── Recent performance charts ─────────────────────────────────────────────────
lookback = st.select_slider(
    "Chart lookback window",
    options=[30, 60, 90, 180, 252, 504],
    value=90,
    format_func=lambda d: f"{d} days",
)

recent_returns = returns.tail(lookback)
recent_dict = {portfolio.name: recent_returns}
if portfolio.benchmark_returns is not None:
    recent_dict[portfolio.benchmark] = portfolio.benchmark_returns.tail(lookback)

st.plotly_chart(
    plot_cumulative_returns(recent_dict, title=f"Cumulative Returns — last {lookback} days"),
    use_container_width=True,
)

st.plotly_chart(
    plot_drawdown(recent_returns, title=f"Drawdown — last {lookback} days"),
    use_container_width=True,
)

# ── VaR term structure ────────────────────────────────────────────────────────
st.subheader("VaR Term Structure (Historical, 95%)")
import pandas as pd  # noqa: E402

horizons = [1, 5, 10, 21]
var_rows = []
for h in horizons:
    v = historical_var(returns.tail(252), confidence=0.95, horizon=h)
    var_rows.append({"Horizon (days)": h, "VaR (%)": f"{v:.2%}", "VaR ($1k)": f"${v * 1000:.2f}"})

st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

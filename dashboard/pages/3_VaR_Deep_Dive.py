"""VaR Deep Dive page — Historical, Parametric, Monte Carlo VaR & CVaR."""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="VaR Deep Dive | QuantRisk", page_icon="⚠️", layout="wide")

from dashboard.sidebar import render_sidebar
from quantrisk.config import settings
from quantrisk.risk.cvar import cvar_summary
from quantrisk.risk.var import (
    historical_var,
    monte_carlo_var,
    parametric_var,
    var_summary,
)
from quantrisk.stress_testing.monte_carlo import mc_var_cvar, simulate_portfolio_paths
from quantrisk.utils.plotting import plot_mc_paths, plot_return_distribution, plot_var_comparison

portfolio = render_sidebar()

st.title("VaR Deep Dive")
st.markdown("---")

# ── Controls ───────────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)
confidence = col_a.selectbox("Confidence level", [0.95, 0.99], index=0)
horizon    = col_b.number_input("Horizon (days)", min_value=1, max_value=252, value=1)
n_sims     = col_c.select_slider(
    "MC simulations", options=[1_000, 5_000, 10_000, 50_000], value=10_000
)

st.markdown("---")

returns = portfolio.returns
weights = portfolio.weights  # dict[str, float] — required by monte_carlo_var and simulate_portfolio_paths

# ── VaR numbers ────────────────────────────────────────────────────────────────
h_var  = historical_var(returns, confidence, horizon)
p_var  = parametric_var(returns, confidence, horizon, distribution="normal")
pt_var = parametric_var(returns, confidence, horizon, distribution="t")

with st.spinner("Running Monte Carlo VaR…"):
    mc_var = monte_carlo_var(
        portfolio.asset_returns,
        weights,
        confidence,
        horizon,
        n_sims,
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Historical VaR",   f"{h_var:.2%}")
c2.metric("Parametric VaR (Normal)", f"{p_var:.2%}")
c3.metric("Parametric VaR (t-dist)", f"{pt_var:.2%}")
c4.metric("Monte Carlo VaR",  f"{mc_var:.2%}")

# ── CVaR ───────────────────────────────────────────────────────────────────────
st.subheader("CVaR / Expected Shortfall")
cvar_df = cvar_summary(returns)
st.dataframe(cvar_df, use_container_width=True)

st.markdown("---")

# ── VaR comparison chart ───────────────────────────────────────────────────────
var_rows = []
for conf in [0.95, 0.99]:
    var_rows.append({"method": "Historical",         "confidence": conf, "var": historical_var(returns, conf, horizon)})
    var_rows.append({"method": "Parametric (Normal)","confidence": conf, "var": parametric_var(returns, conf, horizon)})
    var_rows.append({"method": "Parametric (t)",     "confidence": conf, "var": parametric_var(returns, conf, horizon, "t")})

var_table = pd.DataFrame(var_rows)
st.plotly_chart(
    plot_var_comparison(var_table, title=f"VaR Comparison ({horizon}-day horizon)"),
    use_container_width=True,
)

# ── Return distribution with VaR lines ────────────────────────────────────────
var_lines = {
    f"Hist VaR {confidence:.0%}": h_var,
    f"Param VaR {confidence:.0%}": p_var,
}
st.plotly_chart(
    plot_return_distribution(returns, var_levels=var_lines, title="Return Distribution with VaR"),
    use_container_width=True,
)

# ── Monte Carlo paths ──────────────────────────────────────────────────────────
st.subheader("Monte Carlo Wealth Paths")
mc_horizon = st.slider("Simulation horizon (days)", 21, 252, 126)

with st.spinner("Simulating paths…"):
    paths = simulate_portfolio_paths(
        portfolio.asset_returns,
        weights,
        horizon=mc_horizon,
        n_simulations=n_sims,
    )
    mc_stats = mc_var_cvar(paths, confidence)

st.plotly_chart(
    plot_mc_paths(paths, mc_horizon, confidence, title="Monte Carlo Portfolio Paths"),
    use_container_width=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MC VaR",        f"{mc_stats['var']:.2%}")
c2.metric("MC CVaR",       f"{mc_stats['cvar']:.2%}")
c3.metric("Prob. of Loss", f"{mc_stats['prob_loss']:.2%}")
c4.metric("Median Return", f"{mc_stats['median_return']:.2%}")

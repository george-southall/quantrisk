"""Stress Testing page — historical scenarios and hypothetical shocks."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stress Testing | QuantRisk", page_icon="🔥", layout="wide")

from dashboard.sidebar import render_sidebar
from quantrisk.stress_testing.historical_scenarios import (
    SCENARIOS,
    apply_scenario,
    run_all_scenarios,
)
from quantrisk.stress_testing.hypothetical import apply_hypothetical_shocks
from quantrisk.utils.plotting import plot_scenario_bars

portfolio = render_sidebar()

st.title("Stress Testing")
st.markdown("---")

weights = portfolio.weights

# ── All scenarios summary ──────────────────────────────────────────────────────
st.subheader("Historical Scenario Summary")
summary_df = run_all_scenarios(weights)
st.plotly_chart(
    plot_scenario_bars(summary_df, title="Portfolio P&L Under Historical Stress Scenarios"),
    use_container_width=True,
)

# ── Scenario detail ────────────────────────────────────────────────────────────
st.subheader("Scenario Detail")
scenario_labels = {k: v["name"] for k, v in SCENARIOS.items()}
selected_key = st.selectbox(
    "Select scenario",
    options=list(scenario_labels.keys()),
    format_func=lambda k: scenario_labels[k],
)

result = apply_scenario(weights, selected_key)
scenario_info = SCENARIOS[selected_key]

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Portfolio P&L", f"{result.portfolio_pl:.2%}")
    st.caption(scenario_info["description"])
    if result.issues:
        with st.expander("Assumptions"):
            for issue in result.issues:
                st.caption(f"• {issue}")

with c2:
    detail_df = result.to_dataframe()
    detail_df["weight"]          = detail_df["weight"].map("{:.1%}".format)
    detail_df["shock"]           = detail_df["shock"].map("{:.1%}".format)
    detail_df["pl_contribution"] = detail_df["pl_contribution"].map("{:.2%}".format)
    st.dataframe(detail_df[["ticker", "weight", "shock", "pl_contribution"]], use_container_width=True)

st.markdown("---")

# ── Hypothetical shocks ────────────────────────────────────────────────────────
st.subheader("Hypothetical Shock Analysis")
st.caption("Define custom shocks per ticker (e.g. -0.20 = −20%). Blank = 0%.")

shock_inputs: dict[str, float] = {}
cols = st.columns(min(len(weights), 4))
for i, ticker in enumerate(weights):
    with cols[i % len(cols)]:
        val = st.number_input(
            ticker,
            min_value=-1.0,
            max_value=2.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            key=f"shock_{ticker}",
        )
        if val != 0.0:
            shock_inputs[ticker] = val

if shock_inputs:
    hyp_result = apply_hypothetical_shocks(weights, shock_inputs)
    st.metric("Hypothetical Portfolio P&L", f"{hyp_result.portfolio_pl:.2%}")
    hyp_df = pd.DataFrame([
        {"ticker": t, "weight": f"{weights[t]:.1%}",
         "shock": f"{hyp_result.ticker_shocks[t]:.1%}",
         "p&l": f"{hyp_result.ticker_pl[t]:.2%}"}
        for t in weights
    ])
    st.dataframe(hyp_df, use_container_width=True)
else:
    st.info("Set at least one non-zero shock above to run the analysis.")

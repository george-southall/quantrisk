"""Regime Detection — Hidden Markov Model bull/bear/volatile regime identification."""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Regime Detection | QuantRisk",
    page_icon="📡",
    layout="wide",
)

import plotly.express as px

from dashboard.sidebar import render_sidebar
from quantrisk.config import settings
from quantrisk.regime.hmm import (
    current_regime,
    get_regime_series,
    regime_colour,
    regime_statistics,
)
from quantrisk.utils.plotting import plot_regime_bands

portfolio = render_sidebar()

st.title("Regime Detection")
st.markdown("---")


# ── Configuration ──────────────────────────────────────────────────────────────
with st.expander("Model settings", expanded=True):
    col1, col2, col3 = st.columns(3)
    n_regimes = col1.radio("Number of regimes", [2, 3], horizontal=True,
                           help="2 = Bull / Bear; 3 adds a Volatile state")
    n_iter = col2.slider("HMM iterations", 50, 500, 200, step=50,
                         help="More iterations → better convergence but slower fit")
    rf_rate = col3.number_input(
        "Risk-free rate (%)", 0.0, 20.0,
        value=round(settings.risk_free_rate_fallback * 100, 2),
        step=0.25,
    ) / 100


@st.cache_data(show_spinner="Fitting HMM…")
def _fit_regimes(
    returns_values: tuple,
    returns_index: tuple,
    n_regimes: int,
    n_iter: int,
) -> pd.Series:
    returns = pd.Series(list(returns_values), index=list(returns_index))
    return get_regime_series(returns, n_regimes=n_regimes, n_iter=n_iter)


returns = portfolio.returns.dropna()
regime_series = _fit_regimes(
    returns_values=tuple(returns.values),
    returns_index=tuple(returns.index),
    n_regimes=n_regimes,
    n_iter=n_iter,
)

# ── Current regime badge ───────────────────────────────────────────────────────
cur = current_regime(regime_series)
colour = regime_colour(cur)
_badge_fn = {"Bull": st.success, "Bear": st.error, "Volatile": st.warning}
_badge_fn.get(cur, st.info)(f"**Current Regime: {cur}**  — as of {regime_series.index[-1].date()}")

st.markdown("---")

# ── Regime bands chart ─────────────────────────────────────────────────────────
st.plotly_chart(
    plot_regime_bands(returns, regime_series, title="Portfolio Returns with Regime Bands"),
    use_container_width=True,
)

st.markdown("---")

# ── Regime statistics table ────────────────────────────────────────────────────
st.subheader("Regime Statistics")

stats_df = regime_statistics(returns, regime_series, risk_free_rate=rf_rate)

display_df = stats_df.copy()
display_df["mean_return"] = display_df["mean_return"].map("{:.2%}".format)
display_df["volatility"] = display_df["volatility"].map("{:.2%}".format)
display_df["sharpe_ratio"] = display_df["sharpe_ratio"].map("{:.2f}".format)
display_df["avg_duration_days"] = display_df["avg_duration_days"].map("{:.1f}".format)
display_df["pct_time"] = display_df["pct_time"].map("{:.1%}".format)
display_df["is_current"] = display_df["is_current"].map(lambda x: "◀ current" if x else "")
display_df.columns = [
    "Ann. Return", "Ann. Vol", "Sharpe", "Avg Duration (days)", "% Time", ""
]

st.dataframe(display_df, use_container_width=True)

st.markdown("---")

# ── Regime duration histogram ──────────────────────────────────────────────────
st.subheader("Regime Duration Distribution")

# Compute per-span durations
regime_changes = (regime_series != regime_series.shift()).cumsum()
durations = []
for span_id, span in regime_series.groupby(regime_changes):
    durations.append({"Regime": span.iloc[0], "Duration (days)": len(span)})

dur_df = pd.DataFrame(durations)
_regime_colours = {"Bull": "#00CC96", "Bear": "#EF553B", "Volatile": "#FFA15A"}

fig_hist = px.histogram(
    dur_df,
    x="Duration (days)",
    color="Regime",
    color_discrete_map=_regime_colours,
    barmode="overlay",
    opacity=0.75,
    nbins=30,
    title="Distribution of Regime Durations",
)
fig_hist.update_layout(
    xaxis_title="Duration (trading days)",
    yaxis_title="Count",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=380,
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── Return distribution by regime ─────────────────────────────────────────────
st.subheader("Daily Return Distribution by Regime")

aligned = pd.concat([returns.rename("return"), regime_series], axis=1).dropna()

fig_dist = px.histogram(
    aligned,
    x="return",
    color="regime",
    color_discrete_map=_regime_colours,
    barmode="overlay",
    opacity=0.70,
    nbins=60,
    title="Daily Return Distribution by Regime",
)
fig_dist.update_layout(
    xaxis=dict(title="Daily Return", tickformat=".1%"),
    yaxis_title="Count",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=380,
)
st.plotly_chart(fig_dist, use_container_width=True)

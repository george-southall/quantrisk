"""Portfolio Optimisation — efficient frontier, max Sharpe, min variance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Portfolio Optimisation | QuantRisk",
    page_icon="🎯",
    layout="wide",
)

from dashboard.sidebar import render_sidebar
from quantrisk.config import settings
from quantrisk.portfolio.optimizer import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    target_return_portfolio,
)
from quantrisk.risk.metrics import compute_all_metrics

portfolio = render_sidebar()

st.title("Portfolio Optimisation")
st.markdown("---")


# ── Configuration ──────────────────────────────────────────────────────────────
with st.expander("Optimisation settings", expanded=True):
    col1, col2, col3 = st.columns(3)
    rf_pct = col1.number_input(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=round(settings.risk_free_rate_fallback * 100, 2),
        step=0.25,
        help="Annualised risk-free rate used for Sharpe ratio calculations.",
    )
    rf_rate = rf_pct / 100.0

    n_frontier = col2.slider(
        "Frontier resolution",
        min_value=30,
        max_value=150,
        value=80,
        help="Number of portfolios computed along the efficient frontier.",
    )
    max_wt_pct = col3.slider(
        "Max weight per asset (%)",
        min_value=10,
        max_value=100,
        value=100,
        step=5,
        help="Upper bound on any single asset's weight. Use to limit concentration.",
    )
    max_weight = max_wt_pct / 100.0

if st.button("Run Optimisation", type="primary", use_container_width=True):
    asset_returns = portfolio.asset_returns.dropna()

    with st.spinner("Computing efficient frontier…"):
        try:
            frontier_df = efficient_frontier(
                asset_returns,
                n_points=n_frontier,
                risk_free_rate=rf_rate,
                max_weight=max_weight,
            )
            mv_port = min_variance_portfolio(asset_returns, max_weight=max_weight)
            ms_port = max_sharpe_portfolio(
                asset_returns,
                risk_free_rate=rf_rate,
                max_weight=max_weight,
            )
        except Exception as exc:
            st.error(f"Optimisation failed: {exc}")
            st.stop()

    st.session_state.update({
        "opt_frontier": frontier_df,
        "opt_mv": mv_port,
        "opt_ms": ms_port,
        "opt_rf": rf_rate,
        "opt_asset_returns": asset_returns,
    })

if "opt_frontier" not in st.session_state:
    st.info("Configure settings above and click **Run Optimisation** to begin.")
    st.stop()

# ── Retrieve session state ─────────────────────────────────────────────────────
frontier_df: pd.DataFrame = st.session_state["opt_frontier"]
mv_port: dict = st.session_state["opt_mv"]
ms_port: dict = st.session_state["opt_ms"]
rf_rate_stored: float = st.session_state["opt_rf"]
asset_returns: pd.DataFrame = st.session_state["opt_asset_returns"]

# Current portfolio stats
curr_metrics = compute_all_metrics(portfolio.returns, portfolio.benchmark_returns)
curr_vol = curr_metrics["annualised_volatility"]
curr_ret = curr_metrics["annualised_return"]
curr_sharpe = curr_metrics["sharpe_ratio"]

st.markdown("---")


# ── Efficient Frontier Chart ───────────────────────────────────────────────────
def _build_frontier_chart(
    frontier: pd.DataFrame,
    mv: dict,
    ms: dict,
    curr_v: float,
    curr_r: float,
    rf: float,
    target_port: dict | None = None,
) -> go.Figure:
    fig = go.Figure()

    # ── Efficient frontier (coloured by Sharpe ratio) ──────────────────────────
    fig.add_trace(go.Scatter(
        x=frontier["annualised_volatility"],
        y=frontier["annualised_return"],
        mode="lines+markers",
        name="Efficient Frontier",
        marker=dict(
            size=5,
            color=frontier["sharpe_ratio"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(
                title="Sharpe",
                thickness=12,
                len=0.6,
                x=1.02,
            ),
        ),
        line=dict(color="rgba(100,100,200,0.4)", width=1),
        hovertemplate=(
            "Vol: %{x:.1%}<br>"
            "Return: %{y:.1%}<br>"
            "Sharpe: %{marker.color:.2f}"
            "<extra>Frontier</extra>"
        ),
    ))

    # ── Capital Market Line ────────────────────────────────────────────────────
    cml_slope = ms["sharpe_ratio"]  # = (ms_ret - rf) / ms_vol
    max_cml_vol = frontier["annualised_volatility"].max() * 1.15
    cml_vols = np.array([0.0, max_cml_vol])
    cml_rets = rf + cml_slope * cml_vols
    fig.add_trace(go.Scatter(
        x=cml_vols,
        y=cml_rets,
        mode="lines",
        name="Capital Market Line",
        line=dict(color="rgba(120,120,120,0.7)", width=1.5, dash="dash"),
        hoverinfo="skip",
    ))

    # ── Current portfolio ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[curr_v],
        y=[curr_r],
        mode="markers",
        name="Current Portfolio",
        marker=dict(symbol="circle", size=14, color="#2196F3", line=dict(color="white", width=2)),
        hovertemplate=(
            f"<b>Current Portfolio</b><br>"
            f"Vol: {curr_v:.1%}<br>"
            f"Return: {curr_r:.1%}<br>"
            f"Sharpe: {(curr_r - rf) / curr_v if curr_v > 0 else float('nan'):.2f}"
            "<extra></extra>"
        ),
    ))

    # ── Min Variance ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[mv["annualised_volatility"]],
        y=[mv["annualised_return"]],
        mode="markers",
        name="Min Variance",
        marker=dict(symbol="diamond", size=14, color="#9C27B0", line=dict(color="white", width=2)),
        hovertemplate=(
            f"<b>Min Variance</b><br>"
            f"Vol: {mv['annualised_volatility']:.1%}<br>"
            f"Return: {mv['annualised_return']:.1%}<br>"
            f"Sharpe: {mv['sharpe_ratio']:.2f}"
            "<extra></extra>"
        ),
    ))

    # ── Max Sharpe ────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[ms["annualised_volatility"]],
        y=[ms["annualised_return"]],
        mode="markers",
        name="Max Sharpe",
        marker=dict(symbol="star", size=18, color="#FF9800", line=dict(color="white", width=1.5)),
        hovertemplate=(
            f"<b>Max Sharpe</b><br>"
            f"Vol: {ms['annualised_volatility']:.1%}<br>"
            f"Return: {ms['annualised_return']:.1%}<br>"
            f"Sharpe: {ms['sharpe_ratio']:.2f}"
            "<extra></extra>"
        ),
    ))

    # ── Risk-free rate annotation ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0.0],
        y=[rf],
        mode="markers",
        name=f"Risk-free ({rf:.1%})",
        marker=dict(symbol="circle-open", size=10, color="grey"),
        hoverinfo="skip",
    ))

    # ── Custom target portfolio (optional) ────────────────────────────────────
    if target_port is not None:
        fig.add_trace(go.Scatter(
            x=[target_port["annualised_volatility"]],
            y=[target_port["annualised_return"]],
            mode="markers",
            name="Target Portfolio",
            marker=dict(
                symbol="triangle-up",
                size=15,
                color="#4CAF50",
                line=dict(color="white", width=2),
            ),
            hovertemplate=(
                f"<b>Target Portfolio</b><br>"
                f"Vol: {target_port['annualised_volatility']:.1%}<br>"
                f"Return: {target_port['annualised_return']:.1%}<br>"
                f"Sharpe: {target_port['sharpe_ratio']:.2f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Efficient Frontier",
        xaxis=dict(
            title="Annualised Volatility",
            tickformat=".0%",
            zeroline=False,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            title="Annualised Return",
            tickformat=".0%",
            zeroline=False,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(t=60, b=50, l=60, r=80),
    )
    return fig


# ── Key metrics summary ────────────────────────────────────────────────────────
st.subheader("Portfolio Comparison")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Current Portfolio**")
    st.metric("Return", f"{curr_ret:.2%}")
    st.metric("Volatility", f"{curr_vol:.2%}")
    st.metric("Sharpe", f"{curr_sharpe:.2f}")

with col2:
    st.markdown("**Min Variance**")
    st.metric("Return", f"{mv_port['annualised_return']:.2%}",
              delta=f"{mv_port['annualised_return'] - curr_ret:+.2%}")
    st.metric("Volatility", f"{mv_port['annualised_volatility']:.2%}",
              delta=f"{mv_port['annualised_volatility'] - curr_vol:+.2%}",
              delta_color="inverse")
    st.metric("Sharpe", f"{mv_port['sharpe_ratio']:.2f}",
              delta=f"{mv_port['sharpe_ratio'] - curr_sharpe:+.2f}")

with col3:
    st.markdown("**Max Sharpe**")
    st.metric("Return", f"{ms_port['annualised_return']:.2%}",
              delta=f"{ms_port['annualised_return'] - curr_ret:+.2%}")
    st.metric("Volatility", f"{ms_port['annualised_volatility']:.2%}",
              delta=f"{ms_port['annualised_volatility'] - curr_vol:+.2%}",
              delta_color="inverse")
    st.metric("Sharpe", f"{ms_port['sharpe_ratio']:.2f}",
              delta=f"{ms_port['sharpe_ratio'] - curr_sharpe:+.2f}")

with col4:
    st.markdown("**Frontier range**")
    st.metric("Min Return", f"{frontier_df['annualised_return'].min():.2%}")
    st.metric("Max Return", f"{frontier_df['annualised_return'].max():.2%}")
    st.metric("Best Sharpe", f"{frontier_df['sharpe_ratio'].max():.2f}")

st.markdown("---")


# ── Custom target return ───────────────────────────────────────────────────────
st.subheader("Custom Target Return")

ret_min = float(frontier_df["annualised_return"].min())
ret_max = float(frontier_df["annualised_return"].max())
default_target = float(ms_port["annualised_return"])
default_target = max(ret_min, min(ret_max, default_target))

target_pct = st.slider(
    "Target annualised return",
    min_value=round(ret_min * 100, 1),
    max_value=round(ret_max * 100, 1),
    value=round(default_target * 100, 1),
    step=0.1,
    format="%.1f%%",
)
target_return = target_pct / 100.0

target_port = target_return_portfolio(
    asset_returns,
    target_return=target_return,
    max_weight=max_weight,
)

# ── Main chart ─────────────────────────────────────────────────────────────────
fig = _build_frontier_chart(
    frontier=frontier_df,
    mv=mv_port,
    ms=ms_port,
    curr_v=curr_vol,
    curr_r=curr_ret,
    rf=rf_rate_stored,
    target_port=target_port,
)
st.plotly_chart(fig, use_container_width=True)

if target_port is None:
    st.warning(
        f"Could not find a feasible portfolio at {target_return:.1%} return. "
        "Try a value within the frontier range or relax the max-weight constraint."
    )
else:
    t_col1, t_col2, t_col3 = st.columns(3)
    t_col1.metric("Return (achieved)", f"{target_port['annualised_return']:.2%}")
    t_col2.metric("Volatility", f"{target_port['annualised_volatility']:.2%}")
    t_col3.metric("Sharpe", f"{target_port['sharpe_ratio']:.2f}")

st.markdown("---")


# ── Optimal weights ────────────────────────────────────────────────────────────
st.subheader("Optimal Weights")

tab_ms, tab_mv, tab_target = st.tabs(["Max Sharpe", "Min Variance", "Target Portfolio"])


def _weights_chart_and_table(weights: dict, title: str) -> None:
    """Render a horizontal bar chart and formatted table for a weights dict."""
    wdf = (
        pd.Series(weights, name="Weight")
        .sort_values(ascending=True)
        .reset_index()
        .rename(columns={"index": "Asset"})
    )
    wdf = wdf[wdf["Weight"] > 1e-4]  # drop near-zero allocations

    bar_fig = go.Figure(go.Bar(
        x=wdf["Weight"],
        y=wdf["Asset"],
        orientation="h",
        marker=dict(
            color=wdf["Weight"],
            colorscale="Blues",
            showscale=False,
        ),
        text=[f"{v:.1%}" for v in wdf["Weight"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    bar_fig.update_layout(
        title=title,
        xaxis=dict(tickformat=".0%", range=[0, wdf["Weight"].max() * 1.2]),
        yaxis=dict(automargin=True),
        height=max(250, len(wdf) * 45 + 80),
        margin=dict(t=50, b=30, l=10, r=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    table_df = pd.DataFrame({
        "Asset": list(weights.keys()),
        "Weight": [f"{v:.2%}" for v in weights.values()],
    }).sort_values("Weight", ascending=False).reset_index(drop=True)
    st.dataframe(table_df, use_container_width=True, hide_index=True)


with tab_ms:
    _weights_chart_and_table(ms_port["weights"], "Max Sharpe Portfolio")

with tab_mv:
    _weights_chart_and_table(mv_port["weights"], "Min Variance Portfolio")

with tab_target:
    if target_port is not None:
        _weights_chart_and_table(target_port["weights"], f"Target Return: {target_return:.1%}")
    else:
        st.info("No feasible target portfolio — adjust the slider above.")

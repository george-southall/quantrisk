"""Reusable Plotly chart builders for the QuantRisk dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = px.colors.qualitative.Plotly


def plot_cumulative_returns(
    returns_dict: dict[str, pd.Series],
    title: str = "Cumulative Returns",
    log_scale: bool = False,
) -> go.Figure:
    """Multi-line cumulative wealth chart, one series per strategy / asset."""
    fig = go.Figure()
    for i, (name, returns) in enumerate(returns_dict.items()):
        cumret = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cumret.index,
                y=cumret.values,
                name=name,
                line=dict(color=COLORS[i % len(COLORS)]),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        yaxis_type="log" if log_scale else "linear",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "Drawdown",
    color: str = "#EF553B",
) -> go.Figure:
    """Underwater / drawdown chart as a filled area."""
    from quantrisk.portfolio.returns import drawdown_series

    dd = drawdown_series(returns)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values * 100,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color=color),
            fillcolor="rgba(239,85,59,0.3)",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )
    return fig


def plot_return_distribution(
    returns: pd.Series,
    var_levels: dict[str, float] | None = None,
    title: str = "Return Distribution",
) -> go.Figure:
    """Histogram with optional VaR vertical lines."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns.values * 100,
            nbinsx=60,
            name="Daily Returns",
            marker_color="#636EFA",
            opacity=0.75,
        )
    )
    if var_levels:
        for label, var_val in var_levels.items():
            fig.add_vline(
                x=-abs(var_val) * 100,
                line_dash="dash",
                annotation_text=label,
                annotation_position="top right",
            )
    fig.update_layout(
        title=title,
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
    )
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """Annotated correlation heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(title=title)
    return fig


def plot_weights_pie(
    weights: dict[str, float],
    title: str = "Portfolio Weights",
) -> go.Figure:
    """Donut chart of portfolio weights."""
    fig = go.Figure(
        data=go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.35,
            textinfo="label+percent",
        )
    )
    fig.update_layout(title=title)
    return fig


def plot_rolling_stats(
    rolling_df: pd.DataFrame,
    title: str = "Rolling Statistics",
) -> go.Figure:
    """Three-panel subplot: rolling return, rolling volatility, drawdown."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Rolling Return (ann.)", "Rolling Volatility (ann.)", "Drawdown"],
        vertical_spacing=0.08,
    )

    if "rolling_return" in rolling_df.columns:
        fig.add_trace(
            go.Scatter(
                x=rolling_df.index,
                y=rolling_df["rolling_return"] * 100,
                name="Rolling Return",
                line=dict(color="#636EFA"),
            ),
            row=1,
            col=1,
        )

    if "rolling_volatility" in rolling_df.columns:
        fig.add_trace(
            go.Scatter(
                x=rolling_df.index,
                y=rolling_df["rolling_volatility"] * 100,
                name="Rolling Vol",
                line=dict(color="#EF553B"),
            ),
            row=2,
            col=1,
        )

    if "drawdown" in rolling_df.columns:
        dd = rolling_df["drawdown"]
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values * 100,
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="#AB63FA"),
                fillcolor="rgba(171,99,250,0.3)",
            ),
            row=3,
            col=1,
        )

    fig.update_layout(title=title, hovermode="x unified", height=600)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Vol (%)", row=2, col=1)
    fig.update_yaxes(title_text="DD (%)", row=3, col=1)
    return fig


def plot_scenario_bars(
    scenario_df: pd.DataFrame,
    title: str = "Stress Scenario P&L",
) -> go.Figure:
    """Horizontal bar chart of stress scenario portfolio P&L."""
    losses = scenario_df["portfolio_loss"]
    colors = ["#EF553B" if v < 0 else "#00CC96" for v in losses]
    fig = go.Figure(
        go.Bar(
            x=losses * 100,
            y=scenario_df["scenario"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in losses],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Portfolio P&L (%)",
        yaxis_title="",
        height=max(300, len(scenario_df) * 60),
    )
    return fig


def plot_mc_paths(
    paths: np.ndarray,
    horizon: int,
    confidence: float = 0.95,
    title: str = "Monte Carlo Simulation",
    n_paths_shown: int = 150,
) -> go.Figure:
    """Monte Carlo wealth paths with shaded percentile bands."""
    days = np.arange(1, horizon + 1)
    p05 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig = go.Figure()

    # Sample of individual paths
    n_show = min(n_paths_shown, paths.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(paths.shape[0], n_show, replace=False)
    for i in idx:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=paths[i],
                line=dict(color="rgba(100,100,200,0.04)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Percentile bands
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([p95, p05[::-1]]),
            fill="toself",
            fillcolor="rgba(99,110,250,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="5th–95th pct",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself",
            fillcolor="rgba(99,110,250,0.22)",
            line=dict(color="rgba(255,255,255,0)"),
            name="25th–75th pct",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days, y=p50, name="Median", line=dict(color="#636EFA", width=2)
        )
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Initial Value")

    fig.update_layout(
        title=title,
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=500,
    )
    return fig


def plot_factor_loadings(
    report_df: pd.DataFrame,
    title: str = "Factor Loadings",
) -> go.Figure:
    """Bar chart of factor loadings with significance highlighting."""
    factors = report_df.index.tolist()
    loadings = report_df["loading"].tolist()

    sig_col = "significant" if "significant" in report_df.columns else None
    colors = []
    for i, f in enumerate(factors):
        is_sig = bool(report_df[sig_col].iloc[i]) if sig_col else True
        if not is_sig:
            colors.append("rgba(150,150,150,0.5)")
        elif loadings[i] > 0:
            colors.append("#00CC96")
        else:
            colors.append("#EF553B")

    fig = go.Figure(
        go.Bar(
            x=factors,
            y=loadings,
            marker_color=colors,
            text=[f"{v:.3f}" for v in loadings],
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(title=title, xaxis_title="Factor", yaxis_title="Loading", height=400)
    return fig


def plot_var_comparison(
    var_table: pd.DataFrame,
    title: str = "VaR Comparison by Method",
) -> go.Figure:
    """Grouped bar chart comparing VaR across methods."""
    if var_table.empty:
        return go.Figure()

    # Expect columns: method, confidence, var
    if "method" not in var_table.columns:
        return go.Figure()

    methods = var_table["method"].unique()
    fig = go.Figure()
    for i, method in enumerate(methods):
        subset = var_table[var_table["method"] == method]
        fig.add_trace(
            go.Bar(
                name=method,
                x=subset["confidence"].astype(str),
                y=subset["var"] * 100,
                marker_color=COLORS[i % len(COLORS)],
                text=[f"{v:.2f}%" for v in subset["var"] * 100],
                textposition="outside",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Confidence Level",
        yaxis_title="VaR (%)",
        barmode="group",
    )
    return fig


def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns (%)",
) -> go.Figure:
    """Calendar heatmap of monthly returns."""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = monthly.to_frame("return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month

    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    text_vals = np.where(
        np.isnan(pivot.values),
        "",
        (pivot.values * 100).round(1).astype(str),
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=text_vals,
            texttemplate="%{text}%",
            hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(title=title, yaxis_title="Year", xaxis_title="Month")
    return fig


def plot_weights_history(
    weights_df: pd.DataFrame,
    title: str = "Portfolio Weights Over Time",
) -> go.Figure:
    """Stacked area chart of portfolio weight evolution."""
    fig = go.Figure()
    for i, col in enumerate(weights_df.columns):
        fig.add_trace(
            go.Scatter(
                x=weights_df.index,
                y=weights_df[col],
                name=col,
                stackgroup="one",
                line=dict(color=COLORS[i % len(COLORS)]),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    return fig


def plot_annual_returns_bar(
    annual_df: pd.DataFrame,
    title: str = "Annual Returns by Strategy",
) -> go.Figure:
    """Grouped bar chart of calendar-year returns."""
    fig = go.Figure()
    for i, col in enumerate(annual_df.columns):
        fig.add_trace(
            go.Bar(
                name=col,
                x=annual_df.index.astype(str),
                y=annual_df[col] * 100,
                marker_color=COLORS[i % len(COLORS)],
            )
        )
    fig.add_hline(y=0, line_color="gray", line_dash="dot")
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Return (%)",
        barmode="group",
        hovermode="x unified",
    )
    return fig


def plot_pca_explained_variance(
    explained_ratio: np.ndarray,
    title: str = "PCA Explained Variance",
) -> go.Figure:
    """Bar + cumulative line chart for PCA explained variance."""
    n = len(explained_ratio)
    cumulative = np.cumsum(explained_ratio)
    component_labels = [f"PC{i+1}" for i in range(n)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=component_labels,
            y=explained_ratio * 100,
            name="Individual",
            marker_color="#636EFA",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=component_labels,
            y=cumulative * 100,
            name="Cumulative",
            line=dict(color="#EF553B", width=2),
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.update_layout(title=title, xaxis_title="Component")
    fig.update_yaxes(title_text="Individual (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True, range=[0, 105])
    return fig


def plot_attribution_waterfall(
    summary: pd.Series,
    title: str = "Return Attribution (Annualised)",
) -> go.Figure:
    """Waterfall chart breaking down return by factor contribution."""
    items = summary.items()
    labels, values = zip(*items) if summary.size else ([], [])
    colors = ["#00CC96" if v >= 0 else "#EF553B" for v in values]

    fig = go.Figure(
        go.Bar(
            x=list(labels),
            y=[v * 100 for v in values],
            marker_color=colors,
            text=[f"{v:.2%}" for v in values],
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Contribution (%)",
        height=400,
    )
    return fig

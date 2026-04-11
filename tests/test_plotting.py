"""Smoke tests for plotting functions — verify they return valid Plotly figures."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from quantrisk.utils.plotting import (
    plot_annual_returns_bar,
    plot_attribution_waterfall,
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_drawdown,
    plot_factor_loadings,
    plot_mc_paths,
    plot_monthly_returns_heatmap,
    plot_pca_explained_variance,
    plot_return_distribution,
    plot_rolling_stats,
    plot_scenario_bars,
    plot_var_comparison,
    plot_weights_history,
    plot_weights_pie,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, 500), index=idx)


@pytest.fixture
def multi_returns(returns):
    rng = np.random.default_rng(1)
    return {
        "Portfolio": returns,
        "Benchmark": pd.Series(rng.normal(0.0004, 0.012, 500), index=returns.index),
    }


def _is_figure(obj):
    return isinstance(obj, go.Figure)


class TestPlottingSmoke:
    def test_cumulative_returns(self, multi_returns):
        assert _is_figure(plot_cumulative_returns(multi_returns))

    def test_cumulative_returns_log(self, multi_returns):
        assert _is_figure(plot_cumulative_returns(multi_returns, log_scale=True))

    def test_drawdown(self, returns):
        assert _is_figure(plot_drawdown(returns))

    def test_return_distribution(self, returns):
        assert _is_figure(plot_return_distribution(returns))

    def test_return_distribution_with_var(self, returns):
        assert _is_figure(
            plot_return_distribution(returns, var_levels={"95% VaR": 0.02, "99% VaR": 0.03})
        )

    def test_correlation_heatmap(self):
        corr = pd.DataFrame(
            [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        assert _is_figure(plot_correlation_heatmap(corr))

    def test_weights_pie(self):
        assert _is_figure(plot_weights_pie({"A": 0.4, "B": 0.35, "C": 0.25}))

    def test_rolling_stats(self, returns):
        rolling = pd.DataFrame({
            "rolling_return": returns.rolling(21).mean(),
            "rolling_volatility": returns.rolling(21).std(),
            "drawdown": returns.cumsum(),
        })
        assert _is_figure(plot_rolling_stats(rolling))

    def test_scenario_bars(self):
        df = pd.DataFrame({
            "scenario": ["GFC", "COVID", "Dot-com"],
            "portfolio_loss": [-0.35, -0.28, -0.40],
        })
        assert _is_figure(plot_scenario_bars(df))

    def test_mc_paths(self):
        rng = np.random.default_rng(0)
        paths = np.cumprod(1 + rng.normal(0.0003, 0.01, (500, 63)), axis=1)
        assert _is_figure(plot_mc_paths(paths, horizon=63))

    def test_factor_loadings(self):
        df = pd.DataFrame(
            {"loading": [0.8, 0.3, -0.1], "significant": [True, True, False]},
            index=["Mkt-RF", "SMB", "HML"],
        )
        assert _is_figure(plot_factor_loadings(df))

    def test_var_comparison(self):
        df = pd.DataFrame({
            "method": ["Historical", "Historical", "Parametric", "Parametric"],
            "confidence": [0.95, 0.99, 0.95, 0.99],
            "var": [0.018, 0.025, 0.017, 0.024],
        })
        assert _is_figure(plot_var_comparison(df))

    def test_var_comparison_empty(self):
        assert _is_figure(plot_var_comparison(pd.DataFrame()))

    def test_monthly_returns_heatmap(self, returns):
        assert _is_figure(plot_monthly_returns_heatmap(returns))

    def test_weights_history(self):
        idx = pd.date_range("2020-01-01", periods=12, freq="ME")
        df = pd.DataFrame({"A": [0.5] * 12, "B": [0.3] * 12, "C": [0.2] * 12}, index=idx)
        assert _is_figure(plot_weights_history(df))

    def test_annual_returns_bar(self):
        df = pd.DataFrame(
            {"Strategy A": [0.12, -0.05, 0.18], "Strategy B": [0.09, 0.02, 0.15]},
            index=[2021, 2022, 2023],
        )
        assert _is_figure(plot_annual_returns_bar(df))

    def test_pca_explained_variance(self):
        ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        assert _is_figure(plot_pca_explained_variance(ratios))

    def test_attribution_waterfall(self):
        s = pd.Series({"alpha": 0.02, "Mkt-RF": 0.05, "SMB": -0.01, "residual": 0.003})
        assert _is_figure(plot_attribution_waterfall(s))

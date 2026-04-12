"""Unit tests for mean-variance portfolio optimisation."""

import numpy as np
import pandas as pd
import pytest

from quantrisk.portfolio.optimizer import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    target_return_portfolio,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def returns_df():
    """500-day correlated returns for 4 assets."""
    rng = np.random.default_rng(0)
    cov = np.array([
        [0.0001, 0.00004, 0.00002, 0.00001],
        [0.00004, 0.00016, 0.00003, 0.00002],
        [0.00002, 0.00003, 0.00009, 0.00001],
        [0.00001, 0.00002, 0.00001, 0.00006],
    ])
    L = np.linalg.cholesky(cov)
    raw = rng.standard_normal((500, 4)) @ L.T + np.array([0.0003, 0.0006, 0.0004, 0.0002])
    idx = pd.bdate_range("2018-01-01", periods=500)
    return pd.DataFrame(raw, index=idx, columns=["A", "B", "C", "D"])


def _weights_valid(weights: dict) -> bool:
    vals = list(weights.values())
    return (
        abs(sum(vals) - 1.0) < 1e-5
        and all(v >= -1e-6 for v in vals)
    )


# ── Min Variance ───────────────────────────────────────────────────────────────

class TestMinVariance:
    def test_weights_sum_to_one(self, returns_df):
        result = min_variance_portfolio(returns_df)
        assert _weights_valid(result["weights"])

    def test_returns_expected_keys(self, returns_df):
        result = min_variance_portfolio(returns_df)
        assert {"weights", "annualised_return", "annualised_volatility", "sharpe_ratio"}.issubset(
            result.keys()
        )

    def test_vol_is_positive(self, returns_df):
        result = min_variance_portfolio(returns_df)
        assert result["annualised_volatility"] > 0

    def test_lower_vol_than_equal_weight(self, returns_df):
        mv = min_variance_portfolio(returns_df)
        n = len(returns_df.columns)
        ew_weights = np.ones(n) / n
        cov = returns_df.cov().values * 252
        mv_w = np.array(list(mv["weights"].values()))
        ew_vol = float(np.sqrt(ew_weights @ cov @ ew_weights))
        mv_vol = float(np.sqrt(mv_w @ cov @ mv_w))
        assert mv_vol <= ew_vol + 1e-6

    def test_max_weight_constraint(self, returns_df):
        result = min_variance_portfolio(returns_df, max_weight=0.4)
        assert all(v <= 0.4 + 1e-5 for v in result["weights"].values())

    def test_tickers_match_columns(self, returns_df):
        result = min_variance_portfolio(returns_df)
        assert set(result["weights"].keys()) == set(returns_df.columns)


# ── Max Sharpe ─────────────────────────────────────────────────────────────────

class TestMaxSharpe:
    def test_weights_sum_to_one(self, returns_df):
        result = max_sharpe_portfolio(returns_df)
        assert _weights_valid(result["weights"])

    def test_sharpe_higher_than_equal_weight(self, returns_df):
        ms = max_sharpe_portfolio(returns_df, risk_free_rate=0.02)
        n = len(returns_df.columns)
        ew_w = np.ones(n) / n
        mu = returns_df.mean().values * 252
        cov = returns_df.cov().values * 252
        ew_ret = float(ew_w @ mu)
        ew_vol = float(np.sqrt(ew_w @ cov @ ew_w))
        ew_sharpe = (ew_ret - 0.02) / ew_vol
        assert ms["sharpe_ratio"] >= ew_sharpe - 1e-4

    def test_max_weight_constraint(self, returns_df):
        result = max_sharpe_portfolio(returns_df, max_weight=0.35)
        assert all(v <= 0.35 + 1e-5 for v in result["weights"].values())

    def test_vol_positive(self, returns_df):
        result = max_sharpe_portfolio(returns_df)
        assert result["annualised_volatility"] > 0

    def test_tickers_match_columns(self, returns_df):
        result = max_sharpe_portfolio(returns_df)
        assert set(result["weights"].keys()) == set(returns_df.columns)


# ── Target Return ──────────────────────────────────────────────────────────────

class TestTargetReturn:
    def test_feasible_target_returns_result(self, returns_df):
        mv = min_variance_portfolio(returns_df)
        target = mv["annualised_return"] + 0.01
        result = target_return_portfolio(returns_df, target_return=target)
        assert result is not None

    def test_weights_sum_to_one(self, returns_df):
        mv = min_variance_portfolio(returns_df)
        target = mv["annualised_return"] + 0.01
        result = target_return_portfolio(returns_df, target_return=target)
        assert result is not None
        assert _weights_valid(result["weights"])

    def test_return_close_to_target(self, returns_df):
        mv = min_variance_portfolio(returns_df)
        target = mv["annualised_return"] + 0.02
        result = target_return_portfolio(returns_df, target_return=target)
        if result is not None:
            assert abs(result["annualised_return"] - target) < 0.01

    def test_infeasible_target_returns_none(self, returns_df):
        # A return far beyond any single asset is infeasible
        result = target_return_portfolio(returns_df, target_return=99.0)
        assert result is None

    def test_max_weight_respected(self, returns_df):
        mv = min_variance_portfolio(returns_df)
        target = mv["annualised_return"] + 0.01
        result = target_return_portfolio(returns_df, target_return=target, max_weight=0.4)
        if result is not None:
            assert all(v <= 0.4 + 1e-5 for v in result["weights"].values())


# ── Efficient Frontier ─────────────────────────────────────────────────────────

class TestEfficientFrontier:
    def test_returns_dataframe(self, returns_df):
        df = efficient_frontier(returns_df, n_points=10)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, returns_df):
        df = efficient_frontier(returns_df, n_points=10)
        assert {"annualised_return", "annualised_volatility", "sharpe_ratio"}.issubset(
            df.columns
        )

    def test_non_empty(self, returns_df):
        df = efficient_frontier(returns_df, n_points=10)
        assert len(df) > 0

    def test_sorted_by_vol(self, returns_df):
        df = efficient_frontier(returns_df, n_points=15)
        vols = df["annualised_volatility"].values
        assert np.all(vols[1:] >= vols[:-1] - 1e-8)

    def test_vol_positive(self, returns_df):
        df = efficient_frontier(returns_df, n_points=10)
        assert (df["annualised_volatility"] > 0).all()

    def test_min_point_near_gmv(self, returns_df):
        """Lowest-vol frontier point should be close to the GMV portfolio vol."""
        df = efficient_frontier(returns_df, n_points=20)
        mv = min_variance_portfolio(returns_df)
        assert abs(df["annualised_volatility"].iloc[0] - mv["annualised_volatility"]) < 0.02

    def test_max_weight_applied(self, returns_df):
        df = efficient_frontier(returns_df, n_points=10, max_weight=0.4)
        assert len(df) > 0

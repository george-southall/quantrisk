"""Unit tests for Portfolio class and return calculations."""

import numpy as np
import pandas as pd
import pytest

from quantrisk.portfolio.returns import (
    simple_returns,
    log_returns,
    cumulative_returns,
    annualised_return,
    annualised_volatility,
    downside_deviation,
    max_drawdown,
    max_drawdown_duration,
    excess_returns,
)


class TestReturnCalculations:
    def test_simple_returns_basic(self):
        prices = pd.Series([100, 110, 99])
        r = simple_returns(prices).dropna()
        assert len(r) == 2
        assert abs(r.iloc[0] - 0.10) < 1e-9
        assert abs(r.iloc[1] - (-99 / 110 + 1 - 1)) < 1e-9  # -0.1

    def test_log_returns_basic(self):
        prices = pd.Series([100.0, 110.0])
        r = log_returns(prices).dropna()
        assert abs(r.iloc[0] - np.log(110 / 100)) < 1e-9

    def test_cumulative_returns_starts_at_one(self, normal_returns):
        cum = cumulative_returns(normal_returns.dropna())
        assert abs(cum.iloc[0] - (1 + normal_returns.dropna().iloc[0])) < 1e-9

    def test_annualised_return_flat(self):
        """A zero-return series should give ~0% annualised return."""
        returns = pd.Series([0.0] * 252)
        assert abs(annualised_return(returns)) < 1e-9

    def test_annualised_return_daily(self):
        """252 days of 0.1% daily returns ≈ 28.5% annualised."""
        returns = pd.Series([0.001] * 252)
        ann = annualised_return(returns)
        expected = (1.001 ** 252) - 1
        assert abs(ann - expected) < 1e-6

    def test_annualised_volatility(self):
        """Known volatility: std * sqrt(252)."""
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 500))
        ann_vol = annualised_volatility(returns)
        assert abs(ann_vol - returns.std() * np.sqrt(252)) < 1e-10

    def test_max_drawdown_known(self):
        """
        Prices: 100, 120, 90, 110 → drawdown = (90-120)/120 = -0.25
        """
        prices = pd.Series([100.0, 120.0, 90.0, 110.0])
        returns = simple_returns(prices).dropna()
        mdd = max_drawdown(returns)
        assert abs(mdd - (-0.25)) < 1e-6

    def test_max_drawdown_no_loss(self):
        """Monotonically rising prices → max drawdown = 0."""
        prices = pd.Series([100.0, 110.0, 120.0, 130.0])
        returns = simple_returns(prices).dropna()
        mdd = max_drawdown(returns)
        assert mdd == 0.0

    def test_max_drawdown_duration(self, normal_returns):
        duration = max_drawdown_duration(normal_returns)
        assert duration >= 0
        assert isinstance(duration, int)

    def test_downside_deviation_no_losses(self):
        returns = pd.Series([0.01] * 100)
        dd = downside_deviation(returns, threshold=0.0)
        assert dd == 0.0

    def test_excess_returns_scalar_rf(self, normal_returns):
        rf = 0.05  # 5% annualised
        excess = excess_returns(normal_returns, rf)
        daily_rf = rf / 252
        expected = normal_returns - daily_rf
        pd.testing.assert_series_equal(excess, expected)

    def test_excess_returns_series_rf(self, normal_returns):
        rf_series = pd.Series(0.05, index=normal_returns.index)
        excess = excess_returns(normal_returns, rf_series)
        assert len(excess) == len(normal_returns)


class TestPortfolioClass:
    """Tests for Portfolio that don't require live market data."""

    def test_weight_normalisation(self):
        """Weights should be normalised to sum to 1."""
        from quantrisk.portfolio.portfolio import Portfolio

        p = Portfolio({"A": 2, "B": 3, "C": 5})
        assert abs(sum(p.weights.values()) - 1.0) < 1e-10

    def test_empty_weights_raises(self):
        from quantrisk.portfolio.portfolio import Portfolio

        with pytest.raises(ValueError, match="non-empty"):
            Portfolio({})

    def test_zero_weights_raises(self):
        from quantrisk.portfolio.portfolio import Portfolio

        with pytest.raises(ValueError, match="positive"):
            Portfolio({"A": 0, "B": 0})

    def test_load_required_before_properties(self):
        from quantrisk.portfolio.portfolio import Portfolio

        p = Portfolio({"AAPL": 0.5, "MSFT": 0.5})
        with pytest.raises(RuntimeError, match="load()"):
            _ = p.prices

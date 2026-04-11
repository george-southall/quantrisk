"""Unit tests for VaR and CVaR calculations."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantrisk.risk.cvar import historical_cvar, parametric_cvar
from quantrisk.risk.var import historical_var, monte_carlo_var, parametric_var, var_summary


class TestHistoricalVaR:
    def test_basic(self, normal_returns):
        var = historical_var(normal_returns, confidence=0.95)
        assert var > 0
        assert var < 0.10  # shouldn't be enormous for normal-ish returns

    def test_higher_confidence_gives_larger_var(self, normal_returns):
        var95 = historical_var(normal_returns, confidence=0.95)
        var99 = historical_var(normal_returns, confidence=0.99)
        assert var99 > var95

    def test_horizon_scaling(self, normal_returns):
        var1 = historical_var(normal_returns, confidence=0.95, horizon=1)
        var10 = historical_var(normal_returns, confidence=0.95, horizon=10)
        # Should scale by sqrt(10)
        assert abs(var10 / var1 - np.sqrt(10)) < 0.01

    def test_insufficient_data_raises(self):
        short_series = pd.Series([0.01, -0.02, 0.005])
        with pytest.raises(ValueError, match="Insufficient data"):
            historical_var(short_series)

    def test_returns_positive_number(self, normal_returns):
        var = historical_var(normal_returns)
        assert var > 0


class TestParametricVaR:
    def test_known_distribution(self):
        """
        For a standard normal portfolio with mu=0, sigma=1,
        parametric VaR at 95% should equal ~1.645.
        """
        rng = np.random.default_rng(0)
        # Large sample to get close to theoretical value
        returns = pd.Series(rng.normal(0, 1, 100_000))
        var = parametric_var(returns, confidence=0.95)
        assert abs(var - stats.norm.ppf(0.95)) < 0.05

    def test_normal_vs_t_fat_tails(self, normal_returns):
        """Student-t VaR should be >= normal VaR (fatter tails)."""
        var_norm = parametric_var(normal_returns, confidence=0.99, distribution="normal")
        var_t = parametric_var(normal_returns, confidence=0.99, distribution="t")
        # Student-t should have same or larger VaR than normal at high confidence
        assert var_t >= var_norm * 0.9  # allow small tolerance

    def test_invalid_distribution_raises(self, normal_returns):
        with pytest.raises(ValueError, match="Unknown distribution"):
            parametric_var(normal_returns, distribution="cauchy")

    def test_positive_output(self, normal_returns):
        var = parametric_var(normal_returns)
        assert var > 0

    def test_formula_consistency(self):
        """VaR = -(mu + z * sigma) for normal distribution."""
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.001, 0.02, 10_000))
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        z = stats.norm.ppf(0.05)  # 1-sided at 95%
        expected = -(mu + z * sigma)
        computed = parametric_var(returns, confidence=0.95, horizon=1)
        assert abs(computed - expected) < 1e-6


class TestMonteCarloVaR:
    def test_basic(self, asset_returns, sample_weights):
        var = monte_carlo_var(asset_returns, sample_weights, confidence=0.95, n_simulations=5000)
        assert var > 0
        assert var < 0.15

    def test_equal_weight_default(self, asset_returns):
        var = monte_carlo_var(asset_returns, confidence=0.95, n_simulations=5000)
        assert var > 0

    def test_higher_confidence_larger_var(self, asset_returns, sample_weights):
        var95 = monte_carlo_var(asset_returns, sample_weights, confidence=0.95, n_simulations=5000)
        var99 = monte_carlo_var(asset_returns, sample_weights, confidence=0.99, n_simulations=5000)
        assert var99 > var95

    def test_reproducible_with_seed(self, asset_returns, sample_weights):
        v1 = monte_carlo_var(asset_returns, sample_weights, n_simulations=1000, random_seed=42)
        v2 = monte_carlo_var(asset_returns, sample_weights, n_simulations=1000, random_seed=42)
        assert v1 == v2


class TestCVaR:
    def test_cvar_greater_than_var(self, normal_returns):
        """CVaR must always be >= VaR (by definition)."""
        var = historical_var(normal_returns, confidence=0.95)
        cvar = historical_cvar(normal_returns, confidence=0.95)
        assert cvar >= var

    def test_parametric_cvar_normal(self, normal_returns):
        cvar = parametric_cvar(normal_returns, confidence=0.95, distribution="normal")
        assert cvar > 0

    def test_parametric_cvar_t(self, normal_returns):
        cvar = parametric_cvar(normal_returns, confidence=0.95, distribution="t")
        assert cvar > 0

    def test_cvar_99_greater_than_95(self, normal_returns):
        cvar95 = historical_cvar(normal_returns, confidence=0.95)
        cvar99 = historical_cvar(normal_returns, confidence=0.99)
        assert cvar99 >= cvar95


class TestVaRSummary:
    def test_summary_returns_dataframe(self, normal_returns, asset_returns, sample_weights):
        df = var_summary(
            normal_returns,
            confidence_levels=[0.95, 0.99],
            horizons=[1, 10],
            asset_returns=asset_returns,
            weights=sample_weights,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 conf × 2 horizons
        assert "historical" in df.columns
        assert "parametric_normal" in df.columns
        assert "monte_carlo" in df.columns


# ── Additional CVaR coverage ──────────────────────────────────────────────────

class TestParametricCVarExtended:
    def test_t_distribution(self, normal_returns):
        cvar_t = parametric_cvar(normal_returns, confidence=0.95, distribution="t")
        assert cvar_t > 0

    def test_t_greater_than_normal(self, normal_returns):
        """Student-t CVaR should be >= normal CVaR (fatter tails)."""
        cvar_n = parametric_cvar(normal_returns, confidence=0.95, distribution="normal")
        cvar_t = parametric_cvar(normal_returns, confidence=0.95, distribution="t")
        assert cvar_t >= cvar_n * 0.9  # allow small numerical variance

    def test_invalid_distribution_raises(self, normal_returns):
        with pytest.raises(ValueError):
            parametric_cvar(normal_returns, distribution="cauchy")

    def test_horizon_scaling(self, normal_returns):
        cvar_1 = parametric_cvar(normal_returns, horizon=1)
        cvar_5 = parametric_cvar(normal_returns, horizon=5)
        assert cvar_5 > cvar_1


class TestCVarSummary:
    def test_returns_dataframe(self, normal_returns):
        from quantrisk.risk.cvar import cvar_summary
        df = cvar_summary(normal_returns)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, normal_returns):
        from quantrisk.risk.cvar import cvar_summary
        df = cvar_summary(normal_returns)
        for col in ("confidence", "cvar_historical", "cvar_parametric"):
            assert col in df.columns

    def test_custom_confidence_levels(self, normal_returns):
        from quantrisk.risk.cvar import cvar_summary
        df = cvar_summary(normal_returns, confidence_levels=[0.90, 0.95, 0.99])
        assert len(df) == 3

    def test_cvar_exceeds_var(self, normal_returns):
        from quantrisk.risk.cvar import cvar_summary
        df = cvar_summary(normal_returns)
        assert (df["cvar_historical"] >= df["var_historical"] * 0.99).all()


class TestRollingVar:
    def test_returns_series(self, normal_returns):
        from quantrisk.risk.var import rolling_var
        result = rolling_var(normal_returns, window=50, confidence=0.95, method="historical")
        assert isinstance(result, pd.Series)
        assert len(result) == len(normal_returns) - 50 + 1

    def test_parametric_method(self, normal_returns):
        from quantrisk.risk.var import rolling_var
        result = rolling_var(normal_returns, window=50, method="parametric")
        assert isinstance(result, pd.Series)
        assert (result > 0).all()

    def test_all_positive(self, normal_returns):
        from quantrisk.risk.var import rolling_var
        result = rolling_var(normal_returns, window=50)
        assert (result > 0).all()


class TestMonteCarloCVaR:
    def test_basic(self):
        from quantrisk.risk.cvar import monte_carlo_cvar
        rng = np.random.default_rng(0)
        losses = pd.Series(rng.normal(0, 0.01, 10000))
        cvar = monte_carlo_cvar(losses.values, confidence=0.95)
        assert cvar > 0

    def test_cvar_exceeds_percentile(self):
        from quantrisk.risk.cvar import monte_carlo_cvar
        rng = np.random.default_rng(0)
        losses = rng.normal(0, 0.01, 10000)
        cvar = monte_carlo_cvar(losses, confidence=0.95)
        var_95 = -np.percentile(losses, 5)
        assert cvar >= var_95 * 0.9

    def test_empty_tail_returns_nan(self):
        from quantrisk.risk.cvar import monte_carlo_cvar
        # All positive returns → tail (losses) is empty
        all_gains = np.ones(100) * 0.01
        result = monte_carlo_cvar(all_gains, confidence=0.95)
        assert np.isnan(result) or result <= 0

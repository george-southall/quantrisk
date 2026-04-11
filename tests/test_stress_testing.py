"""Unit tests for stress testing modules."""

import pytest

from quantrisk.stress_testing.historical_scenarios import (
    apply_scenario,
    run_all_scenarios,
    SCENARIOS,
)
from quantrisk.stress_testing.hypothetical import apply_hypothetical_shocks
from quantrisk.stress_testing.monte_carlo import simulate_portfolio_paths, mc_var_cvar


SAMPLE_WEIGHTS = {
    "AAPL": 0.15, "MSFT": 0.15, "JPM": 0.10,
    "XOM": 0.10, "GLD": 0.15, "TLT": 0.15,
    "EEM": 0.10, "VNQ": 0.10,
}


class TestHistoricalScenarios:
    def test_all_scenarios_run(self):
        for key in SCENARIOS:
            result = apply_scenario(SAMPLE_WEIGHTS, key)
            assert result.portfolio_pl != 0.0

    def test_2008_causes_loss(self):
        result = apply_scenario(SAMPLE_WEIGHTS, "2008_gfc")
        assert result.portfolio_pl < 0  # overall portfolio should lose money

    def test_2022_rate_shock_bonds_lose(self):
        result = apply_scenario({"TLT": 0.5, "AAPL": 0.5}, "2022_rate_shock")
        # TLT had -32% shock; AAPL -25% → should be negative
        assert result.portfolio_pl < 0

    def test_pl_contributions_sum_to_total(self):
        result = apply_scenario(SAMPLE_WEIGHTS, "covid_crash")
        total_from_parts = sum(result.ticker_pl.values())
        assert abs(total_from_parts - result.portfolio_pl) < 1e-9

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            apply_scenario(SAMPLE_WEIGHTS, "made_up_scenario")

    def test_missing_ticker_uses_default(self):
        weights = {"UNKNOWN_TICKER_XYZ": 1.0}
        result = apply_scenario(weights, "2008_gfc")
        assert len(result.issues) > 0  # should flag the missing ticker
        assert result.portfolio_pl < 0  # equity default is negative

    def test_run_all_scenarios_returns_df(self):
        df = run_all_scenarios(SAMPLE_WEIGHTS)
        assert len(df) == len(SCENARIOS)
        assert "portfolio_loss" in df.columns

    def test_weights_sum_preserved(self):
        result = apply_scenario(SAMPLE_WEIGHTS, "2008_gfc")
        assert abs(sum(result.weights.values()) - 1.0) < 1e-9

    def test_to_dataframe(self):
        result = apply_scenario(SAMPLE_WEIGHTS, "2008_gfc")
        df = result.to_dataframe()
        assert len(df) == len(SAMPLE_WEIGHTS)
        assert "pl_contribution" in df.columns


class TestHypotheticalShocks:
    def test_custom_shocks_applied(self):
        weights = {"AAPL": 0.5, "GLD": 0.5}
        shocks = {"AAPL": -0.20, "GLD": 0.10}
        result = apply_hypothetical_shocks(weights, shocks)
        expected = 0.5 * (-0.20) + 0.5 * 0.10
        assert abs(result.portfolio_pl - expected) < 1e-9

    def test_default_shock_applied_to_missing(self):
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        shocks = {"AAPL": -0.10}
        result = apply_hypothetical_shocks(weights, shocks, default_shock=-0.05)
        expected = 0.5 * (-0.10) + 0.5 * (-0.05)
        assert abs(result.portfolio_pl - expected) < 1e-9

    def test_zero_shock_no_pl(self):
        weights = {"AAPL": 1.0}
        result = apply_hypothetical_shocks(weights, {}, default_shock=0.0)
        assert result.portfolio_pl == 0.0


class TestMonteCarloSimulation:
    def test_paths_shape(self, asset_returns, sample_weights):
        paths = simulate_portfolio_paths(
            asset_returns, sample_weights, horizon=50, n_simulations=200, random_seed=42
        )
        assert paths.shape == (200, 50)

    def test_paths_start_near_one(self, asset_returns, sample_weights):
        """Paths are cumulative wealth — shouldn't start unreasonably far from 1."""
        paths = simulate_portfolio_paths(
            asset_returns, sample_weights, horizon=1, n_simulations=1000, random_seed=42
        )
        # First column: (1 + one day's return); should be clustered around 1
        assert 0.90 < paths[:, 0].mean() < 1.10

    def test_mc_var_cvar(self, asset_returns, sample_weights):
        paths = simulate_portfolio_paths(
            asset_returns, sample_weights, horizon=30, n_simulations=1000, random_seed=42
        )
        stats = mc_var_cvar(paths, confidence=0.95)
        assert stats["var"] > 0
        assert stats["cvar"] >= stats["var"]
        assert 0 <= stats["prob_loss"] <= 1

    def test_reproducibility(self, asset_returns, sample_weights):
        p1 = simulate_portfolio_paths(
            asset_returns, sample_weights, horizon=20, n_simulations=100, random_seed=7
        )
        p2 = simulate_portfolio_paths(
            asset_returns, sample_weights, horizon=20, n_simulations=100, random_seed=7
        )
        assert (p1 == p2).all()

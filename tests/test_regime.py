"""Unit tests for HMM regime detection."""

import numpy as np
import pandas as pd
import pytest

from quantrisk.regime.hmm import (
    current_regime,
    fit_hmm,
    get_regime_series,
    regime_colour,
    regime_statistics,
)

# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_returns():
    """
    500-day return series with two obvious regimes:
      - First 250 days: positive drift, low vol (bull)
      - Last 250 days: negative drift, high vol (bear)
    """
    rng = np.random.default_rng(7)
    bull = rng.normal(loc=0.001, scale=0.008, size=250)
    bear = rng.normal(loc=-0.001, scale=0.018, size=250)
    data = np.concatenate([bull, bear])
    idx = pd.bdate_range("2018-01-01", periods=500)
    return pd.Series(data, index=idx, name="returns")


# ── fit_hmm ────────────────────────────────────────────────────────────────────

class TestFitHmm:
    def test_returns_three_items(self, synthetic_returns):
        model, states, index = fit_hmm(synthetic_returns, n_regimes=2)
        assert model is not None
        assert len(states) == len(index)

    def test_states_length_matches_clean_returns(self, synthetic_returns):
        _, states, index = fit_hmm(synthetic_returns, n_regimes=2)
        assert len(states) == len(synthetic_returns.dropna())

    def test_state_values_in_range(self, synthetic_returns):
        _, states, _ = fit_hmm(synthetic_returns, n_regimes=2)
        assert set(states).issubset({0, 1})

    def test_three_regimes(self, synthetic_returns):
        _, states, _ = fit_hmm(synthetic_returns, n_regimes=3)
        assert set(states).issubset({0, 1, 2})

    def test_invalid_n_regimes(self, synthetic_returns):
        with pytest.raises(ValueError, match="n_regimes"):
            fit_hmm(synthetic_returns, n_regimes=4)


# ── get_regime_series ──────────────────────────────────────────────────────────

class TestGetRegimeSeries:
    def test_returns_series(self, synthetic_returns):
        result = get_regime_series(synthetic_returns, n_regimes=2)
        assert isinstance(result, pd.Series)

    def test_labels_are_valid_2(self, synthetic_returns):
        result = get_regime_series(synthetic_returns, n_regimes=2)
        assert set(result.unique()).issubset({"Bull", "Bear"})

    def test_labels_are_valid_3(self, synthetic_returns):
        result = get_regime_series(synthetic_returns, n_regimes=3)
        assert set(result.unique()).issubset({"Bull", "Bear", "Volatile"})

    def test_index_aligned(self, synthetic_returns):
        result = get_regime_series(synthetic_returns, n_regimes=2)
        assert len(result) == len(synthetic_returns.dropna())

    def test_bear_has_lower_mean_than_bull(self, synthetic_returns):
        """State relabelling: Bear mean return < Bull mean return."""
        result = get_regime_series(synthetic_returns, n_regimes=2)
        aligned = pd.concat([synthetic_returns.rename("r"), result], axis=1).dropna()
        bull_mean = aligned.loc[aligned["regime"] == "Bull", "r"].mean()
        bear_mean = aligned.loc[aligned["regime"] == "Bear", "r"].mean()
        assert bear_mean < bull_mean

    def test_deterministic_across_calls(self, synthetic_returns):
        """Same seed → same labels."""
        r1 = get_regime_series(synthetic_returns, n_regimes=2)
        r2 = get_regime_series(synthetic_returns, n_regimes=2)
        pd.testing.assert_series_equal(r1, r2)


# ── regime_statistics ─────────────────────────────────────────────────────────

class TestRegimeStatistics:
    def test_returns_dataframe(self, synthetic_returns):
        rs = get_regime_series(synthetic_returns, n_regimes=2)
        stats = regime_statistics(synthetic_returns, rs)
        assert isinstance(stats, pd.DataFrame)

    def test_expected_columns(self, synthetic_returns):
        rs = get_regime_series(synthetic_returns, n_regimes=2)
        stats = regime_statistics(synthetic_returns, rs)
        expected = {"mean_return", "volatility", "sharpe_ratio",
                    "avg_duration_days", "pct_time", "is_current"}
        assert expected.issubset(stats.columns)

    def test_pct_time_sums_to_one(self, synthetic_returns):
        rs = get_regime_series(synthetic_returns, n_regimes=2)
        stats = regime_statistics(synthetic_returns, rs)
        assert abs(stats["pct_time"].sum() - 1.0) < 0.01

    def test_exactly_one_current(self, synthetic_returns):
        rs = get_regime_series(synthetic_returns, n_regimes=2)
        stats = regime_statistics(synthetic_returns, rs)
        assert stats["is_current"].sum() == 1

    def test_volatility_positive(self, synthetic_returns):
        rs = get_regime_series(synthetic_returns, n_regimes=2)
        stats = regime_statistics(synthetic_returns, rs)
        assert (stats["volatility"] > 0).all()

    def test_avg_duration_positive(self, synthetic_returns):
        rs = get_regime_series(synthetic_returns, n_regimes=2)
        stats = regime_statistics(synthetic_returns, rs)
        assert (stats["avg_duration_days"] > 0).all()

    def test_three_regime_stats(self, synthetic_returns):
        rs = get_regime_series(synthetic_returns, n_regimes=3)
        stats = regime_statistics(synthetic_returns, rs)
        assert len(stats) == len(rs.unique())


# ── current_regime / regime_colour ────────────────────────────────────────────

class TestHelpers:
    def test_current_regime_returns_string(self, synthetic_returns):
        rs = get_regime_series(synthetic_returns, n_regimes=2)
        label = current_regime(rs)
        assert isinstance(label, str)
        assert label in {"Bull", "Bear"}

    def test_current_regime_matches_last_observation(self, synthetic_returns):
        rs = get_regime_series(synthetic_returns, n_regimes=2)
        assert current_regime(rs) == rs.iloc[-1]

    def test_regime_colour_known_labels(self):
        assert regime_colour("Bull") == "#00CC96"
        assert regime_colour("Bear") == "#EF553B"
        assert regime_colour("Volatile") == "#FFA15A"

    def test_regime_colour_unknown_fallback(self):
        colour = regime_colour("Unknown")
        assert colour.startswith("#")

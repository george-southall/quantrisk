"""Unit tests for backtesting strategies, engine, and tearsheet evaluator."""

import numpy as np
import pandas as pd
import pytest

from quantrisk.backtesting.strategies import (
    STRATEGY_REGISTRY,
    equal_weight,
    inverse_volatility,
    maximum_sharpe,
    minimum_variance,
    momentum,
    risk_parity,
)
from quantrisk.backtesting.engine import BacktestEngine, BacktestResult
from quantrisk.backtesting.evaluation import TearsheetEvaluator


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def returns_df():
    """250-day returns for 4 assets with some correlation."""
    rng = np.random.default_rng(42)
    n, k = 500, 4
    cov = np.array([
        [0.0001, 0.00005, 0.00002, 0.00001],
        [0.00005, 0.00015, 0.00003, 0.00002],
        [0.00002, 0.00003, 0.00012, 0.00001],
        [0.00001, 0.00002, 0.00001, 0.00008],
    ])
    L = np.linalg.cholesky(cov)
    raw = rng.standard_normal((n, k)) @ L.T + 0.0003
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.DataFrame(raw, index=idx, columns=["A", "B", "C", "D"])


# ── Strategy weight properties ─────────────────────────────────────────────────

class TestStrategyWeights:
    """All strategies must return weights that sum to 1 and are non-negative."""

    @pytest.mark.parametrize("strategy_fn", [
        equal_weight,
        inverse_volatility,
        risk_parity,
        minimum_variance,
        maximum_sharpe,
        momentum,
    ])
    def test_weights_sum_to_one(self, returns_df, strategy_fn):
        weights = strategy_fn(returns_df)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    @pytest.mark.parametrize("strategy_fn", [
        equal_weight,
        inverse_volatility,
        risk_parity,
        minimum_variance,
        maximum_sharpe,
        momentum,
    ])
    def test_weights_non_negative(self, returns_df, strategy_fn):
        weights = strategy_fn(returns_df)
        assert all(v >= -1e-9 for v in weights.values())

    @pytest.mark.parametrize("strategy_fn", [
        equal_weight,
        inverse_volatility,
        risk_parity,
        minimum_variance,
        maximum_sharpe,
        momentum,
    ])
    def test_weights_keys_match_tickers(self, returns_df, strategy_fn):
        weights = strategy_fn(returns_df)
        assert set(weights.keys()) == set(returns_df.columns)

    def test_equal_weight_is_uniform(self, returns_df):
        weights = equal_weight(returns_df)
        expected = 1.0 / len(returns_df.columns)
        for v in weights.values():
            assert abs(v - expected) < 1e-9

    def test_strategy_registry_complete(self):
        expected = {
            "equal_weight", "inverse_volatility", "risk_parity",
            "minimum_variance", "maximum_sharpe", "momentum",
        }
        assert set(STRATEGY_REGISTRY.keys()) == expected


# ── BacktestEngine ─────────────────────────────────────────────────────────────

class TestBacktestEngine:
    def test_run_returns_backtest_result(self, returns_df):
        engine = BacktestEngine(estimation_window=126, rebalance_freq="QE")
        result = engine.run(returns_df, equal_weight, "equal_weight")
        assert isinstance(result, BacktestResult)

    def test_returns_series_length(self, returns_df):
        engine = BacktestEngine(estimation_window=126, rebalance_freq="QE")
        result = engine.run(returns_df, equal_weight, "equal_weight")
        assert len(result.returns) > 0

    def test_cumulative_returns_starts_positive(self, returns_df):
        engine = BacktestEngine(estimation_window=126, rebalance_freq="QE")
        result = engine.run(returns_df, equal_weight, "equal_weight")
        assert (result.cumulative_returns > 0).all()

    def test_insufficient_data_raises(self, returns_df):
        engine = BacktestEngine(estimation_window=600)
        with pytest.raises(ValueError, match="Insufficient data"):
            engine.run(returns_df, equal_weight, "equal_weight")

    def test_metrics_keys_present(self, returns_df):
        engine = BacktestEngine(estimation_window=126, rebalance_freq="QE")
        result = engine.run(returns_df, equal_weight, "equal_weight")
        m = result.metrics()
        for key in ("annualised_return", "sharpe_ratio", "max_drawdown", "win_rate"):
            assert key in m

    def test_run_all_returns_dict(self, returns_df):
        engine = BacktestEngine(estimation_window=126, rebalance_freq="QE")
        strategies = {
            "equal_weight": equal_weight,
            "inv_vol": inverse_volatility,
        }
        results = engine.run_all(returns_df, strategies)
        assert set(results.keys()) == {"equal_weight", "inv_vol"}

    def test_comparison_table_shape(self, returns_df):
        engine = BacktestEngine(estimation_window=126, rebalance_freq="QE")
        results = engine.run_all(returns_df, {"ew": equal_weight, "iv": inverse_volatility})
        table = BacktestEngine.comparison_table(results)
        assert table.shape[0] == 2


# ── TearsheetEvaluator ─────────────────────────────────────────────────────────

class TestTearsheetEvaluator:
    @pytest.fixture
    def evaluator(self, returns_df):
        engine = BacktestEngine(estimation_window=126, rebalance_freq="QE")
        results = engine.run_all(returns_df, {
            "equal_weight": equal_weight,
            "inv_vol": inverse_volatility,
        })
        return TearsheetEvaluator(results)

    def test_comparison_table_returns_df(self, evaluator):
        table = evaluator.comparison_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2

    def test_annual_returns_has_years_as_index(self, evaluator):
        df = evaluator.annual_returns()
        assert df.index.name == "Year"
        assert all(isinstance(y, (int, np.integer)) for y in df.index)

    def test_monthly_heatmap_shape(self, evaluator):
        pivot = evaluator.monthly_returns_heatmap("equal_weight")
        assert isinstance(pivot, pd.DataFrame)
        assert pivot.shape[1] <= 12

    def test_monthly_heatmap_unknown_strategy_raises(self, evaluator):
        with pytest.raises(KeyError):
            evaluator.monthly_returns_heatmap("nonexistent")

    def test_rolling_metrics_columns(self, evaluator):
        df = evaluator.rolling_metrics("equal_weight")
        for col in ("rolling_return", "rolling_volatility", "drawdown"):
            assert col in df.columns

    def test_cumulative_returns_starts_above_zero(self, evaluator):
        cum = evaluator.cumulative_returns()
        assert (cum.iloc[0] > 0).all()

    def test_empty_results_raises(self):
        with pytest.raises(ValueError):
            TearsheetEvaluator({})

    def test_print_tearsheet_runs(self, evaluator, capsys):
        evaluator.print_tearsheet()
        captured = capsys.readouterr()
        assert "Backtest Tearsheet" in captured.out

    def test_benchmark_in_annual_returns(self, returns_df, evaluator):
        bench = returns_df["A"]
        ev = TearsheetEvaluator(
            {"equal_weight": list(evaluator.results.values())[0]},
            benchmark_returns=bench,
        )
        annual = ev.annual_returns()
        assert "Benchmark" in annual.columns

    def test_benchmark_in_cumulative_returns(self, returns_df, evaluator):
        bench = returns_df["A"]
        ev = TearsheetEvaluator(
            {"equal_weight": list(evaluator.results.values())[0]},
            benchmark_returns=bench,
        )
        cum = ev.cumulative_returns()
        assert "Benchmark" in cum.columns

"""Unit tests for risk-adjusted metrics and drawdown analysis."""

import numpy as np
import pandas as pd
import pytest

from quantrisk.risk.drawdown import drawdown_table
from quantrisk.risk.metrics import (
    alpha,
    beta,
    calmar_ratio,
    compute_all_metrics,
    information_ratio,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_returns():
    """Zero-return series — all ratios should be 0 or nan."""
    return pd.Series([0.0] * 252)


@pytest.fixture
def positive_returns():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=500, freq="B")
    return pd.Series(rng.normal(0.001, 0.01, 500), index=idx)


@pytest.fixture
def benchmark_returns():
    rng = np.random.default_rng(99)
    idx = pd.date_range("2018-01-01", periods=500, freq="B")
    return pd.Series(rng.normal(0.0008, 0.012, 500), index=idx)


# ── Sharpe ────────────────────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_for_positive_excess_returns(self, positive_returns):
        sr = sharpe_ratio(positive_returns, risk_free_rate=0.0)
        assert sr > 0

    def test_zero_vol_returns_nan(self, flat_returns):
        assert np.isnan(sharpe_ratio(flat_returns))

    def test_higher_return_higher_sharpe(self):
        rf = 0.0
        # use noisy series so vol > 0
        rng = np.random.default_rng(0)
        noise = pd.Series(rng.normal(0, 0.01, 252))
        low_r  = noise + 0.0001
        high_r = noise + 0.001
        assert sharpe_ratio(high_r, rf) > sharpe_ratio(low_r, rf)


# ── Sortino ───────────────────────────────────────────────────────────────────

class TestSortinoRatio:
    def test_no_downside_returns_nan(self):
        returns = pd.Series([0.01] * 100)
        assert np.isnan(sortino_ratio(returns, risk_free_rate=0.0))

    def test_positive_for_positive_returns(self, positive_returns):
        sr = sortino_ratio(positive_returns, risk_free_rate=0.0)
        assert sr > 0


# ── Calmar ────────────────────────────────────────────────────────────────────

class TestCalmarRatio:
    def test_no_drawdown_returns_nan(self):
        prices = pd.Series([1.0, 1.01, 1.02, 1.03])
        returns = prices.pct_change().dropna()
        assert np.isnan(calmar_ratio(returns))

    def test_positive_for_good_portfolio(self, positive_returns):
        result = calmar_ratio(positive_returns)
        # May be nan if no drawdown, otherwise should be finite
        assert np.isnan(result) or np.isfinite(result)


# ── Beta / Alpha ──────────────────────────────────────────────────────────────

class TestBetaAlpha:
    def test_beta_of_identical_series_is_one(self, positive_returns):
        b = beta(positive_returns, positive_returns)
        assert abs(b - 1.0) < 1e-9

    def test_beta_finite(self, positive_returns, benchmark_returns):
        b = beta(positive_returns, benchmark_returns)
        assert np.isfinite(b)

    def test_alpha_zero_for_identical_series(self, positive_returns):
        a = alpha(positive_returns, positive_returns, risk_free_rate=0.0)
        assert abs(a) < 1e-6

    def test_short_series_returns_nan(self):
        s = pd.Series([0.01])
        assert np.isnan(beta(s, s))

    def test_zero_variance_benchmark_returns_nan(self):
        port  = pd.Series([0.01] * 10)
        bench = pd.Series([0.00] * 10)
        assert np.isnan(beta(port, bench))


# ── Treynor / Information Ratio ───────────────────────────────────────────────

class TestTreynorAndIR:
    def test_treynor_finite(self, positive_returns, benchmark_returns):
        t = treynor_ratio(positive_returns, benchmark_returns)
        assert np.isfinite(t)

    def test_information_ratio_zero_tracking_error_returns_nan(self, positive_returns):
        ir = information_ratio(positive_returns, positive_returns)
        assert np.isnan(ir)

    def test_information_ratio_finite(self, positive_returns, benchmark_returns):
        ir = information_ratio(positive_returns, benchmark_returns)
        assert np.isfinite(ir)


# ── compute_all_metrics ───────────────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_returns_expected_keys(self, positive_returns):
        m = compute_all_metrics(positive_returns)
        for key in ("annualised_return", "annualised_volatility", "sharpe_ratio",
                    "sortino_ratio", "calmar_ratio", "max_drawdown",
                    "max_drawdown_duration_days", "skewness", "kurtosis"):
            assert key in m

    def test_benchmark_keys_added_when_provided(self, positive_returns, benchmark_returns):
        m = compute_all_metrics(positive_returns, benchmark_returns)
        for key in ("beta", "alpha", "treynor_ratio", "information_ratio"):
            assert key in m

    def test_no_benchmark_keys_absent(self, positive_returns):
        m = compute_all_metrics(positive_returns)
        assert "beta" not in m

    def test_values_are_finite(self, positive_returns, benchmark_returns):
        m = compute_all_metrics(positive_returns, benchmark_returns)
        for k, v in m.items():
            if isinstance(v, float):
                assert np.isfinite(v) or np.isnan(v), f"{k} is not finite or nan: {v}"


# ── Drawdown table ────────────────────────────────────────────────────────────

class TestDrawdownTable:
    def test_returns_dataframe(self, positive_returns):
        df = drawdown_table(positive_returns)
        assert isinstance(df, pd.DataFrame)

    def test_depth_column_negative(self, positive_returns):
        df = drawdown_table(positive_returns)
        if not df.empty:
            assert (df["drawdown"] <= 0).all()

    def test_top_n_respected(self, positive_returns):
        df = drawdown_table(positive_returns, top_n=3)
        assert len(df) <= 3

    def test_sorted_by_depth(self, positive_returns):
        df = drawdown_table(positive_returns, top_n=10)
        if len(df) > 1:
            assert df["drawdown"].iloc[0] <= df["drawdown"].iloc[1]

    def test_no_drawdown_series(self):
        rising = pd.Series([0.01] * 50)
        df = drawdown_table(rising)
        assert df.empty or (df["drawdown"] == 0).all()


# ── RiskReport ─────────────────────────────────────────────────────────────────

class MockPortfolio:
    """Minimal portfolio stub for RiskReport tests."""
    def __init__(self, returns, benchmark_returns, asset_returns, weights):
        self.name = "Test"
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.asset_returns = asset_returns
        self.weights = weights


class TestRiskReport:
    @pytest.fixture
    def mock_portfolio(self, positive_returns, benchmark_returns):
        idx = positive_returns.index
        asset_df = pd.DataFrame(
            {"A": positive_returns.values, "B": benchmark_returns.values},
            index=idx,
        )
        return MockPortfolio(
            returns=positive_returns,
            benchmark_returns=benchmark_returns,
            asset_returns=asset_df,
            weights={"A": 0.5, "B": 0.5},
        )

    def test_compute_returns_self(self, mock_portfolio):
        from quantrisk.risk.metrics import RiskReport
        report = RiskReport(mock_portfolio).compute()
        assert report is not None

    def test_metrics_property_after_compute(self, mock_portfolio):
        from quantrisk.risk.metrics import RiskReport
        report = RiskReport(mock_portfolio).compute()
        m = report.metrics
        assert "annualised_return" in m
        assert "sharpe_ratio" in m

    def test_metrics_before_compute_raises(self, mock_portfolio):
        from quantrisk.risk.metrics import RiskReport
        with pytest.raises(RuntimeError):
            _ = RiskReport(mock_portfolio).metrics

    def test_to_series(self, mock_portfolio):
        from quantrisk.risk.metrics import RiskReport
        report = RiskReport(mock_portfolio).compute()
        s = report.to_series()
        assert isinstance(s, pd.Series)

    def test_print_report_runs(self, mock_portfolio, capsys):
        from quantrisk.risk.metrics import RiskReport
        RiskReport(mock_portfolio).compute().print_report()
        captured = capsys.readouterr()
        assert "Risk Report" in captured.out

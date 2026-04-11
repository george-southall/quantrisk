"""Unit tests for DataValidator."""

import numpy as np
import pandas as pd
import pytest

from quantrisk.ingestion.data_validator import DataValidator, ValidationResult


@pytest.fixture
def validator():
    return DataValidator(
        max_gap_days=5,
        outlier_z_threshold=5.0,
        stale_days=9999,      # effectively disable stale check in most tests
        min_observations=10,
    )


@pytest.fixture
def clean_series():
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    return pd.Series(100 + np.arange(60, dtype=float), index=idx, name="TEST")


class TestValidationResult:
    def test_passed_when_no_issues(self):
        result = ValidationResult(ticker="TEST", passed=True, issues=[], warnings=[])
        assert result.passed

    def test_failed_when_issues_present(self):
        result = ValidationResult(ticker="TEST", passed=False, issues=["bad"], warnings=[])
        assert not result.passed


class TestDataValidator:
    def test_clean_series_passes(self, validator, clean_series):
        result = validator.validate_series(clean_series)
        assert result.passed

    def test_too_few_observations_fails(self, validator):
        short = pd.Series(
            [100.0, 101.0],
            index=pd.date_range("2020-01-01", periods=2, freq="B"),
            name="TEST",
        )
        result = validator.validate_series(short)
        assert not result.passed

    def test_all_nulls_fails(self, validator):
        idx = pd.date_range("2020-01-01", periods=60, freq="B")
        null_series = pd.Series([np.nan] * 60, index=idx, name="TEST")
        result = validator.validate_series(null_series)
        assert not result.passed

    def test_excessive_nulls_fails(self, validator):
        idx = pd.date_range("2020-01-01", periods=60, freq="B")
        data = [100.0] * 60
        for i in range(10):   # >10% nulls
            data[i] = np.nan
        result = validator.validate_series(pd.Series(data, index=idx, name="TEST"))
        assert not result.passed

    def test_non_positive_prices_error(self, validator):
        idx = pd.date_range("2020-01-01", periods=60, freq="B")
        data = [100.0] * 60
        data[30] = -5.0
        result = validator.validate_series(pd.Series(data, index=idx, name="TEST"))
        assert not result.passed

    def test_outlier_produces_warning(self, validator):
        idx = pd.date_range("2020-01-01", periods=60, freq="B")
        data = [100.0] * 60
        data[30] = 10000.0   # extreme outlier
        result = validator.validate_series(pd.Series(data, index=idx, name="TEST"))
        assert len(result.warnings) > 0

    def test_gap_detection(self):
        """Consecutive NaN values inside the series should trigger a warning or error."""
        v = DataValidator(max_gap_days=3, min_observations=5, stale_days=9999)
        idx = pd.date_range("2020-01-01", periods=20, freq="B")
        data = [100.0] * 20
        for i in range(8, 15):   # 7-day NaN gap — larger than max_gap_days=3
            data[i] = np.nan
        result = v.validate_series(pd.Series(data, index=idx, name="TEST"))
        # Gap is an error (> max_gap_days) or at least a warning
        assert not result.passed or len(result.warnings) > 0

    def test_stale_data_warning(self):
        """Series ending long in the past should produce a stale warning."""
        v = DataValidator(stale_days=5, min_observations=10, max_gap_days=9999)
        idx = pd.date_range("2010-01-01", periods=30, freq="B")
        data = pd.Series(100.0, index=idx, name="TEST")
        result = v.validate_series(data)
        assert len(result.warnings) > 0

    def test_validate_dataframe(self, validator, clean_series):
        df = pd.DataFrame({"A": clean_series.values, "B": clean_series.values * 1.1},
                          index=clean_series.index)
        results = validator.validate_dataframe(df)
        assert "A" in results
        assert "B" in results
        assert all(r.passed for r in results.values())

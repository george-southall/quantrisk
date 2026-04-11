"""Data quality checks for price series: gaps, outliers, stale data."""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    ticker: str
    passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.ticker}"]
        for issue in self.issues:
            lines.append(f"  ERROR:   {issue}")
        for warn in self.warnings:
            lines.append(f"  WARNING: {warn}")
        return "\n".join(lines)


class DataValidator:
    """
    Validates a price DataFrame for common data quality problems.

    Parameters
    ----------
    max_gap_days : int
        Maximum allowed consecutive missing trading days before flagging.
    outlier_z_threshold : float
        Z-score threshold for flagging daily return outliers.
    stale_days : int
        How many calendar days since last observation before flagging as stale.
    min_observations : int
        Minimum number of non-null observations required.
    """

    def __init__(
        self,
        max_gap_days: int = 5,
        outlier_z_threshold: float = 5.0,
        stale_days: int = 5,
        min_observations: int = 60,
    ):
        self.max_gap_days = max_gap_days
        self.outlier_z_threshold = outlier_z_threshold
        self.stale_days = stale_days
        self.min_observations = min_observations

    def validate_series(self, series: pd.Series) -> ValidationResult:
        """Validate a single price series."""
        ticker = str(series.name)
        issues: list[str] = []
        warnings: list[str] = []
        stats: dict = {}

        # --- Basic completeness ---
        n_total = len(series)
        n_null = series.isna().sum()
        n_valid = n_total - n_null
        stats["total_observations"] = n_total
        stats["null_count"] = int(n_null)
        stats["valid_count"] = n_valid

        if n_valid < self.min_observations:
            issues.append(
                f"Only {n_valid} valid observations (minimum: {self.min_observations})"
            )

        if n_null > 0:
            null_pct = n_null / n_total * 100
            msg = f"{n_null} null values ({null_pct:.1f}%)"
            (issues if null_pct > 10 else warnings).append(msg)

        # --- Consecutive gaps ---
        if not series.empty:
            gaps = self._find_gaps(series)
            if gaps:
                biggest = max(gaps)
                stats["max_gap_days"] = biggest
                msg = f"Largest gap: {biggest} consecutive missing days"
                (issues if biggest > self.max_gap_days else warnings).append(msg)

        # --- Stale data ---
        if not series.dropna().empty:
            last_date = series.dropna().index[-1]
            days_since = (pd.Timestamp(date.today()) - last_date).days
            stats["days_since_last"] = days_since
            if days_since > self.stale_days:
                warnings.append(f"Last observation is {days_since} days old")

        # --- Return outliers ---
        clean = series.dropna()
        if len(clean) > 2:
            returns = clean.pct_change().dropna()
            z_scores = (returns - returns.mean()) / returns.std()
            outlier_mask = z_scores.abs() > self.outlier_z_threshold
            n_outliers = int(outlier_mask.sum())
            stats["outlier_count"] = n_outliers
            if n_outliers > 0:
                outlier_dates = returns[outlier_mask].index.strftime("%Y-%m-%d").tolist()
                warnings.append(
                    f"{n_outliers} return outliers (|z| > {self.outlier_z_threshold}): "
                    f"{', '.join(outlier_dates[:5])}"
                    + (" ..." if n_outliers > 5 else "")
                )

        # --- Zero or negative prices ---
        non_positive = (clean <= 0).sum()
        if non_positive > 0:
            issues.append(f"{non_positive} non-positive price values")

        # --- Price stats ---
        if not clean.empty:
            stats["min_price"] = float(clean.min())
            stats["max_price"] = float(clean.max())
            stats["last_price"] = float(clean.iloc[-1])

        passed = len(issues) == 0
        result = ValidationResult(
            ticker=ticker, passed=passed, issues=issues, warnings=warnings, stats=stats
        )
        if passed:
            logger.info("Validation PASS: %s", ticker)
        else:
            logger.warning("Validation FAIL: %s — %s", ticker, "; ".join(issues))
        return result

    def validate_dataframe(self, prices: pd.DataFrame) -> dict[str, ValidationResult]:
        """Validate all columns in a price DataFrame."""
        results = {}
        for col in prices.columns:
            results[col] = self.validate_series(prices[col])
        return results

    def summary(self, results: dict[str, ValidationResult]) -> pd.DataFrame:
        """Return a tidy summary DataFrame of all validation results."""
        rows = []
        for ticker, r in results.items():
            row = {"ticker": ticker, "passed": r.passed}
            row.update(r.stats)
            row["issues"] = "; ".join(r.issues) if r.issues else ""
            row["warnings"] = "; ".join(r.warnings) if r.warnings else ""
            rows.append(row)
        return pd.DataFrame(rows).set_index("ticker")

    @staticmethod
    def _find_gaps(series: pd.Series) -> list[int]:
        """Return a list of consecutive-NaN run lengths."""
        gaps = []
        current = 0
        for val in series:
            if pd.isna(val):
                current += 1
            else:
                if current > 0:
                    gaps.append(current)
                current = 0
        if current > 0:
            gaps.append(current)
        return gaps

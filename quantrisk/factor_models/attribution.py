"""
Performance attribution using Fama-French factor models.

Decomposes portfolio returns into:
  - Alpha (daily intercept from OLS)
  - Factor contributions  (loading_i × realised_factor_return_i)
  - Residual              (actual excess return minus explained)
"""

from __future__ import annotations

import pandas as pd

from quantrisk.factor_models.fama_french import FamaFrenchModel

TRADING_DAYS_PER_YEAR = 252


class PerformanceAttribution:
    """
    Decompose portfolio returns into factor contributions and alpha.

    Usage
    -----
    model = FamaFrenchModel(n_factors=3).fit(portfolio_returns)
    attr  = PerformanceAttribution(model).compute()
    print(attr.summary())
    attr.print_summary()
    """

    def __init__(self, ff_model: FamaFrenchModel):
        if ff_model._result is None:
            raise RuntimeError("FamaFrenchModel must be fitted before attribution")
        self.ff_model = ff_model
        self._daily: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self) -> "PerformanceAttribution":
        """
        Compute daily factor contributions.

        For each day:
          alpha_contribution_t     = regression intercept (constant)
          factor_contribution_i_t  = loading_i × factor_return_i_t
          residual_t               = actual_excess_return_t
                                     - alpha - sum(factor_contributions)
        """
        result = self.ff_model._result
        aligned = self.ff_model._aligned   # columns: Rp-Rf, Mkt-RF, SMB, ...
        factors = self.ff_model._factors

        factor_cols = [c for c in aligned.columns if c != "Rp-Rf"]
        params = result.params            # includes "const" + factor loadings
        daily_alpha = float(params["const"])

        rows: dict[str, pd.Series] = {}
        rows["alpha"] = pd.Series(daily_alpha, index=aligned.index)
        for fc in factor_cols:
            rows[fc] = aligned[fc] * float(params[fc])

        attribution_df = pd.DataFrame(rows, index=aligned.index)

        explained = attribution_df.sum(axis=1)
        attribution_df["residual"] = aligned["Rp-Rf"] - explained
        attribution_df["actual_excess_return"] = aligned["Rp-Rf"]
        attribution_df["rf"] = factors.loc[aligned.index, "RF"]
        attribution_df["actual_total_return"] = (
            attribution_df["actual_excess_return"] + attribution_df["rf"]
        )

        self._daily = attribution_df
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def daily(self) -> pd.DataFrame:
        """Daily attribution DataFrame."""
        if self._daily is None:
            raise RuntimeError("Call .compute() first")
        return self._daily

    @property
    def factor_cols(self) -> list[str]:
        """Factor contribution column names (excludes meta columns)."""
        if self._daily is None:
            raise RuntimeError("Call .compute() first")
        exclude = {"residual", "actual_excess_return", "rf", "actual_total_return"}
        return [c for c in self._daily.columns if c not in exclude]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def periodic(self, freq: str = "ME") -> pd.DataFrame:
        """
        Compound daily attribution to a given calendar frequency.

        Uses (1 + r).prod() - 1 for each component column so compounding
        is consistent with how portfolio returns are reported.

        Parameters
        ----------
        freq : str
            Pandas offset alias: 'ME' monthly, 'QE' quarterly, 'YE' annual.
        """
        if self._daily is None:
            raise RuntimeError("Call .compute() first")
        return self._daily.resample(freq).apply(lambda x: (1 + x).prod() - 1)

    def summary(self) -> pd.Series:
        """
        Annualised attribution by component.

        Returns a Series with the annualised contribution of alpha, each
        factor, and the residual.
        """
        if self._daily is None:
            raise RuntimeError("Call .compute() first")

        n = len(self._daily)
        if n == 0:
            return pd.Series(dtype=float)

        ann: dict[str, float] = {}
        for col in self.factor_cols:
            total = float((1 + self._daily[col]).prod())
            ann[col] = float(total ** (TRADING_DAYS_PER_YEAR / n) - 1)

        # Residual
        total_resid = float((1 + self._daily["residual"]).prod())
        ann["residual"] = float(total_resid ** (TRADING_DAYS_PER_YEAR / n) - 1)

        return pd.Series(ann)

    def r_squared(self) -> float:
        """Fraction of return variance explained by the factor model."""
        return self.ff_model.r_squared

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a formatted attribution table to stdout."""
        s = self.summary()
        rep = self.ff_model.report()

        print(f"\n{'='*55}")
        print("  Performance Attribution  (Annualised)")
        print(f"  R² = {self.r_squared():.4f}")
        print(f"{'='*55}")
        print(f"  {'Component':<22} {'Contribution':>12}")
        print(f"  {'-'*36}")

        for col, val in s.items():
            desc = ""
            if col in rep.index and col != "alpha":
                desc = f"  ← {rep.loc[col, 'description']}" if "description" in rep.columns else ""
            label = col if col != "alpha" else "Alpha (skill)"
            print(f"  {label:<22} {val:>12.2%}{desc}")

        total_explained = s.drop("residual", errors="ignore").sum()
        print(f"  {'-'*36}")
        print(f"  {'Total (excl. residual)':<22} {total_explained:>12.2%}")
        print(f"{'='*55}\n")

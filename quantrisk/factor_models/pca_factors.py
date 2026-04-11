"""PCA-based statistical factor model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)


class PCAFactorModel:
    """
    Extract statistical factors from asset returns using PCA.

    The first PC typically represents the broad market factor.
    Subsequent PCs often represent sector or rates sensitivity.

    Usage
    -----
    model = PCAFactorModel(n_components=5).fit(asset_returns)
    model.print_report()
    """

    def __init__(self, n_components: int = 5, variance_threshold: float = 0.90):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self._pca: PCA | None = None
        self._scaler: StandardScaler | None = None
        self._asset_returns: pd.DataFrame | None = None
        self._factors: pd.DataFrame | None = None

    def fit(self, asset_returns: pd.DataFrame) -> "PCAFactorModel":
        """Fit PCA to the asset return matrix."""
        clean = asset_returns.dropna()
        self._asset_returns = clean

        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(clean)

        n_comp = min(self.n_components, clean.shape[1])
        self._pca = PCA(n_components=n_comp)
        factor_scores = self._pca.fit_transform(scaled)

        factor_names = [f"PC{i+1}" for i in range(n_comp)]
        self._factors = pd.DataFrame(factor_scores, index=clean.index, columns=factor_names)

        logger.info(
            "PCA fitted: %d components explain %.1f%% of variance",
            n_comp,
            self._pca.explained_variance_ratio_[:n_comp].sum() * 100,
        )
        return self

    @property
    def factors(self) -> pd.DataFrame:
        self._check_fitted()
        return self._factors

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        self._check_fitted()
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    @property
    def loadings(self) -> pd.DataFrame:
        """Factor loadings matrix: assets × PCs."""
        self._check_fitted()
        n_comp = self._pca.n_components_
        tickers = self._asset_returns.columns
        cols = [f"PC{i+1}" for i in range(n_comp)]
        return pd.DataFrame(
            self._pca.components_.T,
            index=tickers,
            columns=cols,
        )

    def n_components_for_variance(self) -> int:
        """Number of components needed to explain >= variance_threshold."""
        cum = self.cumulative_variance
        for i, v in enumerate(cum):
            if v >= self.variance_threshold:
                return i + 1
        return len(cum)

    def portfolio_factor_exposures(self, weights: dict[str, float]) -> pd.Series:
        """
        Compute the portfolio's exposure to each PC as a weighted average of loadings.
        """
        self._check_fitted()
        w = pd.Series(weights).reindex(self._asset_returns.columns).fillna(0)
        w = w / w.sum()
        return self.loadings.T @ w

    def print_report(self) -> None:
        self._check_fitted()
        n = self.n_components_for_variance()
        print(f"\n{'='*55}")
        print(f"  PCA Factor Model   ({len(self._asset_returns.columns)} assets)")
        print(f"  {n} components explain {self.variance_threshold:.0%} of variance")
        print(f"{'='*55}")
        print(f"  {'PC':<6} {'Expl. Var':>10} {'Cumulative':>12}")
        print(f"  {'-'*30}")
        for i, (ev, cv) in enumerate(
            zip(self.explained_variance_ratio, self.cumulative_variance)
        ):
            marker = " <--" if i + 1 == n else ""
            print(f"  PC{i+1:<4} {ev:>10.1%} {cv:>12.1%}{marker}")
        print("\n  Top asset loadings on PC1 (market factor):")
        pc1 = self.loadings["PC1"].sort_values(ascending=False)
        for ticker, val in pc1.items():
            bar = "+" * int(abs(val) * 20) if val > 0 else "-" * int(abs(val) * 20)
            print(f"    {ticker:<8} {val:>6.3f}  {bar}")
        print()

    def _check_fitted(self) -> None:
        if self._pca is None:
            raise RuntimeError("Call .fit() first")

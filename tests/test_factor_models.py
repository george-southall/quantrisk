"""Unit tests for PCA factor model (Fama-French requires network access)."""

import pytest

from quantrisk.factor_models.pca_factors import PCAFactorModel


class TestPCAFactorModel:
    def test_fit_produces_factors(self, asset_returns):
        model = PCAFactorModel(n_components=3).fit(asset_returns)
        assert model.factors.shape == (len(asset_returns.dropna()), 3)

    def test_loadings_shape(self, asset_returns):
        model = PCAFactorModel(n_components=3).fit(asset_returns)
        assert model.loadings.shape == (4, 3)  # 4 assets, 3 PCs

    def test_explained_variance_sums_to_one(self, asset_returns):
        model = PCAFactorModel(n_components=4).fit(asset_returns)
        assert abs(model.explained_variance_ratio.sum() - 1.0) < 1e-6

    def test_cumulative_variance_monotone(self, asset_returns):
        model = PCAFactorModel(n_components=4).fit(asset_returns)
        cv = model.cumulative_variance
        assert all(cv[i] <= cv[i + 1] for i in range(len(cv) - 1))

    def test_n_components_for_variance(self, asset_returns):
        model = PCAFactorModel(n_components=4, variance_threshold=0.90).fit(asset_returns)
        n = model.n_components_for_variance()
        assert 1 <= n <= 4
        assert model.cumulative_variance[n - 1] >= 0.90

    def test_portfolio_exposures(self, asset_returns, sample_weights):
        model = PCAFactorModel(n_components=3).fit(asset_returns)
        exp = model.portfolio_factor_exposures(sample_weights)
        assert len(exp) == 3

    def test_unfit_model_raises(self):
        model = PCAFactorModel()
        with pytest.raises(RuntimeError, match="fit()"):
            _ = model.factors

    def test_n_components_capped_at_n_assets(self, asset_returns):
        model = PCAFactorModel(n_components=100).fit(asset_returns)
        assert model.factors.shape[1] <= asset_returns.shape[1]

    def test_factors_index_matches_returns(self, asset_returns):
        model = PCAFactorModel(n_components=2).fit(asset_returns)
        assert model.factors.index.equals(asset_returns.dropna().index)

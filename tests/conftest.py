"""Shared pytest fixtures for the QuantRisk test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def normal_returns(rng):
    """250 days of normally distributed daily returns, mean 0.0005, std 0.01."""
    data = rng.normal(loc=0.0005, scale=0.01, size=250)
    idx = pd.bdate_range("2020-01-01", periods=250)
    return pd.Series(data, index=idx, name="port")


@pytest.fixture
def asset_returns(rng):
    """250 days of correlated returns for 4 assets."""
    cov = np.array([
        [0.0001, 0.00005, 0.00002, -0.00001],
        [0.00005, 0.00015, 0.00003, -0.00001],
        [0.00002, 0.00003, 0.00008, 0.000005],
        [-0.00001, -0.00001, 0.000005, 0.00006],
    ])
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((250, 4))
    data = 0.0003 + Z @ L.T
    idx = pd.bdate_range("2020-01-01", periods=250)
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C", "D"])


@pytest.fixture
def sample_weights():
    return {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


@pytest.fixture
def prices_df(asset_returns):
    """Convert returns to price levels starting at 100."""
    return (1 + asset_returns).cumprod() * 100

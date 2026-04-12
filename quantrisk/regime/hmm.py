"""
Hidden Markov Model regime detection for portfolio return series.

Fits a Gaussian HMM to identify latent market regimes (bull / bear / volatile).
States are relabelled after fitting so the lowest-mean state is always "Bear"
and the highest-mean state is always "Bull", making results reproducible
across random initialisations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm

from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252

_REGIME_COLOURS = {
    "Bull": "#00CC96",
    "Bear": "#EF553B",
    "Volatile": "#FFA15A",
}


def _make_labels(n_regimes: int) -> dict[int, str]:
    if n_regimes == 2:
        return {0: "Bear", 1: "Bull"}
    return {0: "Bear", 1: "Volatile", 2: "Bull"}


def _prepare_features(returns: pd.Series, n_regimes: int) -> tuple[np.ndarray, pd.Index]:
    """
    Return (X, index) where X is the 2D feature array passed to hmmlearn.

    For 2 regimes: single feature = return.
    For 3 regimes: two features = [return, rolling-5d vol] to help distinguish
    the volatile regime from directional regimes.
    """
    clean = returns.dropna()
    if n_regimes == 2:
        X = clean.values.reshape(-1, 1)
    else:
        vol5 = clean.rolling(5).std().fillna(clean.std())
        X = np.column_stack([clean.values, vol5.values])
    return X.astype(float), clean.index


def _relabel_by_mean_return(
    states: np.ndarray,
    means: np.ndarray,
    labels: dict[int, str],
) -> np.ndarray:
    """
    Remap raw HMM state integers so that state 0 = lowest mean return (Bear),
    state (n-1) = highest mean return (Bull).
    """
    # means shape: (n_regimes, n_features) — use first feature (return) for ordering
    order = np.argsort(means[:, 0])          # ascending mean return
    remap = {old: new for new, old in enumerate(order)}
    return np.vectorize(remap.__getitem__)(states)


def fit_hmm(
    returns: pd.Series,
    n_regimes: int = 2,
    n_iter: int = 200,
    random_seed: int = 42,
) -> tuple[hmm.GaussianHMM, np.ndarray, pd.Index]:
    """
    Fit a Gaussian HMM to the return series.

    Returns
    -------
    model      : fitted GaussianHMM
    states     : integer state array (relabelled: 0=Bear, 1=Bull or 1=Volatile, 2=Bull)
    index      : DatetimeIndex aligned to the state array
    """
    if n_regimes not in (2, 3):
        raise ValueError("n_regimes must be 2 or 3")

    X, index = _prepare_features(returns, n_regimes)

    cov_type = "full" if n_regimes == 2 else "diag"
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type=cov_type,
        n_iter=n_iter,
        random_state=random_seed,
        verbose=False,
    )
    model.fit(X)

    raw_states = model.predict(X)
    states = _relabel_by_mean_return(raw_states, model.means_, _make_labels(n_regimes))

    logger.info(
        "HMM fitted: %d regimes, converged=%s, score=%.2f",
        n_regimes,
        model.monitor_.converged,
        model.score(X),
    )
    return model, states, index


def get_regime_series(
    returns: pd.Series,
    n_regimes: int = 2,
    n_iter: int = 200,
) -> pd.Series:
    """
    Fit HMM and return a string-labelled pd.Series aligned to the return index.

    Values are: 'Bear', 'Bull' (for n_regimes=2)
             or 'Bear', 'Volatile', 'Bull' (for n_regimes=3).
    """
    _, states, index = fit_hmm(returns, n_regimes=n_regimes, n_iter=n_iter)
    labels = _make_labels(n_regimes)
    labelled = pd.Series(
        [labels[s] for s in states],
        index=index,
        name="regime",
    )
    return labelled


def regime_statistics(
    returns: pd.Series,
    regime_series: pd.Series,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Per-regime summary statistics.

    Columns: mean_return, volatility, sharpe_ratio, avg_duration_days,
             pct_time, is_current_regime.
    """
    aligned = pd.concat([returns, regime_series], axis=1).dropna()
    aligned.columns = ["return", "regime"]

    rows = []
    regimes = aligned["regime"].unique()

    for regime in sorted(regimes, key=lambda r: {"Bear": 0, "Volatile": 1, "Bull": 2}.get(r, 9)):
        mask = aligned["regime"] == regime
        r = aligned.loc[mask, "return"]

        if len(r) > 0:
            ann_ret = float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
        else:
            ann_ret = float("nan")
        ann_vol = float(r.std() * np.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan")
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else float("nan")

        # Average duration: length of consecutive same-regime runs
        # Keep only runs where the regime actually matches
        regime_runs = aligned.groupby(
            (aligned["regime"] != aligned["regime"].shift()).cumsum()
        ).filter(lambda g: g["regime"].iloc[0] == regime)

        if not regime_runs.empty:
            run_groups = regime_runs.groupby(
                (aligned["regime"] != aligned["regime"].shift())
                .reindex(regime_runs.index)
                .cumsum()
            )
            avg_dur = float(np.mean([len(g) for _, g in run_groups]))
        else:
            avg_dur = float("nan")

        rows.append({
            "regime": regime,
            "mean_return": ann_ret,
            "volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "avg_duration_days": avg_dur,
            "pct_time": float(mask.sum() / len(aligned)),
            "is_current": regime == regime_series.iloc[-1],
        })

    return pd.DataFrame(rows).set_index("regime")


def current_regime(regime_series: pd.Series) -> str:
    """Return the label of the most recent regime observation."""
    return str(regime_series.iloc[-1])


def regime_colour(label: str) -> str:
    """Return the Plotly hex colour for a regime label."""
    return _REGIME_COLOURS.get(label, "#999999")

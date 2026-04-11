"""Hypothetical / user-defined stress shocks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class HypotheticalShockResult:
    shocks: dict[str, float]
    weights: dict[str, float]
    ticker_pl: dict[str, float]
    portfolio_pl: float

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "ticker": t,
                "weight": self.weights.get(t, 0),
                "shock": self.shocks.get(t, 0),
                "pl_contribution": self.ticker_pl.get(t, 0),
            }
            for t in set(list(self.weights.keys()) + list(self.shocks.keys()))
        ]
        return pd.DataFrame(rows).sort_values("pl_contribution")


def apply_hypothetical_shocks(
    weights: dict[str, float],
    shocks: dict[str, float],
    default_shock: float = 0.0,
) -> HypotheticalShockResult:
    """
    Apply user-defined percentage shocks to the portfolio.

    Parameters
    ----------
    weights : dict[str, float]
        Portfolio weights.
    shocks : dict[str, float]
        Fractional shocks per ticker (e.g. {'AAPL': -0.20, 'GLD': 0.10}).
        Tickers not listed receive `default_shock`.
    default_shock : float
        Shock applied to any portfolio asset not in `shocks` dict.
    """
    ticker_pl: dict[str, float] = {}
    for ticker, weight in weights.items():
        shock = shocks.get(ticker, default_shock)
        ticker_pl[ticker] = weight * shock

    portfolio_pl = sum(ticker_pl.values())
    return HypotheticalShockResult(
        shocks={t: shocks.get(t, default_shock) for t in weights},
        weights=weights,
        ticker_pl=ticker_pl,
        portfolio_pl=portfolio_pl,
    )

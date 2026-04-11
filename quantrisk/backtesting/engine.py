"""Walk-forward backtesting engine with transaction costs and slippage."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quantrisk.config import settings
from quantrisk.portfolio.returns import (
    annualised_return,
    annualised_volatility,
    max_drawdown,
    max_drawdown_duration,
)
from quantrisk.risk.metrics import sharpe_ratio, sortino_ratio, calmar_ratio
from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


class BacktestResult:
    """Holds the output of a single strategy backtest."""

    def __init__(
        self,
        strategy_name: str,
        returns: pd.Series,
        weights_history: pd.DataFrame,
        turnover: pd.Series,
    ):
        self.strategy_name = strategy_name
        self.returns = returns
        self.weights_history = weights_history
        self.turnover = turnover
        self._metrics: dict | None = None

    @property
    def cumulative_returns(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    def metrics(self, risk_free_rate: float = settings.risk_free_rate_fallback) -> dict:
        if self._metrics is None:
            r = self.returns
            self._metrics = {
                "strategy": self.strategy_name,
                "annualised_return": annualised_return(r),
                "annualised_volatility": annualised_volatility(r),
                "sharpe_ratio": sharpe_ratio(r, risk_free_rate),
                "sortino_ratio": sortino_ratio(r, risk_free_rate),
                "calmar_ratio": calmar_ratio(r),
                "max_drawdown": max_drawdown(r),
                "max_drawdown_duration_days": max_drawdown_duration(r),
                "avg_monthly_turnover": float(
                    self.turnover.resample("ME").sum().mean()
                    if not self.turnover.empty
                    else 0
                ),
                "win_rate": float((r > 0).mean()),
                "best_day": float(r.max()),
                "worst_day": float(r.min()),
            }
        return self._metrics


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Estimates portfolio weights using a lookback window of historical returns,
    then steps forward one rebalance period at a time to compute out-of-sample
    portfolio returns.

    Parameters
    ----------
    estimation_window : int
        Number of trading days used to estimate weights (e.g. 252 = 1 year).
    rebalance_freq : str
        Pandas offset alias for rebalancing frequency: 'ME', 'QE', 'W', 'D'.
    transaction_cost_bps : float
        One-way transaction cost in basis points.
    slippage_bps : float
        Bid/ask slippage in basis points (applied on top of transaction costs).
    """

    def __init__(
        self,
        estimation_window: int = 252,
        rebalance_freq: str = "ME",
        transaction_cost_bps: float = settings.transaction_cost_bps,
        slippage_bps: float = settings.slippage_bps,
    ):
        self.estimation_window = estimation_window
        self.rebalance_freq = rebalance_freq
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps

    def run(
        self,
        asset_returns: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame], dict[str, float]],
        strategy_name: str = "Strategy",
    ) -> BacktestResult:
        """
        Run a walk-forward backtest for a single strategy.

        Parameters
        ----------
        asset_returns : pd.DataFrame
            Full history of daily asset returns.
        strategy_fn : callable
            Function that takes a pd.DataFrame of returns and returns weights dict.
        strategy_name : str
            Label for the strategy.
        """
        clean = asset_returns.dropna()
        tickers = clean.columns.tolist()
        n = len(clean)

        if n <= self.estimation_window:
            raise ValueError(
                f"Insufficient data: need >{self.estimation_window} rows, got {n}"
            )

        # Build rebalance dates
        rebalance_dates = pd.date_range(
            start=clean.index[self.estimation_window],
            end=clean.index[-1],
            freq=self.rebalance_freq,
        )

        portfolio_returns: list[pd.Series] = []
        weights_history: dict[pd.Timestamp, dict] = {}
        turnover_list: dict[pd.Timestamp, float] = {}

        current_weights = {t: 1.0 / len(tickers) for t in tickers}
        prev_date = clean.index[self.estimation_window]

        for rb_date in rebalance_dates:
            # Estimation window: all data up to (not including) this rebalance
            est_end_idx = clean.index.get_loc(
                clean.index[clean.index <= rb_date][-1]
            )
            est_start_idx = max(0, est_end_idx - self.estimation_window)
            est_returns = clean.iloc[est_start_idx:est_end_idx]

            if len(est_returns) < 20:
                continue

            try:
                new_weights = strategy_fn(est_returns)
            except Exception as exc:
                logger.warning("Strategy %s failed at %s: %s", strategy_name, rb_date, exc)
                new_weights = current_weights

            # Normalise
            total = sum(new_weights.values())
            new_weights = {t: v / total for t, v in new_weights.items() if total > 0}

            # Compute turnover
            to = sum(
                abs(new_weights.get(t, 0) - current_weights.get(t, 0)) for t in tickers
            ) / 2
            turnover_list[rb_date] = to

            # Transaction costs (applied as a drag on the rebalance-day return)
            cost_per_unit = (self.transaction_cost_bps + self.slippage_bps) / 10_000
            cost = to * cost_per_unit

            # Hold the current weights for the out-of-sample period
            try:
                next_rb_idx = rebalance_dates.get_loc(rb_date) + 1
                if next_rb_idx < len(rebalance_dates):
                    oos_end = rebalance_dates[next_rb_idx]
                else:
                    oos_end = clean.index[-1]
            except Exception:
                oos_end = clean.index[-1]

            oos_mask = (clean.index > rb_date) & (clean.index <= oos_end)
            oos_returns = clean.loc[oos_mask]

            if not oos_returns.empty:
                w_series = pd.Series(current_weights).reindex(tickers).fillna(0)
                port_oos = oos_returns[tickers].fillna(0) @ w_series
                # Subtract cost on the first day
                port_oos.iloc[0] -= cost
                portfolio_returns.append(port_oos)

            weights_history[rb_date] = new_weights
            current_weights = new_weights

        if not portfolio_returns:
            raise RuntimeError(f"No out-of-sample returns generated for {strategy_name}")

        full_returns = pd.concat(portfolio_returns).sort_index()
        full_returns = full_returns[~full_returns.index.duplicated(keep="first")]
        full_returns.name = strategy_name

        weights_df = pd.DataFrame(weights_history).T.fillna(0)
        turnover_series = pd.Series(turnover_list, name="turnover")

        logger.info(
            "Backtest %s: %d days, ann. return=%.2f%%, Sharpe=%.2f",
            strategy_name,
            len(full_returns),
            annualised_return(full_returns) * 100,
            sharpe_ratio(full_returns),
        )

        return BacktestResult(
            strategy_name=strategy_name,
            returns=full_returns,
            weights_history=weights_df,
            turnover=turnover_series,
        )

    def run_all(
        self,
        asset_returns: pd.DataFrame,
        strategies: dict[str, Callable],
        risk_free_rate: float = settings.risk_free_rate_fallback,
    ) -> dict[str, BacktestResult]:
        """Run multiple strategies and return results keyed by name."""
        results = {}
        for name, fn in strategies.items():
            logger.info("Running backtest: %s", name)
            try:
                results[name] = self.run(asset_returns, fn, name)
            except Exception as exc:
                logger.error("Backtest failed for %s: %s", name, exc)
        return results

    @staticmethod
    def comparison_table(
        results: dict[str, BacktestResult],
        risk_free_rate: float = settings.risk_free_rate_fallback,
    ) -> pd.DataFrame:
        """Return a side-by-side comparison DataFrame of all strategy metrics."""
        rows = [r.metrics(risk_free_rate) for r in results.values()]
        return pd.DataFrame(rows).set_index("strategy")

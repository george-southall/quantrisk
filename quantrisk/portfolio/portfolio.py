"""Core Portfolio class: weights, P&L, rebalancing, benchmark comparison."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quantrisk.config import settings
from quantrisk.ingestion.market_data import fetch_prices
from quantrisk.portfolio.returns import (
    annualised_return,
    annualised_volatility,
    cumulative_returns,
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    rolling_annualised_return,
    rolling_annualised_volatility,
    simple_returns,
)
from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


class Portfolio:
    """
    Represents a static-weight (or periodically rebalanced) multi-asset portfolio.

    Parameters
    ----------
    weights : dict[str, float]
        Mapping of ticker → target weight. Weights are normalised to sum to 1.
    start_date : str
        First date of data, format 'YYYY-MM-DD'.
    end_date : str | None
        Last date of data (defaults to today).
    benchmark : str
        Ticker used as benchmark (default SPY).
    rebalance_freq : str | None
        Pandas offset alias for rebalancing (e.g. 'ME' monthly, 'QE' quarterly).
        None means buy-and-hold (weights drift with prices).
    name : str
        Display name for the portfolio.
    """

    def __init__(
        self,
        weights: dict[str, float],
        start_date: str = settings.default_start_date,
        end_date: str | None = None,
        benchmark: str = settings.default_benchmark,
        rebalance_freq: str | None = None,
        name: str = "Portfolio",
    ):
        if not weights:
            raise ValueError("weights must be non-empty")

        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive number")
        self.weights: dict[str, float] = {k: v / total for k, v in weights.items()}
        self.tickers = list(self.weights.keys())
        self.start_date = start_date
        self.end_date = end_date or date.today().isoformat()
        self.benchmark = benchmark
        self.rebalance_freq = rebalance_freq
        self.name = name

        # Populated on load()
        self._prices: pd.DataFrame | None = None
        self._benchmark_prices: pd.Series | None = None
        self._asset_returns: pd.DataFrame | None = None
        self._portfolio_returns: pd.Series | None = None
        self._benchmark_returns: pd.Series | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self, use_cache: bool = True) -> "Portfolio":
        """Fetch price data for all assets and benchmark, compute returns."""
        all_tickers = list(set(self.tickers + [self.benchmark]))
        logger.info(
            "Loading %d tickers from %s to %s", len(all_tickers), self.start_date, self.end_date
        )

        prices = fetch_prices(all_tickers, self.start_date, self.end_date, use_cache=use_cache)

        # Split benchmark from assets
        bench_col = self.benchmark if self.benchmark in prices.columns else None
        if bench_col:
            self._benchmark_prices = prices[bench_col]
            prices = prices.drop(columns=[bench_col], errors="ignore")

        # Align tickers actually returned — gracefully skip any that failed to download
        available = [t for t in self.tickers if t in prices.columns]
        missing = set(self.tickers) - set(available)
        if missing:
            logger.warning("Tickers not available in price data: %s", missing)

        self._prices = prices[available].copy()
        self._asset_returns = simple_returns(self._prices).dropna(how="all")

        # Reindex weights to available tickers, renormalise
        w = {t: self.weights[t] for t in available}
        total = sum(w.values())
        self.weights = {t: v / total for t, v in w.items()}

        # Portfolio returns
        self._portfolio_returns = self._compute_portfolio_returns()

        if self._benchmark_prices is not None:
            self._benchmark_returns = simple_returns(self._benchmark_prices).dropna()

        logger.info("Loaded %d trading days of data", len(self._portfolio_returns))
        return self

    # ------------------------------------------------------------------
    # Return computations
    # ------------------------------------------------------------------

    def _compute_portfolio_returns(self) -> pd.Series:
        """Compute daily portfolio returns respecting rebalancing frequency."""
        if self._asset_returns is None:
            raise RuntimeError("Call .load() first")

        weight_series = pd.Series(self.weights)

        if self.rebalance_freq is None:
            # Buy-and-hold: let weights drift
            port_returns = (self._asset_returns * weight_series).sum(axis=1)
        else:
            # Periodic rebalancing: reset weights at each rebalance date
            rebalance_dates = pd.date_range(
                start=self._asset_returns.index[0],
                end=self._asset_returns.index[-1],
                freq=self.rebalance_freq,
            )
            segments = []
            prev_date = self._asset_returns.index[0]
            for rb_date in rebalance_dates:
                segment = self._asset_returns.loc[prev_date:rb_date]
                seg_returns = (segment * weight_series).sum(axis=1)
                segments.append(seg_returns)
                prev_date = rb_date + pd.Timedelta(days=1)
            # Remainder after last rebalance
            remainder = self._asset_returns.loc[prev_date:]
            if not remainder.empty:
                segments.append((remainder * weight_series).sum(axis=1))

            port_returns = pd.concat(segments).sort_index()
            port_returns = port_returns[~port_returns.index.duplicated(keep="first")]

        port_returns.name = self.name
        return port_returns

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def prices(self) -> pd.DataFrame:
        self._check_loaded()
        return self._prices

    @property
    def asset_returns(self) -> pd.DataFrame:
        self._check_loaded()
        return self._asset_returns

    @property
    def returns(self) -> pd.Series:
        self._check_loaded()
        return self._portfolio_returns

    @property
    def benchmark_returns(self) -> pd.Series | None:
        return self._benchmark_returns

    @property
    def cumulative_returns(self) -> pd.Series:
        return cumulative_returns(self.returns)

    @property
    def covariance_matrix(self) -> pd.DataFrame:
        """Annualised covariance matrix of asset returns."""
        return self.asset_returns.cov() * TRADING_DAYS_PER_YEAR

    @property
    def correlation_matrix(self) -> pd.DataFrame:
        return self.asset_returns.corr()

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------

    def summary(self, risk_free_rate: float | None = None) -> dict:
        """
        Return a dictionary of key portfolio statistics.

        Parameters
        ----------
        risk_free_rate : float | None
            Annualised risk-free rate (decimal). Defaults to config fallback.
        """
        self._check_loaded()
        rf = risk_free_rate if risk_free_rate is not None else settings.risk_free_rate_fallback

        ret = self.returns
        ann_ret = annualised_return(ret)
        ann_vol = annualised_volatility(ret)
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")
        mdd = max_drawdown(ret)

        result = {
            "name": self.name,
            "tickers": self.tickers,
            "start_date": str(ret.index[0].date()),
            "end_date": str(ret.index[-1].date()),
            "trading_days": len(ret),
            "annualised_return": ann_ret,
            "annualised_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "max_drawdown_duration_days": max_drawdown_duration(ret),
        }

        if self._benchmark_returns is not None:
            b_ret = self._benchmark_returns.reindex(ret.index).dropna()
            aligned_port = ret.reindex(b_ret.index).dropna()
            if len(b_ret) > 1 and len(aligned_port) > 1:
                cov = np.cov(aligned_port, b_ret)
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else float("nan")
                bench_ann = annualised_return(b_ret)
                result["benchmark"] = self.benchmark
                result["benchmark_annualised_return"] = bench_ann
                result["beta"] = beta
                result["alpha"] = ann_ret - (rf + beta * (bench_ann - rf))

        return result

    def rolling_stats(self, window: int = 252) -> pd.DataFrame:
        """Rolling annualised return and volatility."""
        self._check_loaded()
        return pd.DataFrame({
            "rolling_return": rolling_annualised_return(self.returns, window),
            "rolling_volatility": rolling_annualised_volatility(self.returns, window),
            "drawdown": drawdown_series(self.returns),
        })

    def weight_series(self) -> pd.Series:
        """Return a Series of ticker weights, sorted descending."""
        return pd.Series(self.weights).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_summary(self, risk_free_rate: float | None = None) -> None:
        stats = self.summary(risk_free_rate)
        print(f"\n{'='*60}")
        print(f"  {stats['name']}")
        print(f"  {stats['start_date']} → {stats['end_date']}  ({stats['trading_days']} days)")
        print(f"{'='*60}")
        print(f"  Annualised Return:     {stats['annualised_return']:>8.2%}")
        print(f"  Annualised Volatility: {stats['annualised_volatility']:>8.2%}")
        print(f"  Sharpe Ratio:          {stats['sharpe_ratio']:>8.2f}")
        print(f"  Max Drawdown:          {stats['max_drawdown']:>8.2%}")
        print(f"  Max DD Duration:       {stats['max_drawdown_duration_days']:>5d} days")
        if "beta" in stats:
            print(f"  Beta vs {stats['benchmark']}:      {stats['beta']:>8.2f}")
            print(f"  Alpha (Jensen's):      {stats['alpha']:>8.2%}")
        print(f"{'='*60}")
        print("  Weights:")
        for ticker, w in self.weight_series().items():
            print(f"    {ticker:<8} {w:>6.1%}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_loaded(self) -> None:
        if self._prices is None:
            raise RuntimeError(
                "Portfolio data not loaded. Call portfolio.load() first."
            )

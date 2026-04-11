"""Risk-adjusted performance metrics: Sharpe, Sortino, Calmar, Treynor, Beta."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantrisk.config import settings
from quantrisk.portfolio.returns import (
    annualised_return,
    annualised_volatility,
    downside_deviation,
    max_drawdown,
)

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = settings.risk_free_rate_fallback,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio = (Rp - Rf) / σp."""
    ann_ret = annualised_return(returns, periods_per_year)
    ann_vol = annualised_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return float("nan")
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = settings.risk_free_rate_fallback,
    threshold: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Sortino ratio = (Rp - Rf) / downside_deviation.

    Uses only downside volatility, penalising losses without penalising upside.
    """
    ann_ret = annualised_return(returns, periods_per_year)
    dd = downside_deviation(returns, threshold, periods_per_year)
    if dd == 0:
        return float("nan")
    return (ann_ret - risk_free_rate) / dd


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calmar ratio = annualised return / |max drawdown|."""
    ann_ret = annualised_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("nan")
    return ann_ret / mdd


def beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Portfolio beta vs benchmark.

    β = Cov(Rp, Rb) / Var(Rb)
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    bench_var = cov_matrix[1, 1]
    if bench_var == 0:
        return float("nan")
    return float(cov_matrix[0, 1] / bench_var)


def alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = settings.risk_free_rate_fallback,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Jensen's alpha = Rp - [Rf + β(Rb - Rf)].

    Excess return not explained by market exposure.
    """
    b = beta(portfolio_returns, benchmark_returns)
    rp = annualised_return(portfolio_returns, periods_per_year)
    rb = annualised_return(benchmark_returns, periods_per_year)
    return rp - (risk_free_rate + b * (rb - risk_free_rate))


def treynor_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = settings.risk_free_rate_fallback,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Treynor ratio = (Rp - Rf) / β."""
    b = beta(portfolio_returns, benchmark_returns)
    if b == 0 or np.isnan(b):
        return float("nan")
    ann_ret = annualised_return(portfolio_returns, periods_per_year)
    return (ann_ret - risk_free_rate) / b


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Information ratio = active return / tracking error.

    Measures consistency of outperformance over the benchmark.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active.std() * np.sqrt(periods_per_year)
    if tracking_error == 0:
        return float("nan")
    active_return = annualised_return(aligned.iloc[:, 0], periods_per_year) - annualised_return(
        aligned.iloc[:, 1], periods_per_year
    )
    return active_return / tracking_error


def compute_all_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = settings.risk_free_rate_fallback,
) -> dict:
    """Compute a comprehensive suite of risk-adjusted performance metrics."""
    from quantrisk.portfolio.returns import max_drawdown_duration

    metrics = {
        "annualised_return": annualised_return(portfolio_returns),
        "annualised_volatility": annualised_volatility(portfolio_returns),
        "sharpe_ratio": sharpe_ratio(portfolio_returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(portfolio_returns, risk_free_rate),
        "calmar_ratio": calmar_ratio(portfolio_returns),
        "max_drawdown": max_drawdown(portfolio_returns),
        "max_drawdown_duration_days": max_drawdown_duration(portfolio_returns),
        "skewness": float(portfolio_returns.skew()),
        "kurtosis": float(portfolio_returns.kurt()),
    }

    if benchmark_returns is not None:
        metrics["beta"] = beta(portfolio_returns, benchmark_returns)
        metrics["alpha"] = alpha(portfolio_returns, benchmark_returns, risk_free_rate)
        metrics["treynor_ratio"] = treynor_ratio(
            portfolio_returns, benchmark_returns, risk_free_rate
        )
        metrics["information_ratio"] = information_ratio(portfolio_returns, benchmark_returns)

    return metrics


class RiskReport:
    """
    Aggregates all risk metrics for a Portfolio into a structured report.

    Usage:
        report = RiskReport(portfolio).compute()
        print(report.to_dataframe())
    """

    def __init__(self, portfolio, risk_free_rate: float | None = None):
        self.portfolio = portfolio
        self.risk_free_rate = (
            risk_free_rate if risk_free_rate is not None else settings.risk_free_rate_fallback
        )
        self._metrics: dict | None = None
        self._var_table: pd.DataFrame | None = None
        self._cvar_table: pd.DataFrame | None = None

    def compute(self) -> "RiskReport":
        from quantrisk.risk.cvar import cvar_summary
        from quantrisk.risk.var import var_summary

        port_returns = self.portfolio.returns
        bench_returns = self.portfolio.benchmark_returns

        self._metrics = compute_all_metrics(port_returns, bench_returns, self.risk_free_rate)

        self._var_table = var_summary(
            port_returns,
            asset_returns=self.portfolio.asset_returns,
            weights=self.portfolio.weights,
        )
        self._cvar_table = cvar_summary(port_returns)

        return self

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            raise RuntimeError("Call .compute() first")
        return self._metrics

    def to_series(self) -> pd.Series:
        return pd.Series(self.metrics)

    def print_report(self) -> None:
        m = self.metrics
        print(f"\n{'='*55}")
        print(f"  Risk Report: {self.portfolio.name}")
        print(f"{'='*55}")
        print(f"  Annualised Return:        {m['annualised_return']:>8.2%}")
        print(f"  Annualised Volatility:    {m['annualised_volatility']:>8.2%}")
        print(f"  Sharpe Ratio:             {m['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:            {m['sortino_ratio']:>8.2f}")
        print(f"  Calmar Ratio:             {m['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:             {m['max_drawdown']:>8.2%}")
        print(f"  Max DD Duration:          {m['max_drawdown_duration_days']:>5d} days")
        print(f"  Skewness:                 {m['skewness']:>8.2f}")
        print(f"  Excess Kurtosis:          {m['kurtosis']:>8.2f}")
        if "beta" in m:
            print(f"  Beta:                     {m['beta']:>8.2f}")
            print(f"  Jensen's Alpha:           {m['alpha']:>8.2%}")
            print(f"  Treynor Ratio:            {m['treynor_ratio']:>8.2f}")
            print(f"  Information Ratio:        {m['information_ratio']:>8.2f}")
        print(f"{'='*55}\n")
        if self._var_table is not None:
            print("  VaR Table:")
            print(self._var_table.to_string(index=False))
        print()

"""Backtest tearsheet evaluator: comparison table, annual returns, monthly heatmap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantrisk.backtesting.engine import BacktestResult
from quantrisk.config import settings
from quantrisk.portfolio.returns import (
    drawdown_series,
    rolling_annualised_return,
    rolling_annualised_volatility,
)


class TearsheetEvaluator:
    """
    Aggregates multiple BacktestResult objects for side-by-side comparison.

    Parameters
    ----------
    results : dict[str, BacktestResult]
        Strategy name → BacktestResult mapping (from BacktestEngine.run_all).
    benchmark_returns : pd.Series | None
        Optional buy-and-hold benchmark for relative stats and charts.
    risk_free_rate : float
        Annualised risk-free rate used in ratio calculations.
    """

    def __init__(
        self,
        results: dict[str, BacktestResult],
        benchmark_returns: pd.Series | None = None,
        risk_free_rate: float = settings.risk_free_rate_fallback,
    ):
        if not results:
            raise ValueError("results must be non-empty")
        self.results = results
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def comparison_table(self) -> pd.DataFrame:
        """Side-by-side metrics for all strategies."""
        rows = [r.metrics(self.risk_free_rate) for r in self.results.values()]
        df = pd.DataFrame(rows).set_index("strategy")
        ordered = [
            "annualised_return", "annualised_volatility", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown",
            "max_drawdown_duration_days", "avg_monthly_turnover",
            "win_rate", "best_day", "worst_day",
        ]
        return df[[c for c in ordered if c in df.columns]]

    # ------------------------------------------------------------------
    # Calendar returns
    # ------------------------------------------------------------------

    def annual_returns(self) -> pd.DataFrame:
        """
        Calendar-year returns for every strategy (and benchmark if provided).

        Returns a DataFrame with years as the index and strategies as columns.
        """
        yearly: dict[str, pd.Series] = {}
        for name, result in self.results.items():
            yearly[name] = result.returns.resample("YE").apply(
                lambda x: (1 + x).prod() - 1
            )
        df = pd.DataFrame(yearly)
        df.index = df.index.year
        df.index.name = "Year"
        if self.benchmark_returns is not None:
            bench = self.benchmark_returns.resample("YE").apply(
                lambda x: (1 + x).prod() - 1
            )
            bench.index = bench.index.year
            df["Benchmark"] = bench
        return df

    def monthly_returns_heatmap(self, strategy_name: str) -> pd.DataFrame:
        """
        Pivot table of monthly returns for a single strategy.

        Rows = year, columns = Jan … Dec.
        """
        if strategy_name not in self.results:
            raise KeyError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(self.results.keys())}"
            )
        r = self.results[strategy_name].returns
        monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        df = monthly.to_frame("return")
        df["year"] = df.index.year
        df["month"] = df.index.month

        pivot = df.pivot(index="year", columns="month", values="return")
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]
        return pivot

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    def rolling_metrics(self, strategy_name: str, window: int = 252) -> pd.DataFrame:
        """Rolling annualised return, volatility, Sharpe, and drawdown."""
        if strategy_name not in self.results:
            raise KeyError(f"Unknown strategy '{strategy_name}'")
        r = self.results[strategy_name].returns
        roll_ret = rolling_annualised_return(r, window)
        roll_vol = rolling_annualised_volatility(r, window)
        roll_sharpe = (roll_ret - self.risk_free_rate) / roll_vol.replace(0, np.nan)
        return pd.DataFrame({
            "rolling_return": roll_ret,
            "rolling_volatility": roll_vol,
            "rolling_sharpe": roll_sharpe,
            "drawdown": drawdown_series(r),
        })

    # ------------------------------------------------------------------
    # Cumulative returns
    # ------------------------------------------------------------------

    def cumulative_returns(self) -> pd.DataFrame:
        """Cumulative wealth index for all strategies (and benchmark)."""
        cum = {name: r.cumulative_returns for name, r in self.results.items()}
        df = pd.DataFrame(cum)
        if self.benchmark_returns is not None:
            df["Benchmark"] = (1 + self.benchmark_returns).cumprod()
        return df

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_tearsheet(self) -> None:
        table = self.comparison_table()
        strat_names = table.index.tolist()
        col_w = max(12, max(len(n) for n in strat_names) + 2)

        labels = {
            "annualised_return": "Ann. Return",
            "annualised_volatility": "Ann. Vol",
            "sharpe_ratio": "Sharpe",
            "sortino_ratio": "Sortino",
            "calmar_ratio": "Calmar",
            "max_drawdown": "Max DD",
            "max_drawdown_duration_days": "Max DD Days",
            "avg_monthly_turnover": "Avg Monthly TO",
            "win_rate": "Win Rate",
            "best_day": "Best Day",
            "worst_day": "Worst Day",
        }
        fmts = {
            "annualised_return": ".2%", "annualised_volatility": ".2%",
            "sharpe_ratio": ".2f", "sortino_ratio": ".2f", "calmar_ratio": ".2f",
            "max_drawdown": ".2%", "max_drawdown_duration_days": ".0f",
            "avg_monthly_turnover": ".2%", "win_rate": ".2%",
            "best_day": ".2%", "worst_day": ".2%",
        }

        print(f"\n{'='*80}")
        print("  Backtest Tearsheet")
        print(f"{'='*80}")
        header = f"  {'Metric':<25}" + "".join(f"{n:>{col_w}}" for n in strat_names)
        print(header)
        print(f"  {'-'*25}" + "-" * (col_w * len(strat_names)))
        for col in table.columns:
            row = f"  {labels.get(col, col):<25}"
            for name in strat_names:
                val = table.loc[name, col]
                try:
                    row += f"{val:>{col_w}{fmts.get(col, '.4f')}}"
                except (ValueError, TypeError):
                    row += f"{'N/A':>{col_w}}"
            print(row)
        print(f"{'='*80}\n")

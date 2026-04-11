"""Drawdown analysis utilities."""

from __future__ import annotations

import pandas as pd

from quantrisk.portfolio.returns import drawdown_series


def drawdown_table(returns: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """
    Identify the top N drawdown events with their start, trough, end, and depth.

    Returns a DataFrame sorted by drawdown depth (worst first).
    """
    clean = returns.dropna()
    dd = drawdown_series(clean)

    events = []
    in_drawdown = False
    start = None
    trough_idx = None
    trough_val = 0.0

    for dt, val in dd.items():
        if val < 0 and not in_drawdown:
            in_drawdown = True
            start = dt
            trough_idx = dt
            trough_val = val
        elif val < 0 and in_drawdown:
            if val < trough_val:
                trough_val = val
                trough_idx = dt
        elif val == 0 and in_drawdown:
            events.append({
                "start": start,
                "trough": trough_idx,
                "end": dt,
                "drawdown": trough_val,
                "duration_days": (dt - start).days,
                "recovery_days": (dt - trough_idx).days,
            })
            in_drawdown = False
            start = None
            trough_idx = None
            trough_val = 0.0

    # Handle open drawdown at end of series
    if in_drawdown:
        events.append({
            "start": start,
            "trough": trough_idx,
            "end": None,
            "drawdown": trough_val,
            "duration_days": (clean.index[-1] - start).days,
            "recovery_days": None,
        })

    df = pd.DataFrame(events)
    if df.empty:
        return df
    return df.nsmallest(top_n, "drawdown").reset_index(drop=True)


def underwater_chart(returns: pd.Series) -> pd.Series:
    """Return the drawdown series for plotting (same as drawdown_series)."""
    return drawdown_series(returns.dropna())

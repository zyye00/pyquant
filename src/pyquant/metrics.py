"""Performance metrics."""

import numpy as np
import pandas as pd
from typing import Optional


def calc_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> dict:
    """计算年化收益、波动、夏普、最大回撤等基础指标。"""
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("returns is empty")

    nav = (1 + r).cumprod()
    annual_return = nav.iloc[-1] ** (periods_per_year / len(r)) - 1
    annual_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan
    max_drawdown = (nav / nav.cummax() - 1).min()

    result = {
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
    }

    if benchmark_returns is not None:
        aligned_r, aligned_b = r.align(benchmark_returns.dropna().astype(float), join="inner")
        active = aligned_r - aligned_b
        result["excess_return"] = float(active.mean() * periods_per_year)
        result["tracking_error"] = float(active.std(ddof=0) * np.sqrt(periods_per_year))

    return result

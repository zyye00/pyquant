"""Performance metrics."""

import numpy as np
import pandas as pd


def calc_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
    *,
    annual_risk_free_rate: float = 0.0,
) -> dict:
    """Calculate metrics from equally spaced periodic simple returns.

    Volatility uses population standard deviation (ddof=0). Sharpe uses
    arithmetic mean excess returns over a constant periodic risk-free rate,
    converted from the effective annual rate. Drawdown is a non-negative loss and
    includes initial wealth of one. return_volatility_ratio preserves the
    CAGR/volatility report-comparison convention, separately from Sharpe.
    Benchmark metrics use only shared non-missing dates; excess_return is
    annualized arithmetic active return, not the difference of two CAGRs.
    Undefined ratios (zero volatility) are returned as NaN.
    """
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive and finite")
    if not np.isfinite(annual_risk_free_rate) or annual_risk_free_rate <= -1:
        raise ValueError("annual_risk_free_rate must be finite and greater than -1")
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("returns is empty")

    nav = (1 + r).cumprod()
    annual_return = nav.iloc[-1] ** (periods_per_year / len(r)) - 1
    annual_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    risk_free = (1 + annual_risk_free_rate) ** (1 / periods_per_year) - 1
    sharpe = (r - risk_free).mean() * periods_per_year / annual_vol if annual_vol else np.nan
    max_drawdown = (1 - nav / nav.cummax().clip(lower=1.0)).max()

    result = {
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "return_volatility_ratio": float(annual_return / annual_vol) if annual_vol else np.nan,
        "max_drawdown": float(max_drawdown),
    }

    if benchmark_returns is not None:
        aligned_r, aligned_b = r.align(
            benchmark_returns.dropna().astype(float), join="inner"
        )
        active = aligned_r - aligned_b
        if active.empty:
            raise ValueError("returns and benchmark_returns have no shared valid dates")
        result["excess_return"] = float(active.mean() * periods_per_year)
        result["tracking_error"] = float(active.std(ddof=0) * np.sqrt(periods_per_year))
        result["information_ratio"] = (
            result["excess_return"] / result["tracking_error"]
            if result["tracking_error"] else np.nan
        )

    return result

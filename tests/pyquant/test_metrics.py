import pandas as pd

from pyquant import calc_metrics


def test_calc_metrics_returns_basic_keys():
    out = calc_metrics(pd.Series([0.01, -0.005, 0.02]))

    assert {"annual_return", "annual_vol", "sharpe", "max_drawdown"} <= set(out)

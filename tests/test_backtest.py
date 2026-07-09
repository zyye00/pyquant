import pandas as pd

from pyquant import run_backtest


def test_run_backtest_uses_previous_period_weights():
    returns = pd.DataFrame({"A": [0.10, 0.20], "B": [0.0, 0.0]})
    weights = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})

    out = run_backtest(returns, weights)

    assert out.tolist() == [0.0, 0.20]

import pandas as pd
import pytest
import vectorbt as vbt

from pyquant import run_backtest


def test_run_backtest_returns_vectorbt_portfolio_and_keeps_initial_build_free():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    close = pd.DataFrame({"A": [10.0, 20.0, 20.0], "B": [10.0, 10.0, 10.0]}, index=dates)
    weights = pd.DataFrame(
        {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=dates
    )
    fees = pd.DataFrame(0.0005, index=dates, columns=close.columns)
    fees.iloc[0] = 0.0

    portfolio = run_backtest(close, weights, fees=fees)

    assert isinstance(portfolio, vbt.Portfolio)
    assert portfolio.orders.records_readable.iloc[0]["Fees"] == 0.0
    assert portfolio.value().iloc[-1] == pytest.approx(1_998_001.0, rel=1e-6)


def test_run_backtest_carries_target_weights_between_rebalance_dates():
    dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29"])
    close = pd.DataFrame({"A": [10.0, 20.0, 20.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=[dates[0]])

    portfolio = run_backtest(close, weights)

    assert portfolio.value().iloc[-1] == pytest.approx(2_000_000.0)


def test_run_backtest_rejects_invalid_target_weights():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    close = pd.DataFrame({"A": 10.0}, index=dates)
    weights = pd.DataFrame({"A": [0.5, 1.0]}, index=dates)

    with pytest.raises(ValueError, match="sum to 1"):
        run_backtest(close, weights)

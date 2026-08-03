import pandas as pd
import pytest

from pyquant import run_backtest


def test_run_backtest_charges_one_way_turnover_after_initial_construction():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    prices = pd.DataFrame({"A": 10.0, "B": 10.0}, index=dates)
    weights = pd.DataFrame(
        {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]},
        index=dates,
    )

    out = run_backtest(prices, weights, fee_rate=0.001)

    assert out.loc["2024-01-31", "turnover"] == 0.0
    assert out.loc["2024-01-31", "transaction_cost"] == 0.0
    assert out.loc["2024-02-29", "turnover"] == 1.0
    assert out.loc["2024-02-29", "transaction_cost"] == 0.001
    assert out.loc["2024-02-29", "total_return"] == pytest.approx(-0.001)
    assert out.loc["2024-03-29", "turnover"] == 0.0


def test_run_backtest_uses_drifted_weights():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    prices = pd.DataFrame(
        {"A": [10.0, 20.0], "B": [10.0, 10.0]},
        index=dates,
    )
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=dates)

    out = run_backtest(prices, weights)

    assert out.loc["2024-02-29", "price_return"] == pytest.approx(0.5)
    assert out.loc["2024-02-29", "turnover"] == pytest.approx(1.0 / 6.0)


def test_run_backtest_keeps_cash_distributions_until_rebalance():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    prices = pd.DataFrame({"A": 10.0}, index=dates)
    weights = pd.DataFrame({"A": 1.0}, index=dates)
    cash_events = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-02-15")],
            "symbol": ["A"],
            "cash_per_share": [1.0],
        }
    )

    out = run_backtest(
        prices,
        weights,
        cash_events,
        fee_rate=0.001,
    )

    assert out.loc["2024-02-29", "dividend_cash"] == pytest.approx(0.1)
    assert out.loc["2024-02-29", "turnover"] == pytest.approx(1.0 / 11.0)
    assert out.loc["2024-02-29", "transaction_cost"] == pytest.approx(1.0 / 11_000.0)


def test_run_backtest_rejects_invalid_target_weights():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    prices = pd.DataFrame({"A": 10.0}, index=dates)
    weights = pd.DataFrame({"A": [0.5, 1.0]}, index=dates)

    with pytest.raises(ValueError, match="sum to 1"):
        run_backtest(prices, weights)

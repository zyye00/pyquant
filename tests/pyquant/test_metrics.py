import pandas as pd
import numpy as np
import pytest

from pyquant import calc_metrics


def test_calc_metrics_returns_basic_keys():
    out = calc_metrics(pd.Series([0.01, -0.005, 0.02]))

    assert {"annual_return", "annual_vol", "sharpe", "max_drawdown"} <= set(out)


@pytest.mark.parametrize("returns, expected", [([-0.1, 0.0], 0.1), ([0.2, -0.25], 0.25), ([0.1, 0.2], 0.0)])
def test_drawdown_includes_initial_wealth_and_later_peaks(returns, expected):
    assert calc_metrics(pd.Series(returns), periods_per_year=12)["max_drawdown"] == pytest.approx(expected)


def test_sharpe_is_distinct_from_compound_return_volatility_ratio():
    out = calc_metrics(pd.Series([0.1, -0.1]), periods_per_year=12)
    assert out["annual_return"] == pytest.approx(0.99**6 - 1)
    assert out["annual_vol"] == pytest.approx(0.1 * np.sqrt(12))
    assert out["sharpe"] == pytest.approx(0.0)
    assert out["return_volatility_ratio"] == pytest.approx((0.99**6 - 1) / (0.1 * np.sqrt(12)))


def test_sharpe_converts_effective_annual_risk_free_rate():
    out = calc_metrics(pd.Series([0.0, 0.02]), periods_per_year=12, annual_risk_free_rate=1.01**12 - 1)
    assert out["sharpe"] == pytest.approx(0.0, abs=1e-12)


def test_information_ratio_uses_aligned_active_returns():
    out = calc_metrics(
        pd.Series([0.9, 0.03, 0.01], index=["unused", "a", "b"]),
        pd.Series([0.01, 0.02, np.nan], index=["a", "b", "unused"]),
        periods_per_year=12,
    )
    assert out["excess_return"] == pytest.approx(0.005 * 12)
    assert out["tracking_error"] == pytest.approx(0.015 * np.sqrt(12))
    assert out["information_ratio"] == pytest.approx(np.sqrt(12) / 3)


def test_undefined_ratios_are_nan():
    out = calc_metrics(pd.Series([0.0, 0.0]), pd.Series([0.0, 0.0]))
    for key in ["sharpe", "return_volatility_ratio", "information_ratio"]:
        assert np.isnan(out[key])
    returns = pd.Series([0.01, -0.02])
    assert np.isnan(calc_metrics(returns, returns)["information_ratio"])


def test_benchmark_requires_overlapping_valid_dates():
    with pytest.raises(ValueError, match="shared valid dates"):
        calc_metrics(pd.Series([0.1], index=["a"]), pd.Series([0.1], index=["b"]))


@pytest.mark.parametrize("kwargs", [{"periods_per_year": 0}, {"annual_risk_free_rate": -1}])
def test_invalid_annualization_parameters(kwargs):
    with pytest.raises(ValueError):
        calc_metrics(pd.Series([0.1]), **kwargs)

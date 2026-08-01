import pandas as pd
import pytest

from pyquant.data.intraday import (
    MINUTE_DAY_INCOMPLETE,
    MINUTE_DAY_NO_DATA_CONFIRMED,
    MINUTE_DAY_VALID,
    calculate_daily_intraday_volatility,
    normalize_minute_bars,
)


def make_minute(datetimes, closes):
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(datetimes),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 100.0,
            "total_turnover": 1_000.0,
        }
    )


def test_daily_volatility_never_includes_overnight_returns():
    minute = normalize_minute_bars(
        make_minute(
            [
                "2024-01-02 09:31",
                "2024-01-02 09:32",
                "2024-01-02 09:33",
                "2024-01-03 09:31",
                "2024-01-03 09:32",
                "2024-01-03 09:33",
            ],
            [100.0, 110.0, 100.0, 200.0, 220.0, 200.0],
        ),
        "sh.600000",
        "2024-01-02",
        "2024-01-03",
    )

    out = calculate_daily_intraday_volatility(
        minute,
        "600000.SH",
        pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    assert out["return_count"].tolist() == [2, 2]
    assert out["status"].tolist() == [MINUTE_DAY_VALID, MINUTE_DAY_VALID]
    assert out.loc[0, "volatility"] == pytest.approx(out.loc[1, "volatility"])


def test_daily_status_distinguishes_empty_and_incomplete_days():
    minute = normalize_minute_bars(
        make_minute(["2024-01-02 09:31", "2024-01-02 09:32"], [10.0, 10.1]),
        "600000.SH",
        "2024-01-02",
        "2024-01-03",
    )

    out = calculate_daily_intraday_volatility(
        minute,
        "600000.SH",
        pd.to_datetime(["2024-01-02", "2024-01-03"]),
        min_bars_per_day=3,
    )

    assert out["status"].tolist() == [
        MINUTE_DAY_INCOMPLETE,
        MINUTE_DAY_NO_DATA_CONFIRMED,
    ]


def test_minute_normalization_rejects_conflicting_duplicate_timestamps():
    data = make_minute(
        ["2024-01-02 09:31", "2024-01-02 09:31"],
        [10.0, 11.0],
    )

    with pytest.raises(ValueError, match="conflicting duplicate"):
        normalize_minute_bars(data, "600000.SH", "2024-01-02", "2024-01-02")

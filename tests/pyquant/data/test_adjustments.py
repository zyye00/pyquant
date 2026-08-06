import pandas as pd
import pytest

from pyquant.data.adjustments import build_back_adjusted_close
from pyquant.data.duckdb import connect_database, initialize_database
from pyquant.data.loader import load_dataset
from pyquant.data.store import write_stock_adjust_factor_request


def test_back_adjustment_applies_latest_factor_and_non_trading_events_forward():
    price = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "symbol": ["600000.SH"] * 3,
            "close": [10.0, 8.0, 9.0],
        }
    )
    factors = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "operate_date": ["2020-01-02"],
            "back_adjust_factor": [1.25],
        }
    )
    coverage = pd.DataFrame(
        {"symbol": ["600000.SH"], "start": ["2019-01-01"], "end": ["2020-12-31"]}
    )

    out = build_back_adjusted_close(price, factors, coverage)

    assert out["adjusted_close"].tolist() == [10.0, 10.0, 11.25]


def test_back_adjustment_rejects_missing_factor_coverage():
    price = pd.DataFrame(
        {"date": [pd.Timestamp("2020-01-01")], "symbol": ["600000.SH"], "close": [10.0]}
    )

    with pytest.raises(ValueError, match="coverage"):
        build_back_adjusted_close(
            price,
            pd.DataFrame(columns=["symbol", "operate_date", "back_adjust_factor"]),
            pd.DataFrame(columns=["symbol", "start", "end"]),
        )


def test_adjustment_factor_store_and_loader_expose_canonical_rows(tmp_path):
    initialize_database(tmp_path / "pyquant.duckdb")
    with connect_database(tmp_path / "pyquant.duckdb") as connection:
        write_stock_adjust_factor_request(
            connection,
            "sh.600000",
            pd.DataFrame(
                {
                    "operate_date": ["2020-01-02"],
                    "fore_adjust_factor": [0.8],
                    "back_adjust_factor": [1.25],
                    "adjust_factor": [1.25],
                }
            ),
            "1990-01-01",
            "2020-12-31",
        )

    factors = load_dataset("stock_adjust_factor", data_root=tmp_path)
    coverage = load_dataset("stock_adjust_factor_coverage", data_root=tmp_path)
    assert factors.loc[0, "symbol"] == "600000.SH"
    assert factors.loc[0, "back_adjust_factor"] == 1.25
    assert coverage.to_dict("records") == [
        {"symbol": "600000.SH", "start": pd.Timestamp("1990-01-01"), "end": pd.Timestamp("2020-12-31")}
    ]

from copy import deepcopy
from importlib.resources import files

import pandas as pd
import pytest
import yaml

from pyquant.data.catalog import DuckDBStorage, get_dataset_spec
from pyquant.data.duckdb import connect_database
from pyquant.data.sources.rqdata import (
    extract_changed_snapshots,
    rqdata_symbol_to_project,
)
from pyquant.data.updater import _run_update_dataset
from strategies.dividend_low_vol import timing as TIMING


STRATEGY_CONFIG = files("strategies.dividend_low_vol").joinpath("config.yaml")
calculate_bp_spread = TIMING.calculate_bp_spread
backtest_timing = TIMING.backtest_valuation_spread_timing


def make_config(
    *,
    index_code: str = "H30269",
    trim_ratio: float = 0.25,
    band_window_months: int = 6,
    band_std_multiplier: float = 1.5,
) -> dict:
    return {
        "strategy_3": {
            "index_code": index_code,
            "trim_ratio": trim_ratio,
            "band_window_months": band_window_months,
            "band_std_multiplier": band_std_multiplier,
        }
    }


def make_spread_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
            "2024-03-29",
            "2024-04-30",
            "2024-05-31",
            "2024-06-28",
        ]
    )
    symbols = list("ABCDEFGH")
    rows = []
    for position, date in enumerate(dates):
        low_center = 1.0
        high_center = 1.0 if position < 5 else 4.0
        bp_values = [
            low_center - 0.5,
            low_center,
            low_center,
            low_center + 0.5,
            high_center - 0.5,
            high_center,
            high_center,
            high_center + 0.5,
        ]
        rows.extend(
            {
                "date": date,
                "symbol": symbol,
                "pb_mrq": 1.0 / bp,
            }
            for symbol, bp in zip(symbols, bp_values, strict=True)
        )
    price = pd.DataFrame(rows)
    constituents = pd.DataFrame(
        {
            "effective_date": pd.Timestamp("2024-01-02"),
            "index_code": "H30269",
            "symbol": list("ABCD"),
        }
    )
    return price, constituents


def test_rqdata_symbols_convert_to_project_format():
    assert rqdata_symbol_to_project("600000.XSHG") == "600000.SH"
    assert rqdata_symbol_to_project("000001.XSHE") == "000001.SZ"

    with pytest.raises(ValueError, match="Unsupported RQData"):
        rqdata_symbol_to_project("00700.XHKG")


def test_changed_snapshots_drop_consecutive_duplicate_sets():
    history = {
        pd.Timestamp("2024-01-02"): ["600000.XSHG", "000001.XSHE"],
        pd.Timestamp("2024-01-03"): ["000001.XSHE", "600000.XSHG"],
        pd.Timestamp("2024-02-01"): ["600000.XSHG", "000002.XSHE"],
    }

    out = extract_changed_snapshots(history, "H30269")

    assert out["effective_date"].unique().tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-02-01"),
    ]
    assert out.to_dict("records") == [
        {
            "effective_date": pd.Timestamp("2024-01-02"),
            "index_code": "H30269",
            "symbol": "000001.SZ",
        },
        {
            "effective_date": pd.Timestamp("2024-01-02"),
            "index_code": "H30269",
            "symbol": "600000.SH",
        },
        {
            "effective_date": pd.Timestamp("2024-02-01"),
            "index_code": "H30269",
            "symbol": "000002.SZ",
        },
        {
            "effective_date": pd.Timestamp("2024-02-01"),
            "index_code": "H30269",
            "symbol": "600000.SH",
        },
    ]


def test_update_index_constituents_replaces_database_snapshots(tmp_path):
    class FakeRqdata:
        def __init__(self):
            self.initialized = False

        def init(self):
            self.initialized = True

        def index_components(self, code, start_date, end_date):
            assert (code, start_date, end_date) == (
                "H30269.XSHG",
                "2024-01-01",
                "2024-01-31",
            )
            return {
                pd.Timestamp("2024-01-02"): [
                    "600000.XSHG",
                    "000001.XSHE",
                ]
            }

    client = FakeRqdata()
    data_root = tmp_path / "data"
    out = _run_update_dataset(
        "index_constituents",
        start="2024-01-01",
        end="2024-01-31",
        pool=["H30269"],
        client=client,
        data_root=data_root,
    )

    assert client.initialized
    assert out[["code", "status", "row_count"]].to_dict("records") == [
        {"code": "H30269", "status": "success", "row_count": 2}
    ]
    with connect_database(data_root / "pyquant.duckdb") as connection:
        assert connection.execute(
            "SELECT index_code, symbol FROM api.index_constituents ORDER BY symbol"
        ).fetchall() == [
            ("H30269", "000001.SZ"),
            ("H30269", "600000.SH"),
        ]
    assert not (tmp_path / "index_constituents").exists()


def test_bp_spread_trims_groups_and_generates_six_month_signal():
    price, constituents = make_spread_inputs()
    original_price = price.copy(deep=True)
    original_constituents = constituents.copy(deep=True)

    out = calculate_bp_spread(price, constituents, make_config())

    assert out.columns.tolist() == TIMING.SPREAD_COLUMNS
    assert out["constituent_count"].eq(4).all()
    assert out["non_constituent_count"].eq(4).all()
    assert out["constituent_trimmed_count"].eq(2).all()
    assert out["non_constituent_trimmed_count"].eq(2).all()
    assert out["bp_spread"].iloc[:5].tolist() == pytest.approx([0.0] * 5)
    assert out["bp_spread"].iloc[-1] == pytest.approx(-3.0)
    assert out["lower_band"].iloc[-1] == pytest.approx(-2.1770509831)
    assert out["bearish_signal"].tolist() == [False] * 5 + [True]
    pd.testing.assert_frame_equal(price, original_price)
    pd.testing.assert_frame_equal(constituents, original_constituents)


def test_bp_spread_uses_latest_snapshot_and_latest_valid_monthly_pb():
    price, constituents = make_spread_inputs()
    price = pd.concat(
        [
            price,
            pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-06-03")],
                    "symbol": ["A"],
                    "pb_mrq": [100.0],
                }
            ),
        ],
        ignore_index=True,
    )
    constituents = pd.concat(
        [
            constituents,
            pd.DataFrame(
                {
                    "effective_date": pd.Timestamp("2024-04-15"),
                    "index_code": "H30269",
                    "symbol": list("ABCE"),
                }
            ),
        ],
        ignore_index=True,
    )

    out = calculate_bp_spread(price, constituents, make_config())

    assert out.loc["2024-03-29", "constituent_effective_date"] == pd.Timestamp(
        "2024-01-02"
    )
    assert out.loc["2024-04-30", "constituent_effective_date"] == pd.Timestamp(
        "2024-04-15"
    )


def test_timing_signal_applies_to_following_month():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    signal = pd.DataFrame(
        {"bearish_signal": [False, True, False]},
        index=dates,
    )
    benchmark = pd.Series(
        [100.0, 90.0, 99.0],
        index=dates,
        name="H20269",
    )

    out = backtest_timing(signal, benchmark)

    assert out.columns.tolist() == TIMING.BACKTEST_COLUMNS
    assert out["bearish_position"].tolist() == [False, False, True]
    assert out["benchmark_return"].iloc[1:].tolist() == pytest.approx([-0.1, 0.1])
    assert out["cash_timing_return"].iloc[1:].tolist() == pytest.approx([-0.1, 0.0])
    assert out["short_timing_return"].iloc[1:].tolist() == pytest.approx([-0.1, -0.1])
    assert out.loc["2024-03-29", "signal_date"] == pd.Timestamp("2024-02-29")


def test_strategy_3_validates_snapshot_and_config():
    price, constituents = make_spread_inputs()
    constituents["effective_date"] = pd.Timestamp("2025-01-01")

    with pytest.raises(ValueError, match="No constituent snapshot"):
        calculate_bp_spread(price, constituents, make_config())

    invalid = deepcopy(make_config())
    invalid["strategy_3"]["trim_ratio"] = 0.5
    with pytest.raises(ValueError, match="trim_ratio"):
        calculate_bp_spread(price, constituents, invalid)


def test_strategy_3_config_and_dataset_catalog():
    with STRATEGY_CONFIG.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    dataset = get_dataset_spec("index_constituents")

    assert config["strategy_3"] == {
        "index_code": "H30269",
        "trim_ratio": 0.1,
        "band_window_months": 6,
        "band_std_multiplier": 1.5,
    }
    assert dataset.storage == DuckDBStorage(
        path="data/pyquant.duckdb",
        relation="api.index_constituents",
        requires_dates=False,
    )
    assert dataset.primary_key == ("effective_date", "index_code", "symbol")


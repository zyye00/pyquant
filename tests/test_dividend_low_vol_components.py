import importlib.util
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest


COMPONENT_PATH = (
    Path(__file__).parents[1]
    / "strategies"
    / "cross_sectional"
    / "dividend_low_vol"
    / "components.py"
)
SPEC = importlib.util.spec_from_file_location("dividend_low_vol_components", COMPONENT_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPONENTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPONENTS)
UNIVERSE_OUTPUT_COLUMNS = COMPONENTS.UNIVERSE_OUTPUT_COLUMNS
build_dividend_universe = COMPONENTS.build_dividend_universe


def make_config(
    start: str = "2024-11-01",
    end: str = "2024-12-31",
    market_cap_keep_ratio: float = 1.0,
    amount_keep_ratio: float = 1.0,
    payout_exclude_ratio: float = 0.0,
) -> dict:
    return {
        "backtest": {"start": start, "end": end, "rebalance": "monthly"},
        "universe": {
            "lookback_days": 240,
            "market_cap_keep_ratio": market_cap_keep_ratio,
            "amount_keep_ratio": amount_keep_ratio,
            "dividend_years": 3,
            "payout_exclude_ratio": payout_exclude_ratio,
        },
    }


def make_price(
    symbols: list[str],
    end: str = "2024-12-31",
    pe_by_symbol: dict[str, float] | None = None,
) -> pd.DataFrame:
    dates = pd.bdate_range("2023-12-01", end)
    pe_by_symbol = pe_by_symbol or {}
    return pd.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "close": 10.0,
                "amount": float(index + 1) * 1_000_000,
                "pe_ttm": pe_by_symbol.get(symbol, 10.0),
            }
            for index, symbol in enumerate(symbols)
            for date in dates
        ]
    )


def make_shares(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbols,
            "publish_date": pd.Timestamp("2023-01-01"),
            "total_shares": [float(index + 1) * 100 for index in range(len(symbols))],
        }
    )


def make_queries(symbols: list[str], years: range = range(2021, 2025)) -> pd.DataFrame:
    return pd.DataFrame(
        [(symbol, year) for symbol in symbols for year in years],
        columns=["symbol", "year"],
    )


def make_dividends(
    symbols: list[str],
    values: dict[str, list[float]] | None = None,
    announce_dates: dict[int, str] | None = None,
) -> pd.DataFrame:
    years = [2021, 2022, 2023, 2024]
    announce_dates = announce_dates or {year: f"{year}-04-30" for year in years}
    rows = []
    for symbol in symbols:
        symbol_values = (values or {}).get(symbol, [1.0, 2.0, 3.0, 4.0])
        rows.extend(
            {
                "symbol": symbol,
                "year": year,
                "announce_date": announce_dates[year],
                "cash_dividend_after_tax": value,
            }
            for year, value in zip(years, symbol_values)
        )
    return pd.DataFrame(rows)


def test_build_dividend_universe_builds_expected_monthly_members():
    symbols = [f"S{index:02d}" for index in range(25)]
    price = make_price(symbols)
    dividends = make_dividends(symbols)
    queries = make_queries(symbols)
    shares = make_shares(symbols)
    original_inputs = [data.copy(deep=True) for data in [price, dividends, queries, shares]]

    out = build_dividend_universe(
        price,
        dividends,
        queries,
        shares,
        make_config(market_cap_keep_ratio=0.8, amount_keep_ratio=0.8),
    )

    assert out.index.names == ["date", "symbol"]
    assert out.columns.tolist() == UNIVERSE_OUTPUT_COLUMNS
    assert set(out.index.get_level_values("date")) == {
        pd.Timestamp("2024-11-29"),
        pd.Timestamp("2024-12-31"),
    }
    assert set(out.xs("2024-12-31").index) == set(symbols[5:])
    assert len(out) == 40
    assert out.loc[("2024-12-31", "S05"), "dividend_yield_ttm"] == pytest.approx(0.4)
    assert out.loc[("2024-12-31", "S05"), "payout_ratio"] == pytest.approx(4.0)
    assert out.loc[("2024-12-31", "S05"), "dividend_growth_slope"] == pytest.approx(1.0)
    assert out["in_universe"].all()
    for data, original in zip([price, dividends, queries, shares], original_inputs):
        pd.testing.assert_frame_equal(data, original)


def test_announcement_date_controls_december_continuous_dividend_window():
    symbols = ["A"]
    dividends = make_dividends(
        symbols,
        announce_dates={
            2021: "2021-04-30",
            2022: "2022-04-30",
            2023: "2023-04-30",
            2024: "2025-01-02",
        },
    )

    out = build_dividend_universe(
        make_price(symbols),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        make_config(),
    )

    assert out.index.get_level_values("date").unique().tolist() == [
        pd.Timestamp("2024-11-29")
    ]


def test_announced_dividend_is_available_before_payment_date():
    symbols = ["A"]
    dividends = make_dividends(symbols)
    dividends["announce_date"] = dividends["announce_date"].mask(
        dividends["year"].eq(2024), "2024-12-15"
    )
    dividends["payment_date"] = dividends["announce_date"].mask(
        dividends["year"].eq(2024), "2025-01-02"
    )

    out = build_dividend_universe(
        make_price(symbols),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        make_config(),
    )

    assert out.index.get_level_values("date").unique().tolist() == [
        pd.Timestamp("2024-11-29"),
        pd.Timestamp("2024-12-31"),
    ]


def test_growth_filter_removes_negative_slope_and_keeps_zero_slope():
    symbols = ["FLAT", "DOWN"]
    dividends = make_dividends(
        symbols,
        values={"FLAT": [1.0, 1.0, 1.0, 0.0], "DOWN": [3.0, 2.0, 1.0, 0.0]},
    )

    out = build_dividend_universe(
        make_price(symbols, end="2024-11-29"),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        make_config(end="2024-11-29"),
    )

    assert out.index.get_level_values("symbol").tolist() == ["FLAT"]
    assert out.iloc[0]["dividend_growth_slope"] == pytest.approx(0.0, abs=1e-12)


def test_payout_filter_removes_negative_and_highest_five_percent():
    symbols = [f"S{index:02d}" for index in range(20)]
    pe_by_symbol = {symbol: 10.0 for symbol in symbols}
    pe_by_symbol["S00"] = -10.0
    pe_by_symbol["S19"] = 100.0

    out = build_dividend_universe(
        make_price(symbols, end="2024-11-29", pe_by_symbol=pe_by_symbol),
        make_dividends(symbols),
        make_queries(symbols),
        make_shares(symbols),
        make_config(end="2024-11-29", payout_exclude_ratio=0.05),
    )

    selected = set(out.index.get_level_values("symbol"))
    assert len(selected) == 18
    assert "S00" not in selected
    assert "S19" not in selected


def test_market_rank_ties_use_symbol_order():
    symbols = ["E", "D", "C", "B", "A"]
    price = make_price(symbols, end="2024-11-29")
    price["amount"] = 1_000_000.0
    shares = make_shares(symbols)
    shares["total_shares"] = 100.0

    out = build_dividend_universe(
        price,
        make_dividends(symbols),
        make_queries(symbols),
        shares,
        make_config(
            end="2024-11-29",
            market_cap_keep_ratio=0.8,
            amount_keep_ratio=0.8,
        ),
    )

    assert set(out.index.get_level_values("symbol")) == {"A", "B", "C", "D"}


def test_total_shares_use_only_published_values():
    symbols = ["A"]
    shares = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "publish_date": ["2023-01-01", "2024-12-15"],
            "total_shares": [100.0, 1_000.0],
        }
    )

    out = build_dividend_universe(
        make_price(symbols, end="2024-11-29"),
        make_dividends(symbols),
        make_queries(symbols),
        shares,
        make_config(end="2024-11-29"),
    )

    assert out.iloc[0]["avg_market_cap_240d"] == pytest.approx(1_000.0)


def test_missing_dividend_query_coverage_raises_clear_error():
    symbols = ["A"]

    with pytest.raises(ValueError, match="Dividend query coverage missing"):
        build_dividend_universe(
            make_price(symbols, end="2024-11-29"),
            make_dividends(symbols),
            make_queries(symbols, range(2021, 2024)),
            make_shares(symbols),
            make_config(end="2024-11-29"),
        )


def test_missing_total_share_symbol_raises_clear_error():
    symbols = ["A", "B"]

    with pytest.raises(ValueError, match="Total-share data missing for 1 price symbols"):
        build_dividend_universe(
            make_price(symbols, end="2024-11-29"),
            make_dividends(symbols),
            make_queries(symbols),
            make_shares(["A"]),
            make_config(end="2024-11-29"),
        )


def test_duplicate_price_key_and_invalid_config_raise_clear_errors():
    symbols = ["A"]
    price = make_price(symbols, end="2024-11-29")
    duplicate_price = pd.concat([price, price.iloc[[0]]], ignore_index=True)
    inputs = [make_dividends(symbols), make_queries(symbols), make_shares(symbols)]

    with pytest.raises(ValueError, match=r"duplicate \(date, symbol\)"):
        build_dividend_universe(duplicate_price, *inputs, make_config(end="2024-11-29"))

    with pytest.raises(ValueError, match="price missing required columns: \\['pe_ttm'\\]"):
        build_dividend_universe(
            price.drop(columns="pe_ttm"),
            *inputs,
            make_config(end="2024-11-29"),
        )

    invalid_config = deepcopy(make_config(end="2024-11-29"))
    invalid_config["universe"]["market_cap_keep_ratio"] = 0
    with pytest.raises(ValueError, match="keep ratios"):
        build_dividend_universe(price, *inputs, invalid_config)

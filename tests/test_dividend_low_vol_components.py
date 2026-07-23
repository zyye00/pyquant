import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from pyquant.io import load_config


STRATEGY_DIR = (
    Path(__file__).parents[1]
    / "strategies"
    / "dividend_low_vol"
)
SPEC = importlib.util.spec_from_file_location(
    "dividend_low_vol_components", STRATEGY_DIR / "components.py"
)
assert SPEC is not None and SPEC.loader is not None
COMPONENTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPONENTS)

CONSTITUENT_COLUMNS = COMPONENTS.CONSTITUENT_COLUMNS
INDEX_COLUMNS = COMPONENTS.INDEX_COLUMNS
calculate_index = COMPONENTS.calculate_dividend_low_vol_index
calculate_rebalanced_index = COMPONENTS.calculate_dividend_low_vol_rebalanced_index
calculate_monthly_rebalanced_index = (
    COMPONENTS.calculate_dividend_low_vol_monthly_rebalanced_index
)
select_constituents = COMPONENTS.select_dividend_low_vol_constituents
select_download_symbols = COMPONENTS.select_dividend_low_vol_download_symbols


def make_config(
    *,
    market_lookback_days: int = 4,
    market_cap_keep_ratio: float = 1.0,
    amount_keep_ratio: float = 1.0,
    payout_exclude_ratio: float = 0.0,
    dividend_yield_lookback_days: int = 6,
    dividend_top_n: int = 3,
    volatility_lookback_days: int = 4,
    final_n: int = 2,
) -> dict:
    return {
        "universe": {
            "lookback_days": market_lookback_days,
            "market_cap_keep_ratio": market_cap_keep_ratio,
            "amount_keep_ratio": amount_keep_ratio,
            "dividend_years": 3,
            "payout_exclude_ratio": payout_exclude_ratio,
        },
        "selection": {
            "dividend_yield_lookback_days": dividend_yield_lookback_days,
            "dividend_top_n": dividend_top_n,
            "volatility_lookback_days": volatility_lookback_days,
            "final_n": final_n,
        },
    }


def make_price(
    symbols: list[str],
    *,
    closes: dict[str, list[float]] | None = None,
    amounts: dict[str, float] | None = None,
    pe_ttm: dict[str, float] | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    dates = dates if dates is not None else pd.bdate_range("2024-11-18", periods=10)
    rows = []
    for index, symbol in enumerate(symbols):
        symbol_closes = (closes or {}).get(symbol, [10.0] * len(dates))
        for date, close in zip(dates, symbol_closes, strict=True):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": close,
                    "amount": (amounts or {}).get(symbol, float(index + 1) * 1_000),
                    "pe_ttm": (pe_ttm or {}).get(symbol, 10.0),
                }
            )
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ms]")
    return out


def make_shares(
    symbols: list[str],
    values: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "symbol": symbols,
            "publish_date": pd.Timestamp("2023-01-01"),
            "total_shares": [
                (values or {}).get(symbol, float(index + 1) * 100)
                for index, symbol in enumerate(symbols)
            ],
        }
    )
    out["publish_date"] = pd.to_datetime(out["publish_date"]).astype("datetime64[ms]")
    return out


def make_queries(
    symbols: list[str],
    years: range = range(2021, 2025),
) -> pd.DataFrame:
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
            for year, value in zip(years, symbol_values, strict=True)
        )
    out = pd.DataFrame(rows)
    out["announce_date"] = pd.to_datetime(out["announce_date"])
    return out


def make_index_price(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(rows, columns=["date", "symbol", "close"])
    out["date"] = pd.to_datetime(out["date"])
    return out


def make_index_queries(symbols: list[str]) -> pd.DataFrame:
    return make_queries(symbols, range(2023, 2025))


def empty_index_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["symbol", "payment_date", "cash_dividend_after_tax"]
    )


def test_download_symbols_require_720_valid_prices_by_as_of_date():
    dates = pd.bdate_range("2021-01-01", periods=7)
    price = make_price(
        ["QUALIFIED", "SHORT", "FUTURE"],
        dates=dates,
        closes={
            "QUALIFIED": [10.0] * 7,
            "SHORT": [10.0] * 5 + [None, 10.0],
            "FUTURE": [None] * 6 + [10.0],
        },
    )
    original = price.copy(deep=True)

    out = select_download_symbols(
        price,
        dates[5],
        make_config(dividend_yield_lookback_days=6),
    )

    assert out == ["QUALIFIED"]
    pd.testing.assert_frame_equal(price, original)


def test_full_75_then_50_selection_order_and_weights():
    symbols = [f"S{index:03d}" for index in range(80)]
    dividend_values = {
        symbol: [cash, cash, cash, cash]
        for symbol, cash in zip(
            symbols,
            [float(80 - index) for index in range(80)],
            strict=True,
        )
    }
    price = make_price(symbols)
    dividends = make_dividends(symbols, dividend_values)
    queries = make_queries(symbols)
    shares = make_shares(symbols)
    originals = [frame.copy(deep=True) for frame in [price, dividends, queries, shares]]
    config = make_config(dividend_top_n=75, final_n=50)

    out = select_constituents(
        price,
        dividends,
        queries,
        shares,
        "2024-11-29",
        config,
    )

    assert out.index.tolist() == symbols[:50]
    assert out.index.name == "symbol"
    assert out.columns.tolist() == CONSTITUENT_COLUMNS
    assert out["dividend_yield_rank"].tolist() == list(range(1, 51))
    assert out["volatility_rank"].tolist() == list(range(1, 51))
    assert out["weight"].sum() == pytest.approx(1.0)
    assert out.loc["S000", "weight"] > out.loc["S049", "weight"]
    assert out["as_of_date"].eq(pd.Timestamp("2024-11-29")).all()
    assert out["price_date"].eq(pd.Timestamp("2024-11-29")).all()
    for frame, original in zip([price, dividends, queries, shares], originals, strict=True):
        pd.testing.assert_frame_equal(frame, original)


def test_market_cap_and_amount_top_80_percent_use_symbol_tie_order():
    symbols = ["E", "D", "C", "B", "A"]
    price = make_price(symbols, amounts={symbol: 1_000.0 for symbol in symbols})
    shares = make_shares(symbols, {symbol: 100.0 for symbol in symbols})
    config = make_config(
        market_cap_keep_ratio=0.8,
        amount_keep_ratio=0.8,
        dividend_top_n=4,
        final_n=4,
    )

    out = select_constituents(
        price,
        make_dividends(symbols),
        make_queries(symbols),
        shares,
        "2024-11-29",
        config,
    )

    assert set(out.index) == {"A", "B", "C", "D"}


def test_market_snapshot_uses_common_dates_and_exact_date_population():
    dates = pd.bdate_range("2024-11-25", periods=5)
    price = pd.DataFrame(
        [
            *(
                {
                    "date": date,
                    "symbol": "A",
                    "close": 10.0,
                    "amount": amount,
                    "pe_ttm": 10.0,
                }
                for date, amount in zip(
                    dates,
                    [999.0, 10.0, 20.0, None, 40.0],
                    strict=True,
                )
            ),
            *(
                {
                    "date": date,
                    "symbol": "B",
                    "close": 20.0,
                    "amount": amount,
                    "pe_ttm": 10.0,
                }
                for date, amount in [(dates[0], 1_000.0), (dates[1], 100.0), (dates[4], 300.0)]
            ),
            *(
                {
                    "date": date,
                    "symbol": "C",
                    "close": 30.0,
                    "amount": 500.0,
                    "pe_ttm": 10.0,
                }
                for date in dates[1:]
            ),
            *(
                {
                    "date": date,
                    "symbol": "OFF_DATE",
                    "close": 40.0,
                    "amount": 600.0,
                    "pe_ttm": 10.0,
                }
                for date in dates[:-1]
            ),
        ]
    )
    price["date"] = pd.to_datetime(price["date"]).astype("datetime64[ms]")
    shares = make_shares(["A", "B"], {"A": 100.0, "B": 200.0})

    snapshot = COMPONENTS._market_snapshot(
        COMPONENTS._prepare_price(price),
        COMPONENTS._prepare_shares(shares),
        dates[-1],
        4,
    ).set_index("symbol")

    assert snapshot.index.tolist() == ["A", "B", "C"]
    assert snapshot.loc["A", "avg_amount_240d"] == pytest.approx(70.0 / 3.0)
    assert snapshot.loc["B", "avg_amount_240d"] == pytest.approx(200.0)
    assert snapshot.loc["A", "avg_market_cap_240d"] == pytest.approx(1_000.0)
    assert snapshot.loc["B", "avg_market_cap_240d"] == pytest.approx(4_000.0)
    assert pd.isna(snapshot.loc["C", "avg_market_cap_240d"])
    assert COMPONENTS._top_symbols(
        snapshot.reset_index(), "avg_market_cap_240d", 2 / 3
    ) == {"A", "B"}


def test_market_snapshot_requires_a_full_market_calendar_window():
    price = make_price(["A"], dates=pd.bdate_range("2024-11-25", periods=3))

    with pytest.raises(ValueError, match="Only 3 market dates are available"):
        COMPONENTS._market_snapshot(
            COMPONENTS._prepare_price(price),
            COMPONENTS._prepare_shares(make_shares(["A"])),
            pd.Timestamp("2024-11-27"),
            4,
        )


def test_payout_filters_negative_and_highest_five_percent_after_continuity():
    symbols = [f"S{index:02d}" for index in range(21)]
    pe_ttm = {symbol: 10.0 for symbol in symbols}
    pe_ttm["S00"] = -10.0
    pe_ttm["S19"] = 100.0
    dividends = make_dividends(symbols)
    dividends.loc[dividends["symbol"].eq("S20") & dividends["year"].eq(2021), "cash_dividend_after_tax"] = 0.0
    config = make_config(
        payout_exclude_ratio=0.05,
        dividend_top_n=18,
        final_n=18,
    )

    out = select_constituents(
        make_price(symbols, pe_ttm=pe_ttm),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        "2024-11-29",
        config,
    )

    assert len(out) == 18
    assert "S00" not in out.index
    assert "S19" not in out.index
    assert "S20" not in out.index


def test_negative_dividend_growth_is_removed_and_zero_growth_is_kept():
    symbols = ["DOWN", "FLAT"]
    dividends = make_dividends(
        symbols,
        {"DOWN": [4.0, 3.0, 2.0, 1.0], "FLAT": [1.0, 1.0, 1.0, 1.0]},
    )

    out = select_constituents(
        make_price(symbols),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        "2024-11-29",
        make_config(dividend_top_n=1, final_n=1),
    )

    assert out.index.tolist() == ["FLAT"]
    assert out.loc["FLAT", "dividend_growth_slope"] == pytest.approx(0.0, abs=1e-12)


def test_low_volatility_stage_runs_after_dividend_yield_ranking():
    symbols = ["HIGH_STABLE", "MID_VOLATILE", "LOW_STABLE", "OUTSIDE"]
    dividends = make_dividends(
        symbols,
        {
            "HIGH_STABLE": [4.0] * 4,
            "MID_VOLATILE": [3.0] * 4,
            "LOW_STABLE": [2.0] * 4,
            "OUTSIDE": [1.0] * 4,
        },
    )
    closes = {
        "HIGH_STABLE": [10.0] * 10,
        "MID_VOLATILE": [10.0, 12.0, 9.0, 13.0, 8.0, 14.0, 7.0, 15.0, 6.0, 16.0],
        "LOW_STABLE": [10.0] * 10,
        "OUTSIDE": [10.0] * 10,
    }

    out = select_constituents(
        make_price(symbols, closes=closes),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        "2024-11-29",
        make_config(dividend_top_n=3, final_n=2),
    )

    assert out.index.tolist() == ["HIGH_STABLE", "LOW_STABLE"]
    assert out.loc["HIGH_STABLE", "dividend_yield_rank"] == 1
    assert "OUTSIDE" not in out.index


def test_future_dividend_announcement_and_share_publication_are_not_visible():
    dividends = make_dividends(
        ["A"],
        {"A": [1.0, 1.0, 1.0, 100.0]},
        {
            2021: "2021-04-30",
            2022: "2022-04-30",
            2023: "2024-02-01",
            2024: "2025-01-02",
        },
    )
    shares = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "publish_date": pd.to_datetime(["2023-01-01", "2024-12-15"]),
            "total_shares": [100.0, 1_000.0],
        }
    )
    shares["publish_date"] = pd.to_datetime(shares["publish_date"]).astype("datetime64[ms]")

    out = select_constituents(
        make_price(["A"]),
        dividends,
        make_queries(["A"]),
        shares,
        "2024-11-29",
        make_config(dividend_top_n=1, final_n=1),
    )

    assert out.loc["A", "avg_market_cap_240d"] == pytest.approx(1_000.0)
    assert out.loc["A", "dividend_yield_ttm"] == pytest.approx(0.1)
    assert out.loc["A", "avg_dividend_yield_3y"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("as_of", "expected_years"),
    [
        ("2013-12-20", [2010, 2011, 2012]),
        ("2013-12-21", [2011, 2012, 2013]),
    ],
)
def test_december_annual_dividend_cutoff(as_of, expected_years):
    assert list(COMPONENTS._continuous_dividend_years(pd.Timestamp(as_of), 3)) == expected_years
    metrics = pd.DataFrame(
        {"symbol": ["A"], "close": [10.0], "pe_ttm": [10.0]}
    )
    dividends = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "year": [2010, 2011, 2012, 2013],
            "announce_date": pd.to_datetime(
                ["2010-06-01", "2011-06-01", "2012-06-01", "2013-06-01"]
            ),
            "cash_dividend_after_tax": [1.0, 1.0, 1.0, 0.0],
        }
    )

    out = COMPONENTS._add_dividend_metrics(
        metrics,
        dividends,
        pd.Timestamp(as_of),
        3,
    )

    assert bool(out.loc[0, "consecutive_dividends"]) == (
        expected_years == [2010, 2011, 2012]
    )


def test_missing_query_coverage_and_insufficient_history_raise():
    inputs = [make_price(["A"]), make_dividends(["A"]), make_shares(["A"])]
    config = make_config(dividend_top_n=1, final_n=1)

    with pytest.raises(ValueError, match="Dividend query coverage missing"):
        select_constituents(
            inputs[0],
            inputs[1],
            make_queries(["A"], range(2021, 2023)),
            inputs[2],
            "2024-11-29",
            config,
        )

    short_price = inputs[0].groupby("symbol").tail(5)
    with pytest.raises(ValueError, match="price-history filters"):
        select_constituents(
            short_price,
            inputs[1],
            make_queries(["A"]),
            inputs[2],
            "2024-11-29",
            config,
        )


def test_query_coverage_is_checked_after_price_history_filter():
    dates = pd.bdate_range("2024-11-18", periods=6)
    price = pd.concat(
        [
            make_price(["QUALIFIED"], dates=dates),
            make_price(["SHORT"], dates=dates[:5]),
        ],
        ignore_index=True,
    )

    out = select_constituents(
        price,
        make_dividends(["QUALIFIED", "SHORT"]),
        make_queries(["QUALIFIED"], range(2021, 2024)),
        make_shares(["QUALIFIED", "SHORT"]),
        dates[-1],
        make_config(dividend_top_n=1, final_n=1),
    )

    assert out.index.tolist() == ["QUALIFIED"]


def test_duplicate_price_key_and_invalid_config_raise():
    price = make_price(["A"])
    duplicate_price = pd.concat([price, price.iloc[[0]]], ignore_index=True)
    inputs = [make_dividends(["A"]), make_queries(["A"]), make_shares(["A"])]
    config = make_config(dividend_top_n=1, final_n=1)

    with pytest.raises(ValueError, match=r"duplicate \(date, symbol\)"):
        select_constituents(
            duplicate_price,
            *inputs,
            "2024-11-29",
            config,
        )

    invalid_config = deepcopy(config)
    invalid_config["selection"]["dividend_top_n"] = 0
    with pytest.raises(ValueError, match="dividend_top_n must be positive"):
        select_constituents(
            price,
            *inputs,
            "2024-11-29",
            invalid_config,
        )


def test_strategy_config_contains_original_index_parameters_only():
    config = load_config(STRATEGY_DIR / "config.yaml")

    assert "backtest" not in config
    assert config["selection"] == {
        "dividend_yield_lookback_days": 720,
        "dividend_top_n": 75,
        "final_n": 50,
        "volatility_lookback_days": 240,
    }
    assert config["data"] == {
        "start_date": "2013-12-16",
        "end_date": "2023-06-16",
        "pool": "all",
    }


def test_notebooks_split_downloads_from_calculation():
    notebooks = {
        name: json.loads((STRATEGY_DIR / name).read_text(encoding="utf-8"))
        for name in ["download.ipynb", "strategy_1_monthly_rebalance.ipynb"]
    }
    for notebook in notebooks.values():
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "notebook_cell", "exec")

    assert "update_dataset" in str(notebooks["download.ipynb"])
    strategy_notebook = str(notebooks["strategy_1_monthly_rebalance.ipynb"])
    assert "update_dataset" not in strategy_notebook
    assert "csindex_daily" in str(notebooks["download.ipynb"])
    assert "csindex_daily" in strategy_notebook
    assert "calculate_dividend_low_vol_monthly_rebalanced_index" in strategy_notebook
    assert "official_index_job.wait()" in str(notebooks["download.ipynb"])


def test_monthly_rebalanced_index_uses_next_trading_day_after_month_end():
    symbols = [f"S{index:03d}" for index in range(3)]
    dates = pd.bdate_range("2024-01-02", periods=45)
    config = make_config(
        market_lookback_days=4,
        dividend_yield_lookback_days=6,
        dividend_top_n=3,
        volatility_lookback_days=4,
        final_n=2,
    )
    strategy_config = {"universe": config["universe"], "strategy_1": config["selection"]}
    dividends = make_dividends(symbols)
    dividends["payment_date"] = pd.NaT

    index, constituents = calculate_monthly_rebalanced_index(
        make_price(symbols, dates=dates),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        dates[0],
        dates[-1],
        strategy_config,
    )

    assert index.index.min() == pd.Timestamp("2024-02-01")
    assert constituents.index.get_level_values("effective_date").min() == pd.Timestamp(
        "2024-02-01"
    )


def test_fixed_quantity_price_index_and_suspension_forward_fill():
    price = make_index_price(
        [
            ("2024-01-02", "A", 10.0),
            ("2024-01-02", "B", 20.0),
            ("2024-01-03", "A", 11.0),
            ("2024-01-04", "A", 12.0),
            ("2024-01-04", "B", 18.0),
        ]
    )
    constituents = pd.DataFrame(
        {"symbol": ["A", "B"], "weight": [0.5, 0.5]}
    ).set_index("symbol")

    out = calculate_index(
        price,
        empty_index_dividends(),
        make_index_queries(["A", "B"]),
        constituents,
        "2024-01-02",
        "2024-01-04",
    )

    assert out.columns.tolist() == INDEX_COLUMNS
    assert out.index.name == "date"
    assert out["price_index"].tolist() == pytest.approx([1.0, 1.05, 1.05])
    assert out["price_return"].tolist() == pytest.approx([0.0, 0.05, 0.0])
    pd.testing.assert_series_equal(
        out["price_index"], out["total_return_index"], check_names=False
    )


def test_all_constituents_can_forward_fill_on_a_market_trading_day():
    price = make_index_price(
        [
            ("2024-01-01", "A", 10.0),
            ("2024-01-02", "MARKET_CALENDAR", 1.0),
            ("2024-01-03", "A", 11.0),
            ("2024-01-03", "MARKET_CALENDAR", 1.0),
        ]
    )
    constituents = pd.DataFrame({"symbol": ["A"], "weight": [1.0]}).set_index(
        "symbol"
    )

    out = calculate_index(
        price,
        empty_index_dividends(),
        make_index_queries(["A"]),
        constituents,
        "2024-01-02",
        "2024-01-03",
    )

    assert out.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert out["price_index"].tolist() == pytest.approx([1.0, 1.1])


def test_total_return_credits_payment_and_moves_weekend_payment_forward():
    price = make_index_price(
        [
            ("2024-01-05", "A", 10.0),
            ("2024-01-08", "A", 10.0),
            ("2024-01-09", "A", 10.0),
        ]
    )
    dividends = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "payment_date": pd.to_datetime(["2024-01-06", "2024-01-09"]),
            "cash_dividend_after_tax": [1.0, 1.0],
        }
    )
    constituents = pd.DataFrame({"symbol": ["A"], "weight": [1.0]}).set_index(
        "symbol"
    )

    out = calculate_index(
        price,
        dividends,
        make_index_queries(["A"]),
        constituents,
        "2024-01-05",
        "2024-01-09",
    )

    assert out.loc["2024-01-05", "dividend_cash"] == 0.0
    assert out.loc["2024-01-08", "dividend_cash"] == pytest.approx(0.1)
    assert out.loc["2024-01-08", "total_return"] == pytest.approx(0.1)
    assert out.loc["2024-01-08", "price_index"] == pytest.approx(1.0)
    assert out.loc["2024-01-08", "total_return_index"] == pytest.approx(1.1)
    assert out.loc["2024-01-09", "dividend_cash"] == pytest.approx(0.1)
    assert out.loc["2024-01-09", "total_return_index"] == pytest.approx(1.21)


def test_no_dividend_indices_match_and_unit_base_segments_can_link():
    price = make_index_price(
        [
            ("2024-01-02", "A", 10.0),
            ("2024-01-03", "A", 11.0),
            ("2024-01-04", "A", 12.0),
        ]
    )
    queries = make_index_queries(["A"])
    constituents = pd.DataFrame({"symbol": ["A"], "weight": [1.0]}).set_index(
        "symbol"
    )
    first = calculate_index(
        price,
        empty_index_dividends(),
        queries,
        constituents,
        "2024-01-02",
        "2024-01-03",
    )
    second = calculate_index(
        price,
        empty_index_dividends(),
        queries,
        constituents,
        "2024-01-03",
        "2024-01-04",
    )
    second["price_index"] = (
        first["price_index"].iloc[-1] * (1.0 + second["price_return"]).cumprod()
    )
    second["total_return_index"] = (
        first["total_return_index"].iloc[-1]
        * (1.0 + second["total_return"]).cumprod()
    )
    chained = pd.concat([first, second.iloc[1:]])
    full = calculate_index(
        price,
        empty_index_dividends(),
        queries,
        constituents,
        "2024-01-02",
        "2024-01-04",
    )

    pd.testing.assert_series_equal(chained["price_index"], full["price_index"])
    pd.testing.assert_series_equal(
        chained["total_return_index"], full["total_return_index"]
    )


def test_annual_rebalanced_index_uses_second_friday_and_links_segments():
    dates = pd.bdate_range("2023-11-20", "2024-12-20")
    price = make_price(
        ["A", "B"],
        closes={
            "A": [10.0 + index * 0.01 for index in range(len(dates))],
            "B": [20.0 + index * 0.02 for index in range(len(dates))],
        },
        dates=dates,
    )
    dividends = make_dividends(
        ["A", "B"],
        {"A": [2.0, 2.0, 2.0, 2.0], "B": [1.0, 1.0, 1.0, 1.0]},
    )
    dividends = pd.concat(
        [
            dividends,
            pd.DataFrame(
                {
                    "symbol": ["A", "B"],
                    "year": [2020, 2020],
                    "announce_date": pd.to_datetime(["2020-04-30", "2020-04-30"]),
                    "cash_dividend_after_tax": [2.0, 1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    dividends["payment_date"] = pd.NaT
    index, constituents = calculate_rebalanced_index(
        price,
        dividends,
        make_queries(["A", "B"], range(2020, 2025)),
        make_shares(["A", "B"]),
        "2023-01-01",
        "2024-12-20",
        make_config(dividend_top_n=2, final_n=1),
    )

    assert index.index.is_unique
    assert index.index[[0, -1]].tolist() == [
        pd.Timestamp("2023-12-11"),
        pd.Timestamp("2024-12-20"),
    ]
    assert index.loc["2023-12-11", "price_index"] == pytest.approx(1.0)
    assert index.loc["2024-12-16", "price_return"] != 0.0
    assert constituents.index.names == ["effective_date", "symbol"]
    assert constituents.index.get_level_values("effective_date").unique().tolist() == [
        pd.Timestamp("2023-12-11"),
        pd.Timestamp("2024-12-16"),
    ]
    assert constituents.loc[(pd.Timestamp("2024-12-16"), "A"), "as_of_date"] == pd.Timestamp(
        "2024-12-13"
    )


def test_index_validates_coverage_effective_prices_and_duplicate_events():
    price = make_index_price(
        [
            ("2024-01-02", "A", 10.0),
            ("2024-01-03", "A", 10.0),
            ("2024-01-03", "B", 20.0),
        ]
    )
    constituents = pd.DataFrame(
        {"symbol": ["A", "B"], "weight": [0.5, 0.5]}
    ).set_index("symbol")
    single_constituent = pd.DataFrame(
        {"symbol": ["A"], "weight": [1.0]}
    ).set_index("symbol")

    with pytest.raises(ValueError, match="No price is available on or before"):
        calculate_index(
            price,
            empty_index_dividends(),
            make_index_queries(["A", "B"]),
            constituents,
            "2024-01-02",
            "2024-01-03",
        )

    with pytest.raises(ValueError, match="Dividend query coverage missing"):
        calculate_index(
            price[price["symbol"].eq("A")],
            empty_index_dividends(),
            make_queries(["A"], range(2024, 2025)),
            single_constituent,
            "2024-01-02",
            "2024-01-03",
        )

    duplicated_dividends = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "payment_date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "cash_dividend_after_tax": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate event keys"):
        calculate_index(
            price[price["symbol"].eq("A")],
            duplicated_dividends,
            make_index_queries(["A"]),
            single_constituent,
            "2024-01-02",
            "2024-01-03",
        )

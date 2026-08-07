from copy import deepcopy
from importlib.resources import files

import pandas as pd
import pytest
import yaml

from pyquant import build_div_low_vol_universe
from strategies.div_low_vol import components as COMPONENTS


STRATEGY_CONFIG = files("strategies.div_low_vol").joinpath("config.yaml")

CONSTITUENT_COLUMNS = COMPONENTS.CONSTITUENT_COLUMNS
INDEX_COLUMNS = COMPONENTS.INDEX_COLUMNS
calculate_index = COMPONENTS.calculate_div_low_vol_index
calculate_rebalanced_index = COMPONENTS.calculate_div_low_vol_rebalanced_index
calculate_monthly_rebalanced_index = (
    COMPONENTS.calculate_div_low_vol_monthly_rebalanced_index
)
calculate_volatility_groups = COMPONENTS.calculate_traditional_volatility_group_indices
select_constituents = COMPONENTS.select_div_low_vol_constituents
select_download_symbols = COMPONENTS.select_div_low_vol_download_symbols
build_minute_requests = COMPONENTS.build_intraday_minute_requests
calculate_high_frequency_factor = COMPONENTS.calculate_high_frequency_volatility_factor
select_high_frequency_constituents = (
    COMPONENTS.select_high_frequency_div_low_vol_constituents
)
calculate_high_frequency_index = (
    COMPONENTS.calculate_high_frequency_div_low_vol_monthly_rebalanced_index
)
calculate_high_frequency_groups = (
    COMPONENTS.calculate_high_frequency_volatility_candidate_group_indices
)


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
                "cash_dividend_before_tax": value,
            }
            for year, value in zip(years, symbol_values, strict=True)
        )
    out = pd.DataFrame(rows)
    out["announce_date"] = pd.to_datetime(out["announce_date"])
    out["operate_date"] = out["announce_date"] + pd.Timedelta(days=1)
    return out


def make_index_price(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(rows, columns=["date", "symbol", "close"])
    out["date"] = pd.to_datetime(out["date"])
    return out


def make_index_queries(symbols: list[str]) -> pd.DataFrame:
    return make_queries(symbols, range(2023, 2025))


def empty_index_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["symbol", "operate_date", "payment_date", "cash_dividend_before_tax"]
    )


def make_adjustment_data(
    symbols: list[str],
    start: str = "1990-01-01",
    end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = pd.DataFrame(
        columns=["symbol", "operate_date", "back_adjust_factor"]
    )
    coverage = pd.DataFrame(
        {"symbol": symbols, "start": start, "end": end}
    )
    return factors, coverage


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
    for frame, original in zip(
        [price, dividends, queries, shares], originals, strict=True
    ):
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


def test_percentage_rankings_exclude_missing_metrics_before_counting():
    symbols = ["A", "B", "C", "D"]
    price = make_price(
        symbols,
        amounts={"A": 400.0, "B": 300.0, "C": 200.0, "D": 100.0},
    )
    config = make_config(market_cap_keep_ratio=0.5, amount_keep_ratio=0.5)

    out = build_div_low_vol_universe(
        price,
        make_dividends(symbols),
        make_queries(symbols),
        make_shares(["A", "B", "C"], {"A": 400.0, "B": 300.0, "C": 200.0}),
        "2024-11-29",
        config["universe"],
        price_history_lookback_days=1,
    )

    assert out.index.tolist() == ["A"]


def test_public_universe_uses_common_dates_and_exact_date_population():
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
                for date, amount in [
                    (dates[0], 1_000.0),
                    (dates[1], 100.0),
                    (dates[4], 300.0),
                ]
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

    snapshot = build_div_low_vol_universe(
        price,
        make_dividends(["A", "B"]),
        make_queries(["A", "B"]),
        shares,
        dates[-1],
        {
            "lookback_days": 4,
            "market_cap_keep_ratio": 1.0,
            "amount_keep_ratio": 1.0,
            "dividend_years": 3,
            "payout_exclude_ratio": 0.0,
        },
        price_history_lookback_days=1,
    )

    assert snapshot.index.tolist() == ["A", "B"]
    assert snapshot.loc["A", "avg_amount_240d"] == pytest.approx(70.0 / 3.0)
    assert snapshot.loc["B", "avg_amount_240d"] == pytest.approx(200.0)
    assert snapshot.loc["A", "avg_market_cap_240d"] == pytest.approx(1_000.0)
    assert snapshot.loc["B", "avg_market_cap_240d"] == pytest.approx(4_000.0)


def test_public_universe_requires_a_full_market_calendar_window():
    price = make_price(["A"], dates=pd.bdate_range("2024-11-25", periods=3))

    with pytest.raises(ValueError, match="Only 3 market dates are available"):
        build_div_low_vol_universe(
            price,
            make_dividends(["A"]),
            make_queries(["A"]),
            make_shares(["A"]),
            pd.Timestamp("2024-11-27"),
            {
                "lookback_days": 4,
                "market_cap_keep_ratio": 1.0,
                "amount_keep_ratio": 1.0,
                "dividend_years": 3,
                "payout_exclude_ratio": 0.0,
            },
            price_history_lookback_days=1,
        )


def test_payout_filters_negative_and_highest_five_percent_after_continuity():
    symbols = [f"S{index:02d}" for index in range(21)]
    pe_ttm = {symbol: 10.0 for symbol in symbols}
    pe_ttm["S00"] = -10.0
    pe_ttm["S19"] = 100.0
    dividends = make_dividends(symbols)
    dividends.loc[
        dividends["symbol"].eq("S20") & dividends["year"].eq(2021),
        "cash_dividend_before_tax",
    ] = 0.0
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


def test_payout_percentage_excludes_missing_values_before_counting():
    symbols = [f"S{index:02d}" for index in range(40)]
    pe_ttm = {"S37": 90.0, "S38": 100.0, "S39": float("nan")}
    config = make_config(payout_exclude_ratio=0.05)

    out = build_div_low_vol_universe(
        make_price(symbols, pe_ttm=pe_ttm),
        make_dividends(symbols),
        make_queries(symbols),
        make_shares(symbols),
        "2024-11-29",
        config["universe"],
        price_history_lookback_days=1,
    )

    assert "S37" in out.index
    assert "S38" not in out.index
    assert "S39" not in out.index


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
            2023: "2023-04-30",
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
    shares["publish_date"] = pd.to_datetime(shares["publish_date"]).astype(
        "datetime64[ms]"
    )

    out = select_constituents(
        make_price(["A"], dates=pd.bdate_range("2023-12-18", periods=10)),
        dividends,
        make_queries(["A"]),
        shares,
        "2023-12-29",
        make_config(dividend_top_n=1, final_n=1),
    )

    assert out.loc["A", "avg_market_cap_240d"] == pytest.approx(1_000.0)
    assert out.loc["A", "dividend_yield_ttm"] == pytest.approx(0.09)
    assert out.loc["A", "avg_dividend_yield_3y"] == pytest.approx(0.09)


@pytest.mark.parametrize(
    ("as_of", "is_eligible"),
    [("2013-12-20", True), ("2013-12-21", False)],
)
def test_public_universe_applies_december_annual_dividend_cutoff(as_of, is_eligible):
    as_of_date = pd.Timestamp(as_of)
    price = pd.DataFrame(
        {
            "date": [as_of_date],
            "symbol": ["A"],
            "close": [10.0],
            "amount": [1_000.0],
            "pe_ttm": [10.0],
        }
    )
    price["date"] = pd.to_datetime(price["date"]).astype("datetime64[ms]")
    dividends = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "year": [2010, 2011, 2012, 2013],
            "announce_date": pd.to_datetime(
                ["2010-06-01", "2011-06-01", "2012-06-01", "2013-06-01"]
            ),
            "cash_dividend_before_tax": [1.0, 1.0, 1.0, 0.0],
        }
    )
    shares = make_shares(["A"])
    shares["publish_date"] = pd.Timestamp("2009-01-01").as_unit("ms")
    out = build_div_low_vol_universe(
        price,
        dividends,
        make_queries(["A"], range(2010, 2014)),
        shares,
        as_of_date,
        {
            "lookback_days": 1,
            "market_cap_keep_ratio": 1.0,
            "amount_keep_ratio": 1.0,
            "dividend_years": 3,
            "payout_exclude_ratio": 0.0,
        },
        price_history_lookback_days=1,
    )

    assert ("A" in out.index) is is_eligible


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
    with STRATEGY_CONFIG.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    assert "backtest" not in config
    assert config["selection"] == {
        "dividend_yield_lookback_days": 720,
        "dividend_top_n": 75,
        "final_n": 50,
        "volatility_lookback_days": 240,
    }
    assert config["data"] == {
        "start_date": "2014-01-30",
        "end_date": "2023-06-30",
        "pool": "all",
    }


def test_strategy_config_defines_component_output_columns():
    with STRATEGY_CONFIG.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    assert config["output_columns"]["constituent"] == COMPONENTS.CONSTITUENT_COLUMNS
    assert config["output_columns"]["candidate"] == COMPONENTS.CANDIDATE_COLUMNS
    assert config["output_columns"]["index"] == COMPONENTS.INDEX_COLUMNS
    assert config["output_columns"]["monthly_index"] == COMPONENTS.MONTHLY_INDEX_COLUMNS


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
    strategy_config = {
        "universe": config["universe"],
        "strategy_1": config["selection"],
    }
    dividends = make_dividends(symbols)
    dividends["payment_date"] = pd.NaT
    dividends.loc[dividends.index[0], "payment_date"] = pd.Timestamp("2024-02-15")
    factors, coverage = make_adjustment_data(symbols)

    index, constituents = calculate_monthly_rebalanced_index(
        make_price(symbols, dates=dates),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        dates[0],
        dates[-1],
        strategy_config,
        factors,
        coverage,
    )

    assert index.index.min() == pd.Timestamp("2024-01-31")
    assert constituents.index.get_level_values("effective_date").min() == pd.Timestamp(
        "2024-01-31"
    )
    assert index.index.equals(pd.Index(index.index.unique(), name="date"))
    assert index.loc[pd.Timestamp("2024-02-29"), "total_return"] == pytest.approx(0.0)


def test_monthly_rebalance_charges_turnover_cost_after_initial_construction(
    monkeypatch,
):
    symbols = ["A", "B", "C"]
    dates = pd.bdate_range("2024-01-02", periods=45)
    config = make_config(
        market_lookback_days=4,
        dividend_yield_lookback_days=6,
        dividend_top_n=3,
        volatility_lookback_days=4,
        final_n=2,
    )
    strategy_config = {
        "universe": config["universe"],
        "strategy_1": {**config["selection"], "transaction_cost_rate": 0.001},
    }
    dividends = make_dividends(symbols)
    dividends["payment_date"] = pd.NaT
    factors, coverage = make_adjustment_data(symbols)

    def select_constituents(*args):
        as_of_date = pd.Timestamp(args[4])
        symbol = "A" if as_of_date.month == 1 else "B"
        return pd.DataFrame({"weight": [1.0]}, index=pd.Index([symbol], name="symbol"))

    monkeypatch.setattr(
        COMPONENTS, "select_div_low_vol_constituents", select_constituents
    )
    index, _ = calculate_monthly_rebalanced_index(
        make_price(symbols, dates=dates),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        dates[0],
        dates[-1],
        strategy_config,
        factors,
        coverage,
    )

    assert index.loc[pd.Timestamp("2024-02-29"), "total_return"] == pytest.approx(
        -0.00099950025
    )


def test_monthly_rebalance_uses_last_valid_price_before_month_end(monkeypatch):
    dates = pd.to_datetime(
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-28", "2024-02-29"]
    )
    symbols = ["A", "B"]
    config = make_config(
        market_lookback_days=4,
        dividend_yield_lookback_days=4,
        dividend_top_n=2,
        volatility_lookback_days=4,
        final_n=1,
    )
    strategy_config = {"universe": config["universe"], "strategy_1": config["selection"]}
    dividends = make_dividends(symbols)
    dividends["payment_date"] = pd.NaT
    factors, coverage = make_adjustment_data(symbols)
    price = make_price(
        symbols,
        dates=dates,
        closes={"A": [10.0, 10.0, 20.0, 20.0, 20.0], "B": [10.0] * len(dates)},
    )
    price = price.loc[~(price["symbol"].eq("A") & price["date"].eq(dates[-1]))]

    def select_constituents(*args):
        symbol = "A" if pd.Timestamp(args[4]).month == 1 else "B"
        return pd.DataFrame({"weight": [1.0]}, index=pd.Index([symbol], name="symbol"))

    monkeypatch.setattr(
        COMPONENTS, "select_div_low_vol_constituents", select_constituents
    )
    index, _ = calculate_monthly_rebalanced_index(
        price,
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        dates[0],
        dates[-1],
        strategy_config,
        factors,
        coverage,
    )

    assert index.loc[pd.Timestamp("2024-02-29"), "total_return"] == pytest.approx(1.0)
    assert index.loc[pd.Timestamp("2024-02-29"), "total_return_index"] == pytest.approx(
        2.0
    )


def test_strategy_1_config_defaults_to_zero_transaction_cost():
    config = yaml.safe_load(STRATEGY_CONFIG.read_text(encoding="utf-8"))

    assert config["strategy_1"]["transaction_cost_rate"] == 0.0


def test_monthly_rebalance_includes_dividend_cash_in_turnover(monkeypatch):
    symbols = ["A", "B", "C"]
    dates = pd.bdate_range("2024-01-02", periods=45)
    config = make_config(
        market_lookback_days=4,
        dividend_yield_lookback_days=6,
        dividend_top_n=3,
        volatility_lookback_days=4,
        final_n=2,
    )
    strategy_config = {
        "universe": config["universe"],
        "strategy_1": {**config["selection"], "transaction_cost_rate": 0.001},
    }
    dividends = make_dividends(symbols)
    dividends["payment_date"] = pd.NaT
    dividends.loc[dividends.index[0], "payment_date"] = pd.Timestamp("2024-02-15")
    factors, coverage = make_adjustment_data(symbols)

    def select_constituents(*args):
        return pd.DataFrame({"weight": [1.0]}, index=pd.Index(["A"], name="symbol"))

    monkeypatch.setattr(
        COMPONENTS, "select_div_low_vol_constituents", select_constituents
    )
    index, _ = calculate_monthly_rebalanced_index(
        make_price(symbols, dates=dates),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        dates[0],
        dates[-1],
        strategy_config,
        factors,
        coverage,
    )

    assert index.loc[pd.Timestamp("2024-02-29"), "total_return"] == pytest.approx(0.0)


def test_monthly_rebalanced_index_uses_back_adjusted_close(monkeypatch):
    symbols = ["A", "B", "C"]
    dates = pd.bdate_range("2024-01-02", periods=45)
    config = make_config(
        market_lookback_days=4,
        dividend_yield_lookback_days=6,
        dividend_top_n=3,
        volatility_lookback_days=4,
        final_n=1,
    )
    strategy_config = {
        "universe": config["universe"],
        "strategy_1": config["selection"],
    }
    dividends = make_dividends(symbols)
    dividends["payment_date"] = pd.NaT
    factors = pd.DataFrame(
        {
            "symbol": ["A"],
            "operate_date": [pd.Timestamp("2024-02-15")],
            "back_adjust_factor": [1.5],
        }
    )
    coverage = pd.DataFrame({"symbol": symbols, "start": "1990-01-01", "end": "2024-12-31"})
    closes = {
        "A": [10.0 if date < pd.Timestamp("2024-02-15") else 20.0 for date in dates]
    }

    def select_constituents(*args):
        return pd.DataFrame({"weight": [1.0]}, index=pd.Index(["A"], name="symbol"))

    monkeypatch.setattr(
        COMPONENTS, "select_div_low_vol_constituents", select_constituents
    )
    index, _ = calculate_monthly_rebalanced_index(
        make_price(symbols, dates=dates, closes=closes),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        dates[0],
        dates[-1],
        strategy_config,
        factors,
        coverage,
    )

    assert index.loc["2024-02-29", "total_return"] == pytest.approx(2.0)
    assert index.loc["2024-02-29", "total_return_index"] == pytest.approx(3.0)


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
    constituents = pd.DataFrame({"symbol": ["A", "B"], "weight": [0.5, 0.5]}).set_index(
        "symbol"
    )

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
    constituents = pd.DataFrame({"symbol": ["A"], "weight": [1.0]}).set_index("symbol")

    out = calculate_index(
        price,
        empty_index_dividends(),
        make_index_queries(["A"]),
        constituents,
        "2024-01-02",
        "2024-01-03",
    )

    assert out.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
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
            "cash_dividend_before_tax": [1.0, 1.0],
        }
    )
    constituents = pd.DataFrame({"symbol": ["A"], "weight": [1.0]}).set_index("symbol")

    out = calculate_index(
        price,
        dividends,
        make_index_queries(["A"]),
        constituents,
        "2024-01-05",
        "2024-01-09",
    )

    assert out.loc["2024-01-05", "dividend_cash"] == 0.0
    assert out.loc["2024-01-08", "dividend_cash"] == pytest.approx(0.09)
    assert out.loc["2024-01-08", "total_return"] == pytest.approx(0.09)
    assert out.loc["2024-01-08", "price_index"] == pytest.approx(1.0)
    assert out.loc["2024-01-08", "total_return_index"] == pytest.approx(1.09)
    assert out.loc["2024-01-09", "dividend_cash"] == pytest.approx(0.09)
    assert out.loc["2024-01-09", "total_return_index"] == pytest.approx(1.1881)


def test_no_dividend_indices_match_and_unit_base_segments_can_link():
    price = make_index_price(
        [
            ("2024-01-02", "A", 10.0),
            ("2024-01-03", "A", 11.0),
            ("2024-01-04", "A", 12.0),
        ]
    )
    queries = make_index_queries(["A"])
    constituents = pd.DataFrame({"symbol": ["A"], "weight": [1.0]}).set_index("symbol")
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
        first["total_return_index"].iloc[-1] * (1.0 + second["total_return"]).cumprod()
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
                    "cash_dividend_before_tax": [2.0, 1.0],
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
    assert constituents.loc[
        (pd.Timestamp("2024-12-16"), "A"), "as_of_date"
    ] == pd.Timestamp("2024-12-13")


def test_index_validates_coverage_effective_prices_and_duplicate_events():
    price = make_index_price(
        [
            ("2024-01-02", "A", 10.0),
            ("2024-01-03", "A", 10.0),
            ("2024-01-03", "B", 20.0),
        ]
    )
    constituents = pd.DataFrame({"symbol": ["A", "B"], "weight": [0.5, 0.5]}).set_index(
        "symbol"
    )
    single_constituent = pd.DataFrame({"symbol": ["A"], "weight": [1.0]}).set_index(
        "symbol"
    )

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
            "cash_dividend_before_tax": [1.0, 1.0],
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


def test_intraday_requests_use_exact_trading_day_lookback_and_candidate_cap():
    calendar = pd.bdate_range("2024-01-01", periods=25)
    symbols = [f"{index:06d}.SH" for index in range(151)]

    out = build_minute_requests(
        symbols,
        calendar[-1],
        calendar,
        lookback_trading_days=20,
        max_candidates=150,
    )

    assert len(out) == 150
    assert out[0].start_date == calendar[-20].date()
    assert out[0].end_date == calendar[-1].date()


def test_high_frequency_factor_is_market_cap_neutralized():
    dates = pd.bdate_range("2024-01-01", periods=20)
    symbols = ["A", "B", "C"]
    daily = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "volatility": base + slope * index,
            }
            for symbol, base, slope in [
                ("A", 0.01, 0.0001),
                ("B", 0.02, 0.0004),
                ("C", 0.03, 0.0009),
            ]
            for index, trade_date in enumerate(dates)
        ]
    )
    market_cap = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "total_market_cap": value,
            }
            for symbol, value in zip(
                symbols,
                [100.0, 200.0, 500.0],
                strict=True,
            )
            for trade_date in dates
        ]
    )

    out = calculate_high_frequency_factor(
        daily,
        market_cap,
        dates[-1],
    )

    residual = out["minute_return_volatility"].to_numpy()
    assert residual.sum() == pytest.approx(0.0, abs=1e-12)
    assert residual @ out["log_market_cap"].to_numpy() == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_high_frequency_factor_excludes_symbols_with_missing_lookback_days():
    dates = pd.bdate_range("2024-01-01", periods=20)
    daily = pd.DataFrame(
        [
            {"symbol": symbol, "trade_date": date, "volatility": 0.01 + index / 10_000}
            for symbol in ["A", "B", "C"]
            for index, date in enumerate(dates)
            if not (symbol == "C" and date == dates[-1])
        ]
    )
    market_cap = pd.DataFrame(
        [
            {"symbol": symbol, "trade_date": date, "total_market_cap": float(index + 1)}
            for symbol, index in zip(["A", "B", "C"], range(1, 4), strict=True)
            for date in dates
        ]
    )

    out = calculate_high_frequency_factor(daily, market_cap, dates[-1])

    assert out.index.tolist() == ["A", "B"]


def test_high_frequency_constituents_sort_factor_and_weight_by_dividend_yield(
    monkeypatch,
):
    config = make_config(dividend_top_n=3, final_n=2)
    strategy_config = {
        "universe": config["universe"],
        "selection": {
            "dividend_yield_lookback_days": 6,
            "dividend_top_n": 3,
            "final_n": 2,
            "lookback_trading_days": 20,
            "min_valid_days": 20,
            "transaction_cost_rate": 0.0,
        },
    }
    metrics = pd.DataFrame(
        {
            "symbol": ["B", "A", "C"],
            "as_of_date": pd.Timestamp("2024-11-29"),
            "price_date": pd.Timestamp("2024-11-29"),
            "avg_market_cap_240d": [100.0, 100.0, 100.0],
            "avg_amount_240d": [100.0, 100.0, 100.0],
            "dividend_yield_ttm": [0.1, 0.1, 0.1],
            "payout_ratio": [0.5, 0.5, 0.5],
            "dividend_growth_slope": [0.0, 0.0, 0.0],
            "dividend_yield_rank": [2, 1, 3],
            "avg_dividend_yield_3y": [2.0, 1.0, 3.0],
            "minute_return_volatility": [0.1, 0.1, 0.2],
            "intraday_volatility_cv": [0.2, 0.3, 0.4],
            "log_market_cap": [1.0, 2.0, 3.0],
        }
    )
    monkeypatch.setattr(
        COMPONENTS,
        "_calculate_high_frequency_candidate_metrics",
        lambda *args, **kwargs: metrics.copy(),
    )

    out = select_high_frequency_constituents(
        make_price(["A", "B", "C"]),
        make_dividends(["A", "B", "C"]),
        make_queries(["A", "B", "C"]),
        make_shares(["A", "B", "C"]),
        pd.DataFrame(columns=["symbol", "trade_date", "volatility"]),
        "2024-11-29",
        strategy_config,
    )

    assert out.index.tolist() == ["A", "B"]
    assert out["volatility_rank"].tolist() == [1, 2]
    assert out["weight"].tolist() == pytest.approx([1 / 3, 2 / 3])


def test_high_frequency_constituents_run_candidate_and_factor_pipeline():
    symbols = ["A", "B", "C"]
    dates = pd.bdate_range("2024-11-04", periods=20)
    base = make_config(
        market_lookback_days=4,
        dividend_yield_lookback_days=6,
        dividend_top_n=3,
        final_n=2,
    )
    strategy_config = {
        "universe": base["universe"],
        "selection": {
            "dividend_yield_lookback_days": 6,
            "dividend_top_n": 3,
            "final_n": 2,
            "lookback_trading_days": 20,
            "min_valid_days": 20,
            "transaction_cost_rate": 0.0,
        },
    }
    daily_volatility = pd.DataFrame(
        [
            {"symbol": symbol, "trade_date": date, "volatility": 0.01 + index * (0.001 + offset / 10_000)}
            for offset, symbol in enumerate(symbols)
            for index, date in enumerate(dates)
        ]
    )

    out = select_high_frequency_constituents(
        make_price(symbols, dates=dates),
        make_dividends(symbols),
        make_queries(symbols),
        make_shares(symbols),
        daily_volatility,
        dates[-1],
        strategy_config,
    )

    assert len(out) == 2
    assert out["weight"].sum() == pytest.approx(1.0)
    assert out["minute_return_volatility"].is_monotonic_increasing


def test_high_frequency_monthly_index_uses_shared_rebalance_backtest(monkeypatch):
    symbols = [f"S{index:03d}" for index in range(3)]
    dates = pd.bdate_range("2024-01-02", periods=45)
    base = make_config(
        market_lookback_days=4,
        dividend_yield_lookback_days=6,
        dividend_top_n=3,
        final_n=2,
    )
    config = {"universe": base["universe"], "strategy_2": {
        "dividend_yield_lookback_days": 6,
        "dividend_top_n": 3,
        "final_n": 2,
        "lookback_trading_days": 20,
        "min_valid_days": 20,
        "transaction_cost_rate": 0.0,
    }}
    dividends = make_dividends(symbols)
    factors, coverage = make_adjustment_data(symbols)

    def fake_select(*args):
        date = pd.Timestamp(args[5])
        symbol = "S000" if date.month == 1 else "S001"
        return pd.DataFrame(
            {"weight": [1.0]}, index=pd.Index([symbol], name="symbol")
        )

    monkeypatch.setattr(COMPONENTS, "select_high_frequency_div_low_vol_constituents", fake_select)
    index, constituents = calculate_high_frequency_index(
        make_price(symbols, dates=dates),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        pd.DataFrame(columns=["symbol", "trade_date", "volatility"]),
        dates[0],
        dates[-1],
        config,
        factors,
        coverage,
    )

    assert len(index) == 3
    assert constituents.index.get_level_values("symbol").tolist() == [
        "S000",
        "S001",
        "S001",
    ]


def test_high_frequency_candidate_groups_keep_factor_ic_metadata(monkeypatch):
    symbols = [f"S{index:02d}" for index in range(10)]
    dates = pd.bdate_range("2024-01-02", periods=45)
    base = make_config(market_lookback_days=4, dividend_yield_lookback_days=6)
    config = {"universe": base["universe"], "strategy_2": {
        "dividend_yield_lookback_days": 6,
        "dividend_top_n": 10,
        "final_n": 2,
        "lookback_trading_days": 20,
        "min_valid_days": 20,
        "transaction_cost_rate": 0.0,
    }}
    dividends = make_dividends(symbols)
    factors, coverage = make_adjustment_data(symbols)

    def fake_metrics(*args):
        return pd.DataFrame({
            "symbol": symbols,
            "minute_return_volatility": [float(index) for index in range(10)],
        })

    monkeypatch.setattr(
        COMPONENTS,
        "_calculate_high_frequency_candidate_metrics",
        fake_metrics,
    )
    groups = calculate_high_frequency_groups(
        make_price(symbols, dates=dates),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        pd.DataFrame(columns=["symbol", "trade_date", "volatility"]),
        dates[0],
        dates[-1],
        config,
        factors,
        coverage,
    )

    assert groups.columns.tolist() == list(COMPONENTS.TRADITIONAL_VOLATILITY_GROUP_COLUMNS)
    assert groups.iloc[0].eq(1.0).all()
    assert groups.attrs["factor_ic"].columns.tolist() == ["ic", "rank_ic"]


def test_traditional_volatility_groups_are_balanced_and_stably_sorted():
    symbols = [f"S{index:02d}" for index in range(11)]
    volatility = {symbol: float(index // 2) for index, symbol in enumerate(symbols)}

    groups = COMPONENTS._assign_traditional_volatility_groups(symbols, volatility)

    assert groups.index.tolist() == sorted(symbols, key=lambda symbol: (volatility[symbol], symbol))
    assert groups.value_counts().sort_index().tolist() == [3, 2, 2, 2, 2]
    assert groups.iloc[0] == 1
    assert groups.iloc[-1] == 5


def test_traditional_volatility_group_indices_use_full_dividend_pool_and_long_short(
    monkeypatch,
):
    symbols = [f"S{index:02d}" for index in range(10)]
    dates = pd.bdate_range("2024-01-02", periods=45)
    config = make_config(
        market_lookback_days=4,
        dividend_yield_lookback_days=6,
        dividend_top_n=3,
        volatility_lookback_days=4,
        final_n=2,
    )
    strategy_config = {
        "universe": config["universe"],
        "strategy_1": config["selection"],
    }
    dividends = make_dividends(symbols)
    factors, coverage = make_adjustment_data(symbols)

    def fake_universe(*args):
        return pd.DataFrame(index=pd.Index(symbols, name="symbol"))

    def fake_volatility_snapshots(adjusted_price, rebalance_dates, lookback_days):
        return {
            date: {symbol: float(symbol[1:]) for symbol in symbols}
            for date in rebalance_dates
        }

    monkeypatch.setattr(COMPONENTS, "build_div_low_vol_universe", fake_universe)
    monkeypatch.setattr(
        COMPONENTS,
        "_calculate_traditional_volatility_snapshots",
        fake_volatility_snapshots,
    )

    result = calculate_volatility_groups(
        make_price(symbols, dates=dates),
        dividends,
        make_queries(symbols),
        make_shares(symbols),
        dates[0],
        dates[-1],
        strategy_config,
        factors,
        coverage,
    )

    expected_columns = list(COMPONENTS.TRADITIONAL_VOLATILITY_GROUP_COLUMNS)
    assert set(result) == {"all_a", "dividend_pool"}
    for nav in result.values():
        assert nav.columns.tolist() == expected_columns
        assert nav.iloc[0].eq(1.0).all()
        group_returns = nav.iloc[:, :5].pct_change().fillna(0.0)
        assert nav[expected_columns[-1]].iloc[1] == pytest.approx(
            (1 + group_returns.iloc[1, 0] - group_returns.iloc[1, 4])
        )

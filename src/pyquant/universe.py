"""Universe construction helpers."""

from math import floor
from typing import Optional

import numpy as np
import pandas as pd


def build_universe(
    price: pd.DataFrame,
    symbols: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """从标准行情长表生成逐日股票池。"""
    required = {"date", "symbol"}
    missing = required - set(price.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    df = price.loc[:, ["date", "symbol"]].copy()

    if symbols is not None:
        df = df[df["symbol"].isin([str(symbol) for symbol in symbols])]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    out = df.drop_duplicates().sort_values(["date", "symbol"])
    out["in_universe"] = True
    return out.set_index(["date", "symbol"])


def build_dividend_low_vol_universe(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    config: dict,
    price_history_lookback_days: int,
    prepared: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build the point-in-time dividend low-volatility universe.

    This applies only the sample-space and dividend-quality filters. The
    strategy remains responsible for factor ranking and portfolio weights.
    """
    _validate_dividend_low_vol_config(config, price_history_lookback_days)
    as_of_date = pd.Timestamp(as_of_date)
    prepared = prepared or prepare_dividend_low_vol_universe_inputs(
        price, dividends, dividend_queries, shares
    )
    price_data = prepared["price"]
    price_data = price_data[price_data["date"].le(as_of_date)]
    if price_data.empty:
        raise ValueError(f"No price data is available on or before {as_of_date.date()}")
    dividend_data = prepared["dividends"]
    query_data = prepared["dividend_queries"]
    market_data = prepared["market_data"]
    snapshot = _dividend_low_vol_market_snapshot(
        price_data, market_data, as_of_date, config["lookback_days"]
    )
    symbols = sorted(
        _top_dividend_low_vol_symbols(
            snapshot, "avg_market_cap_240d", config["market_cap_keep_ratio"]
        )
        & _top_dividend_low_vol_symbols(
            snapshot, "avg_amount_240d", config["amount_keep_ratio"]
        )
        & set(
            prepared["price_counts"].loc[
                lambda data: data["date"].eq(as_of_date)
                & data["observation_count"].ge(price_history_lookback_days),
                "symbol",
            ]
        )
    )
    if not symbols:
        raise ValueError("No symbols passed the market, liquidity, and price-history filters")
    _require_dividend_low_vol_query_coverage(
        query_data,
        symbols,
        _dividend_low_vol_continuous_years(as_of_date, config["dividend_years"]),
        f"selection at {as_of_date.date()}",
    )
    metrics = _add_dividend_low_vol_metrics(
        snapshot[snapshot["symbol"].isin(symbols)].copy(),
        dividend_data,
        as_of_date,
        config["dividend_years"],
    )
    metrics = metrics[metrics["consecutive_dividends"]].copy()
    high_payout = set(
        metrics.dropna(subset=["payout_ratio"])
        .sort_values(["payout_ratio", "symbol"], ascending=[False, True])
        .head(floor(len(metrics) * config["payout_exclude_ratio"]))["symbol"]
    )
    return metrics.loc[
        metrics["payout_ratio"].ge(0)
        & metrics["dividend_growth_slope"].ge(0)
        & ~metrics["symbol"].isin(high_payout)
    ].set_index("symbol")


def prepare_dividend_low_vol_universe_inputs(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Prepare reusable inputs for repeated dividend low-volatility selections."""
    price_data = _prepare_dividend_low_vol_price(price)
    share_data = _prepare_dividend_low_vol_shares(shares)
    market_data = pd.merge_asof(
        price_data.sort_values(["date", "symbol"]),
        share_data.sort_values(["publish_date", "symbol"]),
        left_on="date",
        right_on="publish_date",
        by="symbol",
        direction="backward",
    )
    market_data["total_market_cap"] = market_data["close"] * market_data["total_shares"]
    price_counts = price_data[["date", "symbol"]].copy()
    price_counts["observation_count"] = price_counts.groupby("symbol").cumcount() + 1
    return {
        "price": price_data,
        "dividends": _prepare_dividend_low_vol_dividends(dividends),
        "dividend_queries": _prepare_dividend_low_vol_queries(dividend_queries),
        "market_data": market_data,
        "price_counts": price_counts,
    }


def _validate_dividend_low_vol_config(config: dict, price_history_days: int) -> None:
    required = {
        "lookback_days",
        "market_cap_keep_ratio",
        "amount_keep_ratio",
        "dividend_years",
        "payout_exclude_ratio",
    }
    _require_dividend_low_vol_columns(config, required, "config")
    if config["lookback_days"] <= 0 or config["dividend_years"] < 2:
        raise ValueError("lookback_days must be positive and dividend_years at least 2")
    if price_history_days <= 0:
        raise ValueError("price_history_lookback_days must be positive")
    for name in ["market_cap_keep_ratio", "amount_keep_ratio"]:
        if not 0 < config[name] <= 1:
            raise ValueError("Universe keep ratios must be in (0, 1]")
    if not 0 <= config["payout_exclude_ratio"] < 1:
        raise ValueError("payout_exclude_ratio must be in [0, 1)")


def _prepare_dividend_low_vol_price(price: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "close", "amount", "pe_ttm"}
    _require_dividend_low_vol_columns(price, required, "price")
    out = price.loc[:, sorted(required)].copy()
    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("price contains duplicate (date, symbol) rows")
    out.loc[out["amount"].lt(0), "amount"] = np.nan
    return out[out["close"].gt(0)].sort_values(["symbol", "date"]).reset_index(drop=True)


def _prepare_dividend_low_vol_dividends(dividends: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "year", "announce_date", "cash_dividend_after_tax"}
    _require_dividend_low_vol_columns(dividends, required, "dividends")
    return dividends.loc[:, sorted(required)].dropna(
        subset=["announce_date", "cash_dividend_after_tax"]
    )


def _prepare_dividend_low_vol_queries(dividend_queries: pd.DataFrame) -> pd.DataFrame:
    if {"symbol", "year"}.issubset(dividend_queries.columns):
        return dividend_queries[["symbol", "year"]].drop_duplicates().sort_values(["symbol", "year"])
    required = {"symbol", "start", "end"}
    _require_dividend_low_vol_columns(dividend_queries, required, "dividend_queries")
    out = dividend_queries.loc[:, sorted(required)].copy()
    out = out.loc[out["start"] <= out["end"]]
    out = out.loc[out.index.repeat(out["end"].dt.year - out["start"].dt.year + 1)].copy()
    out["year"] = out.groupby(level=0).cumcount() + out["start"].dt.year
    return out[["symbol", "year"]].drop_duplicates().sort_values(["symbol", "year"])


def _prepare_dividend_low_vol_shares(shares: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "publish_date", "total_shares"}
    _require_dividend_low_vol_columns(shares, required, "shares")
    out = shares.copy()
    if "report_date" in out:
        out = out.sort_values(["symbol", "publish_date", "report_date"])
    out = out.dropna(subset=["publish_date", "total_shares"])
    return out[out["total_shares"] > 0].drop_duplicates(
        ["symbol", "publish_date"], keep="last"
    )


def _dividend_low_vol_market_snapshot(
    price: pd.DataFrame,
    market_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    dates = (
        price.loc[price["date"].le(as_of_date), "date"]
        .drop_duplicates()
        .sort_values()
        .tail(lookback_days)
    )
    if len(dates) < lookback_days:
        raise ValueError(f"Only {len(dates)} market dates are available; at least {lookback_days} are required")
    snapshot = price[price["date"].eq(as_of_date)].copy()
    if snapshot.empty:
        raise ValueError(f"No price data is available on {as_of_date.date()}")
    average = market_data[market_data["date"].isin(dates)].groupby("symbol", as_index=False).agg(
        avg_market_cap_240d=("total_market_cap", "mean"),
        avg_amount_240d=("amount", "mean"),
    )
    return snapshot.merge(average, on="symbol", how="left")


def _add_dividend_low_vol_metrics(
    metrics: pd.DataFrame,
    dividends: pd.DataFrame,
    as_of_date: pd.Timestamp,
    dividend_years: int,
) -> pd.DataFrame:
    visible = dividends[dividends["announce_date"].le(as_of_date)]
    annual = visible.groupby(["symbol", "year"])["cash_dividend_after_tax"].sum()
    trailing = visible[
        visible["announce_date"] > as_of_date - pd.Timedelta(days=365)
    ].groupby("symbol")["cash_dividend_after_tax"].sum()
    years = _dividend_low_vol_continuous_years(as_of_date, dividend_years)
    metrics["consecutive_dividends"] = metrics["symbol"].map(
        lambda symbol: all(annual.get((symbol, year), 0.0) > 0 for year in years)
    )

    def growth(symbol: str) -> float:
        current_announced = annual.get((symbol, as_of_date.year), 0.0) > 0
        first_year = as_of_date.year - dividend_years + int(current_announced)
        values = [annual.get((symbol, year), 0.0) for year in range(first_year, first_year + dividend_years)]
        slope = float(np.polyfit(np.arange(dividend_years), values, 1)[0])
        return 0.0 if np.isclose(slope, 0.0, atol=1e-12) else slope

    metrics["dividend_growth_slope"] = metrics["symbol"].map(growth)
    metrics["dividend_yield_ttm"] = metrics["symbol"].map(trailing).fillna(0.0).div(metrics["close"])
    metrics["payout_ratio"] = metrics["dividend_yield_ttm"] * metrics["pe_ttm"]
    return metrics


def _dividend_low_vol_continuous_years(as_of_date: pd.Timestamp, dividend_years: int) -> range:
    start = as_of_date.year - dividend_years + (as_of_date.month == 12 and as_of_date.day >= 21)
    return range(start, start + dividend_years)


def _require_dividend_low_vol_query_coverage(
    queries: pd.DataFrame, symbols: list[str], years: range, context: str
) -> None:
    missing = sorted(
        {(symbol, year) for symbol in symbols for year in years}
        - set(queries.itertuples(index=False, name=None))
    )
    if missing:
        raise ValueError(
            f"Dividend query coverage missing for {len(missing)} symbol-years during {context}; examples: {missing[:5]}"
        )


def _top_dividend_low_vol_symbols(
    data: pd.DataFrame, column: str, keep_ratio: float
) -> set[str]:
    return set(
        data.sort_values([column, "symbol"], ascending=[False, True])
        .head(floor(len(data) * keep_ratio))["symbol"]
    )


def _require_dividend_low_vol_columns(
    data: pd.DataFrame | dict, required: set[str], name: str
) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

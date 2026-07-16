"""Strategy-specific components for the dividend low-volatility strategy."""

from math import floor

import numpy as np
import pandas as pd


UNIVERSE_OUTPUT_COLUMNS = [
    "avg_market_cap_240d",
    "avg_amount_240d",
    "dividend_yield_ttm",
    "payout_ratio",
    "dividend_growth_slope",
    "in_universe",
]


def build_dividend_universe(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build the monthly dividend stock universe from announced dividends."""
    settings, start, end = _validate_config(config)
    price_data = _prepare_price(price)
    dividend_data = _prepare_dividends(dividends)
    query_data = _prepare_dividend_queries(dividend_queries)
    share_data = _prepare_shares(shares)

    missing_share_symbols = sorted(
        set(price_data["symbol"]) - set(share_data["symbol"])
    )
    if missing_share_symbols:
        raise ValueError(
            "Total-share data missing for "
            f"{len(missing_share_symbols)} price symbols; examples: "
            f"{missing_share_symbols[:5]}"
        )

    daily = _add_rolling_market_metrics(
        price_data,
        share_data,
        settings["lookback_days"],
    )
    trading_dates = daily["date"].drop_duplicates().sort_values()
    rebalance_dates = trading_dates.groupby(trading_dates.dt.to_period("M")).max()
    if start is not None:
        rebalance_dates = rebalance_dates[rebalance_dates >= start]
    if end is not None:
        rebalance_dates = rebalance_dates[rebalance_dates <= end]

    completed_queries = set(query_data.itertuples(index=False, name=None))
    results = []
    for rebalance_date in rebalance_dates:
        selected = _build_monthly_universe(
            daily,
            dividend_data,
            completed_queries,
            pd.Timestamp(rebalance_date),
            settings,
        )
        if not selected.empty:
            results.append(selected)
    if not results:
        return _empty_universe()

    return (
        pd.concat(results, ignore_index=True)
        .set_index(["date", "symbol"])
        .sort_index()[UNIVERSE_OUTPUT_COLUMNS]
    )


def calc_factor(
    price: pd.DataFrame,
    universe: pd.DataFrame,
    config: dict,
) -> pd.Series:
    """Calculate the strategy factor."""
    raise NotImplementedError("dividend_low_vol factor logic is not implemented yet")


def _validate_config(config: dict) -> tuple[dict, pd.Timestamp | None, pd.Timestamp | None]:
    try:
        settings = config["universe"]
        backtest = config["backtest"]
        lookback_days = int(settings["lookback_days"])
        dividend_years = int(settings["dividend_years"])
        market_cap_keep_ratio = float(settings["market_cap_keep_ratio"])
        amount_keep_ratio = float(settings["amount_keep_ratio"])
        payout_exclude_ratio = float(settings["payout_exclude_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid dividend-universe configuration") from exc

    if lookback_days <= 0 or dividend_years < 2:
        raise ValueError("lookback_days must be positive and dividend_years at least 2")
    if not 0 < market_cap_keep_ratio <= 1 or not 0 < amount_keep_ratio <= 1:
        raise ValueError("Universe keep ratios must be in (0, 1]")
    if not 0 <= payout_exclude_ratio < 1:
        raise ValueError("payout_exclude_ratio must be in [0, 1)")
    if backtest.get("rebalance") != "monthly":
        raise ValueError("Dividend universe requires monthly rebalancing")

    start = pd.Timestamp(backtest["start"]) if backtest.get("start") else None
    end = pd.Timestamp(backtest["end"]) if backtest.get("end") else None
    if start is not None and end is not None and start > end:
        raise ValueError("backtest.start must not be after backtest.end")
    return {
        "lookback_days": lookback_days,
        "dividend_years": dividend_years,
        "market_cap_keep_ratio": market_cap_keep_ratio,
        "amount_keep_ratio": amount_keep_ratio,
        "payout_exclude_ratio": payout_exclude_ratio,
    }, start, end


def _prepare_price(price: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "close", "amount", "pe_ttm"}
    _require_columns(price, required, "price")
    out = price.loc[:, sorted(required)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if out["symbol"].isna().any():
        raise ValueError("price.symbol must not contain missing values")
    out["symbol"] = out["symbol"].astype(str)
    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("price contains duplicate (date, symbol) rows")
    for column in ["close", "amount", "pe_ttm"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _prepare_dividends(dividends: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "year", "announce_date", "cash_dividend_after_tax"}
    _require_columns(dividends, required, "dividends")
    out = dividends.loc[:, sorted(required)].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out["announce_date"] = pd.to_datetime(out["announce_date"], errors="coerce")
    out["cash_dividend_after_tax"] = pd.to_numeric(
        out["cash_dividend_after_tax"], errors="coerce"
    )
    return out.dropna(subset=["announce_date", "cash_dividend_after_tax"])


def _prepare_dividend_queries(dividend_queries: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "year"}
    _require_columns(dividend_queries, required, "dividend_queries")
    out = dividend_queries.loc[:, sorted(required)].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    return out.drop_duplicates().sort_values(["symbol", "year"])


def _prepare_shares(shares: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "publish_date", "total_shares"}
    _require_columns(shares, required, "shares")
    columns = sorted(required | ({"report_date"} & set(shares.columns)))
    out = shares.loc[:, columns].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["publish_date"] = pd.to_datetime(out["publish_date"], errors="coerce")
    out["total_shares"] = pd.to_numeric(out["total_shares"], errors="coerce")
    if "report_date" in out:
        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
        out = out.sort_values(["symbol", "publish_date", "report_date"])
    out = out.dropna(subset=["publish_date", "total_shares"])
    out = out[out["total_shares"] > 0]
    return out.drop_duplicates(["symbol", "publish_date"], keep="last")


def _add_rolling_market_metrics(
    price: pd.DataFrame,
    shares: pd.DataFrame,
    lookback_days: int,
) -> pd.DataFrame:
    daily = pd.merge_asof(
        price.sort_values(["date", "symbol"]),
        shares.sort_values(["publish_date", "symbol"]),
        left_on="date",
        right_on="publish_date",
        by="symbol",
        direction="backward",
    ).sort_values(["symbol", "date"])
    daily["total_market_cap"] = daily["close"] * daily["total_shares"]
    for source, target in [
        ("total_market_cap", "avg_market_cap_240d"),
        ("amount", "avg_amount_240d"),
    ]:
        daily[target] = (
            daily.groupby("symbol", sort=False)[source]
            .rolling(lookback_days, min_periods=lookback_days)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return daily


def _build_monthly_universe(
    daily: pd.DataFrame,
    dividends: pd.DataFrame,
    completed_queries: set[tuple[str, int]],
    rebalance_date: pd.Timestamp,
    settings: dict,
) -> pd.DataFrame:
    snapshot = daily.loc[
        daily["date"].eq(rebalance_date),
        [
            "symbol",
            "close",
            "pe_ttm",
            "avg_market_cap_240d",
            "avg_amount_240d",
        ],
    ].dropna(subset=["avg_market_cap_240d", "avg_amount_240d"])
    if snapshot.empty:
        return pd.DataFrame()

    market_symbols = _top_symbols(
        snapshot,
        "avg_market_cap_240d",
        settings["market_cap_keep_ratio"],
    )
    amount_symbols = _top_symbols(
        snapshot,
        "avg_amount_240d",
        settings["amount_keep_ratio"],
    )
    ranked_symbols = sorted(market_symbols & amount_symbols)
    if not ranked_symbols:
        return pd.DataFrame()

    year = rebalance_date.year
    required_queries = {
        (symbol, query_year)
        for symbol in ranked_symbols
        for query_year in range(year - settings["dividend_years"], year + 1)
    }
    missing_queries = sorted(required_queries - completed_queries)
    if missing_queries:
        raise ValueError(
            "Dividend query coverage missing for "
            f"{len(missing_queries)} symbol-years at {rebalance_date.date()}; "
            f"examples: {missing_queries[:5]}"
        )

    metrics = snapshot[snapshot["symbol"].isin(ranked_symbols)].copy()
    visible = dividends[dividends["announce_date"] <= rebalance_date]
    annual_dividends = visible.groupby(["symbol", "year"])[
        "cash_dividend_after_tax"
    ].sum()
    trailing = visible[
        visible["announce_date"] > rebalance_date - pd.Timedelta(days=365)
    ].groupby("symbol")["cash_dividend_after_tax"].sum()

    dividend_years = settings["dividend_years"]
    continuous_start = year - dividend_years + (1 if rebalance_date.month == 12 else 0)
    continuous_years = range(continuous_start, continuous_start + dividend_years)
    metrics["consecutive_dividends"] = metrics["symbol"].map(
        lambda symbol: all(
            annual_dividends.get((symbol, item), 0.0) > 0 for item in continuous_years
        )
    )
    metrics = metrics[metrics["consecutive_dividends"]].copy()
    if metrics.empty:
        return pd.DataFrame()

    def dividend_growth(symbol: str) -> float:
        current_paid = annual_dividends.get((symbol, year), 0.0) > 0
        first_year = year - dividend_years + (1 if current_paid else 0)
        values = [
            annual_dividends.get((symbol, item), 0.0)
            for item in range(first_year, first_year + dividend_years)
        ]
        return float(np.polyfit(np.arange(dividend_years), values, 1)[0])

    metrics["dividend_growth_slope"] = metrics["symbol"].map(dividend_growth)
    metrics["dividend_yield_ttm"] = metrics["symbol"].map(trailing).fillna(0.0) / metrics[
        "close"
    ]
    metrics["payout_ratio"] = metrics["dividend_yield_ttm"] * metrics["pe_ttm"]

    high_payout_count = floor(len(metrics) * settings["payout_exclude_ratio"])
    high_payout_symbols = set(
        metrics.dropna(subset=["payout_ratio"])
        .sort_values(["payout_ratio", "symbol"], ascending=[False, True])
        .head(high_payout_count)["symbol"]
    )
    selected = metrics[
        metrics["payout_ratio"].ge(0)
        & metrics["dividend_growth_slope"].ge(0)
        & ~metrics["symbol"].isin(high_payout_symbols)
    ].copy()
    selected["date"] = rebalance_date
    selected["in_universe"] = True
    return selected[["date", "symbol", *UNIVERSE_OUTPUT_COLUMNS]]


def _top_symbols(data: pd.DataFrame, column: str, keep_ratio: float) -> set[str]:
    count = floor(len(data) * keep_ratio)
    return set(
        data.sort_values([column, "symbol"], ascending=[False, True])
        .head(count)["symbol"]
        .tolist()
    )


def _require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _empty_universe() -> pd.DataFrame:
    out = pd.DataFrame(columns=UNIVERSE_OUTPUT_COLUMNS)
    out.index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
    return out

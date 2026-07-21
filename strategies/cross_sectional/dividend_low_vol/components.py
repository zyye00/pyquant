"""Original dividend low-volatility index components."""

from math import floor

import numpy as np
import pandas as pd


CONSTITUENT_COLUMNS = [
    "as_of_date",
    "price_date",
    "avg_market_cap_240d",
    "avg_amount_240d",
    "dividend_yield_ttm",
    "payout_ratio",
    "dividend_growth_slope",
    "avg_dividend_yield_3y",
    "volatility_240d",
    "dividend_yield_rank",
    "volatility_rank",
    "weight",
]
INDEX_COLUMNS = [
    "price_return",
    "total_return",
    "dividend_cash",
    "price_index",
    "total_return_index",
]


def select_dividend_low_vol_download_symbols(
    price: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    config: dict,
) -> list[str]:
    """Return symbols with enough valid price observations for selection.

    The result is intended to limit dividend and share downloads to securities
    that can satisfy the strategy's dividend-yield lookback at ``as_of_date``.
    """
    settings = _validate_selection_config(config)
    as_of = pd.Timestamp(as_of_date)
    price_data = _prepare_price(price)
    counts = price_data.loc[price_data["date"].le(as_of)].groupby("symbol").size()
    return sorted(
        counts.loc[counts.ge(settings["dividend_yield_lookback_days"])].index.tolist()
    )


def select_dividend_low_vol_constituents(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    config: dict,
) -> pd.DataFrame:
    """Select one point-in-time constituent snapshot and dividend-yield weights."""
    settings = _validate_selection_config(config)
    as_of = pd.Timestamp(as_of_date)
    price_data = _prepare_price(price)
    price_data = price_data[price_data["date"] <= as_of]
    if price_data.empty:
        raise ValueError(f"No price data is available on or before {as_of.date()}")
    dividend_data = _prepare_selection_dividends(dividends)
    query_data = _prepare_dividend_queries(dividend_queries)
    share_data = _prepare_shares(shares)

    daily = _add_rolling_market_metrics(
        price_data,
        share_data,
        settings["market_lookback_days"],
    )
    snapshot = (
        daily.sort_values(["symbol", "date"])
        .groupby("symbol", sort=False)
        .tail(1)
        .dropna(subset=["avg_market_cap_240d", "avg_amount_240d"])
    )
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
    price_history_symbols = set(
        price_data.groupby("symbol")
        .size()
        .loc[lambda counts: counts.ge(settings["dividend_yield_lookback_days"])]
        .index
    )
    eligible_symbols = sorted(market_symbols & amount_symbols & price_history_symbols)
    if not eligible_symbols:
        raise ValueError("No symbols passed the market, liquidity, and price-history filters")

    required_years = _continuous_dividend_years(as_of, settings["dividend_years"])
    _require_query_coverage(
        query_data,
        eligible_symbols,
        required_years,
        f"selection at {as_of.date()}",
    )
    metrics = _add_dividend_metrics(
        snapshot[snapshot["symbol"].isin(eligible_symbols)].copy(),
        dividend_data,
        as_of,
        settings["dividend_years"],
    )
    metrics = metrics[metrics["consecutive_dividends"]].copy()
    high_payout_count = floor(len(metrics) * settings["payout_exclude_ratio"])
    high_payout_symbols = set(
        metrics.dropna(subset=["payout_ratio"])
        .sort_values(["payout_ratio", "symbol"], ascending=[False, True])
        .head(high_payout_count)["symbol"]
    )
    metrics = metrics[
        metrics["payout_ratio"].ge(0)
        & metrics["dividend_growth_slope"].ge(0)
        & ~metrics["symbol"].isin(high_payout_symbols)
    ].copy()

    candidate_price = price_data[price_data["symbol"].isin(metrics["symbol"])]
    metrics["avg_dividend_yield_3y"] = metrics["symbol"].map(
        _average_ttm_dividend_yield(
            candidate_price,
            dividend_data[dividend_data["announce_date"] <= as_of],
            settings["dividend_yield_lookback_days"],
        )
    )
    metrics = metrics.dropna(subset=["avg_dividend_yield_3y"])
    metrics = metrics.sort_values(
        ["avg_dividend_yield_3y", "symbol"], ascending=[False, True]
    ).head(settings["dividend_top_n"])
    metrics["dividend_yield_rank"] = np.arange(1, len(metrics) + 1)

    candidate_price = price_data[price_data["symbol"].isin(metrics["symbol"])]
    metrics["volatility_240d"] = metrics["symbol"].map(
        _price_volatility(candidate_price, settings["volatility_lookback_days"])
    )
    metrics = metrics.dropna(subset=["volatility_240d"])
    if len(metrics) < settings["final_n"]:
        raise ValueError(
            f"Only {len(metrics)} eligible symbols remain; "
            f"at least {settings['final_n']} are required"
        )
    metrics = metrics.sort_values(
        ["volatility_240d", "symbol"], ascending=[True, True]
    ).head(settings["final_n"])
    metrics["volatility_rank"] = np.arange(1, len(metrics) + 1)
    weight_total = metrics["avg_dividend_yield_3y"].sum()
    if not np.isfinite(weight_total) or weight_total <= 0:
        raise ValueError("Selected dividend yields must sum to a positive value")
    metrics["weight"] = metrics["avg_dividend_yield_3y"] / weight_total
    metrics["as_of_date"] = as_of
    metrics["price_date"] = metrics["date"]
    return metrics.set_index("symbol")[CONSTITUENT_COLUMNS]


def calculate_dividend_low_vol_index(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    constituents: pd.DataFrame,
    effective_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    price_base_value: float = 1000.0,
    total_return_base_value: float = 1000.0,
) -> pd.DataFrame:
    """Calculate one fixed-constituent price and total-return index segment."""
    effective = pd.Timestamp(effective_date)
    end = pd.Timestamp(end_date)
    if effective > end:
        raise ValueError("effective_date must not be after end_date")
    if price_base_value <= 0 or total_return_base_value <= 0:
        raise ValueError("Index base values must be positive")
    members = _prepare_constituents(constituents)
    price_data = _prepare_index_price(price)
    query_data = _prepare_dividend_queries(dividend_queries)
    symbols = members.index.tolist()
    _require_query_coverage(
        query_data,
        symbols,
        range(effective.year - 1, end.year + 1),
        f"index period {effective.date()} to {end.date()}",
    )

    member_price = price_data[price_data["symbol"].isin(symbols)]
    calendar = pd.Index(
        price_data.loc[price_data["date"].between(effective, end), "date"]
        .drop_duplicates()
        .sort_values(),
        name="date",
    )
    if effective not in calendar:
        raise ValueError(f"effective_date is not present in the price calendar: {effective.date()}")
    prices = (
        member_price[member_price["date"].le(end)]
        .pivot(index="date", columns="symbol", values="close")
        .reindex(columns=symbols)
        .reindex(
            price_data.loc[price_data["date"].le(end), "date"]
            .drop_duplicates()
            .sort_values()
        )
        .ffill()
        .reindex(calendar)
    )
    missing = prices.loc[effective][prices.loc[effective].isna()].index.tolist()
    if missing:
        raise ValueError(
            "No price is available on or before effective_date for "
            f"{len(missing)} constituents; examples: {missing[:5]}"
        )
    prices = prices.ffill()
    normalized_shares = members["weight"] / prices.loc[effective]
    portfolio_value = prices.mul(normalized_shares, axis="columns").sum(axis=1)

    dividend_cash = _index_dividend_cash(
        dividends,
        normalized_shares,
        calendar,
        effective,
        end,
    )
    price_return = portfolio_value.pct_change(fill_method=None).fillna(0.0)
    total_return = (
        (portfolio_value + dividend_cash)
        .div(portfolio_value.shift(1))
        .sub(1.0)
        .fillna(0.0)
    )
    out = pd.DataFrame(
        {
            "price_return": price_return,
            "total_return": total_return,
            "dividend_cash": dividend_cash,
            "price_index": price_base_value
            * portfolio_value.div(portfolio_value.iloc[0]),
            "total_return_index": total_return_base_value
            * (1.0 + total_return).cumprod(),
        },
        index=calendar,
    )
    return out[INDEX_COLUMNS]


def _validate_selection_config(config: dict) -> dict:
    try:
        universe = config["universe"]
        selection = config["selection"]
        settings = {
            "market_lookback_days": int(universe["lookback_days"]),
            "market_cap_keep_ratio": float(universe["market_cap_keep_ratio"]),
            "amount_keep_ratio": float(universe["amount_keep_ratio"]),
            "dividend_years": int(universe["dividend_years"]),
            "payout_exclude_ratio": float(universe["payout_exclude_ratio"]),
            "dividend_yield_lookback_days": int(
                selection["dividend_yield_lookback_days"]
            ),
            "dividend_top_n": int(selection["dividend_top_n"]),
            "volatility_lookback_days": int(selection["volatility_lookback_days"]),
            "final_n": int(selection["final_n"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid dividend-low-volatility selection configuration") from exc
    if settings["market_lookback_days"] <= 0 or settings["dividend_years"] < 2:
        raise ValueError("lookback_days must be positive and dividend_years at least 2")
    for name in ["market_cap_keep_ratio", "amount_keep_ratio"]:
        if not 0 < settings[name] <= 1:
            raise ValueError("Universe keep ratios must be in (0, 1]")
    if not 0 <= settings["payout_exclude_ratio"] < 1:
        raise ValueError("payout_exclude_ratio must be in [0, 1)")
    for name in [
        "dividend_yield_lookback_days",
        "dividend_top_n",
        "volatility_lookback_days",
        "final_n",
    ]:
        if settings[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if settings["dividend_top_n"] < settings["final_n"]:
        raise ValueError("dividend_top_n must not be smaller than final_n")
    return settings


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
    out = out[out["close"].gt(0) & out["amount"].ge(0)]
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _prepare_index_price(price: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "close"}
    _require_columns(price, required, "price")
    out = price.loc[:, sorted(required)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["symbol"] = out["symbol"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("price contains duplicate (date, symbol) rows")
    return out.dropna(subset=["close"]).sort_values(["date", "symbol"])


def _prepare_selection_dividends(dividends: pd.DataFrame) -> pd.DataFrame:
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
    if {"symbol", "year"}.issubset(dividend_queries.columns):
        out = dividend_queries[["symbol", "year"]].copy()
        out["symbol"] = out["symbol"].astype(str)
        out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
        return out.drop_duplicates().sort_values(["symbol", "year"])
    required = {"symbol", "start", "end"}
    _require_columns(dividend_queries, required, "dividend_queries")
    out = dividend_queries.loc[:, sorted(required)].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["start"] = pd.to_datetime(out["start"], errors="raise")
    out["end"] = pd.to_datetime(out["end"], errors="raise")
    out = out.loc[out["start"] <= out["end"]]
    out = out.loc[
        out.index.repeat(out["end"].dt.year - out["start"].dt.year + 1)
    ].copy()
    out["year"] = out.groupby(level=0).cumcount() + out["start"].dt.year
    return out[["symbol", "year"]].drop_duplicates().sort_values(["symbol", "year"])


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


def _prepare_constituents(constituents: pd.DataFrame) -> pd.DataFrame:
    out = constituents.copy()
    if out.index.name != "symbol":
        if "symbol" not in out:
            raise ValueError("constituents must have a symbol index or column")
        out = out.set_index("symbol")
    if out.index.has_duplicates:
        raise ValueError("constituents contain duplicate symbols")
    _require_columns(out, {"weight"}, "constituents")
    out.index = out.index.astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    if out["weight"].isna().any() or not out["weight"].gt(0).all():
        raise ValueError("constituent weights must be positive numbers")
    if not np.isclose(out["weight"].sum(), 1.0):
        raise ValueError("constituent weights must sum to 1")
    return out


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


def _add_dividend_metrics(
    metrics: pd.DataFrame,
    dividends: pd.DataFrame,
    as_of: pd.Timestamp,
    dividend_years: int,
) -> pd.DataFrame:
    visible = dividends[dividends["announce_date"] <= as_of]
    annual = visible.groupby(["symbol", "year"])["cash_dividend_after_tax"].sum()
    trailing = visible[
        visible["announce_date"] > as_of - pd.Timedelta(days=365)
    ].groupby("symbol")["cash_dividend_after_tax"].sum()
    continuous_years = _continuous_dividend_years(as_of, dividend_years)
    metrics["consecutive_dividends"] = metrics["symbol"].map(
        lambda symbol: all(
            annual.get((symbol, year), 0.0) > 0 for year in continuous_years
        )
    )

    def growth(symbol: str) -> float:
        current_announced = annual.get((symbol, as_of.year), 0.0) > 0
        first_year = as_of.year - dividend_years + (1 if current_announced else 0)
        values = [
            annual.get((symbol, year), 0.0)
            for year in range(first_year, first_year + dividend_years)
        ]
        slope = float(np.polyfit(np.arange(dividend_years), values, 1)[0])
        return 0.0 if np.isclose(slope, 0.0, atol=1e-12) else slope

    metrics["dividend_growth_slope"] = metrics["symbol"].map(growth)
    metrics["dividend_yield_ttm"] = metrics["symbol"].map(trailing).fillna(0.0).div(
        metrics["close"]
    )
    metrics["payout_ratio"] = metrics["dividend_yield_ttm"] * metrics["pe_ttm"]
    return metrics


def _current_year_is_available(as_of: pd.Timestamp) -> bool:
    """Treat December 21 onward as the month-end annual-data cutoff."""
    return as_of.month == 12 and as_of.day >= 21


def _continuous_dividend_years(as_of: pd.Timestamp, dividend_years: int) -> range:
    start = as_of.year - dividend_years + _current_year_is_available(as_of)
    return range(start, start + dividend_years)


def _average_ttm_dividend_yield(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    lookback_days: int,
) -> dict[str, float]:
    values = {}
    for symbol, history in price.groupby("symbol", sort=False):
        history = history.dropna(subset=["close"])
        history = history[history["close"] > 0].tail(lookback_days)
        if len(history) < lookback_days:
            continue
        events = dividends[dividends["symbol"] == symbol].sort_values("announce_date")
        event_dates = events["announce_date"].to_numpy(dtype="datetime64[ns]")
        cash = events["cash_dividend_after_tax"].to_numpy(dtype=float)
        cumulative = np.concatenate(([0.0], np.cumsum(cash)))
        dates = history["date"].to_numpy(dtype="datetime64[ns]")
        right = np.searchsorted(event_dates, dates, side="right")
        left = np.searchsorted(
            event_dates,
            dates - np.timedelta64(365, "D"),
            side="right",
        )
        trailing_cash = cumulative[right] - cumulative[left]
        values[str(symbol)] = float((trailing_cash / history["close"].to_numpy()).mean())
    return values


def _price_volatility(price: pd.DataFrame, lookback_days: int) -> dict[str, float]:
    values = {}
    for symbol, history in price.groupby("symbol", sort=False):
        close = history["close"].dropna()
        close = close[close > 0].tail(lookback_days)
        if len(close) < lookback_days:
            continue
        values[str(symbol)] = float(close.pct_change(fill_method=None).dropna().std())
    return values


def _index_dividend_cash(
    dividends: pd.DataFrame,
    normalized_shares: pd.Series,
    calendar: pd.Index,
    effective: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    required = {"symbol", "payment_date", "cash_dividend_after_tax"}
    _require_columns(dividends, required, "dividends")
    event_key = [
        column
        for column in [
            "symbol",
            "year",
            "announce_date",
            "record_date",
            "operate_date",
            "payment_date",
            "cash_dividend_after_tax",
        ]
        if column in dividends
    ]
    if dividends.duplicated(event_key).any():
        raise ValueError(f"dividends contain duplicate event keys: {event_key}")
    events = dividends.loc[:, sorted(required)].copy()
    events["symbol"] = events["symbol"].astype(str)
    events["payment_date"] = pd.to_datetime(events["payment_date"], errors="coerce")
    events["cash_dividend_after_tax"] = pd.to_numeric(
        events["cash_dividend_after_tax"], errors="coerce"
    )
    events = events[
        events["symbol"].isin(normalized_shares.index)
        & events["payment_date"].gt(effective)
        & events["payment_date"].le(end)
        & events["cash_dividend_after_tax"].gt(0)
    ]
    cash = pd.Series(0.0, index=calendar, name="dividend_cash")
    for event in events.itertuples(index=False):
        position = calendar.searchsorted(event.payment_date, side="left")
        if position < len(calendar):
            cash.iloc[position] += (
                normalized_shares[event.symbol] * event.cash_dividend_after_tax
            )
    return cash


def _require_query_coverage(
    queries: pd.DataFrame,
    symbols: list[str],
    years: range,
    context: str,
) -> None:
    completed = set(queries.itertuples(index=False, name=None))
    required = {(symbol, year) for symbol in symbols for year in years}
    missing = sorted(required - completed)
    if missing:
        raise ValueError(
            f"Dividend query coverage missing for {len(missing)} symbol-years "
            f"during {context}; examples: {missing[:5]}"
        )


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

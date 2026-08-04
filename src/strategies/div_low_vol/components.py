"""Original dividend low-volatility index components."""

from importlib.resources import files

import numpy as np
import pandas as pd
import yaml

from pyquant import (
    DIVIDEND_AFTER_TAX_RATIO,
    MinuteRequest,
    build_div_low_vol_universe,
    get_period_end_dates,
    normalize_query_years,
    prepare_div_low_vol_universe_inputs,
    run_backtest,
)


config_file = files("strategies.div_low_vol").joinpath("config.yaml")
with config_file.open(encoding="utf-8") as _config_stream:
    _OUTPUT_COLUMNS = (yaml.safe_load(_config_stream) or {})["output_columns"]

CONSTITUENT_COLUMNS = _OUTPUT_COLUMNS["constituent"]
CANDIDATE_COLUMNS = _OUTPUT_COLUMNS["candidate"]
INDEX_COLUMNS = _OUTPUT_COLUMNS["index"]
MONTHLY_INDEX_COLUMNS = _OUTPUT_COLUMNS["monthly_index"]


def select_div_low_vol_download_symbols(
    price: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    config: dict,
) -> list[str]:
    """Return symbols with enough valid price observations for selection.

    The result is intended to limit dividend and share downloads to securities
    that can satisfy the strategy's dividend-yield lookback at ``as_of_date``.
    """
    _validate_selection_config(config)
    as_of_date = pd.Timestamp(as_of_date)
    price_data = _prepare_price(price)
    counts = price_data.loc[price_data["date"].le(as_of_date)].groupby("symbol").size()
    return sorted(
        counts.loc[
            counts.ge(config["selection"]["dividend_yield_lookback_days"])
        ].index.tolist()
    )


def select_div_low_vol_constituents(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    config: dict,
    prepared: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Select one point-in-time constituent snapshot and dividend-yield weights."""
    prepared = prepared or prepare_div_low_vol_universe_inputs(
        price, dividends, dividend_queries, shares
    )
    metrics = select_div_low_vol_candidates(
        price,
        dividends,
        dividend_queries,
        shares,
        as_of_date,
        config,
        prepared,
    ).reset_index()
    selection = config["selection"]
    price_data = prepared["price"]
    price_data = price_data[price_data["date"] <= pd.Timestamp(as_of_date)]
    candidate_price = price_data[price_data["symbol"].isin(metrics["symbol"])]
    metrics["volatility_240d"] = metrics["symbol"].map(
        _price_volatility(candidate_price, selection["volatility_lookback_days"])
    )
    metrics = metrics.dropna(subset=["volatility_240d"])
    if len(metrics) < selection["final_n"]:
        raise ValueError(
            f"Only {len(metrics)} eligible symbols remain; "
            f"at least {selection['final_n']} are required"
        )
    metrics = metrics.sort_values(
        ["volatility_240d", "symbol"], ascending=[True, True]
    ).head(selection["final_n"])
    metrics["volatility_rank"] = np.arange(1, len(metrics) + 1)
    weight_total = metrics["avg_dividend_yield_3y"].sum()
    if not np.isfinite(weight_total) or weight_total <= 0:
        raise ValueError("Selected dividend yields must sum to a positive value")
    metrics["weight"] = metrics["avg_dividend_yield_3y"] / weight_total
    return metrics.set_index("symbol")[CONSTITUENT_COLUMNS]


def select_div_low_vol_candidates(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    config: dict,
    prepared: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Select the dividend-ranked candidates before applying a volatility factor."""
    _validate_selection_config(config)
    selection = config["selection"]
    as_of_date = pd.Timestamp(as_of_date)
    prepared = prepared or prepare_div_low_vol_universe_inputs(
        price, dividends, dividend_queries, shares
    )
    price_data = prepared["price"]
    price_data = price_data[price_data["date"] <= as_of_date]
    dividend_data = prepared["dividends"]
    metrics = build_div_low_vol_universe(
        price_data,
        dividends,
        dividend_queries,
        shares,
        as_of_date,
        config["universe"],
        selection["dividend_yield_lookback_days"],
        prepared,
    ).reset_index()

    candidate_price = price_data[price_data["symbol"].isin(metrics["symbol"])]
    metrics["avg_dividend_yield_3y"] = metrics["symbol"].map(
        _average_ttm_dividend_yield(
            candidate_price,
            dividend_data[dividend_data["announce_date"] <= as_of_date],
            selection["dividend_yield_lookback_days"],
        )
    )
    metrics = metrics.dropna(subset=["avg_dividend_yield_3y"])
    metrics = metrics.sort_values(
        ["avg_dividend_yield_3y", "symbol"], ascending=[False, True]
    ).head(selection["dividend_top_n"])
    metrics["dividend_yield_rank"] = np.arange(1, len(metrics) + 1)
    metrics["as_of_date"] = as_of_date
    metrics["price_date"] = metrics["date"]
    return metrics.set_index("symbol")[CANDIDATE_COLUMNS]


def build_intraday_minute_requests(
    symbols: list[str],
    signal_date: str | pd.Timestamp,
    trading_dates: pd.Index,
    *,
    lookback_trading_days: int = 20,
    max_candidates: int = 150,
    retain_raw: bool = True,
) -> list[MinuteRequest]:
    """Build one 20-trading-day minute request per ranked candidate."""
    if lookback_trading_days <= 0 or max_candidates <= 0:
        raise ValueError("lookback_trading_days and max_candidates must be positive")
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_dates, errors="raise"))
    if calendar.hasnans:
        raise ValueError("trading_dates must not contain invalid values")
    calendar = calendar.normalize().drop_duplicates().sort_values()
    signal_date = pd.Timestamp(signal_date).normalize()
    eligible = calendar[calendar <= signal_date]
    if signal_date not in calendar:
        raise ValueError("signal_date must be a trading date")
    if len(eligible) < lookback_trading_days:
        raise ValueError("Not enough trading dates for the intraday lookback")
    symbols = list(dict.fromkeys(map(str, symbols)))[:max_candidates]
    return [
        MinuteRequest(
            symbol,
            eligible[-lookback_trading_days],
            eligible[-1],
            retain_raw,
        )
        for symbol in symbols
    ]


def calculate_high_frequency_volatility_factor(
    daily_volatility: pd.DataFrame,
    market_cap: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    *,
    lookback_trading_days: int = 20,
    min_valid_days: int = 20,
) -> pd.DataFrame:
    """Calculate and market-cap-neutralize the report's 20-day intraday factor."""
    if not 2 <= min_valid_days <= lookback_trading_days:
        raise ValueError(
            "min_valid_days must be at least 2 and no larger than the lookback"
        )
    _require_columns(
        daily_volatility,
        {"symbol", "trade_date", "volatility"},
        "daily_volatility",
    )
    _require_columns(
        market_cap,
        {"symbol", "trade_date", "total_market_cap"},
        "market_cap",
    )
    if daily_volatility.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("daily_volatility contains duplicate symbol-date rows")
    if market_cap.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("market_cap contains duplicate symbol-date rows")
    as_of_date = pd.Timestamp(as_of_date).normalize()
    volatility = daily_volatility.copy()
    volatility["trade_date"] = pd.to_datetime(volatility["trade_date"], errors="raise")
    volatility["volatility"] = pd.to_numeric(volatility["volatility"], errors="coerce")
    caps = market_cap.copy()
    caps["trade_date"] = pd.to_datetime(caps["trade_date"], errors="raise")
    caps["total_market_cap"] = pd.to_numeric(caps["total_market_cap"], errors="coerce")
    window_dates = (
        caps.loc[caps["trade_date"].le(as_of_date), "trade_date"]
        .drop_duplicates()
        .sort_values()
        .tail(lookback_trading_days)
    )
    if len(window_dates) < lookback_trading_days:
        raise ValueError("Not enough market-cap trading dates for the factor lookback")
    volatility = volatility[
        volatility["trade_date"].isin(window_dates) & volatility["volatility"].gt(0)
    ].sort_values(["symbol", "trade_date"])
    rows = []
    for symbol, history in volatility.groupby("symbol", sort=False):
        values = history["volatility"].dropna().tail(lookback_trading_days)
        if len(values) < min_valid_days:
            continue
        mean = values.mean()
        factor = values.std(ddof=1) / mean
        if np.isfinite(factor) and mean > 0:
            rows.append((str(symbol), factor))
    factors = pd.DataFrame(
        rows,
        columns=["symbol", "intraday_volatility_cv"],
    )
    caps = (
        caps[caps["trade_date"].le(as_of_date) & caps["total_market_cap"].gt(0)]
        .sort_values(["symbol", "trade_date"])
        .groupby("symbol", sort=False)
        .tail(1)
    )
    factors = factors.merge(
        caps[["symbol", "total_market_cap"]],
        on="symbol",
        how="inner",
        validate="one_to_one",
    )
    if len(factors) < 2:
        raise ValueError(
            "At least two eligible symbols are required for neutralization"
        )
    factors["log_market_cap"] = np.log(factors["total_market_cap"])
    design = np.column_stack(
        [np.ones(len(factors)), factors["log_market_cap"].to_numpy()]
    )
    fitted = (
        design
        @ np.linalg.lstsq(
            design,
            factors["intraday_volatility_cv"].to_numpy(),
            rcond=None,
        )[0]
    )
    factors["minute_return_volatility"] = factors["intraday_volatility_cv"] - fitted
    factors["as_of_date"] = as_of_date
    return factors.set_index("symbol")[
        [
            "as_of_date",
            "intraday_volatility_cv",
            "log_market_cap",
            "minute_return_volatility",
        ]
    ].sort_index()


def calculate_div_low_vol_index(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    constituents: pd.DataFrame,
    effective_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    prepared: dict | None = None,
) -> pd.DataFrame:
    """Calculate one unit-based fixed-constituent index segment."""
    effective = pd.Timestamp(effective_date)
    end = pd.Timestamp(end_date)
    if effective > end:
        raise ValueError("effective_date must not be after end_date")
    members = _prepare_constituents(constituents)
    prepared = prepared or _prepare_index_inputs(price, dividends, dividend_queries)
    price_data = prepared["price"]
    symbols = members.index.tolist()
    _require_query_coverage_from_set(
        prepared["query_coverage"],
        symbols,
        range(effective.year - 1, end.year + 1),
        f"index period {effective.date()} to {end.date()}",
    )

    calendar = pd.Index(
        price_data.loc[price_data["date"].between(effective, end), "date"]
        .drop_duplicates()
        .sort_values(),
        name="date",
    )
    if effective not in calendar:
        raise ValueError(
            f"effective_date is not present in the price calendar: {effective.date()}"
        )
    prices = (
        prepared["prices"].reindex(columns=symbols).loc[:end].ffill().reindex(calendar)
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
        prepared["dividend_events"],
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
            "price_index": portfolio_value.div(portfolio_value.iloc[0]),
            "total_return_index": (1.0 + total_return).cumprod(),
        },
        index=calendar,
    )
    return out[INDEX_COLUMNS]


def calculate_div_low_vol_rebalanced_index(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate annually rebalanced price and total-return indices.

    Constituents are selected at each December's second-Friday close and take
    effect on the next available price-calendar date.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    price_data = _prepare_index_price(price)
    calendar = pd.Index(
        price_data.loc[price_data["date"].between(start, end), "date"]
        .drop_duplicates()
        .sort_values(),
        name="date",
    )
    schedule = _annual_rebalance_schedule(calendar, start, end)
    if not schedule:
        raise ValueError("No annual rebalance effective date falls within the period")

    return _calculate_div_low_vol_rebalanced_index(
        price,
        dividends,
        dividend_queries,
        shares,
        end,
        config,
        schedule,
    )


def calculate_div_low_vol_monthly_rebalanced_index(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the strategy 1 monthly-rebalanced total-return index."""
    try:
        strategy_config = {
            "universe": config["universe"],
            "selection": config["strategy_1"],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing strategy_1 configuration") from exc
    _validate_selection_config(strategy_config)
    transaction_cost_rate = strategy_config["selection"].get(
        "transaction_cost_rate", 0.001
    )
    if not 0 <= transaction_cost_rate < 1:
        raise ValueError("transaction_cost_rate must be in [0, 1)")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    price_data = _prepare_index_price(price)
    calendar = pd.Index(
        price_data.loc[price_data["date"].between(start, end), "date"]
        .drop_duplicates()
        .sort_values(),
        name="date",
    )
    rebalance_dates = get_period_end_dates(calendar)
    if rebalance_dates.empty:
        raise ValueError("No monthly rebalance effective date falls within the period")
    return _calculate_div_low_vol_monthly_index(
        price,
        dividends,
        dividend_queries,
        shares,
        strategy_config,
        rebalance_dates,
        transaction_cost_rate,
    )


def _calculate_div_low_vol_monthly_index(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    config: dict,
    rebalance_dates: pd.DatetimeIndex,
    transaction_cost_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_div_low_vol_universe_inputs(
        price, dividends, dividend_queries, shares
    )
    index_inputs = _prepare_index_inputs(price, dividends, dividend_queries)
    constituent_snapshots = []
    constituent_weights = {}

    for rebalance_date in rebalance_dates:
        constituents = select_div_low_vol_constituents(
            price,
            dividends,
            dividend_queries,
            shares,
            rebalance_date,
            config,
            prepared,
        )
        constituent_weights[rebalance_date] = constituents["weight"]
        constituent_snapshots.append(
            constituents.reset_index()
            .assign(effective_date=rebalance_date)
            .set_index(["effective_date", "symbol"])
        )
    target_weights = pd.DataFrame(constituent_weights).T.fillna(0.0)
    target_weights.index.name = "date"
    cash_events = index_inputs["dividend_events"].rename(
        columns={
            "payment_date": "date",
            "cash_dividend_before_tax": "cash_per_share",
        }
    )
    cash_events["cash_per_share"] = (
        cash_events["cash_per_share"] * DIVIDEND_AFTER_TAX_RATIO
    )
    cash_events = cash_events.loc[
        cash_events["date"].notna() & cash_events["cash_per_share"].gt(0)
    ]
    index = run_backtest(
        index_inputs["prices"],
        target_weights,
        cash_events[["date", "symbol", "cash_per_share"]],
        fee_rate=transaction_cost_rate,
    )
    return index[MONTHLY_INDEX_COLUMNS], pd.concat(constituent_snapshots).sort_index()


def _calculate_div_low_vol_rebalanced_index(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    end_date: pd.Timestamp,
    config: dict,
    schedule: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_div_low_vol_universe_inputs(
        price, dividends, dividend_queries, shares
    )
    index_inputs = _prepare_index_inputs(price, dividends, dividend_queries)
    index_segments = []
    constituent_snapshots = []
    for position, (as_of_date, effective_date) in enumerate(schedule):
        constituents = select_div_low_vol_constituents(
            price,
            dividends,
            dividend_queries,
            shares,
            as_of_date,
            config,
            prepared,
        )
        constituent_snapshots.append(
            constituents.reset_index()
            .assign(effective_date=effective_date)
            .set_index(["effective_date", "symbol"])
        )
        segment_end = (
            schedule[position + 1][1] if position + 1 < len(schedule) else end_date
        )
        segment = calculate_div_low_vol_index(
            price,
            dividends,
            dividend_queries,
            constituents,
            effective_date,
            segment_end,
            index_inputs,
        )
        if index_segments:
            segment["price_index"] = (
                index_segments[-1]["price_index"].iloc[-1]
                * (1.0 + segment["price_return"]).cumprod()
            )
            segment["total_return_index"] = (
                index_segments[-1]["total_return_index"].iloc[-1]
                * (1.0 + segment["total_return"]).cumprod()
            )
            segment = segment.iloc[1:]
        index_segments.append(segment)
    return (
        pd.concat(index_segments),
        pd.concat(constituent_snapshots).sort_index(),
    )


def _annual_rebalance_schedule(
    calendar: pd.Index,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    schedule = []
    for year in range(start.year, end.year + 1):
        as_of_date = pd.date_range(f"{year}-12-01", periods=2, freq="W-FRI")[1]
        position = calendar.searchsorted(as_of_date, side="right")
        if position < len(calendar):
            schedule.append((as_of_date, calendar[position]))
    return schedule


def _validate_selection_config(config: dict) -> None:
    try:
        universe = config["universe"]
        selection = config["selection"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Invalid dividend-low-volatility selection configuration"
        ) from exc
    if universe["lookback_days"] <= 0 or universe["dividend_years"] < 2:
        raise ValueError("lookback_days must be positive and dividend_years at least 2")
    for name in ["market_cap_keep_ratio", "amount_keep_ratio"]:
        if not 0 < universe[name] <= 1:
            raise ValueError("Universe keep ratios must be in (0, 1]")
    if not 0 <= universe["payout_exclude_ratio"] < 1:
        raise ValueError("payout_exclude_ratio must be in [0, 1)")
    for name in [
        "dividend_yield_lookback_days",
        "dividend_top_n",
        "volatility_lookback_days",
        "final_n",
    ]:
        if selection[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if selection["dividend_top_n"] < selection["final_n"]:
        raise ValueError("dividend_top_n must not be smaller than final_n")


def _prepare_price(price: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "close", "amount", "pe_ttm"}
    _require_columns(price, required, "price")
    out = price.loc[:, sorted(required)].copy()
    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("price contains duplicate (date, symbol) rows")
    out.loc[out["amount"].lt(0), "amount"] = np.nan
    out = out[out["close"].gt(0)]
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _prepare_index_price(price: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "close"}
    _require_columns(price, required, "price")
    out = price.loc[:, sorted(required)].copy()
    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("price contains duplicate (date, symbol) rows")
    return out.dropna(subset=["close"]).sort_values(["date", "symbol"])


def _prepare_index_inputs(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
) -> dict:
    price_data = _prepare_index_price(price)
    required = {"symbol", "payment_date", "cash_dividend_before_tax"}
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
            "cash_dividend_before_tax",
        ]
        if column in dividends
    ]
    if dividends.duplicated(event_key).any():
        raise ValueError(f"dividends contain duplicate event keys: {event_key}")
    events = dividends.loc[:, sorted(required)].copy()
    return {
        "price": price_data,
        "prices": price_data.pivot(index="date", columns="symbol", values="close"),
        "dividend_events": events,
        "query_coverage": set(
            normalize_query_years(dividend_queries).itertuples(index=False, name=None)
        ),
    }


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
        cash = (
            events["cash_dividend_before_tax"].to_numpy(dtype=float)
            * DIVIDEND_AFTER_TAX_RATIO
        )
        cumulative = np.concatenate(([0.0], np.cumsum(cash)))
        dates = history["date"].to_numpy(dtype="datetime64[ns]")
        right = np.searchsorted(event_dates, dates, side="right")
        left = np.searchsorted(
            event_dates,
            dates - np.timedelta64(365, "D"),
            side="right",
        )
        trailing_cash = cumulative[right] - cumulative[left]
        values[str(symbol)] = float(
            (trailing_cash / history["close"].to_numpy()).mean()
        )
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
    events = dividends.loc[
        dividends["symbol"].isin(normalized_shares.index)
        & dividends["payment_date"].gt(effective)
        & dividends["payment_date"].le(end)
        & dividends["cash_dividend_before_tax"].gt(0)
    ]
    cash = pd.Series(0.0, index=calendar, name="dividend_cash")
    for event in events.itertuples(index=False):
        position = calendar.searchsorted(event.payment_date, side="left")
        if position < len(calendar):
            cash.iloc[position] += (
                normalized_shares[event.symbol]
                * event.cash_dividend_before_tax
                * DIVIDEND_AFTER_TAX_RATIO
            )
    return cash


def _require_query_coverage_from_set(
    completed: set[tuple],
    symbols: list[str],
    years: range,
    context: str,
) -> None:
    missing = sorted(
        {(symbol, year) for symbol in symbols for year in years} - completed
    )
    if missing:
        raise ValueError(
            f"Dividend query coverage missing for {len(missing)} symbol-years "
            f"during {context}; examples: {missing[:5]}"
        )


def _require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

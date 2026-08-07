"""Original dividend low-volatility index components."""

from importlib.resources import files

import numpy as np
import pandas as pd
import yaml

from pyquant import (
    DIVIDEND_AFTER_TAX_RATIO,
    build_back_adjusted_close,
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
TRADITIONAL_VOLATILITY_GROUP_COLUMNS = (
    "第1组（低波）",
    "第2组",
    "第3组",
    "第4组",
    "第5组（高波）",
    "多空对冲（低波-高波）",
)
HIGH_FREQUENCY_CONSTITUENT_COLUMNS = _OUTPUT_COLUMNS[
    "high_frequency_constituent"
]


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
    volatility_price: pd.DataFrame | None = None,
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
    volatility_data = _prepare_index_price(
        volatility_price if volatility_price is not None else prepared["price"]
    )
    volatility_data = volatility_data[
        volatility_data["date"] <= pd.Timestamp(as_of_date)
    ]
    candidate_price = volatility_data[
        volatility_data["symbol"].isin(metrics["symbol"])
    ]
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
    _validate_candidate_config(config)
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


def select_high_frequency_div_low_vol_constituents(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    config: dict,
    prepared: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Select strategy-2 constituents from the high-frequency factor."""
    metrics = _calculate_high_frequency_candidate_metrics(
        price,
        dividends,
        dividend_queries,
        shares,
        daily_volatility,
        as_of_date,
        config,
        prepared,
    )
    selection = config["selection"]
    metrics = metrics.sort_values(
        ["minute_return_volatility", "symbol"], ascending=[True, True]
    ).head(selection["final_n"])
    if len(metrics) < selection["final_n"]:
        raise ValueError(
            f"Only {len(metrics)} eligible symbols remain on "
            f"{pd.Timestamp(as_of_date).date()}; at least "
            f"{selection['final_n']} are required"
        )
    metrics["volatility_rank"] = np.arange(1, len(metrics) + 1)
    weight_total = metrics["avg_dividend_yield_3y"].sum()
    if not np.isfinite(weight_total) or weight_total <= 0:
        raise ValueError("Selected dividend yields must sum to a positive value")
    metrics["weight"] = metrics["avg_dividend_yield_3y"] / weight_total
    return metrics.set_index("symbol")[list(HIGH_FREQUENCY_CONSTITUENT_COLUMNS)]


def _calculate_high_frequency_candidate_metrics(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    config: dict,
    prepared: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    _validate_strategy_2_config(config)
    prepared = prepared or prepare_div_low_vol_universe_inputs(
        price, dividends, dividend_queries, shares
    )
    candidates = select_div_low_vol_candidates(
        price,
        dividends,
        dividend_queries,
        shares,
        as_of_date,
        config,
        prepared,
    ).reset_index()
    market_cap = prepared["market_data"].loc[
        :, ["symbol", "date", "total_market_cap"]
    ].rename(columns={"date": "trade_date"})
    factor = calculate_high_frequency_volatility_factor(
        daily_volatility.loc[
            daily_volatility["symbol"].astype(str).isin(candidates["symbol"])
        ],
        market_cap.loc[market_cap["symbol"].astype(str).isin(candidates["symbol"])],
        as_of_date,
        lookback_trading_days=config["selection"]["lookback_trading_days"],
        min_valid_days=config["selection"]["min_valid_days"],
    ).drop(columns="as_of_date").reset_index()
    metrics = candidates.merge(factor, on="symbol", how="inner", validate="one_to_one")
    if len(metrics) < config["selection"]["final_n"]:
        raise ValueError(
            f"Only {len(metrics)} eligible symbols remain on "
            f"{pd.Timestamp(as_of_date).date()}; at least "
            f"{config['selection']['final_n']} are required"
        )
    return metrics


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
    adjustment_factors: pd.DataFrame,
    adjustment_factor_coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate strategy 1 using BaoStock back-adjusted monthly prices."""
    try:
        strategy_config = {
            "universe": config["universe"],
            "selection": config["strategy_1"],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing strategy_1 configuration") from exc
    _validate_selection_config(strategy_config)
    transaction_cost_rate = strategy_config["selection"].get(
        "transaction_cost_rate", 0.0
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
        adjustment_factors,
        adjustment_factor_coverage,
    )


def calculate_high_frequency_div_low_vol_monthly_rebalanced_index(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    config: dict,
    adjustment_factors: pd.DataFrame,
    adjustment_factor_coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate strategy 2 with monthly high-frequency-volatility selection."""
    strategy_config = _strategy_2_selection_config(config)
    _validate_strategy_2_config(strategy_config)
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
    prepared = prepare_div_low_vol_universe_inputs(
        price, dividends, dividend_queries, shares
    )
    return _calculate_monthly_index_with_selector(
        price,
        dividends,
        dividend_queries,
        shares,
        strategy_config,
        rebalance_dates,
        strategy_config["selection"]["transaction_cost_rate"],
        adjustment_factors,
        adjustment_factor_coverage,
        lambda date, adjusted, shared: select_high_frequency_div_low_vol_constituents(
            price,
            dividends,
            dividend_queries,
            shares,
            daily_volatility,
            date,
            strategy_config,
            shared,
        ),
        prepared,
    )


def calculate_high_frequency_volatility_candidate_group_indices(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    config: dict,
    adjustment_factors: pd.DataFrame,
    adjustment_factor_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate factor groups and IC within monthly top-dividend candidates."""
    strategy_config = _strategy_2_selection_config(config)
    _validate_strategy_2_config(strategy_config)
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
    prepared = prepare_div_low_vol_universe_inputs(
        price, dividends, dividend_queries, shares
    )
    adjusted_price = build_back_adjusted_close(
        price,
        adjustment_factors,
        adjustment_factor_coverage,
    ).loc[:, ["date", "symbol", "adjusted_close"]].rename(
        columns={"adjusted_close": "close"}
    )
    monthly_prices = (
        adjusted_price.pivot(index="date", columns="symbol", values="close")
        .ffill()
        .reindex(rebalance_dates)
    )
    snapshots = []
    groups = []
    for rebalance_date in rebalance_dates:
        metrics = _calculate_high_frequency_candidate_metrics(
            price,
            dividends,
            dividend_queries,
            shares,
            daily_volatility,
            rebalance_date,
            strategy_config,
            prepared,
        )
        factor = metrics.set_index("symbol")["minute_return_volatility"]
        groups.append(
            _assign_factor_groups(
                factor.index.tolist(),
                factor.to_dict(),
                rebalance_date,
                "high-frequency candidates",
            )
        )
        snapshots.append(factor)
    nav = _run_factor_group_backtest(monthly_prices, rebalance_dates, groups)
    nav.attrs["factor_ic"] = _calculate_factor_ic(
        monthly_prices, rebalance_dates, snapshots, groups
    )
    return nav


def calculate_traditional_volatility_group_indices(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    config: dict,
    adjustment_factors: pd.DataFrame,
    adjustment_factor_coverage: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculate monthly traditional-volatility quintile NAVs.

    The returned ``all_a`` pool contains every stock with a month-end quote and
    a complete volatility lookback.  ``dividend_pool`` applies the dividend
    low-volatility universe quality filters, but deliberately does not apply
    the strategy's top-dividend ranking or final-N volatility selection.
    """
    try:
        strategy_config = {
            "universe": config["universe"],
            "selection": config["strategy_1"],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing strategy_1 configuration") from exc
    _validate_selection_config(strategy_config)
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

    prepared = prepare_div_low_vol_universe_inputs(
        price, dividends, dividend_queries, shares
    )
    adjusted_price = build_back_adjusted_close(
        price,
        adjustment_factors,
        adjustment_factor_coverage,
    ).loc[:, ["date", "symbol", "adjusted_close"]].rename(
        columns={"adjusted_close": "close"}
    )
    adjusted_price["symbol"] = adjusted_price["symbol"].astype(str)
    adjusted_price = adjusted_price.sort_values(["date", "symbol"])
    monthly_prices = adjusted_price.pivot(
        index="date", columns="symbol", values="close"
    ).ffill().reindex(rebalance_dates)
    volatility_snapshots = _calculate_traditional_volatility_snapshots(
        adjusted_price,
        rebalance_dates,
        strategy_config["selection"]["volatility_lookback_days"],
    )

    all_a_groups = []
    dividend_groups = []
    for rebalance_date in rebalance_dates:
        volatility = volatility_snapshots[rebalance_date]
        month_symbols = set(
            price_data.loc[
                price_data["date"].eq(rebalance_date), "symbol"
            ].astype(str)
        )
        all_a_groups.append(
            _assign_traditional_volatility_groups(
                sorted(month_symbols), volatility, rebalance_date, "all-A"
            )
        )
        dividend_universe = build_div_low_vol_universe(
            price,
            dividends,
            dividend_queries,
            shares,
            rebalance_date,
            strategy_config["universe"],
            strategy_config["selection"]["dividend_yield_lookback_days"],
            prepared,
        )
        dividend_symbols = sorted(
            month_symbols.intersection(dividend_universe.index.astype(str))
        )
        dividend_groups.append(
            _assign_traditional_volatility_groups(
                dividend_symbols, volatility, rebalance_date, "dividend pool"
            )
        )

    all_a_nav = _run_traditional_volatility_group_backtest(
        monthly_prices, rebalance_dates, all_a_groups
    )
    dividend_nav = _run_traditional_volatility_group_backtest(
        monthly_prices, rebalance_dates, dividend_groups
    )
    all_a_nav.attrs["factor_ic"] = _calculate_traditional_volatility_ic(
        monthly_prices,
        rebalance_dates,
        volatility_snapshots,
        all_a_groups,
    )
    dividend_nav.attrs["factor_ic"] = _calculate_traditional_volatility_ic(
        monthly_prices,
        rebalance_dates,
        volatility_snapshots,
        dividend_groups,
    )
    return {"all_a": all_a_nav, "dividend_pool": dividend_nav}


def _assign_traditional_volatility_groups(
    symbols: list[str],
    volatility: dict[str, float],
    as_of_date: pd.Timestamp | None = None,
    pool_name: str = "pool",
) -> pd.Series:
    """Sort traditional volatility low-to-high into five balanced groups."""
    return _assign_factor_groups(symbols, volatility, as_of_date, pool_name)


def _assign_factor_groups(
    symbols: list[str],
    factor: dict[str, float],
    as_of_date: pd.Timestamp | None = None,
    pool_name: str = "pool",
) -> pd.Series:
    """Sort a factor low-to-high and assign five balanced groups."""
    ranked = pd.DataFrame(
        {
            "symbol": [str(symbol) for symbol in symbols],
            "factor": [factor.get(str(symbol), np.nan) for symbol in symbols],
        }
    ).dropna(subset=["factor"])
    ranked = ranked.loc[np.isfinite(ranked["factor"])]
    ranked = ranked.sort_values(["factor", "symbol"], kind="mergesort")
    if len(ranked) < 5:
        date_text = "" if as_of_date is None else f" on {pd.Timestamp(as_of_date).date()}"
        raise ValueError(
            f"Only {len(ranked)} symbols in {pool_name}{date_text} have a complete "
            "volatility lookback; at least 5 are required"
        )
    ranked["group"] = np.floor(np.arange(len(ranked)) * 5 / len(ranked)).astype(int) + 1
    return ranked.set_index("symbol")["group"].astype(int)


def _calculate_traditional_volatility_snapshots(
    adjusted_price: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    lookback_days: int,
) -> dict[pd.Timestamp, dict[str, float]]:
    """Calculate exact trailing price-volatility values in one pass per symbol."""
    if lookback_days < 2:
        raise ValueError("lookback_days must be at least 2")
    dates = pd.DatetimeIndex(rebalance_dates)
    date_set = set(dates)
    snapshots = {date: {} for date in dates}
    for symbol, history in adjusted_price.groupby("symbol", sort=False):
        history = history.loc[:, ["date", "close"]].dropna(subset=["close"])
        history = history.loc[history["close"].gt(0)].sort_values("date")
        if len(history) < lookback_days:
            continue
        close = pd.Series(
            history["close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(history["date"]),
        )
        returns = close.pct_change(fill_method=None)
        volatility = returns.rolling(lookback_days - 1).std()
        for date, value in volatility.items():
            if date in date_set and pd.notna(value):
                snapshots[date][str(symbol)] = float(value)
    return snapshots


def _run_traditional_volatility_group_backtest(
    monthly_prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    groups: list[pd.Series],
) -> pd.DataFrame:
    return _run_factor_group_backtest(monthly_prices, rebalance_dates, groups)


def _run_factor_group_backtest(
    monthly_prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    groups: list[pd.Series],
) -> pd.DataFrame:
    symbols = sorted({symbol for snapshot in groups for symbol in snapshot.index})
    prices = monthly_prices.reindex(index=rebalance_dates, columns=symbols)
    group_returns = {}
    for group_number, column in enumerate(TRADITIONAL_VOLATILITY_GROUP_COLUMNS[:5], 1):
        target_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=symbols)
        for rebalance_date, snapshot in zip(rebalance_dates, groups):
            members = snapshot.index[snapshot.eq(group_number)]
            if len(members) == 0:
                raise ValueError(
                    f"Volatility group {group_number} is empty on {rebalance_date.date()}"
                )
            target_weights.loc[rebalance_date, members] = 1.0 / len(members)
        portfolio = run_backtest(prices, target_weights, freq="ME")
        group_returns[column] = portfolio.returns().reindex(rebalance_dates).fillna(0.0)

    returns = pd.DataFrame(group_returns, index=rebalance_dates)
    returns[TRADITIONAL_VOLATILITY_GROUP_COLUMNS[5]] = (
        returns[TRADITIONAL_VOLATILITY_GROUP_COLUMNS[0]]
        - returns[TRADITIONAL_VOLATILITY_GROUP_COLUMNS[4]]
    )
    nav = (1.0 + returns).cumprod()
    nav.index.name = None
    return nav.loc[:, list(TRADITIONAL_VOLATILITY_GROUP_COLUMNS)]


def _calculate_traditional_volatility_ic(
    monthly_prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    volatility_snapshots: dict[pd.Timestamp, dict[str, float]],
    groups: list[pd.Series],
) -> pd.DataFrame:
    return _calculate_factor_ic(
        monthly_prices,
        rebalance_dates,
        [volatility_snapshots[date] for date in rebalance_dates],
        groups,
    )


def _calculate_factor_ic(
    monthly_prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    factor_snapshots: list[dict[str, float] | pd.Series],
    groups: list[pd.Series],
) -> pd.DataFrame:
    records = []
    for position, rebalance_date in enumerate(rebalance_dates[:-1]):
        symbols = groups[position].index
        factor = pd.Series(
            {
                symbol: factor_snapshots[position].get(symbol, np.nan)
                for symbol in symbols
            },
            dtype=float,
        )
        current = monthly_prices.iloc[position].reindex(symbols)
        following = monthly_prices.iloc[position + 1].reindex(symbols)
        forward_return = following.div(current).sub(1.0)
        aligned = pd.concat(
            [factor.rename("factor"), forward_return.rename("forward_return")],
            axis=1,
        ).dropna()
        records.append(
            {
                "date": rebalance_date,
                "ic": _safe_correlation(
                    aligned["factor"], aligned["forward_return"]
                ),
                "rank_ic": _safe_correlation(
                    aligned["factor"].rank(), aligned["forward_return"].rank()
                ),
            }
        )
    return pd.DataFrame.from_records(records).set_index("date")


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.std(ddof=1) == 0 or right.std(ddof=1) == 0:
        return np.nan
    return float(left.corr(right))


def _calculate_div_low_vol_monthly_index(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    config: dict,
    rebalance_dates: pd.DatetimeIndex,
    transaction_cost_rate: float,
    adjustment_factors: pd.DataFrame,
    adjustment_factor_coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _calculate_monthly_index_with_selector(
        price,
        dividends,
        dividend_queries,
        shares,
        config,
        rebalance_dates,
        transaction_cost_rate,
        adjustment_factors,
        adjustment_factor_coverage,
        lambda date, adjusted, prepared: select_div_low_vol_constituents(
            price,
            dividends,
            dividend_queries,
            shares,
            date,
            config,
            prepared,
            adjusted,
        ),
    )


def _calculate_monthly_index_with_selector(
    price: pd.DataFrame,
    dividends: pd.DataFrame,
    dividend_queries: pd.DataFrame,
    shares: pd.DataFrame,
    config: dict,
    rebalance_dates: pd.DatetimeIndex,
    transaction_cost_rate: float,
    adjustment_factors: pd.DataFrame,
    adjustment_factor_coverage: pd.DataFrame,
    selector,
    prepared: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_div_low_vol_universe_inputs(
        price, dividends, dividend_queries, shares
    ) if prepared is None else prepared
    adjusted_price = build_back_adjusted_close(
        price,
        adjustment_factors,
        adjustment_factor_coverage,
    ).loc[:, ["date", "symbol", "adjusted_close"]].rename(
        columns={"adjusted_close": "close"}
    )
    constituent_snapshots = []
    constituent_weights = {}

    for rebalance_date in rebalance_dates:
        constituents = selector(rebalance_date, adjusted_price, prepared)
        constituent_weights[rebalance_date] = constituents["weight"]
        constituent_snapshots.append(
            constituents.reset_index()
            .assign(effective_date=rebalance_date)
            .set_index(["effective_date", "symbol"])
        )
    target_weights = pd.DataFrame(constituent_weights).T.fillna(0.0)
    target_weights.index.name = "date"
    adjusted_prices = adjusted_price.pivot(
        index="date", columns="symbol", values="close"
    )
    monthly_prices = adjusted_prices.ffill().reindex(rebalance_dates)
    monthly_prices = monthly_prices.reindex(columns=target_weights.columns)
    fee_matrix = pd.DataFrame(
        transaction_cost_rate / 2.0,
        index=monthly_prices.index,
        columns=monthly_prices.columns,
    )
    fee_matrix.iloc[0] = 0.0
    portfolio = run_backtest(
        monthly_prices,
        target_weights,
        fees=fee_matrix,
        freq="ME",
    )
    values = portfolio.value()
    returns = portfolio.returns().rename("total_return")
    index = pd.DataFrame(
        {
            "total_return": returns,
            "total_return_index": values / values.iloc[0],
        }
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
    _validate_candidate_config(config)
    selection = config["selection"]
    for name in ["volatility_lookback_days", "final_n"]:
        if selection[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if selection["dividend_top_n"] < selection["final_n"]:
        raise ValueError("dividend_top_n must not be smaller than final_n")


def _validate_candidate_config(config: dict) -> None:
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
    for name in ["dividend_yield_lookback_days", "dividend_top_n"]:
        if selection[name] <= 0:
            raise ValueError(f"{name} must be positive")


def _validate_strategy_2_config(config: dict) -> None:
    _validate_candidate_config(config)
    selection = config["selection"]
    for name in ["final_n", "lookback_trading_days", "min_valid_days"]:
        if selection[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if selection["dividend_top_n"] < selection["final_n"]:
        raise ValueError("dividend_top_n must not be smaller than final_n")
    if selection["min_valid_days"] > selection["lookback_trading_days"]:
        raise ValueError("min_valid_days must not exceed lookback_trading_days")
    transaction_cost_rate = selection.get("transaction_cost_rate", 0.0)
    if not 0 <= transaction_cost_rate < 1:
        raise ValueError("transaction_cost_rate must be in [0, 1)")


def _strategy_2_selection_config(config: dict) -> dict:
    try:
        if "strategy_2" in config:
            return {"universe": config["universe"], "selection": config["strategy_2"]}
        return config
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing strategy_2 configuration") from exc


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
    event_columns = sorted(required | {"operate_date"}.intersection(dividends.columns))
    events = dividends.loc[:, event_columns].copy()
    return {
        "price": price_data,
        "prices": price_data.pivot(index="date", columns="symbol", values="close"),
        "dividend_events": events,
        "query_coverage": set(
            normalize_query_years(dividend_queries).itertuples(index=False, name=None)
        ),
    }


def _prepare_index_cash_events(
    dividends: pd.DataFrame,
    date_column: str,
    symbols: pd.Index | None = None,
    require_dates: bool = False,
) -> pd.DataFrame:
    if date_column not in dividends:
        raise ValueError(f"dividends must contain {date_column}")
    events = dividends.loc[:, ["symbol", date_column, "cash_dividend_before_tax"]].copy()
    if symbols is not None:
        events = events.loc[events["symbol"].isin(symbols)]
    positive = events["cash_dividend_before_tax"].gt(0)
    if require_dates and events.loc[positive, date_column].isna().any():
        raise ValueError(
            f"Positive dividends for held symbols must contain {date_column}"
        )
    events = events.rename(
        columns={
            date_column: "date",
            "cash_dividend_before_tax": "cash_per_share",
        }
    )
    events["cash_per_share"] = (
        events["cash_per_share"] * DIVIDEND_AFTER_TAX_RATIO
    )
    return events.loc[
        events["date"].notna() & events["cash_per_share"].gt(0),
        ["date", "symbol", "cash_per_share"],
    ]


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

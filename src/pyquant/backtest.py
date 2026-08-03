"""Minimal weight return backtest."""

import pandas as pd


REBALANCED_BACKTEST_COLUMNS = [
    "price_return",
    "total_return",
    "dividend_cash",
    "turnover",
    "transaction_cost",
    "price_index",
    "total_return_index",
]


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cash_events: pd.DataFrame | None = None,
    fee_rate: float = 0.0,
) -> pd.DataFrame:
    """Backtest periodic target weights with cash events and one-way costs.

    Each weight row is formed at that row's close and held until the next
    weight date. Cash events are credited over ``(start, end]`` and reinvested
    at the ending rebalance. Initial portfolio construction is cost-free.
    """
    if fee_rate < 0:
        raise ValueError("fee_rate must not be negative")
    price_data = _prepare_price_panel(prices)
    weight_data = _prepare_target_weights(target_weights)
    cash_data = _prepare_cash_events(cash_events)
    rebalance_prices = (
        price_data.reindex(price_data.index.union(weight_data.index))
        .sort_index()
        .ffill()
        .reindex(weight_data.index)
    )
    result = pd.DataFrame(
        0.0,
        index=weight_data.index,
        columns=REBALANCED_BACKTEST_COLUMNS[:-2],
    )

    for position, start in enumerate(weight_data.index[:-1]):
        end = weight_data.index[position + 1]
        held_weights = weight_data.loc[start]
        held_symbols = held_weights[held_weights.gt(0)].index
        interval_prices = rebalance_prices.loc[[start, end], held_symbols]
        invalid = interval_prices.columns[
            interval_prices.isna().any() | interval_prices.le(0).any()
        ].tolist()
        if invalid:
            raise ValueError(
                "Missing or non-positive prices for held symbols; "
                f"examples: {invalid[:5]}"
            )
        normalized_shares = held_weights[held_symbols].div(interval_prices.loc[start])
        end_positions = interval_prices.loc[end].mul(normalized_shares)
        dividend_cash = _interval_cash(
            cash_data,
            normalized_shares,
            start,
            end,
        )
        gross_value = end_positions.sum() + dividend_cash
        if gross_value <= 0:
            raise ValueError("Portfolio gross value must remain positive")

        pre_trade_weights = end_positions.div(gross_value)
        target = weight_data.loc[end]
        symbols = pre_trade_weights.index.union(target.index)
        turnover = 0.5 * (
            pre_trade_weights.reindex(symbols, fill_value=0.0)
            .sub(target.reindex(symbols, fill_value=0.0))
            .abs()
            .sum()
            + dividend_cash / gross_value
        )
        transaction_cost = turnover * fee_rate
        result.loc[end] = [
            end_positions.sum() - 1.0,
            gross_value * (1.0 - transaction_cost) - 1.0,
            dividend_cash,
            turnover,
            transaction_cost,
        ]

    result["price_index"] = (1.0 + result["price_return"]).cumprod()
    result["total_return_index"] = (1.0 + result["total_return"]).cumprod()
    return result[REBALANCED_BACKTEST_COLUMNS]


def _prepare_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a DataFrame")
    out = prices.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.hasnans:
        raise ValueError("prices index must not contain invalid dates")
    if out.index.has_duplicates:
        raise ValueError("prices index must be unique")
    if out.columns.has_duplicates:
        raise ValueError("prices columns must be unique")
    out.columns = out.columns.astype(str)
    return out.apply(pd.to_numeric, errors="coerce").sort_index()


def _prepare_target_weights(target_weights: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(target_weights, pd.DataFrame):
        raise TypeError("target_weights must be a DataFrame")
    out = target_weights.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if out.empty or out.index.hasnans:
        raise ValueError("target_weights must contain valid dates")
    if out.index.has_duplicates:
        raise ValueError("target_weights index must be unique")
    if out.columns.has_duplicates:
        raise ValueError("target_weights columns must be unique")
    out.columns = out.columns.astype(str)
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0).sort_index()
    if out.lt(0).any().any():
        raise ValueError("target_weights must not be negative")
    if not out.sum(axis=1).sub(1.0).abs().le(1e-10).all():
        raise ValueError("Each target_weights row must sum to 1")
    return out


def _prepare_cash_events(cash_events: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["date", "symbol", "cash_per_share"]
    if cash_events is None:
        return pd.DataFrame(columns=columns)
    missing = sorted(set(columns) - set(cash_events))
    if missing:
        raise ValueError(f"cash_events missing required columns: {missing}")
    out = cash_events.loc[:, columns].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    out["cash_per_share"] = pd.to_numeric(out["cash_per_share"], errors="coerce")
    if out[["date", "symbol", "cash_per_share"]].isna().any().any():
        raise ValueError("cash_events must not contain invalid values")
    if out["cash_per_share"].lt(0).any():
        raise ValueError("cash_per_share must not be negative")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _interval_cash(
    cash_events: pd.DataFrame,
    normalized_shares: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    events = cash_events.loc[
        cash_events["symbol"].isin(normalized_shares.index)
        & cash_events["date"].gt(start)
        & cash_events["date"].le(end)
        & cash_events["cash_per_share"].gt(0)
    ]
    return float(
        events["symbol"].map(normalized_shares).mul(events["cash_per_share"]).sum()
    )

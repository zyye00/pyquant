"""VectorBT-backed target-weight portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import vectorbt as vbt


def run_backtest(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    execution_price: pd.DataFrame | None = None,
    fees: float | pd.DataFrame = 0.0,
    slippage: float = 0.0,
    init_cash: float = 1_000_000.0,
    freq: str | pd.Timedelta = "ME",
) -> vbt.Portfolio:
    """Run a target-weight portfolio with VectorBT.

    ``target_weights`` is interpreted as a portfolio percentage at each
    timestamp.  All orders share one cash account and are sequenced
    automatically so switches between constituents are fully funded.  The
    A scalar fee is applied after the free initial construction.  A fee matrix
    can override this row-by-row behavior for more specific execution rules.
    """
    import vectorbt as vbt

    close_data = _prepare_panel(close, "close")
    weights = _prepare_target_weights(target_weights, close_data.columns)
    if not weights.index.isin(close_data.index).all():
        raise ValueError("target_weights dates must be present in close")
    close_data = close_data.reindex(close_data.index.union(weights.index)).sort_index()
    close_data = close_data.ffill().reindex(columns=weights.columns)
    held_weights = weights.reindex(close_data.index).ffill().fillna(0.0)
    if (close_data.isna() & held_weights.gt(0)).any().any():
        raise ValueError("close is missing while a target symbol is held")
    close_data = close_data.bfill()
    if close_data.isna().any().any() or close_data.le(0).any().any():
        raise ValueError("close must contain positive prices for all target symbols")
    price = None
    if execution_price is not None:
        price = _prepare_panel(execution_price, "execution_price")
        if not price.index.equals(close_data.index) or not price.columns.equals(
            close_data.columns
        ):
            price = price.reindex(index=close_data.index, columns=close_data.columns)
        if price.isna().any().any() or price.le(0).any().any():
            raise ValueError("execution_price must contain positive prices")
    if isinstance(fees, (int, float, np.number)):
        fee_data = pd.DataFrame(
            float(fees), index=close_data.index, columns=close_data.columns
        )
        fee_data.iloc[0] = 0.0
    else:
        fee_data = fees.reindex(index=close_data.index, columns=close_data.columns)
        if fee_data.isna().any().any():
            raise ValueError("fees must cover all close dates and symbols")
    if fee_data.lt(0).any().any():
        raise ValueError("fees must not be negative")
    if slippage < 0:
        raise ValueError("slippage must not be negative")
    if init_cash <= 0:
        raise ValueError("init_cash must be positive")

    return vbt.Portfolio.from_orders(
        close_data,
        size=held_weights,
        size_type="targetpercent",
        direction="longonly",
        price=price,
        fees=fee_data,
        slippage=slippage,
        init_cash=init_cash,
        cash_sharing=True,
        call_seq="auto",
        group_by=True,
        freq=freq,
    )


def _prepare_panel(data: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame")
    out = data.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if out.empty or out.index.hasnans:
        raise ValueError(f"{name} must contain valid dates")
    if out.index.has_duplicates:
        raise ValueError(f"{name} index must be unique")
    if out.columns.has_duplicates:
        raise ValueError(f"{name} columns must be unique")
    out.columns = out.columns.astype(str)
    return out.apply(pd.to_numeric, errors="coerce").sort_index()


def _prepare_target_weights(
    target_weights: pd.DataFrame,
    close_columns: pd.Index,
) -> pd.DataFrame:
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
    unsupported = sorted(set(out.columns) - set(close_columns))
    if unsupported:
        raise ValueError(f"target_weights symbols missing from close: {unsupported}")
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0).sort_index()
    if out.lt(0).any().any():
        raise ValueError("target_weights must not be negative")
    if not out.sum(axis=1).sub(1.0).abs().le(1e-10).all():
        raise ValueError("Each target_weights row must sum to 1")
    return out

"""Price adjustments derived from corporate-action factors."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def build_back_adjusted_close(
    price: pd.DataFrame,
    factors: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Attach BaoStock back-adjusted closes to a long price table.

    ``back_adjust_factor`` is cumulative and applies from its
    ``operate_date`` onward.  Dates before the first corporate action use a
    factor of one.  A factor event on a non-trading day naturally takes effect
    on the next available price because prices are matched with the latest
    event on or before each price date.

    The factor query coverage must fully contain every requested price range;
    silently extrapolating outside a downloaded range would produce a
    misleading adjusted series.
    """
    _require_columns(price, {"date", "symbol", "close"}, "price")
    _require_columns(
        factors,
        {"symbol", "operate_date", "back_adjust_factor"},
        "factors",
    )
    _require_columns(coverage, {"symbol", "start", "end"}, "coverage")
    if price.duplicated(["date", "symbol"]).any():
        raise ValueError("price contains duplicate (date, symbol) rows")
    if factors.duplicated(["symbol", "operate_date"]).any():
        raise ValueError("factors contain duplicate (symbol, operate_date) rows")
    if coverage.duplicated(["symbol", "start", "end"]).any():
        raise ValueError("coverage contains duplicate ranges")

    prices = price.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").astype(
        "datetime64[ns]"
    )
    prices["symbol"] = prices["symbol"].astype(str)
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    if prices[["date", "close"]].isna().any().any():
        raise ValueError("price dates and closes must not be missing")
    factors_data = factors.copy()
    factors_data["operate_date"] = pd.to_datetime(
        factors_data["operate_date"], errors="coerce"
    ).astype("datetime64[ns]")
    factors_data["symbol"] = factors_data["symbol"].astype(str)
    factors_data["back_adjust_factor"] = pd.to_numeric(
        factors_data["back_adjust_factor"], errors="coerce"
    )
    if factors_data[["operate_date", "back_adjust_factor"]].isna().any().any():
        raise ValueError("factor dates and back_adjust_factor must not be missing")
    if (~np.isfinite(factors_data["back_adjust_factor"])).any() or (
        ~factors_data["back_adjust_factor"].gt(0)
    ).any():
        raise ValueError("back_adjust_factor must contain positive finite numbers")
    ranges = coverage.copy()
    ranges["symbol"] = ranges["symbol"].astype(str)
    ranges["start"] = pd.to_datetime(ranges["start"], errors="coerce").astype(
        "datetime64[ns]"
    )
    ranges["end"] = pd.to_datetime(ranges["end"], errors="coerce").astype(
        "datetime64[ns]"
    )
    if ranges[["start", "end"]].isna().any().any():
        raise ValueError("coverage dates must not be missing")
    if (ranges["start"] > ranges["end"]).any():
        raise ValueError("coverage start must not be after end")

    factors_by_symbol = {
        symbol: group.sort_values("operate_date")
        for symbol, group in factors_data.groupby("symbol", sort=False)
    }
    rows: list[pd.DataFrame] = []
    for symbol, group in prices.groupby("symbol", sort=False):
        group = group.copy()
        requested_start = group["date"].min()
        requested_end = group["date"].max()
        if not _ranges_cover(
            ranges.loc[ranges["symbol"].eq(symbol)], requested_start, requested_end
        ):
            raise ValueError(
                f"adjustment-factor coverage does not contain price range for {symbol}"
            )
        events = factors_by_symbol.get(symbol)
        if events is None:
            factor = pd.Series(1.0, index=group.index)
        else:
            matched = pd.merge_asof(
                group[["date"]].sort_values("date"),
                events[["operate_date", "back_adjust_factor"]].rename(
                    columns={"operate_date": "date"}
                ).sort_values("date"),
                on="date",
                direction="backward",
            )
            factor = matched["back_adjust_factor"].fillna(1.0)
            factor.index = group.sort_values("date").index
            factor = factor.reindex(group.index)
        group["adjusted_close"] = group["close"] * factor
        rows.append(group)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)


def _ranges_cover(
    ranges: pd.DataFrame,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> bool:
    cursor = requested_start
    for first, last in ranges.sort_values("start")[["start", "end"]].itertuples(
        index=False, name=None
    ):
        if last < cursor:
            continue
        if first > cursor:
            return False
        cursor = max(cursor, last + pd.Timedelta(days=1))
        if cursor > requested_end:
            return True
    return cursor > requested_end


def _require_columns(data: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

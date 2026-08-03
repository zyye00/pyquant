"""Canonical one-minute bars and reusable daily intraday features."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from pyquant.data.identifiers import normalize_security_symbol


MINUTE_DAY_VALID = 1
MINUTE_DAY_NO_DATA_CONFIRMED = 2
MINUTE_DAY_INCOMPLETE = 3
MINUTE_DAY_FAILED = 4
MINUTE_COLUMNS = [
    "symbol",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "total_turnover",
]
MINUTE_VALUE_COLUMNS = MINUTE_COLUMNS[2:]


def normalize_minute_bars(
    data: pd.DataFrame,
    symbol: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Validate and normalize one security's unadjusted one-minute bars."""
    symbol = normalize_security_symbol(symbol)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must not be after end_date")
    missing = sorted({"datetime", *MINUTE_VALUE_COLUMNS} - set(data))
    if missing:
        raise ValueError(f"Minute bars missing required columns: {missing}")
    out = data.loc[:, ["datetime", *MINUTE_VALUE_COLUMNS]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="raise")
    if (
        out["datetime"].isna().any()
        or not pd.api.types.is_datetime64_any_dtype(out["datetime"])
        or isinstance(out["datetime"].dtype, pd.DatetimeTZDtype)
    ):
        raise ValueError("Minute-bar datetimes must be valid timezone-naive values")
    dates = out["datetime"].dt.normalize()
    if not dates.between(start, end).all():
        raise ValueError("Minute bars contain timestamps outside the requested range")
    for column in MINUTE_VALUE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out["close"].isna().any() or out["close"].le(0).any():
        raise ValueError("Minute-bar close values must be positive")
    duplicates = out.duplicated("datetime", keep=False)
    if duplicates.any():
        conflicts = (
            out.loc[duplicates]
            .groupby("datetime", sort=False)[MINUTE_VALUE_COLUMNS]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicts.any():
            raise ValueError("Minute bars contain conflicting duplicate timestamps")
        out = out.drop_duplicates("datetime")
    out.insert(0, "symbol", symbol)
    out[["open", "high", "low", "close"]] = out[
        ["open", "high", "low", "close"]
    ].astype("float32")
    out[["volume", "total_turnover"]] = out[["volume", "total_turnover"]].astype(
        "float64"
    )
    return out.sort_values("datetime").reset_index(drop=True)


def calculate_daily_intraday_volatility(
    minute: pd.DataFrame,
    symbol: str,
    trading_dates: Collection[object],
    *,
    min_bars_per_day: int | None = None,
) -> pd.DataFrame:
    """Calculate within-day minute-return volatility and per-day status."""
    symbol = normalize_security_symbol(symbol)
    if min_bars_per_day is not None and min_bars_per_day <= 0:
        raise ValueError("min_bars_per_day must be positive")
    dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates), errors="raise"))
    if dates.hasnans:
        raise ValueError("trading_dates must not contain missing values")
    dates = dates.normalize().drop_duplicates().sort_values()
    if minute.empty:
        return pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": dates,
                "volatility": np.nan,
                "bar_count": 0,
                "return_count": 0,
                "status": MINUTE_DAY_NO_DATA_CONFIRMED,
            }
        )
    required = {"symbol", "datetime", "close"}
    missing = sorted(required - set(minute))
    if missing:
        raise ValueError(f"Minute bars missing required columns: {missing}")
    if set(minute["symbol"]) != {symbol}:
        raise ValueError("Minute bars do not match the requested symbol")
    out = minute.loc[:, ["symbol", "datetime", "close"]].copy()
    out["trade_date"] = out["datetime"].dt.normalize()
    if not out["trade_date"].isin(dates).all():
        raise ValueError("Minute bars contain dates outside the trading calendar")
    out["minute_return"] = out.groupby(["symbol", "trade_date"], sort=False)[
        "close"
    ].pct_change(fill_method=None)
    grouped = out.groupby("trade_date", sort=False)
    daily = pd.DataFrame(
        {
            "volatility": grouped["minute_return"].std(ddof=1),
            "bar_count": grouped.size(),
            "return_count": grouped["minute_return"].count(),
        }
    ).reindex(dates)
    daily.index.name = "trade_date"
    daily["bar_count"] = daily["bar_count"].fillna(0).astype(int)
    daily["return_count"] = daily["return_count"].fillna(0).astype(int)
    daily["status"] = MINUTE_DAY_VALID
    daily.loc[daily["bar_count"].eq(0), "status"] = MINUTE_DAY_NO_DATA_CONFIRMED
    incomplete = daily["bar_count"].gt(0) & ~np.isfinite(daily["volatility"])
    if min_bars_per_day is not None:
        incomplete |= daily["bar_count"].between(1, min_bars_per_day - 1)
    daily.loc[incomplete, "status"] = MINUTE_DAY_INCOMPLETE
    daily.insert(0, "symbol", symbol)
    return daily.reset_index()[
        [
            "symbol",
            "trade_date",
            "volatility",
            "bar_count",
            "return_count",
            "status",
        ]
    ]

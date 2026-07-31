"""Dataset loading and field standardization."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import pandas as pd

from pyquant.data.catalog import (
    DatasetSpec,
    get_dataset_spec,
)
from pyquant.data.duckdb import load_relation
from pyquant.data.identifiers import normalize_security_symbol
from pyquant.data.resources import load_source_protocols


_PRICE = load_source_protocols()["price"]


def load_dataset(
    name: str,
    *,
    start: str | None = None,
    end: str | None = None,
    symbols: Collection[str] | None = None,
    data_root: Path = Path("data"),
) -> pd.DataFrame:
    """Load a catalog dataset with canonical columns."""
    dataset = get_dataset_spec(name)
    storage = dataset.storage
    if storage.requires_dates and (start is None or end is None):
        raise ValueError(f"Dataset {name!r} requires explicit start and end dates")
    start_at = pd.Timestamp(start) if start is not None else None
    end_at = pd.Timestamp(end) if end is not None else None
    if start_at is not None and end_at is not None and start_at > end_at:
        raise ValueError("start must not be after end")
    relation_symbols = symbols
    if storage.allowed_symbols:
        allowed = set(storage.allowed_symbols)
        if symbols is not None and not set(symbols).issubset(allowed):
            unsupported = sorted(set(symbols) - allowed)
            raise ValueError(
                f"Dataset {name!r} does not support symbols: {unsupported}"
            )
        relation_symbols = symbols or storage.allowed_symbols
    out = load_relation(
        storage.relation,
        list(dataset.columns),
        database_path=storage.resolve_path(data_root),
        date_column=dataset.date_column,
        start=start_at,
        end=end_at,
        symbols=relation_symbols,
        normalize_symbols=storage.normalize_symbols,
    )
    out = _canonicalize_dataset(out, dataset)
    if symbols is not None:
        requested = {
            normalize_security_symbol(symbol)
            if storage.normalize_symbols
            else str(symbol)
            for symbol in symbols
        }
        out = out[out["symbol"].isin(requested)]
    if dataset.primary_key and out.duplicated(list(dataset.primary_key)).any():
        raise ValueError(
            f"Dataset {name!r} contains duplicate primary keys: "
            f"{list(dataset.primary_key)}"
        )
    columns = [column for column in dataset.columns if column in out]
    order = list(dataset.primary_key) or columns[:1]
    return out[columns].sort_values(order).reset_index(drop=True)


def standardize_price(
    data: pd.DataFrame,
    field_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Standardize price field names, dates, and symbols."""
    rename_map = _PRICE["default_field_map"] | (field_map or {})
    out = data.rename(
        columns={key: value for key, value in rename_map.items() if key in data.columns}
    ).copy()
    missing = [column for column in ["date", "symbol", "close"] if column not in out]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    ordered = [column for column in _PRICE["columns"] if column in out]
    extras = [column for column in out if column not in ordered]
    return out[ordered + extras].sort_values(["date", "symbol"]).reset_index(drop=True)


def get_period_end_dates(
    dates: Collection[object] | pd.Index,
    frequency: str = "M",
) -> pd.DatetimeIndex:
    """Return the last available date in each calendar period."""
    name = dates.name if isinstance(dates, pd.Index) else None
    index = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce"))
    if index.hasnans:
        raise ValueError("dates must not contain invalid values")
    index = index.drop_duplicates().sort_values()
    if index.empty:
        return pd.DatetimeIndex([], name=name)
    period_ends = (
        pd.Series(index, index=index).groupby(index.to_period(frequency)).max()
    )
    return pd.DatetimeIndex(period_ends.array, name=name)


def normalize_query_years(queries: pd.DataFrame) -> pd.DataFrame:
    """Normalize query coverage to unique ``symbol, year`` rows."""
    if {"symbol", "year"}.issubset(queries.columns):
        out = queries[["symbol", "year"]].copy()
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        if out[["symbol", "year"]].isna().any().any():
            raise ValueError("queries symbol and year must not contain invalid values")
        out["symbol"] = out["symbol"].astype(str)
        out["year"] = out["year"].astype(int)
        return out.drop_duplicates().sort_values(["symbol", "year"])

    required = {"symbol", "start", "end"}
    missing = sorted(required - set(queries))
    if missing:
        raise ValueError(f"queries missing required columns: {missing}")
    out = queries.loc[:, sorted(required)].copy()
    out["start"] = pd.to_datetime(out["start"], errors="coerce")
    out["end"] = pd.to_datetime(out["end"], errors="coerce")
    if out[["symbol", "start", "end"]].isna().any().any():
        raise ValueError("queries ranges must not contain invalid values")
    out["symbol"] = out["symbol"].astype(str)
    out = out.loc[out["start"].le(out["end"])]
    out = out.loc[
        out.index.repeat(out["end"].dt.year - out["start"].dt.year + 1)
    ].copy()
    out["year"] = out.groupby(level=0).cumcount() + out["start"].dt.year
    return out[["symbol", "year"]].drop_duplicates().sort_values(["symbol", "year"])


def _canonicalize_dataset(
    data: pd.DataFrame,
    dataset: DatasetSpec,
) -> pd.DataFrame:
    out = data.copy()
    missing = sorted(set(dataset.required) - set(out))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    if "symbol" in out:
        if out["symbol"].isna().any():
            raise ValueError("Dataset symbol must not contain missing values")
        out["symbol"] = out["symbol"].astype(str)
    for column in dataset.date_columns:
        if column in out and not pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in dataset.numeric_columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out

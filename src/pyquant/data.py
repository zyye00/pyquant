"""Dataset catalog, loading, and field standardization."""

from __future__ import annotations

from collections.abc import Collection
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd

from pyquant.io import load_config


DEFAULT_DATASET_CATALOG = Path(__file__).parents[2] / "configs/datasets.yaml"
PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
DEFAULT_PRICE_FIELD_MAP = {
    "trade_date": "date",
    "datetime": "date",
    "ticker": "symbol",
    "code": "symbol",
    "sec_code": "symbol",
    "vol": "volume",
    "money": "amount",
    "turnover": "amount",
}


def load_dataset(
    name: str,
    *,
    start: str | None = None,
    end: str | None = None,
    symbols: Collection[str] | None = None,
    adjustment: str | None = None,
) -> pd.DataFrame:
    """Load a catalog dataset with canonical columns."""
    catalog = _load_dataset_catalog()
    dataset = _get_dataset(catalog, name)
    storage = dataset["storage"]
    kind = storage["kind"]
    if kind != "table" and (start is None or end is None):
        raise ValueError(f"Dataset {name!r} requires explicit start and end dates")
    start_at, end_at = _validate_date_range(start, end)
    adjustment_name = adjustment or "none"
    if "{adjustment}" not in storage["path"] and adjustment is not None:
        raise ValueError(f"Dataset {name!r} does not support adjustment")

    paths = _dataset_paths(storage, symbols, adjustment_name, start_at, end_at)
    if not paths:
        raise FileNotFoundError(f"No files found for dataset {name!r}")
    frames = [
        _read_dataset_file(path, dataset, storage, start_at, end_at)
        for path in paths
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=dataset["columns"])
    out = pd.concat(frames, ignore_index=True)
    out = _canonicalize_dataset(out, dataset)
    if symbols is not None:
        out = out[out["symbol"].isin({str(symbol) for symbol in symbols})]
    key = dataset.get("primary_key", [])
    if key and out.duplicated(key).any():
        raise ValueError(f"Dataset {name!r} contains duplicate primary keys: {key}")
    columns = [column for column in dataset["columns"] if column in out]
    return out[columns].sort_values(key or columns[:1]).reset_index(drop=True)


def update_dataset(
    name: str,
    *,
    start: str,
    end: str | None = None,
    symbols: Collection[str] | None = None,
    pool: str | None = None,
    pool_date: str | None = None,
    adjustment: str | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Update a catalog dataset through its configured source."""
    from pyquant._data_update import update_dataset as _update_dataset

    return _update_dataset(
        name,
        start=start,
        end=end,
        symbols=symbols,
        pool=pool,
        pool_date=pool_date,
        adjustment=adjustment,
        max_tasks=max_tasks,
    )


def load_price(
    path: str | Path,
    field_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a standalone price file and standardize its fields."""
    path = Path(path)
    if path.suffix == ".csv":
        data = pd.read_csv(path)
    elif path.suffix in {".parquet", ".pq"}:
        data = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported price file type: {path.suffix}")
    return standardize_price(data, field_map=field_map)


def standardize_price(
    data: pd.DataFrame,
    field_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Standardize price field names, dates, and symbols."""
    rename_map = DEFAULT_PRICE_FIELD_MAP | (field_map or {})
    out = data.rename(
        columns={key: value for key, value in rename_map.items() if key in data.columns}
    ).copy()
    missing = [column for column in ["date", "symbol", "close"] if column not in out]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    ordered = [column for column in PRICE_COLUMNS if column in out]
    extras = [column for column in out if column not in ordered]
    return out[ordered + extras].sort_values(["date", "symbol"]).reset_index(drop=True)


def _load_dataset_catalog(path: str | Path = DEFAULT_DATASET_CATALOG) -> dict[str, Any]:
    catalog = load_config(path)
    if catalog.get("version") != 1 or not isinstance(catalog.get("datasets"), dict):
        raise ValueError("Invalid dataset catalog")
    return catalog


def _get_dataset(catalog: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        dataset = catalog["datasets"][name]
    except KeyError as exc:
        available = ", ".join(sorted(catalog["datasets"]))
        raise ValueError(f"Unknown dataset {name!r}; available: {available}") from exc
    required = {"source", "storage", "columns", "required", "field_map"}
    missing = sorted(required - set(dataset))
    if missing:
        raise ValueError(f"Dataset {name!r} catalog entry missing keys: {missing}")
    return dataset


def _validate_date_range(
    start: str | None, end: str | None
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_at = pd.Timestamp(start) if start is not None else None
    end_at = pd.Timestamp(end) if end is not None else None
    if start_at is not None and end_at is not None and start_at > end_at:
        raise ValueError("start must not be after end")
    return start_at, end_at


def _dataset_paths(
    storage: dict[str, str],
    symbols: Collection[str] | None,
    adjustment: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> list[Path]:
    template = storage["path"]
    if storage["kind"] == "table":
        path = Path(template)
        return [path] if path.exists() else []
    symbol_values = [str(symbol) for symbol in symbols] if symbols else ["*"]
    years: list[int | str] = ["*"]
    if "{year}" in template and start is not None and end is not None:
        years = list(range(start.year, end.year + 1))
    paths = {
        Path(path)
        for symbol in symbol_values
        for year in years
        for path in glob(
            template.format(adjustment=adjustment, symbol=symbol, year=year)
        )
    }
    return sorted(paths)


def _read_dataset_file(
    path: Path,
    dataset: dict[str, Any],
    storage: dict[str, str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    data = pd.read_parquet(path)
    if storage.get("symbol_from") == "stem":
        data["symbol"] = path.stem
    elif storage.get("symbol_from") == "parent":
        data["symbol"] = path.parent.name
    date_column = dataset.get("date_column")
    source_date = next(
        (source for source, target in dataset["field_map"].items() if target == date_column),
        date_column,
    )
    if source_date in data:
        values = pd.to_datetime(data[source_date], errors="coerce")
        if start is not None:
            data = data[values >= start]
            values = values.loc[data.index]
        if end is not None:
            data = data[values <= end]
    return data


def _canonicalize_dataset(
    data: pd.DataFrame, dataset: dict[str, Any]
) -> pd.DataFrame:
    rename_map = {
        source: target
        for source, target in dataset["field_map"].items()
        if source in data and target not in data
    }
    out = data.rename(columns=rename_map).copy()
    missing = sorted(set(dataset["required"]) - set(out))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    if "symbol" in out:
        if out["symbol"].isna().any():
            raise ValueError("Dataset symbol must not contain missing values")
        out["symbol"] = out["symbol"].astype(str)
    for column in dataset.get("date_columns", []):
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in dataset.get("numeric_columns", []):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out

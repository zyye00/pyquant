"""Validated dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


DEFAULT_CATALOG_PATH = Path(__file__).parents[3] / "configs/dataset_catalog.yaml"
KNOWN_SOURCES = {"akshare", "baostock", "generated", "rqdata"}
KNOWN_UPDATE_KINDS = {
    "csindex_history",
    "dividend",
    "adjust_factor",
    "history",
    "index_constituents",
    "profit_quarterly",
    "stock_pb",
}


def resolve_data_path(path: str | Path, data_root: Path) -> Path:
    """Resolve a catalog path against one data root."""
    path = Path(path)
    if path.is_absolute():
        return path
    try:
        path = path.relative_to("data")
    except ValueError:
        pass
    return data_root / path


@dataclass(frozen=True)
class DuckDBStorage:
    """DuckDB relation storage settings."""

    path: str
    relation: str
    requires_dates: bool
    normalize_symbols: bool = True
    allowed_symbols: tuple[str, ...] = ()
    kind: str = "duckdb"

    def resolve_path(self, data_root: Path) -> Path:
        """Resolve the database path for one data root."""
        return resolve_data_path(self.path, data_root)


StorageSpec = DuckDBStorage


@dataclass(frozen=True)
class UpdateSpec:
    """Dataset update settings."""

    kind: str
    pool: bool
    target: str | None = None
    source_codes: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable, validated dataset definition."""

    name: str
    description: str
    source: str
    storage: StorageSpec
    columns: tuple[str, ...]
    required: tuple[str, ...]
    primary_key: tuple[str, ...]
    date_column: str | None
    date_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    update: UpdateSpec | None


def _storage_from_mapping(name: str, values: Mapping[str, Any]) -> StorageSpec:
    kind = values.get("kind")
    path = values.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"Dataset {name!r} storage requires path")
    requires_dates = bool(values.get("requires_dates", kind != "table"))
    normalize_symbols = bool(values.get("normalize_symbols", True))
    if kind == "duckdb":
        relation = values.get("relation")
        if not isinstance(relation, str) or not relation:
            raise ValueError(f"Dataset {name!r} DuckDB storage requires relation")
        return DuckDBStorage(
            path=path,
            relation=relation,
            requires_dates=requires_dates,
            normalize_symbols=normalize_symbols,
            allowed_symbols=tuple(values.get("allowed_symbols", ())),
        )
    if kind != "duckdb":
        raise ValueError(f"Dataset {name!r} has unknown storage kind {kind!r}")


def _update_from_mapping(
    name: str,
    source: str,
    values: Mapping[str, Any] | None,
) -> UpdateSpec | None:
    if values is None:
        return None
    kind = values.get("kind")
    if kind not in KNOWN_UPDATE_KINDS:
        raise ValueError(f"Dataset {name!r} has unknown update kind {kind!r}")
    if source == "generated":
        raise ValueError(f"Generated dataset {name!r} cannot define updates")
    allowed = {
        "akshare": {"csindex_history"},
        "baostock": {"adjust_factor", "dividend", "history", "profit_quarterly"},
        "rqdata": {"index_constituents", "stock_pb"},
    }
    if kind not in allowed[source]:
        raise ValueError(
            f"Dataset {name!r} update kind {kind!r} does not match source {source!r}"
        )
    target = values.get("target")
    if kind == "history" and target not in {"index", "stock"}:
        raise ValueError(f"Dataset {name!r} history update requires target")
    source_codes = values.get("source_codes", {})
    if not isinstance(source_codes, dict):
        raise ValueError(f"Dataset {name!r} source_codes must be a mapping")
    if kind == "index_constituents" and not source_codes:
        raise ValueError(f"Dataset {name!r} index constituents require source_codes")
    return UpdateSpec(
        kind=kind,
        target=target,
        pool=bool(values.get("pool", False)),
        source_codes=MappingProxyType(dict(source_codes)),
    )


def dataset_spec_from_mapping(name: str, values: Mapping[str, Any]) -> DatasetSpec:
    """Build one validated dataset definition."""
    source = values.get("source")
    if source not in KNOWN_SOURCES:
        raise ValueError(f"Dataset {name!r} has unknown source {source!r}")
    columns = tuple(values.get("columns", ()))
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"Dataset {name!r} columns must be unique and non-empty")
    required = tuple(values.get("required", ()))
    primary_key = tuple(values.get("primary_key", ()))
    date_column = values.get("date_column")
    date_columns = tuple(values.get("date_columns", ()))
    for label, fields in {"required": required, "primary_key": primary_key}.items():
        missing = sorted(set(fields) - set(columns))
        if missing:
            raise ValueError(
                f"Dataset {name!r} {label} fields are not columns: {missing}"
            )
    if date_column is not None and date_column not in columns:
        raise ValueError(f"Dataset {name!r} date_column is not a column")
    if date_column is not None and date_column not in date_columns:
        raise ValueError(f"Dataset {name!r} date_column must be in date_columns")
    missing_dates = sorted(set(date_columns) - set(columns))
    if missing_dates:
        raise ValueError(
            f"Dataset {name!r} date_columns are not columns: {missing_dates}"
        )
    numeric_columns = tuple(values.get("numeric_columns", ()))
    missing_numeric = sorted(set(numeric_columns) - set(columns))
    if missing_numeric:
        raise ValueError(
            f"Dataset {name!r} numeric_columns are not columns: {missing_numeric}"
        )
    return DatasetSpec(
        name=name,
        description=str(values.get("description", "")),
        source=source,
        storage=_storage_from_mapping(name, values.get("storage", {})),
        columns=columns,
        required=required,
        primary_key=primary_key,
        date_column=date_column,
        date_columns=date_columns,
        numeric_columns=numeric_columns,
        update=_update_from_mapping(name, source, values.get("update")),
    )


with DEFAULT_CATALOG_PATH.open(encoding="utf-8") as stream:
    RAW_CATALOG = yaml.safe_load(stream) or {}
DATASET_SPECS = MappingProxyType(
    {
        name: dataset_spec_from_mapping(name, values)
        for name, values in RAW_CATALOG.items()
    }
)


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return one built-in dataset definition."""
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset {name!r}; available: {available}") from exc

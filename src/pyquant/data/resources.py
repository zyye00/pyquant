"""Static data-layer resource loading."""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


SOURCE_PROTOCOLS_PATH = Path(__file__).parents[3] / "configs/source_protocols.yaml"
_REQUIRED_SOURCES = {"baostock", "csindex", "rqdata", "price"}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@cache
def load_source_protocols() -> Mapping[str, Any]:
    """Load and validate immutable source metadata."""
    with SOURCE_PROTOCOLS_PATH.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    missing = sorted(_REQUIRED_SOURCES - set(data))
    if missing:
        raise ValueError(
            f"Source protocol configuration {SOURCE_PROTOCOLS_PATH} missing sections: {missing}"
        )
    return _freeze(data)


def load_schema_sql() -> str:
    """Load the packaged DuckDB schema definition."""
    return files("pyquant.data").joinpath("sql/schema.sql").read_text(encoding="utf-8")

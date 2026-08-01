"""DuckDB connection, schema, and trusted relation queries."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from pyquant.data.identifiers import normalize_security_symbol
from pyquant.data.resources import load_schema_sql

DEFAULT_DATABASE_PATH = Path("data/pyquant.duckdb")


def get_database_path(data_root: Path = Path("data")) -> Path:
    """Return the DuckDB path for one data root."""
    return data_root / DEFAULT_DATABASE_PATH.name


def connect_database(
    database_path: Path = DEFAULT_DATABASE_PATH,
    *,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open the project DuckDB database."""
    if not read_only:
        database_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(database_path), read_only=read_only)


def initialize_database(
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> Path:
    """Create all persistent schemas, tables, and API views."""
    with connect_database(database_path) as connection:
        connection.execute(load_schema_sql())
    return database_path


def load_relation(
    relation: str,
    columns: Collection[str],
    *,
    database_path: Path = DEFAULT_DATABASE_PATH,
    date_column: str | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    symbols: Collection[str] | None = None,
    normalize_symbols: bool = True,
) -> pd.DataFrame:
    """Load one trusted catalog relation with optional date and symbol filters."""
    if not database_path.exists():
        raise FileNotFoundError(f"DuckDB database does not exist: {database_path}")
    conditions = []
    parameters: list[Any] = []
    if date_column is not None and start is not None:
        conditions.append(f"{date_column} >= ?")
        parameters.append(
            start.to_pydatetime() if date_column == "datetime" else start.date()
        )
    if date_column is not None and end is not None:
        if date_column == "datetime":
            conditions.append(f"{date_column} < ?")
            parameters.append((end.normalize() + pd.Timedelta(days=1)).to_pydatetime())
        else:
            conditions.append(f"{date_column} <= ?")
            parameters.append(end.date())
    if symbols:
        normalized = [
            normalize_security_symbol(symbol) if normalize_symbols else str(symbol)
            for symbol in symbols
        ]
        placeholders = ", ".join("?" for _ in normalized)
        conditions.append(f"symbol IN ({placeholders})")
        parameters.extend(normalized)
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT {', '.join(columns)} FROM {relation}{where}"
    with connect_database(database_path, read_only=True) as connection:
        return connection.execute(query, parameters).df()

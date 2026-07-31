"""Download changed official index-constituent snapshots from RQData."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from pyquant.database import (
    DEFAULT_DATABASE_PATH,
    connect_database,
    initialize_database,
    write_index_constituents,
)

EXCHANGE_PREFIXES = {
    "XSHG": "SH",
    "XSHE": "SZ",
}


def rqdata_symbol_to_project(symbol: str) -> str:
    """Convert one RQData stock identifier to the project symbol format."""
    try:
        code, exchange = str(symbol).rsplit(".", maxsplit=1)
        prefix = EXCHANGE_PREFIXES[exchange]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported RQData stock identifier: {symbol!r}") from exc
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"Unsupported RQData stock identifier: {symbol!r}")
    return f"{code}.{prefix}"


def extract_changed_snapshots(
    history: Mapping[date | datetime | pd.Timestamp, Sequence[str]],
    index_code: str,
) -> pd.DataFrame:
    """Convert daily component history to changed long-form snapshots."""
    if not history:
        raise ValueError("RQData returned no index-constituent history")

    rows = []
    previous: frozenset[str] | None = None
    for effective_date, components in sorted(history.items()):
        current = frozenset(map(str, components))
        if not current:
            raise ValueError(
                f"RQData returned an empty constituent snapshot at {effective_date}"
            )
        if current == previous:
            continue
        rows.extend(
            {
                "effective_date": pd.Timestamp(effective_date),
                "index_code": index_code,
                "symbol": rqdata_symbol_to_project(symbol),
            }
            for symbol in sorted(current)
        )
        previous = current

    out = pd.DataFrame(rows)
    key = ["effective_date", "index_code", "symbol"]
    if out.duplicated(key).any():
        raise ValueError(f"Index constituents contain duplicate keys: {key}")
    return out.sort_values(key).reset_index(drop=True)


def download_index_constituents(
    start: str,
    end: str,
    rqdata_index_code: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    client: object | None = None,
) -> pd.DataFrame:
    """Download changed snapshots and replace the target index in DuckDB."""
    start_at = pd.Timestamp(start)
    end_at = pd.Timestamp(end)
    if start_at > end_at:
        raise ValueError("start must not be after end")

    if client is None:
        import rqdatac

        client = rqdatac
    client.init()
    history = client.index_components(
        rqdata_index_code,
        start_date=start_at.strftime("%Y-%m-%d"),
        end_date=end_at.strftime("%Y-%m-%d"),
    )
    index_code = rqdata_index_code.split(".", 1)[0]
    data = extract_changed_snapshots(history, index_code)
    database_path = Path(database_path)
    initialize_database(database_path)
    with connect_database(database_path) as connection:
        write_index_constituents(connection, index_code, data)
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download changed official index constituents from RQData."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--index-code", required=True)
    parser.add_argument(
        "--database-path",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
    )
    return parser.parse_args()


def main() -> None:
    """Run the RQData index-constituent download."""
    args = _parse_args()
    data = download_index_constituents(
        args.start,
        args.end,
        args.index_code,
        args.database_path,
    )
    snapshot_count = data["effective_date"].nunique()
    print(f"Saved {len(data)} rows across {snapshot_count} changed snapshots")


if __name__ == "__main__":
    main()

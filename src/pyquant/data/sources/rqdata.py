"""RQData market-index constituent history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd

from pyquant.data.resources import load_source_protocols


_RQDATA = load_source_protocols()["rqdata"]


def rqdata_symbol_to_project(symbol: str) -> str:
    """Convert one RQData stock identifier to the project symbol format."""
    try:
        code, exchange = str(symbol).rsplit(".", maxsplit=1)
        prefix = _RQDATA["exchange_prefixes"][exchange]
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


def query_index_constituents(
    start_date: str,
    end_date: str,
    source_index_code: str,
    *,
    client: Any | None = None,
) -> pd.DataFrame:
    """Query RQData and return changed constituent snapshots."""
    start_at = pd.Timestamp(start_date)
    end_at = pd.Timestamp(end_date)
    if start_at > end_at:
        raise ValueError("start must not be after end")
    if client is None:
        try:
            import rqdatac as client
        except ImportError as exc:
            raise ImportError("RQData download requires package 'rqdatac'.") from exc
    try:
        client.init()
        history = client.index_components(
            source_index_code,
            start_date=start_at.strftime("%Y-%m-%d"),
            end_date=end_at.strftime("%Y-%m-%d"),
        )
    except Exception as exc:
        raise RuntimeError(
            f"RQData request failed for {source_index_code}: {exc}"
        ) from exc
    return extract_changed_snapshots(
        history,
        source_index_code.split(".", maxsplit=1)[0],
    )

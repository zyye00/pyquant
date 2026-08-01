"""RQData market-index constituent history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd

from pyquant.data.resources import load_source_protocols


_RQDATA = load_source_protocols()["rqdata"]
_MINUTE = _RQDATA["stock_minute_1m"]
_INITIALIZED_CLIENTS: list[Any] = []


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


def project_symbol_to_rqdata(symbol: str) -> str:
    """Convert one project stock symbol to an RQData order-book identifier."""
    try:
        code, exchange = str(symbol).rsplit(".", maxsplit=1)
        suffix = _RQDATA["project_exchanges"][exchange.upper()]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported project stock identifier: {symbol!r}") from exc
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"Unsupported project stock identifier: {symbol!r}")
    return f"{code}.{suffix}"


def extract_minute_prices(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Reshape one RQData minute response without changing its values."""
    columns = ["datetime", *_MINUTE["fields"]]
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("RQData minute response must be a DataFrame")
    out = data.reset_index()
    datetime_column = next(
        (
            column
            for column in ["datetime", "date", "index", "level_1"]
            if column in out
        ),
        None,
    )
    if datetime_column is None:
        raise ValueError("RQData minute response has no datetime index or column")
    rq_symbol = project_symbol_to_rqdata(symbol)
    for column in ["order_book_id", "level_0"]:
        if column in out and set(out[column].dropna().astype(str)) - {rq_symbol}:
            raise ValueError("RQData minute response contains an unexpected symbol")
    missing = sorted(set(_MINUTE["fields"]) - set(out))
    if missing:
        raise ValueError(f"RQData minute response missing fields: {missing}")
    return out.rename(columns={datetime_column: "datetime"})[columns]


def query_stock_minute_1m(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    client: Any | None = None,
) -> pd.DataFrame:
    """Query one security's unadjusted one-minute bars from RQData."""
    client = _resolve_client(client)
    rq_symbol = project_symbol_to_rqdata(symbol)
    try:
        data = client.get_price(
            order_book_ids=rq_symbol,
            start_date=start_date,
            end_date=end_date,
            frequency=_MINUTE["frequency"],
            fields=list(_MINUTE["fields"]),
            adjust_type=_MINUTE["adjust_type"],
            skip_suspended=_MINUTE["skip_suspended"],
            expect_df=True,
            market=_MINUTE["market"],
        )
    except Exception as exc:
        raise RuntimeError(f"RQData minute request failed for {rq_symbol}: {exc}") from exc
    return extract_minute_prices(data, symbol)


def query_rqdata_trading_dates(
    start_date: str,
    end_date: str,
    *,
    client: Any | None = None,
) -> pd.DatetimeIndex:
    """Return the RQData CN trading calendar for one inclusive interval."""
    client = _resolve_client(client)
    try:
        dates = client.get_trading_dates(
            start_date,
            end_date,
            market=_MINUTE["market"],
        )
    except Exception as exc:
        raise RuntimeError(f"RQData trading-calendar request failed: {exc}") from exc
    out = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="raise"))
    if out.hasnans:
        raise ValueError("RQData trading calendar contains invalid dates")
    return out.normalize().drop_duplicates().sort_values()


def query_rqdata_quota_remaining(*, client: Any | None = None) -> int | None:
    """Return remaining daily bytes, or ``None`` for an unlimited quota."""
    client = _resolve_client(client)
    try:
        quota = client.user.get_quota()
        bytes_limit = int(quota["bytes_limit"])
        bytes_used = int(quota["bytes_used"])
    except Exception as exc:
        raise RuntimeError(f"RQData quota request failed: {exc}") from exc
    return None if bytes_limit == 0 else max(bytes_limit - bytes_used, 0)


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
    client = _resolve_client(client)
    try:
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


def _resolve_client(client: Any | None) -> Any:
    if client is None:
        try:
            import rqdatac as client
        except ImportError as exc:
            raise ImportError("RQData download requires package 'rqdatac'.") from exc
    if not any(initialized is client for initialized in _INITIALIZED_CLIENTS):
        try:
            client.init()
        except Exception as exc:
            raise RuntimeError(f"RQData initialization failed: {exc}") from exc
        _INITIALIZED_CLIENTS.append(client)
    return client

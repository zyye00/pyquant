"""RQData market-index constituent history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd

from pyquant.data.identifiers import normalize_security_symbol
from pyquant.data.resources import load_source_protocols


_RQDATA = load_source_protocols()["rqdata"]
_MINUTE = _RQDATA["stock_minute_1m"]
_PB = _RQDATA["stock_pb_daily"]
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


def query_rqdata_stock_symbols(
    start_date: str,
    end_date: str,
    *,
    client: Any | None = None,
) -> list[str]:
    """Return stocks listed during one inclusive historical interval."""
    start_at = pd.Timestamp(start_date)
    end_at = pd.Timestamp(end_date)
    if start_at > end_at:
        raise ValueError("start_date must not be after end_date")
    client = _resolve_client(client)
    try:
        data = client.all_instruments(
            type=_PB["instrument_type"],
            market=_PB["market"],
        )
    except Exception as exc:
        raise RuntimeError(f"RQData instrument-list request failed: {exc}") from exc
    if not isinstance(data, pd.DataFrame):
        raise TypeError("RQData instrument-list response must be a DataFrame")
    required = {"order_book_id", "listed_date", "de_listed_date"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"RQData instrument-list response missing fields: {missing}")
    listed = data["listed_date"].astype("string")
    listed_at = pd.to_datetime(
        listed.mask(listed.eq("2999-12-31")),
        errors="coerce",
    )
    de_listed = data["de_listed_date"].astype("string")
    open_ended = de_listed.isna() | de_listed.isin(["0000-00-00", ""])
    de_listed_at = pd.to_datetime(de_listed.mask(open_ended), errors="coerce")
    invalid_dates = (listed.ne("2999-12-31") & listed_at.isna()) | (
        ~open_ended & de_listed_at.isna()
    )
    if invalid_dates.any():
        raise ValueError("RQData instrument-list response contains invalid listing dates")
    selected = data.loc[
        listed_at.le(end_at) & (open_ended | de_listed_at.gt(start_at)),
        "order_book_id",
    ]
    if selected.isna().any():
        raise ValueError("RQData instrument-list response contains invalid identifiers")
    try:
        return sorted({rqdata_symbol_to_project(symbol) for symbol in selected})
    except ValueError as exc:
        raise ValueError("RQData instrument-list response contains unsupported identifiers") from exc


def extract_stock_pb_daily(
    data: pd.DataFrame | None,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """Normalize one RQData multi-factor PB response."""
    columns = ["date", "symbol", *_PB["factors"]]
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("RQData PB response must be a DataFrame")
    out = data.reset_index()
    order_book_column = next(
        (column for column in ["order_book_id", "level_0"] if column in out),
        None,
    )
    date_column = next(
        (column for column in ["date", "level_1"] if column in out),
        None,
    )
    if order_book_column is None or date_column is None:
        raise ValueError("RQData PB response must have order_book_id and date")
    missing = sorted(set(_PB["factors"]) - set(out))
    if missing:
        raise ValueError(f"RQData PB response missing factors: {missing}")
    expected = {project_symbol_to_rqdata(normalize_security_symbol(symbol)) for symbol in symbols}
    actual = set(out[order_book_column].dropna().astype(str))
    unexpected = sorted(actual - expected)
    if unexpected:
        raise ValueError(f"RQData PB response contains unexpected symbols: {unexpected}")
    out = out.rename(columns={order_book_column: "rq_symbol", date_column: "date"})
    out["symbol"] = out["rq_symbol"].map(rqdata_symbol_to_project)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out[["date", "symbol"]].isna().any().any():
        raise ValueError("RQData PB response contains invalid identifiers or dates")
    for factor in _PB["factors"]:
        out[factor] = pd.to_numeric(out[factor], errors="coerce").astype("float64")
    out = out[columns]
    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("RQData PB response contains duplicate (date, symbol) rows")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def query_stock_pb_daily(
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    *,
    client: Any | None = None,
) -> pd.DataFrame:
    """Query all configured daily PB factors for a security collection."""
    normalized = sorted({normalize_security_symbol(symbol) for symbol in symbols})
    if not normalized:
        raise ValueError("RQData PB query requires at least one symbol")
    start_at = pd.Timestamp(start_date)
    end_at = pd.Timestamp(end_date)
    if start_at > end_at:
        raise ValueError("start_date must not be after end_date")
    client = _resolve_client(client)
    rq_symbols = [project_symbol_to_rqdata(symbol) for symbol in normalized]
    try:
        data = client.get_factor(
            rq_symbols,
            list(_PB["factors"]),
            start_date=start_at.strftime("%Y-%m-%d"),
            end_date=end_at.strftime("%Y-%m-%d"),
            expect_df=True,
            market=_PB["market"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"RQData PB request failed for {start_date} to {end_date}: {exc}"
        ) from exc
    return extract_stock_pb_daily(data, normalized)


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
        raise RuntimeError(
            f"RQData minute request failed for {rq_symbol}: {exc}"
        ) from exc
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

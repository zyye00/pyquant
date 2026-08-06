"""DuckDB business writes and download coverage."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from pyquant.data.identifiers import normalize_index_code, normalize_security_symbol
from pyquant.data.intraday import (
    MINUTE_DAY_FAILED,
    MINUTE_DAY_INCOMPLETE,
    MINUTE_DAY_NO_DATA_CONFIRMED,
    MINUTE_DAY_VALID,
)
from pyquant.data.resources import load_source_protocols

STOCK_DAILY_FIELD_SET_ID = 1
STOCK_MINUTE_1M_FIELD_SET_ID = 1
BAOSTOCK_INDEX_DAILY_FIELD_SET_ID = 1
CSINDEX_DAILY_FIELD_SET_ID = 2
LEGACY_DIVIDEND_FIELD_SET_ID = 1
DIVIDEND_BEFORE_TAX_FIELD_SET_ID = 2
MINUTE_TASK_PENDING = 0
MINUTE_TASK_RUNNING = 1
MINUTE_TASK_SUCCESS = 2
MINUTE_TASK_PARTIAL = 3
MINUTE_TASK_NO_DATA = 4
MINUTE_TASK_FAILED = 5
MINUTE_TASK_INVALID_CODE = 6
MINUTE_TASK_QUOTA_STOPPED = 7
_RQDATA_PB = load_source_protocols()["rqdata"]["stock_pb_daily"]


@dataclass(frozen=True)
class _EntitySpec:
    reference_table: str
    value_column: str
    id_column: str
    coverage_table: str
    daily_fact_table: str
    normalize: Callable[[object], str]
    deduplicate_daily: bool = False
    id_limit: int | None = None
    id_limit_message: str | None = None


_SECURITY_SPEC = _EntitySpec(
    reference_table="ref.security",
    value_column="symbol",
    id_column="security_id",
    coverage_table="meta.stock_daily_coverage",
    daily_fact_table="core.stock_daily",
    normalize=normalize_security_symbol,
)
_PB_SPEC = _EntitySpec(
    reference_table="ref.security",
    value_column="symbol",
    id_column="security_id",
    coverage_table="meta.stock_pb_daily_coverage",
    daily_fact_table="core.stock_pb_daily",
    normalize=normalize_security_symbol,
)
_INDEX_SPEC = _EntitySpec(
    reference_table="ref.market_index",
    value_column="index_code",
    id_column="index_id",
    coverage_table="meta.index_daily_coverage",
    daily_fact_table="core.index_daily",
    normalize=normalize_index_code,
    deduplicate_daily=True,
    id_limit=np.iinfo(np.uint16).max,
    id_limit_message="ref.market_index has exhausted USMALLINT IDs",
)


@contextmanager
def _transaction(connection: duckdb.DuckDBPyConnection):
    connection.begin()
    try:
        yield
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


@contextmanager
def _registered_frame(
    connection: duckdb.DuckDBPyConnection,
    name: str,
    data: pd.DataFrame,
):
    connection.register(name, data)
    try:
        yield
    finally:
        connection.unregister(name)


def _ensure_reference_ids(
    connection: duckdb.DuckDBPyConnection,
    values: Iterable[object],
    spec: _EntitySpec,
) -> dict[str, int]:
    normalized = sorted({spec.normalize(value) for value in values})
    if not normalized:
        return {}
    existing = dict(
        connection.execute(
            f"SELECT {spec.value_column}, {spec.id_column} FROM {spec.reference_table}"
        ).fetchall()
    )
    next_id = connection.execute(
        f"SELECT COALESCE(MAX({spec.id_column}), 0) + 1 FROM {spec.reference_table}"
    ).fetchone()[0]
    rows = []
    for value in normalized:
        if value not in existing:
            if spec.id_limit is not None and next_id > spec.id_limit:
                raise OverflowError(spec.id_limit_message)
            existing[value] = next_id
            rows.append((next_id, value))
            next_id += 1
    if rows:
        connection.executemany(
            f"INSERT INTO {spec.reference_table} VALUES (?, ?)", rows
        )
    return {value: existing[value] for value in normalized}


def ensure_securities(
    connection: duckdb.DuckDBPyConnection,
    symbols: Iterable[object],
) -> dict[str, int]:
    """Append unseen symbols and return stable symbol-to-ID mappings."""
    return _ensure_reference_ids(connection, symbols, _SECURITY_SPEC)


def ensure_market_indices(
    connection: duckdb.DuckDBPyConnection,
    index_codes: Iterable[object],
) -> dict[str, int]:
    """Append unseen market indices and return stable code-to-ID mappings."""
    return _ensure_reference_ids(connection, index_codes, _INDEX_SPEC)


def _daily_coverage(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    field_set_id: int,
    spec: _EntitySpec,
) -> list[tuple[str, str]]:
    rows = connection.execute(
        f"""
        SELECT CAST(c.start_date AS VARCHAR), CAST(c.end_date AS VARCHAR)
        FROM {spec.coverage_table} AS c
        JOIN {spec.reference_table} AS r USING ({spec.id_column})
        WHERE r.{spec.value_column} = ? AND c.field_set_id = ?
        ORDER BY c.start_date
        """,
        [spec.normalize(code), field_set_id],
    ).fetchall()
    return [(start, end) for start, end in rows]


def stock_daily_coverage(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    field_set_id: int = STOCK_DAILY_FIELD_SET_ID,
) -> list[tuple[str, str]]:
    """Return completed daily ranges for one source code."""
    return _daily_coverage(connection, code, field_set_id, _SECURITY_SPEC)


def stock_adjust_factor_coverage(
    connection: duckdb.DuckDBPyConnection,
    code: str,
) -> list[tuple[str, str]]:
    """Return completed adjustment-factor query ranges for one security."""
    rows = connection.execute(
        """
        SELECT CAST(c.start_date AS VARCHAR), CAST(c.end_date AS VARCHAR)
        FROM meta.stock_adjust_factor_coverage AS c
        JOIN ref.security AS s USING (security_id)
        WHERE s.symbol = ?
        ORDER BY c.start_date
        """,
        [normalize_security_symbol(code)],
    ).fetchall()
    return [(start, end) for start, end in rows]


def stock_pb_coverage(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    field_set_id: int = int(_RQDATA_PB["field_set_id"]),
) -> list[tuple[str, str]]:
    """Return completed RQData PB ranges for one security."""
    return _daily_coverage(connection, code, field_set_id, _PB_SPEC)


def index_daily_coverage(
    connection: duckdb.DuckDBPyConnection,
    index_code: str,
    field_set_id: int,
) -> list[tuple[str, str]]:
    """Return completed daily ranges for one index and field set."""
    return _daily_coverage(connection, index_code, field_set_id, _INDEX_SPEC)


def dividend_coverage(
    connection: duckdb.DuckDBPyConnection,
    field_set_id: int = DIVIDEND_BEFORE_TAX_FIELD_SET_ID,
) -> set[tuple[str, int]]:
    """Return completed dividend source-code years for one field set."""
    return {
        (symbol, int(year))
        for symbol, year in connection.execute(
            """
            SELECT s.symbol, c.query_year
            FROM meta.dividend_coverage AS c
            JOIN ref.security AS s USING (security_id)
            WHERE c.field_set_id = ?
            """,
            [field_set_id],
        ).fetchall()
    }


def share_capital_coverage(
    connection: duckdb.DuckDBPyConnection,
) -> set[tuple[str, int, int]]:
    """Return completed quarterly-share source-code periods."""
    return {
        (symbol, int(year), int(quarter))
        for symbol, year, quarter in connection.execute(
            """
            SELECT s.symbol, c.report_year, c.report_quarter
            FROM meta.share_capital_coverage AS c
            JOIN ref.security AS s USING (security_id)
            """
        ).fetchall()
    }


def completed_minute_days(
    connection: duckdb.DuckDBPyConnection,
    symbols: Iterable[str],
    field_set_id: int = STOCK_MINUTE_1M_FIELD_SET_ID,
) -> set[tuple[str, pd.Timestamp]]:
    """Return minute days whose raw and feature requirements are satisfied."""
    symbols = sorted({normalize_security_symbol(symbol) for symbol in symbols})
    if not symbols:
        return set()
    placeholders = ", ".join("?" for _ in symbols)
    rows = connection.execute(
        f"""
        SELECT s.symbol, d.trade_date
        FROM meta.minute_day_status AS d
        JOIN ref.security AS s USING (security_id)
        WHERE s.symbol IN ({placeholders})
          AND d.field_set_id = ?
          AND d.status IN (?, ?)
          AND d.raw_saved
          AND d.feature_saved
        """,
        [
            *symbols,
            field_set_id,
            MINUTE_DAY_VALID,
            MINUTE_DAY_NO_DATA_CONFIRMED,
        ],
    ).fetchall()
    return {(symbol, pd.Timestamp(trade_date)) for symbol, trade_date in rows}


def create_minute_download_task(
    connection: duckdb.DuckDBPyConnection,
    symbol: str,
    start_date: str,
    end_date: str,
    retain_raw: bool,
    field_set_id: int = STOCK_MINUTE_1M_FIELD_SET_ID,
) -> int:
    """Create one pending minute-download task and return its stable ID."""
    symbol = normalize_security_symbol(symbol)
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start > end:
        raise ValueError("start_date must not be after end_date")
    with _transaction(connection):
        security_id = ensure_securities(connection, [symbol])[symbol]
        existing = connection.execute(
            """
            SELECT task_id
            FROM meta.minute_download_task
            WHERE security_id = ?
              AND start_date = ?
              AND end_date = ?
              AND field_set_id = ?
              AND retain_raw = ?
              AND status = ?
            ORDER BY task_id
            LIMIT 1
            """,
            [
                security_id,
                start,
                end,
                field_set_id,
                retain_raw,
                MINUTE_TASK_PENDING,
            ],
        ).fetchone()
        if existing is not None:
            return int(existing[0])
        task_id = connection.execute(
            "SELECT COALESCE(MAX(task_id), 0) + 1 FROM meta.minute_download_task"
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO meta.minute_download_task
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, now(), now())
            """,
            [
                task_id,
                security_id,
                start,
                end,
                field_set_id,
                retain_raw,
                MINUTE_TASK_PENDING,
            ],
        )
    return int(task_id)


def update_minute_download_task(
    connection: duckdb.DuckDBPyConnection,
    task_id: int,
    status: int,
    *,
    increment_attempts: bool = False,
    rows_received: int | None = None,
    days_received: int | None = None,
    error: Exception | None = None,
) -> None:
    """Update mutable execution fields for one minute-download task."""
    error_type = type(error).__name__ if error is not None else None
    error_message = str(error) if error is not None else None
    changed = connection.execute(
        """
        UPDATE meta.minute_download_task
        SET
            status = ?,
            attempts = attempts + ?,
            rows_received = COALESCE(?, rows_received),
            days_received = COALESCE(?, days_received),
            error_type = ?,
            error_message = ?,
            updated_at = now()
        WHERE task_id = ?
        RETURNING task_id
        """,
        [
            status,
            int(increment_attempts),
            rows_received,
            days_received,
            error_type,
            error_message,
            task_id,
        ],
    ).fetchone()
    if changed is None:
        raise ValueError(f"Unknown minute download task: {task_id}")


def recover_minute_download_tasks(
    connection: duckdb.DuckDBPyConnection,
    max_attempts: int,
    stale_after_seconds: int = 3_600,
) -> None:
    """Recover tasks left running by an interrupted process."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    connection.execute(
        """
        UPDATE meta.minute_download_task
        SET
            status = CASE WHEN attempts < ? THEN ? ELSE ? END,
            error_type = CASE
                WHEN attempts < ? THEN NULL
                ELSE 'InterruptedTask'
            END,
            error_message = CASE
                WHEN attempts < ? THEN NULL
                ELSE 'Task exceeded max_attempts after interruption'
            END,
            updated_at = now()
        WHERE status = ?
          AND updated_at < now() - ? * INTERVAL '1 second'
        """,
        [
            max_attempts,
            MINUTE_TASK_PENDING,
            MINUTE_TASK_FAILED,
            max_attempts,
            max_attempts,
            MINUTE_TASK_RUNNING,
            stale_after_seconds,
        ],
    )


def write_minute_request(
    connection: duckdb.DuckDBPyConnection,
    task_id: int,
    symbol: str,
    minute: pd.DataFrame,
    daily: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    retain_raw: bool = True,
    field_set_id: int = STOCK_MINUTE_1M_FIELD_SET_ID,
) -> None:
    """Atomically replace minute facts, daily features, coverage, and task state."""
    symbol = normalize_security_symbol(symbol)
    required_minute = {
        "symbol",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "total_turnover",
    }
    required_daily = {
        "symbol",
        "trade_date",
        "volatility",
        "bar_count",
        "return_count",
        "status",
    }
    for name, data, required in [
        ("minute", minute, required_minute),
        ("daily", daily, required_daily),
    ]:
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"{name} data missing required columns: {missing}")
        if not data.empty and set(data["symbol"]) != {symbol}:
            raise ValueError(f"{name} data does not match the requested symbol")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must not be after end_date")
    if (
        not daily["status"]
        .isin([MINUTE_DAY_VALID, MINUTE_DAY_NO_DATA_CONFIRMED, MINUTE_DAY_INCOMPLETE])
        .all()
    ):
        raise ValueError("daily data contains unsupported minute-day statuses")
    incoming_minute = minute.drop(columns="symbol").copy()
    incoming_daily = daily.drop(columns="symbol").copy()
    terminal = incoming_daily["status"].isin(
        [MINUTE_DAY_VALID, MINUTE_DAY_NO_DATA_CONFIRMED]
    )
    if (
        terminal.all()
        and incoming_daily["status"].eq(MINUTE_DAY_NO_DATA_CONFIRMED).all()
    ):
        task_status = MINUTE_TASK_NO_DATA
    elif terminal.all():
        task_status = MINUTE_TASK_SUCCESS
    else:
        task_status = MINUTE_TASK_PARTIAL

    with _transaction(connection):
        security_id = ensure_securities(connection, [symbol])[symbol]
        if retain_raw:
            connection.execute(
                """
                DELETE FROM core.stock_minute_1m
                WHERE security_id = ?
                  AND datetime >= ?
                  AND datetime < ?
                """,
                [security_id, start, end + pd.Timedelta(days=1)],
            )
            if not incoming_minute.empty:
                with _registered_frame(
                    connection, "incoming_stock_minute_1m", incoming_minute
                ):
                    connection.execute(
                        """
                        INSERT INTO core.stock_minute_1m
                        SELECT
                            ?,
                            CAST(datetime AS TIMESTAMP),
                            CAST(open AS FLOAT),
                            CAST(high AS FLOAT),
                            CAST(low AS FLOAT),
                            CAST(close AS FLOAT),
                            CAST(volume AS DOUBLE),
                            CAST(total_turnover AS DOUBLE)
                        FROM incoming_stock_minute_1m
                        ORDER BY datetime
                        """,
                        [security_id],
                    )
        connection.execute(
            """
            DELETE FROM feature.intraday_volatility_daily
            WHERE security_id = ?
              AND trade_date >= ?
              AND trade_date <= ?
            """,
            [security_id, start, end],
        )
        if not incoming_daily.empty:
            with _registered_frame(connection, "incoming_minute_daily", incoming_daily):
                feature_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info('feature.intraday_volatility_daily')"
                    ).fetchall()
                }
                targets = ["security_id", "trade_date", "volatility"]
                values = [
                    "?",
                    "CAST(trade_date AS DATE)",
                    "CAST(volatility AS FLOAT)",
                ]
                legacy = {
                    "vol_daily": "CAST(volatility AS FLOAT)",
                    "bar_count": "CAST(bar_count AS USMALLINT)",
                    "return_count": "CAST(return_count AS USMALLINT)",
                    "is_valid": "TRUE",
                }
                for column, expression in legacy.items():
                    if column in feature_columns:
                        targets.append(column)
                        values.append(expression)
                connection.execute(
                    f"""
                    INSERT INTO feature.intraday_volatility_daily (
                        {", ".join(targets)}
                    )
                    SELECT
                        {", ".join(values)}
                    FROM incoming_minute_daily
                    WHERE status = ?
                    """,
                    [security_id, MINUTE_DAY_VALID],
                )
                connection.execute(
                    """
                    DELETE FROM meta.minute_day_status
                    WHERE security_id = ?
                      AND field_set_id = ?
                      AND trade_date IN (
                          SELECT CAST(trade_date AS DATE) FROM incoming_minute_daily
                      )
                    """,
                    [security_id, field_set_id],
                )
                connection.execute(
                    """
                    INSERT INTO meta.minute_day_status
                    SELECT
                        ?,
                        CAST(trade_date AS DATE),
                        ?,
                        CAST(status AS UTINYINT),
                        CAST(bar_count AS USMALLINT),
                        CAST(return_count AS USMALLINT),
                        TRUE,
                        status IN (?, ?),
                        now()
                    FROM incoming_minute_daily
                    """,
                    [
                        security_id,
                        field_set_id,
                        MINUTE_DAY_VALID,
                        MINUTE_DAY_NO_DATA_CONFIRMED,
                    ],
                )
        changed = connection.execute(
            """
            UPDATE meta.minute_download_task
            SET
                status = ?,
                rows_received = ?,
                days_received = ?,
                error_type = NULL,
                error_message = NULL,
                updated_at = now()
            WHERE task_id = ? AND security_id = ?
            RETURNING task_id
            """,
            [
                task_status,
                len(incoming_minute),
                int(incoming_daily["status"].eq(MINUTE_DAY_VALID).sum()),
                task_id,
                security_id,
            ],
        ).fetchone()
        if changed is None:
            raise ValueError(f"Minute task {task_id} does not match {symbol}")


def write_minute_request_failure(
    connection: duckdb.DuckDBPyConnection,
    task_id: int,
    symbol: str,
    trading_dates: Iterable[object],
    error: Exception,
    field_set_id: int = STOCK_MINUTE_1M_FIELD_SET_ID,
) -> None:
    """Atomically record failed minute days and their terminal task attempt."""
    symbol = normalize_security_symbol(symbol)
    dates = sorted({pd.Timestamp(trade_date).date() for trade_date in trading_dates})
    with _transaction(connection):
        task = connection.execute(
            """
            SELECT t.security_id
            FROM meta.minute_download_task AS t
            JOIN ref.security AS s USING (security_id)
            WHERE t.task_id = ? AND s.symbol = ?
            """,
            [task_id, symbol],
        ).fetchone()
        if task is None:
            raise ValueError(f"Minute task {task_id} does not match {symbol}")
        security_id = task[0]
        if dates:
            connection.executemany(
                """
                DELETE FROM meta.minute_day_status
                WHERE security_id = ?
                  AND trade_date = ?
                  AND field_set_id = ?
                """,
                [(security_id, trade_date, field_set_id) for trade_date in dates],
            )
            connection.executemany(
                """
                INSERT INTO meta.minute_day_status
                VALUES (?, ?, ?, ?, NULL, NULL, FALSE, FALSE, now())
                """,
                [
                    (
                        security_id,
                        trade_date,
                        field_set_id,
                        MINUTE_DAY_FAILED,
                    )
                    for trade_date in dates
                ],
            )
        connection.execute(
            """
            UPDATE meta.minute_download_task
            SET
                status = ?,
                error_type = ?,
                error_message = ?,
                updated_at = now()
            WHERE task_id = ?
            """,
            [MINUTE_TASK_FAILED, type(error).__name__, str(error), task_id],
        )


def write_stock_daily_request(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    """Persist one daily source response and its merged coverage atomically."""
    symbol = normalize_security_symbol(code)
    incoming = _prepare_stock_daily(data)
    with _transaction(connection):
        security_id = ensure_securities(connection, [symbol])[symbol]
        _replace_daily_facts(connection, security_id, incoming, _SECURITY_SPEC)
        _replace_daily_coverage(
            connection,
            security_id,
            start_date,
            end_date,
            STOCK_DAILY_FIELD_SET_ID,
            _SECURITY_SPEC,
        )


def write_stock_adjust_factor_request(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    """Replace one adjustment-factor response and merge its coverage."""
    symbol = normalize_security_symbol(code)
    incoming = _prepare_adjust_factors(data)
    if not incoming.empty and incoming["operate_date"].duplicated().any():
        raise ValueError("Adjustment factors contain duplicate operate dates")
    start_at = pd.Timestamp(start_date)
    end_at = pd.Timestamp(end_date)
    if start_at > end_at:
        raise ValueError("adjustment-factor coverage start must not be after end")
    if not incoming.empty and not incoming["operate_date"].between(
        start_at, end_at
    ).all():
        raise ValueError("Adjustment factors contain dates outside the request")
    with _transaction(connection):
        security_id = ensure_securities(connection, [symbol])[symbol]
        connection.execute(
            """
            DELETE FROM core.stock_adjust_factor
            WHERE security_id = ? AND operate_date BETWEEN ? AND ?
            """,
            [security_id, start_at.date(), end_at.date()],
        )
        if not incoming.empty:
            with _registered_frame(connection, "incoming_adjust_factor", incoming):
                connection.execute(
                    """
                    INSERT INTO core.stock_adjust_factor
                    SELECT ?, CAST(operate_date AS DATE), fore_adjust_factor,
                           back_adjust_factor, adjust_factor
                    FROM incoming_adjust_factor
                    """,
                    [security_id],
                )
        rows = connection.execute(
            """
            SELECT start_date, end_date
            FROM meta.stock_adjust_factor_coverage
            WHERE security_id = ?
            """,
            [security_id],
        ).fetchall()
        merged = _merge_date_ranges([*rows, (start_at.date(), end_at.date())])
        connection.execute(
            "DELETE FROM meta.stock_adjust_factor_coverage WHERE security_id = ?",
            [security_id],
        )
        connection.executemany(
            "INSERT INTO meta.stock_adjust_factor_coverage VALUES (?, ?, ?)",
            [(security_id, first, last) for first, last in merged],
        )


def write_stock_pb_request(
    connection: duckdb.DuckDBPyConnection,
    symbols: Iterable[str],
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    field_set_id: int = int(_RQDATA_PB["field_set_id"]),
) -> None:
    """Persist one RQData PB response and its merged coverage atomically."""
    symbols = sorted({normalize_security_symbol(symbol) for symbol in symbols})
    if not symbols:
        raise ValueError("PB write requires at least one symbol")
    start_at = pd.Timestamp(start_date)
    end_at = pd.Timestamp(end_date)
    if start_at > end_at:
        raise ValueError("PB coverage start_date must not be after end_date")
    incoming = _prepare_stock_pb(data)
    if not incoming.empty:
        if not incoming["symbol"].isin(symbols).all():
            raise ValueError("PB response contains an unexpected symbol")
        if not incoming["date"].between(start_at, end_at).all():
            raise ValueError("PB response contains a date outside its request")
    with _transaction(connection):
        security_ids = ensure_securities(connection, symbols)
        for symbol in symbols:
            connection.execute(
                """
                DELETE FROM core.stock_pb_daily
                WHERE security_id = ? AND trade_date BETWEEN ? AND ?
                """,
                [security_ids[symbol], start_at.date(), end_at.date()],
            )
        if not incoming.empty:
            incoming = incoming.assign(
                security_id=incoming["symbol"].map(security_ids).astype("int64")
            )
            with _registered_frame(connection, "incoming_stock_pb", incoming):
                connection.execute(
                    f"""
                    INSERT INTO core.stock_pb_daily (
                        security_id,
                        trade_date,
                        {", ".join(_PB_VALUE_COLUMNS)}
                    )
                    SELECT
                        security_id,
                        CAST(date AS DATE),
                        {", ".join(_PB_FACTORS)}
                    FROM incoming_stock_pb
                    """
                )
        for symbol in symbols:
            _replace_daily_coverage(
                connection,
                security_ids[symbol],
                start_date,
                end_date,
                field_set_id,
                _PB_SPEC,
            )


def write_index_daily_request(
    connection: duckdb.DuckDBPyConnection,
    index_code: str,
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    field_set_id: int,
) -> None:
    """Persist one index response and its field-specific coverage atomically."""
    if field_set_id not in {
        BAOSTOCK_INDEX_DAILY_FIELD_SET_ID,
        CSINDEX_DAILY_FIELD_SET_ID,
    }:
        raise ValueError(f"Unsupported index daily field set: {field_set_id}")
    code = normalize_index_code(index_code)
    incoming = _prepare_index_daily(data)
    with _transaction(connection):
        index_id = ensure_market_indices(connection, [code])[code]
        if field_set_id == BAOSTOCK_INDEX_DAILY_FIELD_SET_ID:
            _replace_daily_facts(connection, index_id, incoming, _INDEX_SPEC)
        elif not incoming.empty:
            with _registered_frame(connection, "incoming_index_daily", incoming):
                connection.execute(
                    """
                    UPDATE core.index_daily AS target
                    SET close = source.close
                    FROM incoming_index_daily AS source
                    WHERE target.index_id = ?
                      AND target.trade_date = CAST(source.date AS DATE)
                    """,
                    [index_id],
                )
                connection.execute(
                    """
                    INSERT INTO core.index_daily (
                        index_id,
                        trade_date,
                        close
                    )
                    SELECT
                        ?,
                        CAST(source.date AS DATE),
                        source.close
                    FROM incoming_index_daily AS source
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM core.index_daily AS target
                        WHERE target.index_id = ?
                          AND target.trade_date = CAST(source.date AS DATE)
                    )
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY CAST(source.date AS DATE)
                        ORDER BY CAST(source.date AS DATE)
                    ) = 1
                    """,
                    [index_id, index_id],
                )
        _replace_daily_coverage(
            connection,
            index_id,
            start_date,
            end_date,
            field_set_id,
            _INDEX_SPEC,
        )


def write_index_constituents(
    connection: duckdb.DuckDBPyConnection,
    index_code: str,
    data: pd.DataFrame,
) -> None:
    """Replace all changed constituent snapshots for one market index."""
    code = normalize_index_code(index_code)
    required = {"effective_date", "index_code", "symbol"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Index constituents missing required columns: {missing}")
    incoming = data.loc[:, ["effective_date", "index_code", "symbol"]].copy()
    incoming["index_code"] = incoming["index_code"].map(normalize_index_code)
    if set(incoming["index_code"]) - {code}:
        raise ValueError("Index constituent rows do not match the target index")
    incoming["symbol"] = incoming["symbol"].map(normalize_security_symbol)
    incoming["effective_date"] = pd.to_datetime(
        incoming["effective_date"], errors="raise"
    )
    if incoming[["effective_date", "symbol"]].isna().any().any():
        raise ValueError("Index constituents must not contain missing keys")
    incoming = incoming.drop_duplicates(["effective_date", "index_code", "symbol"])
    with _transaction(connection):
        index_id = ensure_market_indices(connection, [code])[code]
        security_ids = ensure_securities(connection, incoming["symbol"])
        incoming["security_id"] = incoming["symbol"].map(security_ids)
        connection.execute(
            "DELETE FROM core.index_constituent WHERE index_id = ?",
            [index_id],
        )
        if not incoming.empty:
            with _registered_frame(connection, "incoming_index_constituent", incoming):
                connection.execute(
                    """
                    INSERT INTO core.index_constituent
                    SELECT
                        ?,
                        CAST(effective_date AS DATE),
                        security_id
                    FROM incoming_index_constituent
                    """,
                    [index_id],
                )


def write_dividend_request(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    year: int,
    data: pd.DataFrame,
) -> None:
    """Persist one before-tax dividend response and its coverage atomically."""
    symbol = normalize_security_symbol(code)
    incoming = data.loc[
        data["cash_dividend_before_tax"].notna(),
        [
            "announce_date",
            "record_date",
            "operate_date",
            "payment_date",
            "cash_dividend_before_tax",
        ],
    ].copy()
    if incoming["cash_dividend_before_tax"].lt(0).any():
        raise ValueError("cash_dividend_before_tax must not be negative")
    with _transaction(connection):
        security_id = ensure_securities(connection, [symbol])[symbol]
        if not incoming.empty:
            with _registered_frame(connection, "incoming_dividend", incoming):
                connection.execute(
                    """
                    INSERT INTO core.dividend
                    SELECT DISTINCT
                        ?,
                        CAST(announce_date AS DATE),
                        CAST(record_date AS DATE),
                        CAST(operate_date AS DATE),
                        CAST(payment_date AS DATE),
                        CAST(cash_dividend_before_tax AS FLOAT)
                    FROM incoming_dividend AS i
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM core.dividend AS d
                        WHERE d.security_id = ?
                          AND d.announce_date IS NOT DISTINCT FROM CAST(i.announce_date AS DATE)
                          AND d.record_date IS NOT DISTINCT FROM CAST(i.record_date AS DATE)
                          AND d.ex_date IS NOT DISTINCT FROM CAST(i.operate_date AS DATE)
                          AND d.payment_date IS NOT DISTINCT FROM CAST(i.payment_date AS DATE)
                          AND d.cash_dividend_before_tax IS NOT DISTINCT FROM
                              CAST(i.cash_dividend_before_tax AS FLOAT)
                    )
                    """,
                    [security_id, security_id],
                )
        connection.execute(
            """
            INSERT OR IGNORE INTO meta.dividend_coverage
            VALUES (?, ?, ?)
            """,
            [security_id, year, DIVIDEND_BEFORE_TAX_FIELD_SET_ID],
        )


def write_share_capital_request(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    year: int,
    quarter: int,
    data: pd.DataFrame,
) -> None:
    """Persist one quarterly-share response and its coverage atomically."""
    symbol = normalize_security_symbol(code)
    incoming = _prepare_share_capital(data)
    with _transaction(connection):
        security_id = ensure_securities(connection, [symbol])[symbol]
        if not incoming.empty:
            with _registered_frame(connection, "incoming_share_capital", incoming):
                connection.execute(
                    """
                    DELETE FROM core.share_capital_quarterly
                    WHERE security_id = ?
                      AND report_date IN (
                          SELECT CAST(report_date AS DATE)
                          FROM incoming_share_capital
                      )
                    """,
                    [security_id],
                )
                connection.execute(
                    """
                    INSERT INTO core.share_capital_quarterly
                    SELECT DISTINCT
                        ?,
                        CAST(report_date AS DATE),
                        CAST(publish_date AS DATE),
                        CAST(total_shares AS BIGINT)
                    FROM incoming_share_capital
                    """,
                    [security_id],
                )
        connection.execute(
            """
            INSERT OR IGNORE INTO meta.share_capital_coverage
            VALUES (?, ?, ?)
            """,
            [security_id, year, quarter],
        )


_DAILY_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "pe_ttm",
    "ps_ttm",
    "pcf_ncf_ttm",
    "is_st",
]
_DAILY_VALUE_COLUMNS = _DAILY_COLUMNS[1:]
_DAILY_RENAMES = {
    "peTTM": "pe_ttm",
    "psTTM": "ps_ttm",
    "pcfNcfTTM": "pcf_ncf_ttm",
    "isST": "is_st",
}
_PB_FACTORS = tuple(_RQDATA_PB["factors"])
_PB_VALUE_COLUMNS = _PB_FACTORS


def _prepare_daily(data: pd.DataFrame) -> pd.DataFrame:
    out = data.rename(columns=_DAILY_RENAMES).copy()
    for column in _DAILY_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    return out[_DAILY_COLUMNS]


def _prepare_stock_daily(data: pd.DataFrame) -> pd.DataFrame:
    return _prepare_daily(data)


def _prepare_adjust_factors(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "operate_date",
        "fore_adjust_factor",
        "back_adjust_factor",
        "adjust_factor",
    ]
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)
    missing = sorted(set(columns) - set(data))
    if missing:
        raise ValueError(f"Adjustment factors missing columns: {missing}")
    out = data[columns].copy()
    out["operate_date"] = pd.to_datetime(out["operate_date"], errors="coerce")
    if out["operate_date"].isna().any():
        raise ValueError("Adjustment-factor dates must not be missing")
    for column in columns[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any() or (~out[column].gt(0)).any():
            raise ValueError(f"{column} must contain positive numbers")
    return out.sort_values("operate_date").reset_index(drop=True)


def _prepare_stock_pb(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "symbol", *_PB_FACTORS]
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)
    missing = sorted(set(columns) - set(data))
    if missing:
        raise ValueError(f"PB data missing columns: {missing}")
    out = data[columns].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("PB dates must not be missing")
    out["symbol"] = out["symbol"].map(normalize_security_symbol)
    for factor in _PB_FACTORS:
        out[factor] = pd.to_numeric(out[factor], errors="coerce").astype("float64")
    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("PB data contains duplicate (date, symbol) rows")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _prepare_index_daily(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=_DAILY_COLUMNS)
    for column in ["date", "close"]:
        if column not in data:
            raise ValueError(f"Index daily data missing required column: {column}")
    out = _prepare_daily(data)
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if out["date"].isna().any():
        raise ValueError("Index daily dates must not be missing")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if out["close"].isna().any():
        raise ValueError("Index daily close values must not be missing")
    return (
        out[_DAILY_COLUMNS]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _prepare_share_capital(data: pd.DataFrame) -> pd.DataFrame:
    out = data[["publish_date", "report_date", "total_shares"]].copy()
    shares = pd.to_numeric(out["total_shares"], errors="coerce")
    finite = shares.notna() & np.isfinite(shares)
    invalid = (~finite & out["total_shares"].notna()) | shares.lt(0)
    invalid |= finite & ~np.isclose(shares, np.round(shares), rtol=0, atol=1e-6)
    invalid |= shares.gt(np.iinfo(np.int64).max)
    if invalid.any():
        raise ValueError("total_shares contains invalid non-integer values")
    out["total_shares"] = shares.round().astype("Int64")
    return out.dropna(subset=["report_date"])


def _replace_daily_facts(
    connection: duckdb.DuckDBPyConnection,
    entity_id: int,
    incoming: pd.DataFrame,
    spec: _EntitySpec,
) -> None:
    if incoming.empty:
        return
    with _registered_frame(connection, "incoming_daily", incoming):
        connection.execute(
            f"""
            DELETE FROM {spec.daily_fact_table}
            WHERE {spec.id_column} = ?
              AND trade_date IN (
                  SELECT CAST(date AS DATE) FROM incoming_daily
              )
            """,
            [entity_id],
        )
        deduplication = (
            """
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY CAST(date AS DATE)
                ORDER BY CAST(date AS DATE)
            ) = 1
            """
            if spec.deduplicate_daily
            else ""
        )
        connection.execute(
            f"""
            INSERT INTO {spec.daily_fact_table}
            SELECT
                ?,
                CAST(date AS DATE),
                {", ".join(_DAILY_VALUE_COLUMNS)}
            FROM incoming_daily
            {deduplication}
            """,
            [entity_id],
        )


def _replace_daily_coverage(
    connection: duckdb.DuckDBPyConnection,
    entity_id: int,
    start_date: str,
    end_date: str,
    field_set_id: int,
    spec: _EntitySpec,
) -> None:
    rows = connection.execute(
        f"""
        SELECT start_date, end_date
        FROM {spec.coverage_table}
        WHERE {spec.id_column} = ? AND field_set_id = ?
        ORDER BY start_date
        """,
        [entity_id, field_set_id],
    ).fetchall()
    rows.append((pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()))
    merged = _merge_date_ranges(rows)
    connection.execute(
        f"""
        DELETE FROM {spec.coverage_table}
        WHERE {spec.id_column} = ? AND field_set_id = ?
        """,
        [entity_id, field_set_id],
    )
    connection.executemany(
        f"INSERT INTO {spec.coverage_table} VALUES (?, ?, ?, ?)",
        [
            (entity_id, range_start, range_end, field_set_id)
            for range_start, range_end in merged
        ],
    )


def _merge_date_ranges(
    ranges: Iterable[tuple[object, object]],
) -> list[tuple[object, object]]:
    ordered = sorted(
        (pd.Timestamp(start).date(), pd.Timestamp(end).date()) for start, end in ranges
    )
    merged = []
    for start, end in ordered:
        if start > end:
            raise ValueError("coverage start_date must not be after end_date")
        if not merged or start > pd.Timestamp(merged[-1][1]).date() + pd.Timedelta(
            days=1
        ):
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(values) for values in merged]

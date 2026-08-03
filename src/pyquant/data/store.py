"""DuckDB business writes and download coverage."""

from __future__ import annotations

from collections.abc import Iterable

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


def ensure_securities(
    connection: duckdb.DuckDBPyConnection,
    symbols: Iterable[object],
) -> dict[str, int]:
    """Append unseen symbols and return stable symbol-to-ID mappings."""
    normalized = sorted({normalize_security_symbol(symbol) for symbol in symbols})
    if not normalized:
        return {}
    existing = dict(
        connection.execute("SELECT symbol, security_id FROM ref.security").fetchall()
    )
    next_id = connection.execute(
        "SELECT COALESCE(MAX(security_id), 0) + 1 FROM ref.security"
    ).fetchone()[0]
    rows = []
    for symbol in normalized:
        if symbol not in existing:
            existing[symbol] = next_id
            rows.append((next_id, symbol))
            next_id += 1
    if rows:
        connection.executemany("INSERT INTO ref.security VALUES (?, ?)", rows)
    return {symbol: existing[symbol] for symbol in normalized}


def ensure_market_indices(
    connection: duckdb.DuckDBPyConnection,
    index_codes: Iterable[object],
) -> dict[str, int]:
    """Append unseen market indices and return stable code-to-ID mappings."""
    normalized = sorted({normalize_index_code(code) for code in index_codes})
    if not normalized:
        return {}
    existing = dict(
        connection.execute(
            "SELECT index_code, index_id FROM ref.market_index"
        ).fetchall()
    )
    next_id = connection.execute(
        "SELECT COALESCE(MAX(index_id), 0) + 1 FROM ref.market_index"
    ).fetchone()[0]
    rows = []
    for code in normalized:
        if code not in existing:
            if next_id > np.iinfo(np.uint16).max:
                raise OverflowError("ref.market_index has exhausted USMALLINT IDs")
            existing[code] = next_id
            rows.append((next_id, code))
            next_id += 1
    if rows:
        connection.executemany("INSERT INTO ref.market_index VALUES (?, ?)", rows)
    return {code: existing[code] for code in normalized}


def stock_daily_coverage(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    field_set_id: int = STOCK_DAILY_FIELD_SET_ID,
) -> list[tuple[str, str]]:
    """Return completed daily ranges for one source code."""
    symbol = normalize_security_symbol(code)
    rows = connection.execute(
        """
        SELECT CAST(c.start_date AS VARCHAR), CAST(c.end_date AS VARCHAR)
        FROM meta.stock_daily_coverage AS c
        JOIN ref.security AS s USING (security_id)
        WHERE s.symbol = ? AND c.field_set_id = ?
        ORDER BY c.start_date
        """,
        [symbol, field_set_id],
    ).fetchall()
    return [(start, end) for start, end in rows]


def index_daily_coverage(
    connection: duckdb.DuckDBPyConnection,
    index_code: str,
    field_set_id: int,
) -> list[tuple[str, str]]:
    """Return completed daily ranges for one index and field set."""
    rows = connection.execute(
        """
        SELECT CAST(c.start_date AS VARCHAR), CAST(c.end_date AS VARCHAR)
        FROM meta.index_daily_coverage AS c
        JOIN ref.market_index AS i USING (index_id)
        WHERE i.index_code = ? AND c.field_set_id = ?
        ORDER BY c.start_date
        """,
        [normalize_index_code(index_code), field_set_id],
    ).fetchall()
    return [(start, end) for start, end in rows]


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
    connection.begin()
    try:
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
            connection.commit()
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
        connection.commit()
    except Exception:
        connection.rollback()
        raise
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

    connection.begin()
    registered_minute = False
    registered_daily = False
    try:
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
                connection.register("incoming_stock_minute_1m", incoming_minute)
                registered_minute = True
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
            connection.register("incoming_minute_daily", incoming_daily)
            registered_daily = True
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
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if registered_minute:
            connection.unregister("incoming_stock_minute_1m")
        if registered_daily:
            connection.unregister("incoming_minute_daily")


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
    connection.begin()
    try:
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
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def write_stock_daily_request(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    """Persist one daily source response and its merged coverage atomically."""
    symbol = normalize_security_symbol(code)
    security_id = ensure_securities(connection, [symbol])[symbol]
    incoming = _prepare_stock_daily(data)
    connection.begin()
    try:
        if not incoming.empty:
            connection.register("incoming_stock_daily", incoming)
            connection.execute(
                """
                DELETE FROM core.stock_daily
                WHERE security_id = ?
                  AND trade_date IN (
                      SELECT CAST(date AS DATE) FROM incoming_stock_daily
                  )
                """,
                [security_id],
            )
            connection.execute(
                """
                INSERT INTO core.stock_daily
                SELECT
                    ?,
                    CAST(date AS DATE),
                    open,
                    high,
                    low,
                    close,
                    preclose,
                    volume,
                    amount,
                    turn,
                    peTTM,
                    pbMRQ,
                    psTTM,
                    pcfNcfTTM,
                    isST
                FROM incoming_stock_daily
                """,
                [security_id],
            )
            connection.unregister("incoming_stock_daily")
        _replace_stock_coverage(
            connection,
            security_id,
            start_date,
            end_date,
            STOCK_DAILY_FIELD_SET_ID,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


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
    connection.begin()
    registered = False
    try:
        index_id = ensure_market_indices(connection, [code])[code]
        if not incoming.empty:
            connection.register("incoming_index_daily", incoming)
            registered = True
            if field_set_id == BAOSTOCK_INDEX_DAILY_FIELD_SET_ID:
                connection.execute(
                    """
                    DELETE FROM core.index_daily
                    WHERE index_id = ?
                      AND trade_date IN (
                          SELECT CAST(date AS DATE) FROM incoming_index_daily
                      )
                    """,
                    [index_id],
                )
                connection.execute(
                    """
                    INSERT INTO core.index_daily
                    SELECT
                        ?,
                        CAST(date AS DATE),
                        open,
                        high,
                        low,
                        close,
                        preclose,
                        volume,
                        amount,
                        turn,
                        pe_ttm,
                        pb_mrq,
                        ps_ttm,
                        pcf_ncf_ttm,
                        is_st
                    FROM incoming_index_daily
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY CAST(date AS DATE)
                        ORDER BY CAST(date AS DATE)
                    ) = 1
                    """,
                    [index_id],
                )
            else:
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
        _replace_index_coverage(
            connection,
            index_id,
            start_date,
            end_date,
            field_set_id,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if registered:
            connection.unregister("incoming_index_daily")


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
    connection.begin()
    registered = False
    try:
        index_id = ensure_market_indices(connection, [code])[code]
        security_ids = ensure_securities(connection, incoming["symbol"])
        incoming["security_id"] = incoming["symbol"].map(security_ids)
        connection.execute(
            "DELETE FROM core.index_constituent WHERE index_id = ?",
            [index_id],
        )
        if not incoming.empty:
            connection.register("incoming_index_constituent", incoming)
            registered = True
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
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if registered:
            connection.unregister("incoming_index_constituent")


def write_dividend_request(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    year: int,
    data: pd.DataFrame,
) -> None:
    """Persist one before-tax dividend response and its coverage atomically."""
    symbol = normalize_security_symbol(code)
    security_id = ensure_securities(connection, [symbol])[symbol]
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
    connection.begin()
    try:
        if not incoming.empty:
            connection.register("incoming_dividend", incoming)
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
            connection.unregister("incoming_dividend")
        connection.execute(
            """
            INSERT OR IGNORE INTO meta.dividend_coverage
            VALUES (?, ?, ?)
            """,
            [security_id, year, DIVIDEND_BEFORE_TAX_FIELD_SET_ID],
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def write_share_capital_request(
    connection: duckdb.DuckDBPyConnection,
    code: str,
    year: int,
    quarter: int,
    data: pd.DataFrame,
) -> None:
    """Persist one quarterly-share response and its coverage atomically."""
    symbol = normalize_security_symbol(code)
    security_id = ensure_securities(connection, [symbol])[symbol]
    incoming = _prepare_share_capital(data)
    connection.begin()
    try:
        if not incoming.empty:
            connection.register("incoming_share_capital", incoming)
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
            connection.unregister("incoming_share_capital")
        connection.execute(
            """
            INSERT OR IGNORE INTO meta.share_capital_coverage
            VALUES (?, ?, ?)
            """,
            [security_id, year, quarter],
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _prepare_stock_daily(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
        "isST",
    ]
    out = data.copy()
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    return out[columns]


def _prepare_index_daily(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "pb_mrq",
        "ps_ttm",
        "pcf_ncf_ttm",
        "is_st",
    ]
    out = data.rename(
        columns={
            "peTTM": "pe_ttm",
            "pbMRQ": "pb_mrq",
            "psTTM": "ps_ttm",
            "pcfNcfTTM": "pcf_ncf_ttm",
            "isST": "is_st",
        }
    ).copy()
    if out.empty:
        return pd.DataFrame(columns=columns)
    if "date" not in out:
        raise ValueError("Index daily data missing required column: date")
    if "close" not in out:
        raise ValueError("Index daily data missing required column: close")
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if out["date"].isna().any():
        raise ValueError("Index daily dates must not be missing")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if out["close"].isna().any():
        raise ValueError("Index daily close values must not be missing")
    return (
        out[columns]
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


def _replace_stock_coverage(
    connection: duckdb.DuckDBPyConnection,
    security_id: int,
    start_date: str,
    end_date: str,
    field_set_id: int,
) -> None:
    rows = connection.execute(
        """
        SELECT start_date, end_date
        FROM meta.stock_daily_coverage
        WHERE security_id = ? AND field_set_id = ?
        ORDER BY start_date
        """,
        [security_id, field_set_id],
    ).fetchall()
    rows.append((pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()))
    merged = _merge_date_ranges(rows)
    connection.execute(
        """
        DELETE FROM meta.stock_daily_coverage
        WHERE security_id = ? AND field_set_id = ?
        """,
        [security_id, field_set_id],
    )
    connection.executemany(
        "INSERT INTO meta.stock_daily_coverage VALUES (?, ?, ?, ?)",
        [
            (security_id, range_start, range_end, field_set_id)
            for range_start, range_end in merged
        ],
    )


def _replace_index_coverage(
    connection: duckdb.DuckDBPyConnection,
    index_id: int,
    start_date: str,
    end_date: str,
    field_set_id: int,
) -> None:
    rows = connection.execute(
        """
        SELECT start_date, end_date
        FROM meta.index_daily_coverage
        WHERE index_id = ? AND field_set_id = ?
        ORDER BY start_date
        """,
        [index_id, field_set_id],
    ).fetchall()
    rows.append((pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()))
    connection.execute(
        """
        DELETE FROM meta.index_daily_coverage
        WHERE index_id = ? AND field_set_id = ?
        """,
        [index_id, field_set_id],
    )
    connection.executemany(
        "INSERT INTO meta.index_daily_coverage VALUES (?, ?, ?, ?)",
        [
            (index_id, range_start, range_end, field_set_id)
            for range_start, range_end in _merge_date_ranges(rows)
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

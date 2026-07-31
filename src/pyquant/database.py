"""DuckDB storage, migration, and write helpers."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


DEFAULT_DATABASE_PATH = Path("data/pyquant.duckdb")
STOCK_DAILY_FIELD_SET_ID = 1
BAOSTOCK_INDEX_DAILY_FIELD_SET_ID = 1
CSINDEX_DAILY_FIELD_SET_ID = 2
LEGACY_DIVIDEND_FIELD_SET_ID = 1
DIVIDEND_BEFORE_TAX_FIELD_SET_ID = 2

_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS ref;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS api;
CREATE SCHEMA IF NOT EXISTS feature;

CREATE TABLE IF NOT EXISTS ref.security (
    security_id UINTEGER PRIMARY KEY,
    symbol VARCHAR NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS ref.market_index (
    index_id USMALLINT PRIMARY KEY,
    index_code VARCHAR NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS core.stock_daily (
    security_id UINTEGER NOT NULL,
    trade_date DATE NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    preclose FLOAT,
    volume BIGINT,
    amount DOUBLE,
    turn FLOAT,
    pe_ttm FLOAT,
    pb_mrq FLOAT,
    ps_ttm FLOAT,
    pcf_ncf_ttm FLOAT,
    is_st BOOLEAN
);

CREATE TABLE IF NOT EXISTS core.index_daily (
    index_id USMALLINT NOT NULL,
    trade_date DATE NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close DOUBLE,
    preclose FLOAT,
    volume BIGINT,
    amount DOUBLE,
    turn FLOAT,
    pe_ttm FLOAT,
    pb_mrq FLOAT,
    ps_ttm FLOAT,
    pcf_ncf_ttm FLOAT,
    is_st BOOLEAN
);

CREATE TABLE IF NOT EXISTS core.index_constituent (
    index_id USMALLINT NOT NULL,
    effective_date DATE NOT NULL,
    security_id UINTEGER NOT NULL,
    PRIMARY KEY (index_id, effective_date, security_id)
);

CREATE TABLE IF NOT EXISTS core.dividend (
    security_id UINTEGER NOT NULL,
    announce_date DATE,
    record_date DATE,
    ex_date DATE,
    payment_date DATE,
    cash_dividend_before_tax FLOAT
);

CREATE TABLE IF NOT EXISTS core.share_capital_quarterly (
    security_id UINTEGER NOT NULL,
    report_date DATE,
    publish_date DATE,
    total_shares BIGINT
);

CREATE TABLE IF NOT EXISTS meta.stock_daily_coverage (
    security_id UINTEGER NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    field_set_id UTINYINT NOT NULL,
    PRIMARY KEY (security_id, start_date, end_date, field_set_id)
);

CREATE TABLE IF NOT EXISTS meta.index_daily_coverage (
    index_id USMALLINT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    field_set_id UTINYINT NOT NULL,
    PRIMARY KEY (index_id, start_date, end_date, field_set_id)
);

CREATE TABLE IF NOT EXISTS meta.dividend_coverage (
    security_id UINTEGER NOT NULL,
    query_year USMALLINT NOT NULL,
    field_set_id UTINYINT NOT NULL,
    PRIMARY KEY (security_id, query_year, field_set_id)
);

CREATE TABLE IF NOT EXISTS meta.share_capital_coverage (
    security_id UINTEGER NOT NULL,
    report_year USMALLINT NOT NULL,
    report_quarter UTINYINT NOT NULL,
    PRIMARY KEY (security_id, report_year, report_quarter)
);

CREATE TABLE IF NOT EXISTS feature.intraday_volatility_daily (
    security_id UINTEGER NOT NULL,
    trade_date DATE NOT NULL,
    vol_daily FLOAT,
    bar_count USMALLINT,
    return_count USMALLINT,
    is_valid BOOLEAN NOT NULL
);

CREATE OR REPLACE VIEW api.stock_daily AS
SELECT
    s.symbol,
    d.trade_date AS date,
    d.open,
    d.high,
    d.low,
    d.close,
    d.preclose,
    d.volume,
    d.amount,
    d.turn,
    100.0 * (d.close / d.preclose - 1.0) AS pct_chg,
    d.pe_ttm,
    d.pb_mrq,
    d.ps_ttm,
    d.pcf_ncf_ttm,
    d.is_st
FROM core.stock_daily AS d
JOIN ref.security AS s USING (security_id);

CREATE OR REPLACE VIEW api.index_daily AS
SELECT
    d.trade_date AS date,
    i.index_code AS symbol,
    d.open,
    d.high,
    d.low,
    d.close,
    d.preclose,
    d.volume,
    d.amount,
    d.turn,
    d.close / d.preclose - 1.0 AS pct_chg,
    d.pe_ttm,
    d.pb_mrq,
    d.ps_ttm,
    d.pcf_ncf_ttm,
    d.is_st
FROM core.index_daily AS d
JOIN ref.market_index AS i USING (index_id);

CREATE OR REPLACE VIEW api.index_constituents AS
SELECT
    c.effective_date,
    i.index_code,
    s.symbol
FROM core.index_constituent AS c
JOIN ref.market_index AS i USING (index_id)
JOIN ref.security AS s USING (security_id);

CREATE OR REPLACE VIEW api.dividend AS
SELECT
    s.symbol,
    d.announce_date,
    d.record_date,
    d.ex_date AS operate_date,
    d.payment_date,
    d.cash_dividend_before_tax
FROM core.dividend AS d
JOIN ref.security AS s USING (security_id);

CREATE OR REPLACE VIEW api.share_capital_quarterly AS
SELECT
    s.symbol,
    q.publish_date,
    q.report_date,
    q.total_shares
FROM core.share_capital_quarterly AS q
JOIN ref.security AS s USING (security_id);

CREATE OR REPLACE VIEW api.stock_daily_coverage AS
SELECT
    s.symbol,
    c.start_date AS start,
    c.end_date AS end,
    c.field_set_id
FROM meta.stock_daily_coverage AS c
JOIN ref.security AS s USING (security_id);

CREATE OR REPLACE VIEW api.dividend_coverage AS
SELECT
    s.symbol,
    c.query_year AS year
FROM meta.dividend_coverage AS c
JOIN ref.security AS s USING (security_id)
WHERE c.field_set_id = 2;

CREATE OR REPLACE VIEW api.share_capital_coverage AS
SELECT
    s.symbol,
    c.report_year AS year,
    c.report_quarter AS quarter
FROM meta.share_capital_coverage AS c
JOIN ref.security AS s USING (security_id);

CREATE OR REPLACE VIEW api.daily_market_cap AS
SELECT
    s.symbol,
    m.trade_date,
    m.close,
    m.publish_date,
    m.total_shares,
    m.close * m.total_shares AS total_market_cap
FROM (
    SELECT
        p.security_id,
        p.trade_date,
        p.close,
        q.publish_date,
        q.total_shares
    FROM core.stock_daily AS p
    ASOF LEFT JOIN core.share_capital_quarterly AS q
        ON p.security_id = q.security_id
       AND p.trade_date >= q.publish_date
) AS m
JOIN ref.security AS s USING (security_id);
"""


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
        connection.execute(_SCHEMA_SQL)
    return database_path


def normalize_security_symbol(symbol: object) -> str:
    """Convert source or API codes to the project security-symbol format."""
    value = str(symbol).strip()
    if len(value) == 9 and value[2] == ".":
        exchange, code = value.split(".", 1)
        if exchange.lower() in {"sh", "sz", "bj"} and code.isdigit():
            return f"{code}.{exchange.upper()}"
    if len(value) == 9 and value[6] == ".":
        code, exchange = value.split(".", 1)
        if code.isdigit() and exchange.upper() in {"SH", "SZ", "BJ"}:
            return f"{code}.{exchange.upper()}"
    raise ValueError(f"Unsupported security symbol: {symbol!r}")


def normalize_index_code(index_code: object) -> str:
    """Normalize one supported market-index identifier."""
    value = str(index_code).strip()
    if len(value) == 6 and value[0].upper() == "H" and value[1:].isdigit():
        return value.upper()
    if len(value) == 9 and value[2] == ".":
        exchange, code = value.split(".", 1)
        if exchange.lower() in {"sh", "sz", "bj"} and code.isdigit():
            return f"{exchange.lower()}.{code}"
    if len(value) == 9 and value[6] == ".":
        code, exchange = value.split(".", 1)
        if code.isdigit() and exchange.upper() in {"SH", "SZ", "BJ"}:
            return f"{exchange.lower()}.{code}"
    raise ValueError(f"Unsupported market index code: {index_code!r}")


def ensure_securities(
    connection: duckdb.DuckDBPyConnection,
    symbols: Iterable[object],
) -> dict[str, int]:
    """Append unseen symbols and return stable symbol-to-ID mappings."""
    normalized = sorted({normalize_security_symbol(symbol) for symbol in symbols})
    if not normalized:
        return {}
    existing = dict(
        connection.execute(
            "SELECT symbol, security_id FROM ref.security"
        ).fetchall()
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
        parameters.append(start.date())
    if date_column is not None and end is not None:
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
    incoming = incoming.drop_duplicates(
        ["effective_date", "index_code", "symbol"]
    )
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


def migrate_legacy_data(
    data_root: Path = Path("data"),
    *,
    stock_batch_size: int = 200,
) -> dict[str, int]:
    """Migrate supported legacy Parquet data without importing tax-after dividends."""
    database_path = get_database_path(data_root)
    initialize_database(database_path)
    legacy_root = data_root / "legacy_parquet_backup"
    stock_dir = _legacy_dataset_dir(data_root, legacy_root, "stock_daily")
    stock_files = sorted(
        path for path in stock_dir.glob("*.parquet") if path.name != "queries.parquet"
    )
    dividend_dir = _legacy_dataset_dir(data_root, legacy_root, "dividend")
    share_dir = _legacy_dataset_dir(
        data_root, legacy_root, "stock_profit_quarterly"
    )
    index_dir = _legacy_dataset_dir(data_root, legacy_root, "index_daily")
    csindex_dir = _legacy_dataset_dir(data_root, legacy_root, "csindex_daily")
    constituent_dir = _legacy_dataset_dir(
        data_root, legacy_root, "index_constituents"
    )
    symbols = [path.stem for path in stock_files]
    for path in [
        dividend_dir / "data.parquet",
        dividend_dir / "queries.parquet",
        share_dir / "data.parquet",
        share_dir / "queries.parquet",
    ]:
        if path.exists():
            symbols.extend(
                pd.read_parquet(path, columns=["code"])["code"].dropna().astype(str)
            )

    with connect_database(database_path) as connection:
        connection.execute(_SCHEMA_SQL)
        ensure_securities(connection, symbols)
        connection.execute("DELETE FROM core.dividend")
        connection.execute("DELETE FROM meta.dividend_coverage")
        _migrate_stock_daily(connection, stock_files, stock_batch_size)
        _migrate_stock_coverage(connection, stock_dir / "queries.parquet")
        _migrate_legacy_dividend_coverage(
            connection, dividend_dir / "queries.parquet"
        )
        _migrate_share_capital(
            connection,
            share_dir / "data.parquet",
            share_dir / "queries.parquet",
        )
        _migrate_index_data(
            connection,
            index_dir,
            csindex_dir,
            constituent_dir,
        )
        return {
            "securities": connection.execute(
                "SELECT COUNT(*) FROM ref.security"
            ).fetchone()[0],
            "stock_daily": connection.execute(
                "SELECT COUNT(*) FROM core.stock_daily"
            ).fetchone()[0],
            "dividend": connection.execute(
                "SELECT COUNT(*) FROM core.dividend"
            ).fetchone()[0],
            "share_capital_quarterly": connection.execute(
                "SELECT COUNT(*) FROM core.share_capital_quarterly"
            ).fetchone()[0],
            "market_indices": connection.execute(
                "SELECT COUNT(*) FROM ref.market_index"
            ).fetchone()[0],
            "index_daily": connection.execute(
                "SELECT COUNT(*) FROM core.index_daily"
            ).fetchone()[0],
            "index_constituents": connection.execute(
                "SELECT COUNT(*) FROM core.index_constituent"
            ).fetchone()[0],
        }


def migrate_legacy_index_data(
    data_root: Path = Path("data"),
) -> dict[str, int]:
    """Migrate only legacy index prices and constituents into DuckDB."""
    database_path = get_database_path(data_root)
    initialize_database(database_path)
    legacy_root = data_root / "legacy_parquet_backup"
    with connect_database(database_path) as connection:
        _migrate_index_data(
            connection,
            _legacy_dataset_dir(data_root, legacy_root, "index_daily"),
            _legacy_dataset_dir(data_root, legacy_root, "csindex_daily"),
            _legacy_dataset_dir(data_root, legacy_root, "index_constituents"),
        )
        return {
            "market_indices": connection.execute(
                "SELECT COUNT(*) FROM ref.market_index"
            ).fetchone()[0],
            "index_daily": connection.execute(
                "SELECT COUNT(*) FROM core.index_daily"
            ).fetchone()[0],
            "index_constituents": connection.execute(
                "SELECT COUNT(*) FROM core.index_constituent"
            ).fetchone()[0],
        }


def validate_database(
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict[str, int]:
    """Run structural duplicate and dividend-contract checks."""
    with connect_database(database_path, read_only=True) as connection:
        duplicate_stock = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT security_id, trade_date
                FROM core.stock_daily
                GROUP BY security_id, trade_date
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        duplicate_shares = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT security_id, report_date
                FROM core.share_capital_quarterly
                GROUP BY security_id, report_date
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        duplicate_index_daily = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT index_id, trade_date
                FROM core.index_daily
                GROUP BY index_id, trade_date
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        duplicate_constituents = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT index_id, effective_date, security_id
                FROM core.index_constituent
                GROUP BY index_id, effective_date, security_id
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        invalid_shares = connection.execute(
            """
            SELECT COUNT(*)
            FROM core.share_capital_quarterly
            WHERE total_shares < 0
            """
        ).fetchone()[0]
        dividend_rows = connection.execute(
            "SELECT COUNT(*) FROM core.dividend"
        ).fetchone()[0]
        before_tax_coverage = connection.execute(
            """
            SELECT COUNT(*)
            FROM meta.dividend_coverage
            WHERE field_set_id = 2
            """
        ).fetchone()[0]
    return {
        "duplicate_stock_daily_keys": duplicate_stock,
        "duplicate_share_capital_keys": duplicate_shares,
        "duplicate_index_daily_keys": duplicate_index_daily,
        "duplicate_index_constituent_keys": duplicate_constituents,
        "invalid_share_capital_rows": invalid_shares,
        "dividend_rows": dividend_rows,
        "before_tax_dividend_coverage": before_tax_coverage,
    }


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


def _legacy_dataset_dir(
    data_root: Path,
    legacy_root: Path,
    name: str,
) -> Path:
    raw_path = data_root / "raw" / name
    return raw_path if raw_path.exists() else legacy_root / name


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
        (pd.Timestamp(start).date(), pd.Timestamp(end).date())
        for start, end in ranges
    )
    merged = []
    for start, end in ordered:
        if start > end:
            raise ValueError("coverage start_date must not be after end_date")
        if not merged or start > pd.Timestamp(merged[-1][1]).date() + pd.Timedelta(days=1):
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(values) for values in merged]


def _migrate_stock_daily(
    connection: duckdb.DuckDBPyConnection,
    files: list[Path],
    batch_size: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("stock_batch_size must be positive")
    connection.execute("DELETE FROM core.stock_daily")
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE stock_daily_migration (
            raw_symbol VARCHAR,
            trade_date DATE,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            preclose FLOAT,
            volume BIGINT,
            amount DOUBLE,
            turn FLOAT,
            pe_ttm FLOAT,
            pb_mrq FLOAT,
            ps_ttm FLOAT,
            pcf_ncf_ttm FLOAT,
            is_st BOOLEAN
        )
        """
    )
    for offset in range(0, len(files), batch_size):
        paths = [str(path.resolve()) for path in files[offset : offset + batch_size]]
        connection.execute(
            """
            INSERT INTO stock_daily_migration
            SELECT
                regexp_extract(filename, '([^/]+)\\.parquet$', 1),
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
            FROM read_parquet(?, filename = true)
            """,
            [paths],
        )
    connection.execute(
        """
        INSERT INTO core.stock_daily
        SELECT
            s.security_id,
            m.trade_date,
            m.open,
            m.high,
            m.low,
            m.close,
            m.preclose,
            m.volume,
            m.amount,
            m.turn,
            m.pe_ttm,
            m.pb_mrq,
            m.ps_ttm,
            m.pcf_ncf_ttm,
            m.is_st
        FROM stock_daily_migration AS m
        JOIN ref.security AS s
          ON s.symbol =
             upper(substr(m.raw_symbol, 4)) || '.' || upper(substr(m.raw_symbol, 1, 2))
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY s.security_id, m.trade_date
            ORDER BY s.security_id
        ) = 1
        ORDER BY s.security_id, m.trade_date
        """
    )
    connection.execute("DROP TABLE stock_daily_migration")


def _migrate_stock_coverage(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> None:
    connection.execute("DELETE FROM meta.stock_daily_coverage")
    if not path.exists():
        return
    coverage = pd.read_parquet(path)
    coverage["symbol"] = coverage["code"].map(normalize_security_symbol)
    mappings = dict(
        connection.execute("SELECT symbol, security_id FROM ref.security").fetchall()
    )
    rows = []
    for symbol, ranges in coverage.groupby("symbol", sort=False):
        for start, end in _merge_date_ranges(ranges[["start", "end"]].itertuples(
            index=False, name=None
        )):
            rows.append(
                (mappings[symbol], start, end, STOCK_DAILY_FIELD_SET_ID)
            )
    if rows:
        connection.executemany(
            "INSERT INTO meta.stock_daily_coverage VALUES (?, ?, ?, ?)", rows
        )


def _migrate_legacy_dividend_coverage(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> None:
    connection.execute(
        "DELETE FROM meta.dividend_coverage WHERE field_set_id = ?",
        [LEGACY_DIVIDEND_FIELD_SET_ID],
    )
    if not path.exists():
        return
    coverage = pd.read_parquet(path)
    mappings = dict(
        connection.execute("SELECT symbol, security_id FROM ref.security").fetchall()
    )
    rows = set()
    for row in coverage.itertuples(index=False):
        symbol = normalize_security_symbol(row.code)
        for year in range(pd.Timestamp(row.start).year, pd.Timestamp(row.end).year + 1):
            rows.add(
                (mappings[symbol], year, LEGACY_DIVIDEND_FIELD_SET_ID)
            )
    if rows:
        connection.executemany(
            "INSERT INTO meta.dividend_coverage VALUES (?, ?, ?)", sorted(rows)
        )


def _migrate_share_capital(
    connection: duckdb.DuckDBPyConnection,
    data_path: Path,
    coverage_path: Path,
) -> None:
    connection.execute("DELETE FROM core.share_capital_quarterly")
    connection.execute("DELETE FROM meta.share_capital_coverage")
    mappings = dict(
        connection.execute("SELECT symbol, security_id FROM ref.security").fetchall()
    )
    if data_path.exists():
        data = pd.read_parquet(data_path)
        data["security_id"] = data["code"].map(
            lambda code: mappings[normalize_security_symbol(code)]
        )
        prepared = _prepare_share_capital(data)
        prepared["security_id"] = data.loc[prepared.index, "security_id"]
        connection.register("legacy_share_capital", prepared)
        connection.execute(
            """
            INSERT INTO core.share_capital_quarterly
            SELECT
                security_id,
                CAST(report_date AS DATE),
                CAST(publish_date AS DATE),
                CAST(total_shares AS BIGINT)
            FROM legacy_share_capital
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY security_id, CAST(report_date AS DATE)
                ORDER BY publish_date DESC NULLS LAST
            ) = 1
            ORDER BY security_id, report_date
            """
        )
        connection.unregister("legacy_share_capital")
    if coverage_path.exists():
        coverage = pd.read_parquet(coverage_path)
        rows = {
            (
                mappings[normalize_security_symbol(row.code)],
                pd.Timestamp(row.start).year,
                pd.Timestamp(row.start).quarter,
            )
            for row in coverage.itertuples(index=False)
        }
        if rows:
            connection.executemany(
                "INSERT INTO meta.share_capital_coverage VALUES (?, ?, ?)",
                sorted(rows),
            )


def _migrate_index_data(
    connection: duckdb.DuckDBPyConnection,
    index_dir: Path,
    csindex_dir: Path,
    constituent_dir: Path,
) -> None:
    connection.execute("DELETE FROM core.index_daily")
    connection.execute("DELETE FROM core.index_constituent")
    connection.execute("DELETE FROM meta.index_daily_coverage")
    for directory, field_set_id in [
        (index_dir, BAOSTOCK_INDEX_DAILY_FIELD_SET_ID),
        (csindex_dir, CSINDEX_DAILY_FIELD_SET_ID),
    ]:
        for path in sorted(
            file
            for file in directory.glob("*.parquet")
            if file.name != "queries.parquet"
        ):
            data = pd.read_parquet(path)
            if data.empty:
                continue
            write_index_daily_request(
                connection,
                path.stem,
                data,
                pd.Timestamp(data["date"].min()).strftime("%Y-%m-%d"),
                pd.Timestamp(data["date"].max()).strftime("%Y-%m-%d"),
                field_set_id,
            )
    grouped: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(constituent_dir.glob("*.parquet")):
        data = pd.read_parquet(path)
        if data.empty:
            continue
        for code, group in data.groupby("index_code", sort=False):
            grouped.setdefault(normalize_index_code(code), []).append(group)
    for code, frames in grouped.items():
        write_index_constituents(
            connection,
            code,
            pd.concat(frames, ignore_index=True),
        )

"""DuckDB business writes and download coverage."""

from __future__ import annotations

from collections.abc import Iterable

import duckdb
import numpy as np
import pandas as pd

from pyquant.data.identifiers import normalize_index_code, normalize_security_symbol

STOCK_DAILY_FIELD_SET_ID = 1
BAOSTOCK_INDEX_DAILY_FIELD_SET_ID = 1
CSINDEX_DAILY_FIELD_SET_ID = 2
LEGACY_DIVIDEND_FIELD_SET_ID = 1
DIVIDEND_BEFORE_TAX_FIELD_SET_ID = 2


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

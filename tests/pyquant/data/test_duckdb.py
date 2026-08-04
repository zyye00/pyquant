import pandas as pd
import duckdb
import pytest

from pyquant.data.duckdb import connect_database, initialize_database, load_relation
from pyquant.data.store import create_minute_download_task, write_minute_request


def test_initialize_database_exposes_empty_api_relation(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)

    out = load_relation(
        "api.stock_daily",
        ["date", "symbol", "close"],
        database_path=database_path,
        date_column="date",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-31"),
    )

    assert out.empty
    assert out.columns.tolist() == ["date", "symbol", "close"]
    with connect_database(database_path, read_only=True) as connection:
        stock_columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'core' AND table_name = 'stock_daily'
                """
            ).fetchall()
        }
        assert "pb_mrq" not in stock_columns
        assert connection.execute(
            "SELECT * FROM api.stock_pb_daily"
        ).fetchall() == []


def test_initialize_database_preserves_legacy_intraday_table(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("CREATE SCHEMA feature")
        connection.execute(
            """
            CREATE TABLE feature.intraday_volatility_daily (
                security_id UINTEGER,
                trade_date DATE,
                vol_daily FLOAT,
                bar_count USMALLINT,
                return_count USMALLINT,
                is_valid BOOLEAN
            )
            """
        )
        connection.execute(
            """
            INSERT INTO feature.intraday_volatility_daily
            VALUES (1, '2024-01-02', 0.1, 240, 239, TRUE)
            """
        )

    initialize_database(database_path)

    with duckdb.connect(str(database_path)) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info('feature.intraday_volatility_daily')"
            ).fetchall()
        }
        count = connection.execute(
            "SELECT COUNT(*) FROM feature.intraday_volatility_daily"
        ).fetchone()
    assert "volatility" in columns
    assert count == (1,)

    minute = pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 3,
            "datetime": pd.to_datetime(
                [
                    "2024-01-03 09:31",
                    "2024-01-03 09:32",
                    "2024-01-03 09:33",
                ]
            ),
            "open": [10.0, 10.1, 10.2],
            "high": [10.0, 10.1, 10.2],
            "low": [10.0, 10.1, 10.2],
            "close": [10.0, 10.1, 10.2],
            "volume": [100.0] * 3,
            "total_turnover": [1_000.0] * 3,
        }
    )
    daily = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": pd.to_datetime(["2024-01-03"]),
            "volatility": [0.001],
            "bar_count": [3],
            "return_count": [2],
            "status": [1],
        }
    )
    with connect_database(database_path) as connection:
        task_id = create_minute_download_task(
            connection,
            "600000.SH",
            "2024-01-03",
            "2024-01-03",
            True,
        )
        write_minute_request(
            connection,
            task_id,
            "600000.SH",
            minute,
            daily,
            "2024-01-03",
            "2024-01-03",
        )
        written = connection.execute(
            """
            SELECT volatility, vol_daily, bar_count, return_count, is_valid
            FROM feature.intraday_volatility_daily
            WHERE trade_date = '2024-01-03'
            """
        ).fetchone()
    assert written[:2] == pytest.approx((0.001, 0.001))
    assert written[2:] == (3, 2, True)


def test_initialize_database_removes_legacy_baostock_pb_columns(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("CREATE SCHEMA core")
        connection.execute(
            """
            CREATE TABLE core.stock_daily (
                security_id UINTEGER,
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
        connection.execute(
            """
            CREATE TABLE core.index_daily AS
            SELECT
                1::USMALLINT AS index_id,
                DATE '2024-01-02' AS trade_date,
                1::FLOAT AS open,
                1::FLOAT AS high,
                1::FLOAT AS low,
                1::DOUBLE AS close,
                1::FLOAT AS preclose,
                1::BIGINT AS volume,
                1::DOUBLE AS amount,
                1::FLOAT AS turn,
                1::FLOAT AS pe_ttm,
                1::FLOAT AS pb_mrq,
                1::FLOAT AS ps_ttm,
                1::FLOAT AS pcf_ncf_ttm,
                FALSE AS is_st
            """
        )
        connection.execute(
            """
            INSERT INTO core.stock_daily
            VALUES (1, DATE '2024-01-02', 1, 1, 1, 10, 9, 1, 1, 1, 1, 2, 1, 1, FALSE)
            """
        )

    initialize_database(database_path)

    with duckdb.connect(str(database_path), read_only=True) as connection:
        stock_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('core.stock_daily')").fetchall()
        }
        index_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('core.index_daily')").fetchall()
        }
        assert "pb_mrq" not in stock_columns
        assert "pb_mrq" not in index_columns
        assert connection.execute("SELECT COUNT(*) FROM core.stock_daily").fetchone() == (1,)

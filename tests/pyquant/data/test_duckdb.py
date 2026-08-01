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

from pathlib import Path

import pandas as pd
import pytest

from pyquant.database import (
    BAOSTOCK_INDEX_DAILY_FIELD_SET_ID,
    CSINDEX_DAILY_FIELD_SET_ID,
    connect_database,
    ensure_market_indices,
    ensure_securities,
    get_database_path,
    initialize_database,
    migrate_legacy_index_data,
    migrate_legacy_data,
    normalize_index_code,
    normalize_security_symbol,
    validate_database,
    write_dividend_request,
    write_index_constituents,
    write_index_daily_request,
    write_share_capital_request,
)


def make_stock_daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.5],
            "high": [11.0, 11.5],
            "low": [9.0, 9.5],
            "close": [10.5, 11.0],
            "preclose": [10.0, 10.5],
            "volume": [100, 200],
            "amount": [1_000.0, 2_000.0],
            "turn": [1.0, 2.0],
            "pctChg": [5.0, 4.7619],
            "peTTM": [8.0, 9.0],
            "pbMRQ": [1.0, 1.1],
            "psTTM": [2.0, 2.1],
            "pcfNcfTTM": [3.0, 3.1],
            "isST": [False, False],
        }
    )


def write_legacy_sample(data_root: Path) -> None:
    stock_dir = data_root / "raw/stock_daily"
    dividend_dir = data_root / "raw/dividend"
    share_dir = data_root / "raw/stock_profit_quarterly"
    for path in [stock_dir, dividend_dir, share_dir]:
        path.mkdir(parents=True)
    make_stock_daily().to_parquet(stock_dir / "sh.600000.parquet", index=False)
    pd.DataFrame(
        {
            "code": ["sh.600000"],
            "start": pd.to_datetime(["2024-01-02"]),
            "end": pd.to_datetime(["2024-01-03"]),
        }
    ).to_parquet(stock_dir / "queries.parquet", index=False)
    pd.DataFrame(
        {
            "code": ["sh.600000"],
            "year": [2022],
            "announce_date": pd.to_datetime(["2022-05-01"]),
            "record_date": pd.to_datetime(["2022-05-10"]),
            "operate_date": pd.to_datetime(["2022-05-11"]),
            "payment_date": pd.to_datetime(["2022-05-20"]),
            "cash_dividend_after_tax": ["0.1或0.2"],
        }
    ).to_parquet(dividend_dir / "data.parquet", index=False)
    pd.DataFrame(
        {
            "code": ["sh.600000"],
            "start": pd.to_datetime(["2022-01-01"]),
            "end": pd.to_datetime(["2022-12-31"]),
        }
    ).to_parquet(dividend_dir / "queries.parquet", index=False)
    pd.DataFrame(
        {
            "code": ["sh.600000"],
            "year": [2022],
            "quarter": [1],
            "publish_date": pd.to_datetime(["2022-04-30"]),
            "report_date": pd.to_datetime(["2022-03-31"]),
            "total_shares": [123_456_789.0],
        }
    ).to_parquet(share_dir / "data.parquet", index=False)
    pd.DataFrame(
        {
            "code": ["sh.600000"],
            "start": pd.to_datetime(["2022-01-01"]),
            "end": pd.to_datetime(["2022-03-31"]),
        }
    ).to_parquet(share_dir / "queries.parquet", index=False)


def test_security_ids_are_stable_and_symbols_are_normalized(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)

    with connect_database(database_path) as connection:
        first = ensure_securities(connection, ["sh.600000", "sz.000001"])
        second = ensure_securities(connection, ["bj.920001", "sh.600000"])

    assert normalize_security_symbol("sh.600000") == "600000.SH"
    assert second["600000.SH"] == first["600000.SH"]
    assert second["920001.BJ"] > max(first.values())


def test_migration_keeps_legacy_tax_after_dividends_out_of_core(tmp_path):
    data_root = tmp_path / "data"
    write_legacy_sample(data_root)

    migrated = migrate_legacy_data(data_root, stock_batch_size=1)
    database_path = get_database_path(data_root)

    assert migrated == {
        "securities": 1,
        "stock_daily": 2,
        "dividend": 0,
        "share_capital_quarterly": 1,
        "market_indices": 0,
        "index_daily": 0,
        "index_constituents": 0,
    }
    assert validate_database(database_path) == {
        "duplicate_stock_daily_keys": 0,
        "duplicate_share_capital_keys": 0,
        "duplicate_index_daily_keys": 0,
        "duplicate_index_constituent_keys": 0,
        "invalid_share_capital_rows": 0,
        "dividend_rows": 0,
        "before_tax_dividend_coverage": 0,
    }
    with connect_database(database_path, read_only=True) as connection:
        assert connection.execute(
            "SELECT query_year, field_set_id FROM meta.dividend_coverage"
        ).fetchall() == [(2022, 1)]
        stock_row = connection.execute(
            "SELECT symbol, date, pct_chg FROM api.stock_daily ORDER BY date"
        ).fetchall()[0]
        assert stock_row[:2] == ("600000.SH", pd.Timestamp("2024-01-02").date())
        assert stock_row[2] == pytest.approx(5.0)
        assert connection.execute(
            "SELECT symbol, total_shares FROM api.share_capital_quarterly"
        ).fetchall() == [("600000.SH", 123_456_789)]
        core_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info('core.stock_daily')"
            ).fetchall()
        }
        assert "pct_chg" not in core_columns
    assert (data_root / "raw/dividend/data.parquet").exists()
    backup_root = data_root / "legacy_parquet_backup"
    backup_root.mkdir()
    for name in ["stock_daily", "dividend", "stock_profit_quarterly"]:
        (data_root / "raw" / name).rename(backup_root / name)

    assert migrate_legacy_data(data_root, stock_batch_size=1) == migrated


def test_index_ids_are_stable_and_field_set_coverage_isolated(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)
    full = make_stock_daily().iloc[:1]
    close_only = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-03"]), "close": [12.25]}
    )

    with connect_database(database_path) as connection:
        first = ensure_market_indices(connection, ["sh.000300", "H30269"])
        second = ensure_market_indices(connection, ["000300.SH", "H20269"])
        write_index_daily_request(
            connection,
            "sh.000300",
            full,
            "2024-01-02",
            "2024-01-02",
            BAOSTOCK_INDEX_DAILY_FIELD_SET_ID,
        )
        write_index_daily_request(
            connection,
            "H30269",
            close_only,
            "2024-01-03",
            "2024-01-03",
            CSINDEX_DAILY_FIELD_SET_ID,
        )
        coverage = connection.execute(
            """
            SELECT i.index_code, c.field_set_id
            FROM meta.index_daily_coverage AS c
            JOIN ref.market_index AS i USING (index_id)
            ORDER BY i.index_code
            """
        ).fetchall()

    assert normalize_index_code("000300.SH") == "sh.000300"
    assert second["sh.000300"] == first["sh.000300"]
    assert second["H20269"] > max(first.values())
    assert coverage == [("H30269", 2), ("sh.000300", 1)]


def test_index_constituents_are_replaced_and_exposed_as_standard_symbols(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)
    first = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "index_code": ["H30269", "H30269"],
            "symbol": ["sh.600000", "000001.SZ"],
        }
    )
    second = first.iloc[[0]].assign(effective_date=pd.Timestamp("2024-02-01"))

    with connect_database(database_path) as connection:
        write_index_constituents(connection, "H30269", first)
        write_index_constituents(connection, "H30269", second)
        rows = connection.execute(
            "SELECT * FROM api.index_constituents"
        ).fetchall()

    assert rows == [
        (pd.Timestamp("2024-02-01").date(), "H30269", "600000.SH")
    ]


def test_close_only_index_write_preserves_full_fields_and_caches_empty_result(
    tmp_path,
):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)
    full = make_stock_daily().iloc[:1]
    close_only = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02"]), "close": [99.5]}
    )

    with connect_database(database_path) as connection:
        write_index_daily_request(
            connection,
            "H30269",
            full,
            "2024-01-02",
            "2024-01-02",
            BAOSTOCK_INDEX_DAILY_FIELD_SET_ID,
        )
        write_index_daily_request(
            connection,
            "H30269",
            close_only,
            "2024-01-02",
            "2024-01-02",
            CSINDEX_DAILY_FIELD_SET_ID,
        )
        write_index_daily_request(
            connection,
            "H20269",
            pd.DataFrame(),
            "2024-01-01",
            "2024-01-31",
            CSINDEX_DAILY_FIELD_SET_ID,
        )
        row = connection.execute(
            "SELECT open, close FROM api.index_daily WHERE symbol = 'H30269'"
        ).fetchone()
        empty_coverage = connection.execute(
            """
            SELECT COUNT(*)
            FROM meta.index_daily_coverage AS c
            JOIN ref.market_index AS i USING (index_id)
            WHERE i.index_code = 'H20269' AND c.field_set_id = 2
            """
        ).fetchone()[0]

    assert row == pytest.approx((10.0, 99.5))
    assert empty_coverage == 1


def test_migrate_legacy_index_data_reads_prices_and_constituents(tmp_path):
    data_root = tmp_path / "data"
    price_dir = data_root / "raw/csindex_daily"
    constituent_dir = data_root / "raw/index_constituents"
    price_dir.mkdir(parents=True)
    constituent_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["H30269", "H30269"],
            "close": [1_000.0, 1_010.0],
        }
    ).to_parquet(price_dir / "H30269.parquet", index=False)
    pd.DataFrame(
        {
            "effective_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "index_code": ["H30269", "H30269"],
            "symbol": ["sh.600000", "sz.000001"],
        }
    ).to_parquet(constituent_dir / "H30269.parquet", index=False)

    migrated = migrate_legacy_index_data(data_root)

    assert migrated == {
        "market_indices": 1,
        "index_daily": 2,
        "index_constituents": 2,
    }
    with connect_database(get_database_path(data_root)) as connection:
        assert connection.execute(
            "SELECT symbol, close FROM api.index_daily ORDER BY date"
        ).fetchall() == [("H30269", 1_000.0), ("H30269", 1_010.0)]
        assert connection.execute(
            "SELECT symbol FROM api.index_constituents ORDER BY symbol"
        ).fetchall() == [("000001.SZ",), ("600000.SH",)]


def test_before_tax_download_populates_fact_and_new_coverage(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)
    data = pd.DataFrame(
        {
            "announce_date": pd.to_datetime(["2024-05-01"]),
            "record_date": pd.to_datetime(["2024-05-10"]),
            "operate_date": pd.to_datetime(["2024-05-11"]),
            "payment_date": pd.to_datetime(["2024-05-20"]),
            "cash_dividend_before_tax": [0.25],
        }
    )

    with connect_database(database_path) as connection:
        write_dividend_request(connection, "sh.600000", 2024, data)
        assert connection.execute(
            "SELECT cash_dividend_before_tax FROM api.dividend"
        ).fetchall() == [(0.25,)]
        assert connection.execute(
            "SELECT query_year, field_set_id FROM meta.dividend_coverage"
        ).fetchall() == [(2024, 2)]


def test_missing_total_shares_are_preserved_as_null(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)
    data = pd.DataFrame(
        {
            "publish_date": pd.to_datetime(["2024-04-30"]),
            "report_date": pd.to_datetime(["2024-03-31"]),
            "total_shares": [float("nan")],
        }
    )

    with connect_database(database_path) as connection:
        write_share_capital_request(connection, "sh.600000", 2024, 1, data)
        assert connection.execute(
            """
            SELECT report_date, total_shares
            FROM core.share_capital_quarterly
            """
        ).fetchall() == [(pd.Timestamp("2024-03-31").date(), None)]

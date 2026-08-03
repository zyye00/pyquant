import pandas as pd
import pytest

from pyquant.data import store as store_module
from pyquant.data.duckdb import (
    connect_database,
    initialize_database,
)
from pyquant.data.identifiers import normalize_index_code, normalize_security_symbol
from pyquant.data.store import (
    BAOSTOCK_INDEX_DAILY_FIELD_SET_ID,
    CSINDEX_DAILY_FIELD_SET_ID,
    MINUTE_TASK_SUCCESS,
    create_minute_download_task,
    ensure_market_indices,
    ensure_securities,
    index_daily_coverage,
    stock_daily_coverage,
    write_dividend_request,
    write_index_constituents,
    write_index_daily_request,
    write_minute_request,
    write_share_capital_request,
    write_stock_daily_request,
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


def test_security_ids_are_stable_and_symbols_are_normalized(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)

    with connect_database(database_path) as connection:
        first = ensure_securities(connection, ["sh.600000", "sz.000001"])
        second = ensure_securities(connection, ["bj.920001", "sh.600000"])

    assert normalize_security_symbol("sh.600000") == "600000.SH"
    assert second["600000.SH"] == first["600000.SH"]
    assert second["920001.BJ"] > max(first.values())


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


def test_daily_coverage_merges_adjacent_stock_and_index_ranges(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)

    with connect_database(database_path) as connection:
        write_stock_daily_request(
            connection,
            "sh.600000",
            make_stock_daily().iloc[:1],
            "2024-01-02",
            "2024-01-03",
        )
        write_stock_daily_request(
            connection,
            "sh.600000",
            pd.DataFrame(),
            "2024-01-04",
            "2024-01-05",
        )
        write_index_daily_request(
            connection,
            "H30269",
            make_stock_daily().iloc[:1],
            "2024-01-02",
            "2024-01-03",
            BAOSTOCK_INDEX_DAILY_FIELD_SET_ID,
        )
        write_index_daily_request(
            connection,
            "H30269",
            pd.DataFrame(),
            "2024-01-04",
            "2024-01-05",
            BAOSTOCK_INDEX_DAILY_FIELD_SET_ID,
        )

        stock_coverage = stock_daily_coverage(connection, "sh.600000")
        index_coverage = index_daily_coverage(
            connection, "H30269", BAOSTOCK_INDEX_DAILY_FIELD_SET_ID
        )

    assert stock_coverage == [("2024-01-02", "2024-01-05")]
    assert index_coverage == [("2024-01-02", "2024-01-05")]


def test_stock_daily_write_preserves_standardized_fact_fields(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)

    with connect_database(database_path) as connection:
        write_stock_daily_request(
            connection,
            "sh.600000",
            make_stock_daily().iloc[:1],
            "2024-01-02",
            "2024-01-02",
        )
        row = connection.execute(
            """
            SELECT open, close, pe_ttm, pb_mrq, ps_ttm, pcf_ncf_ttm, is_st
            FROM api.stock_daily
            """
        ).fetchone()

    assert row[:6] == pytest.approx((10.0, 10.5, 8.0, 1.0, 2.0, 3.0))
    assert row[6] is False


def test_stock_daily_write_rolls_back_reference_and_fact_on_failure(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)

    def fail_coverage(*args):
        raise RuntimeError("coverage write failed")

    monkeypatch.setattr(store_module, "_replace_daily_coverage", fail_coverage)
    with connect_database(database_path) as connection:
        with pytest.raises(RuntimeError, match="coverage write failed"):
            write_stock_daily_request(
                connection,
                "sh.600000",
                make_stock_daily().iloc[:1],
                "2024-01-02",
                "2024-01-02",
            )
        reference_count = connection.execute(
            "SELECT COUNT(*) FROM ref.security"
        ).fetchone()[0]
        fact_count = connection.execute(
            "SELECT COUNT(*) FROM core.stock_daily"
        ).fetchone()[0]

    assert reference_count == 0
    assert fact_count == 0


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
        rows = connection.execute("SELECT * FROM api.index_constituents").fetchall()

    assert rows == [(pd.Timestamp("2024-02-01").date(), "H30269", "600000.SH")]


def test_close_only_index_write_preserves_full_fields_and_caches_empty_result(
    tmp_path,
):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)
    full = make_stock_daily().iloc[:1]
    close_only = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "close": [99.5]})

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


def test_minute_facts_features_status_and_task_commit_together(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)
    minute = pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 3,
            "datetime": pd.to_datetime(
                [
                    "2024-01-02 09:31",
                    "2024-01-02 09:32",
                    "2024-01-02 09:33",
                ]
            ),
            "open": [10.0, 10.1, 10.2],
            "high": [10.0, 10.1, 10.2],
            "low": [10.0, 10.1, 10.2],
            "close": [10.0, 10.1, 10.2],
            "volume": [100.0, 100.0, 100.0],
            "total_turnover": [1_000.0, 1_000.0, 1_000.0],
        }
    )
    daily = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": pd.to_datetime(["2024-01-02"]),
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
            "2024-01-02",
            "2024-01-02",
            True,
        )
        write_minute_request(
            connection,
            task_id,
            "600000.SH",
            minute,
            daily,
            "2024-01-02",
            "2024-01-02",
        )
        raw = connection.execute("SELECT COUNT(*) FROM api.stock_minute_1m").fetchone()
        feature = connection.execute(
            """
            SELECT volatility, bar_count, return_count, status
            FROM api.intraday_volatility_daily
            """
        ).fetchone()
        task = connection.execute(
            """
            SELECT status, attempts, rows_received, days_received
            FROM meta.minute_download_task
            """
        ).fetchone()

    assert raw == (3,)
    assert feature == pytest.approx((0.001, 3, 2, 1))
    assert task == (MINUTE_TASK_SUCCESS, 0, 3, 1)

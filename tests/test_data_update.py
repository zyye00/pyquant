import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pyquant._data_update import (
    BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY,
    BAOSTOCK_SOCKET_TIMEOUT_SECONDS,
    BaostockClient,
    append_request_log,
    clean_baostock_data,
    clean_baostock_dividends,
    clean_baostock_profit,
    clean_csindex_history,
    create_download_lock,
    init_data_storage,
    minute_5_target_path,
    request_count_today,
    resolve_baostock_codes,
    update_dividends,
    update_dataset,
    update_csindex_daily,
    update_history_dataset,
    update_profit_quarterly,
    validate_request_limit,
)
from pyquant.database import (
    connect_database,
    ensure_securities,
    write_stock_daily_request,
)


class FakeResult:
    def __init__(self, fields, rows):
        self.fields = fields
        self.rows = rows
        self.error_code = "0"
        self.error_msg = ""
        self.index = -1

    def next(self):
        self.index += 1
        return self.index < len(self.rows)

    def get_row_data(self):
        return self.rows[self.index]


def test_baostock_client_sets_socket_timeout(monkeypatch):
    socket = SimpleNamespace(
        timeout=None, settimeout=lambda value: setattr(socket, "timeout", value)
    )
    bs = SimpleNamespace(
        login=lambda: SimpleNamespace(error_code="0", error_msg=""),
        logout=lambda: None,
        common=SimpleNamespace(context=SimpleNamespace(default_socket=socket)),
    )
    monkeypatch.setitem(sys.modules, "baostock", bs)

    with BaostockClient():
        pass

    assert socket.timeout == BAOSTOCK_SOCKET_TIMEOUT_SECONDS


class FakeClient:
    def __init__(self):
        self.bs = self
        self.calls = []

    def query_history_k_data_plus(
        self, code, fields, start_date, end_date, frequency, adjustflag
    ):
        self.calls.append((code, start_date, end_date, frequency, adjustflag))
        names = fields.split(",")
        row = {name: "" for name in names}
        row.update(
            {
                "date": start_date,
                "time": "093500000",
                "code": code,
                "open": "10",
                "high": "11",
                "low": "9",
                "close": "10.5",
                "volume": "100",
                "amount": "1000",
                "adjustflag": adjustflag,
                "tradestatus": "1",
                "isST": "0",
            }
        )
        return FakeResult(names, [[row[name] for name in names]])

    def query_hs300_stocks(self, trade_date):
        return FakeResult(["code"], [["sh.600000"], ["sz.000001"]])

    def query_stock_basic(self):
        return FakeResult(
            ["code", "type"],
            [["sh.600000", "1"], ["sz.000001", "1"], ["sh.510050", "2"]],
        )

    def query_trade_dates(self, start_date, end_date):
        return FakeResult(["calendar_date", "is_trading_day"], [[end_date, "1"]])

    def query_dividend_data(self, code, year, yearType):
        self.calls.append((code, year, yearType))
        rows = (
            []
            if year == "2023"
            else [
                [code, "2022-05-01", "2022-05-10", "2022-05-11", "2022-05-20", "0.25"]
            ]
        )
        return FakeResult(
            [
                "code",
                "dividPlanAnnounceDate",
                "dividRegistDate",
                "dividOperateDate",
                "dividPayDate",
                "dividCashPsBeforeTax",
            ],
            rows,
        )

    def query_profit_data(self, code, year, quarter):
        self.calls.append((code, year, quarter))
        rows = (
            [] if quarter == "2" else [[code, "2022-04-30", "2022-03-31", "123456789"]]
        )
        return FakeResult(["code", "pubDate", "statDate", "totalShare"], rows)


class FailingHistoryClient(FakeClient):
    def query_history_k_data_plus(
        self, code, fields, start_date, end_date, frequency, adjustflag
    ):
        if code != "sz.000002":
            return super().query_history_k_data_plus(
                code, fields, start_date, end_date, frequency, adjustflag
            )
        self.calls.append((code, start_date, end_date, frequency, adjustflag))
        result = FakeResult([], [])
        result.error_code = "100"
        result.error_msg = "simulated source failure"
        return result


class FailingRequestHistoryClient(FakeClient):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def query_history_k_data_plus(
        self, code, fields, start_date, end_date, frequency, adjustflag
    ):
        self.calls.append((code, start_date, end_date, frequency, adjustflag))
        raise self.error


class EmptyHistoryClient(FakeClient):
    def query_history_k_data_plus(
        self, code, fields, start_date, end_date, frequency, adjustflag
    ):
        self.calls.append((code, start_date, end_date, frequency, adjustflag))
        return FakeResult(fields.split(","), [])


class StopAfterFirstRequest:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls == 1


class FakeAkshare:
    def __init__(self):
        self.calls = []

    def stock_zh_index_hist_csindex(self, symbol, start_date, end_date):
        self.calls.append((symbol, start_date, end_date))
        return pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "指数代码": [symbol],
                "收盘": [1_234.5],
                "指数中文全称": ["ignored"],
            }
        )


def seed_stock_coverage(root, code, start, end):
    paths = init_data_storage(root)
    with connect_database(paths.database_path) as connection:
        write_stock_daily_request(connection, code, pd.DataFrame(), start, end)


def read_database(root, query, parameters=None):
    paths = init_data_storage(root)
    with connect_database(paths.database_path) as connection:
        return connection.execute(query, parameters or []).fetchall()


def test_init_data_storage_has_no_task_state(tmp_path):
    paths = init_data_storage(tmp_path / "data")

    assert paths.database_path.exists()
    assert (paths.data_root / "staging/migration").exists()
    assert paths.request_log_path.exists()
    assert not (paths.state_dir / "tasks.parquet").exists()


def test_create_download_lock_replaces_stale_lock(tmp_path, monkeypatch):
    root = tmp_path / "data"
    lock_path = init_data_storage(root).lock_path
    lock_path.write_text("12345", encoding="utf-8")

    def process_does_not_exist(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", process_does_not_exist)

    assert create_download_lock(root).read_text(encoding="utf-8") == str(os.getpid())


def test_create_download_lock_rejects_active_lock(tmp_path, monkeypatch):
    root = tmp_path / "data"
    lock_path = init_data_storage(root).lock_path
    lock_path.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)

    with pytest.raises(RuntimeError, match="download lock exists"):
        create_download_lock(root)


def test_update_checks_one_stock_at_a_time_and_counts_multiple_ranges_once(tmp_path):
    root = tmp_path / "data"
    seed_stock_coverage(root, "sh.600000", "2024-01-03", "2024-01-03")
    progress = []

    result = update_history_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2024-01-02",
        "2024-01-05",
        root,
        10,
        client=FakeClient(),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert result[["code", "start_date", "end_date"]].values.tolist() == [
        ["sh.600000", "2024-01-02", "2024-01-02"],
        ["sh.600000", "2024-01-04", "2024-01-05"],
    ]
    assert Path(result.loc[0, "target_path"]).name == "pyquant.duckdb"
    assert progress == [(0, 1), (1, 1)]


def test_history_query_cache_skips_completed_ranges(tmp_path):
    root = tmp_path / "data"
    seed_stock_coverage(root, "sh.600000", "2024-01-02", "2024-01-03")
    client = FakeClient()

    result = update_history_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2024-01-02",
        "2024-01-05",
        root,
        10,
        client=client,
    )

    assert result[["start_date", "end_date"]].values.tolist() == [
        ["2024-01-04", "2024-01-05"]
    ]
    assert client.calls == [("sh.600000", "2024-01-04", "2024-01-05", "d", "3")]
    assert read_database(
        root,
        """
        SELECT start_date, end_date
        FROM meta.stock_daily_coverage
        """,
    ) == [(date(2024, 1, 2), date(2024, 1, 5))]


def test_local_min_max_does_not_hide_query_cache_gap(tmp_path):
    root = tmp_path / "data"
    paths = init_data_storage(root)
    with connect_database(paths.database_path) as connection:
        security_id = ensure_securities(connection, ["sh.600000"])["600000.SH"]
        connection.executemany(
            """
            INSERT INTO core.stock_daily (security_id, trade_date)
            VALUES (?, ?)
            """,
            [
                (security_id, date(2013, 1, 1)),
                (security_id, date(2014, 1, 3)),
            ],
        )
        write_stock_daily_request(
            connection,
            "sh.600000",
            pd.DataFrame(),
            "2014-01-02",
            "2014-01-03",
        )

    result = update_history_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2013-01-01",
        "2014-01-03",
        root,
        10,
        client=FakeClient(),
    )

    assert result[["start_date", "end_date"]].values.tolist() == [
        ["2013-01-01", "2014-01-01"]
    ]


def test_minute_updates_are_partitioned_by_year(tmp_path):
    result = update_history_dataset(
        "stock",
        "5",
        ["sh.600000"],
        "2023-12-29",
        "2024-01-03",
        tmp_path / "data",
        10,
        client=FakeClient(),
    )

    assert result["target_path"].tolist() == [
        str(minute_5_target_path("sh.600000", 2023, tmp_path / "data")),
        str(minute_5_target_path("sh.600000", 2024, tmp_path / "data")),
    ]


def test_update_merges_data_and_skips_covered_range(tmp_path):
    client = FakeClient()
    first = update_history_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2024-01-02",
        "2024-01-02",
        tmp_path / "data",
        10,
        client=client,
    )
    second = update_history_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2024-01-02",
        "2024-01-02",
        tmp_path / "data",
        10,
        client=client,
    )

    assert first["status"].tolist() == ["success"]
    assert second.empty
    assert len(client.calls) == 1


def test_update_caches_successful_empty_history_query(tmp_path):
    root = tmp_path / "data"
    client = EmptyHistoryClient()

    first = update_history_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2024-01-02",
        "2024-01-02",
        root,
        10,
        client=client,
    )
    second = update_history_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2024-01-02",
        "2024-01-02",
        root,
        10,
        client=client,
    )

    assert first[["status", "row_count"]].values.tolist() == [["success", 0]]
    assert second.empty
    assert len(client.calls) == 1


def test_update_reports_completed_stock_count(tmp_path):
    progress = []
    root = tmp_path / "data"
    seed_stock_coverage(root, "sh.600000", "2024-01-02", "2024-01-02")
    client = FakeClient()

    update_history_dataset(
        "stock",
        "d",
        ["sh.600000", "sz.000001"],
        "2024-01-02",
        "2024-01-02",
        root,
        10,
        client=client,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert [call[0] for call in client.calls] == ["sz.000001"]


def test_update_respects_request_limit_without_completing_next_stock(tmp_path):
    paths = init_data_storage(tmp_path / "data")
    progress = []
    append_request_log(
        paths.request_log_path,
        "endpoint",
        "sh.600519",
        "d",
        "2024-01-02",
        "2024-01-02",
    )

    result = update_history_dataset(
        "stock",
        "d",
        ["sh.600000", "sz.000001"],
        "2024-01-02",
        "2024-01-02",
        paths.data_root,
        2,
        client=FakeClient(),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert result["code"].tolist() == ["sh.600000"]
    assert progress == [(0, 2), (1, 2)]
    assert request_count_today(paths.request_log_path) == 2


def test_update_does_not_complete_partially_updated_stock(tmp_path):
    root = tmp_path / "data"
    seed_stock_coverage(root, "sh.600000", "2024-01-03", "2024-01-03")
    progress = []

    result = update_history_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2024-01-02",
        "2024-01-05",
        root,
        10,
        client=FakeClient(),
        progress=lambda completed, total: progress.append((completed, total)),
        max_tasks=1,
    )

    assert len(result) == 1
    assert progress == [(0, 1)]


def test_runtime_error_stops_history_update_and_releases_lock(tmp_path, monkeypatch):
    root = tmp_path / "data"
    progress = []
    calls = []

    def fail(code, *args):
        calls.append(code)
        raise RuntimeError("source failed")

    monkeypatch.setattr("pyquant._data_update.query_baostock_history", fail)
    with pytest.raises(RuntimeError, match="source failed"):
        update_history_dataset(
            "stock",
            "d",
            ["sh.600000", "sz.000001"],
            "2024-01-02",
            "2024-01-02",
            root,
            10,
            client=FakeClient(),
            progress=lambda completed, total: progress.append((completed, total)),
        )

    assert calls == ["sh.600000"]
    assert progress == [(0, 2)]
    assert not init_data_storage(root).lock_path.exists()


@pytest.mark.parametrize(
    "error",
    [OSError("socket closed"), ValueError("invalid source response")],
)
def test_request_error_stops_history_update_and_releases_lock(tmp_path, error):
    root = tmp_path / "data"
    client = FailingRequestHistoryClient(error)

    with pytest.raises(RuntimeError, match=f"BaoStock request failed: {error}"):
        update_history_dataset(
            "stock",
            "d",
            ["sh.600000", "sz.000001"],
            "2024-01-02",
            "2024-01-02",
            root,
            10,
            client=client,
        )

    assert [call[0] for call in client.calls] == ["sh.600000"]
    assert not init_data_storage(root).lock_path.exists()


def test_database_write_failure_stops_history_update(tmp_path, monkeypatch):
    root = tmp_path / "data"
    client = FakeClient()

    def fail_database_write(*args, **kwargs):
        raise OSError("database write failed")

    monkeypatch.setattr(
        "pyquant._data_update.write_stock_daily_request",
        fail_database_write,
    )
    with pytest.raises(OSError, match="database write failed"):
        update_history_dataset(
            "stock",
            "d",
            ["sh.600000", "sz.000001"],
            "2024-01-02",
            "2024-01-02",
            root,
            10,
            client=client,
        )

    assert [call[0] for call in client.calls] == ["sh.600000"]
    assert not init_data_storage(root).lock_path.exists()


def test_update_stop_leaves_partially_updated_stock_incomplete(tmp_path):
    root = tmp_path / "data"
    seed_stock_coverage(root, "sh.600000", "2024-01-03", "2024-01-03")
    checks = iter([True, True, True, False])
    progress = []

    result = update_history_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2024-01-02",
        "2024-01-05",
        root,
        10,
        client=FakeClient(),
        checkpoint=lambda: next(checks),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert len(result) == 1
    assert progress == [(0, 1)]
    assert not init_data_storage(root).lock_path.exists()


def test_clean_baostock_data_removes_source_fields_and_casts_types():
    out = clean_baostock_data(
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "code": ["sh.600000", "sh.600000"],
                "open": ["10.1", "10.2"],
                "volume": ["100", "200"],
                "amount": ["1000.5", "2000.5"],
                "adjustflag": ["2", "2"],
                "tradestatus": ["1", "0"],
                "isST": ["0", "0"],
            }
        )
    )

    assert out.columns.tolist() == ["date", "open", "volume", "amount", "isST"]
    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert str(out["isST"].dtype) == "boolean"


def test_clean_baostock_dividends_keeps_cash_and_implementation_dates():
    out = clean_baostock_dividends(
        pd.DataFrame(
            {
                "code": ["sh.600000"],
                "dividPlanAnnounceDate": ["2022-05-01"],
                "dividRegistDate": ["2022-05-10"],
                "dividOperateDate": ["2022-05-11"],
                "dividPayDate": ["2022-05-20"],
                "dividCashPsBeforeTax": ["0.25"],
            }
        ),
        "sh.600000",
        2022,
    )

    assert out.columns.tolist() == [
        "code",
        "year",
        "announce_date",
        "record_date",
        "operate_date",
        "payment_date",
        "cash_dividend_before_tax",
    ]
    assert pd.api.types.is_datetime64_any_dtype(out["operate_date"])
    assert out.loc[0, "operate_date"] == pd.Timestamp("2022-05-11")
    assert out.loc[0, "cash_dividend_before_tax"] == pytest.approx(0.25)
    assert out["cash_dividend_before_tax"].dtype == "float32"


def test_clean_baostock_dividends_rejects_invalid_before_tax_cash():
    with pytest.raises(ValueError, match="Invalid dividCashPsBeforeTax"):
        clean_baostock_dividends(
            pd.DataFrame(
                {
                    "code": ["sh.600000", "sh.600001"],
                    "dividPlanAnnounceDate": ["2022-05-01"] * 2,
                    "dividCashPsBeforeTax": ["0.1或0.2", "not-a-number"],
                }
            ),
            "sh.600000",
            2022,
        )


def test_clean_baostock_dividends_keeps_empty_before_tax_cash_as_missing():
    out = clean_baostock_dividends(
        pd.DataFrame(
            {
                "code": ["sh.600000", "sh.600001"],
                "dividPlanAnnounceDate": ["2022-05-01"] * 2,
                "dividCashPsBeforeTax": ["", "0.25"],
            }
        ),
        "sh.600000",
        2022,
    )

    assert out["cash_dividend_before_tax"].tolist() == pytest.approx(
        [float("nan"), 0.25], nan_ok=True
    )


def test_clean_baostock_profit_keeps_total_shares_and_dates():
    out = clean_baostock_profit(
        pd.DataFrame(
            {
                "code": ["sh.600000"],
                "pubDate": ["2022-04-30"],
                "statDate": ["2022-03-31"],
                "totalShare": ["123456789"],
            }
        ),
        "sh.600000",
        2022,
        1,
    )

    assert out.columns.tolist() == [
        "code",
        "year",
        "quarter",
        "publish_date",
        "report_date",
        "total_shares",
    ]
    assert pd.api.types.is_datetime64_any_dtype(out["report_date"])
    assert out.loc[0, "report_date"] == pd.Timestamp("2022-03-31")
    assert out.loc[0, "total_shares"] == pytest.approx(123456789)


def test_update_dividends_skips_saved_and_empty_code_years(tmp_path):
    client = FakeClient()
    first = update_dividends(
        ["sh.600000"], 2022, 2023, tmp_path / "data", 10, client=client
    )
    second = update_dividends(
        ["sh.600000"], 2022, 2023, tmp_path / "data", 10, client=client
    )

    assert first["status"].tolist() == ["success", "success"]
    assert second.empty
    assert client.calls == [
        ("sh.600000", "2022", "operate"),
        ("sh.600000", "2023", "operate"),
    ]
    root = tmp_path / "data"
    assert read_database(root, "SELECT COUNT(*) FROM core.dividend") == [(1,)]
    assert read_database(
        root,
        """
        SELECT query_year, field_set_id
        FROM meta.dividend_coverage
        ORDER BY query_year
        """,
    ) == [(2022, 2), (2023, 2)]


def test_update_profit_skips_saved_and_empty_code_quarters(tmp_path):
    client = FakeClient()
    first = update_profit_quarterly(
        ["sh.600000"], "2022-01-01", "2022-12-31", tmp_path / "data", 10, client=client
    )
    second = update_profit_quarterly(
        ["sh.600000"], "2022-01-01", "2022-12-31", tmp_path / "data", 10, client=client
    )

    assert first["status"].tolist() == ["success"] * 4
    assert second.empty
    assert client.calls == [
        ("sh.600000", "2022", "1"),
        ("sh.600000", "2022", "2"),
        ("sh.600000", "2022", "3"),
        ("sh.600000", "2022", "4"),
    ]
    root = tmp_path / "data"
    assert read_database(
        root, "SELECT COUNT(*) FROM core.share_capital_quarterly"
    ) == [(1,)]
    assert read_database(
        root,
        """
        SELECT report_year, report_quarter
        FROM meta.share_capital_coverage
        ORDER BY report_quarter
        """,
    ) == [(2022, 1), (2022, 2), (2022, 3), (2022, 4)]


def test_update_profit_infers_quarters_from_dates(tmp_path):
    client = FakeClient()

    update_profit_quarterly(
        ["sh.600000"], "2022-02-01", "2022-07-01", tmp_path / "data", 10, client=client
    )

    assert client.calls == [
        ("sh.600000", "2022", "1"),
        ("sh.600000", "2022", "2"),
        ("sh.600000", "2022", "3"),
    ]


def test_update_dividends_reports_completed_stock_count(tmp_path):
    progress = []

    update_dividends(
        ["sh.600000", "sz.000001"],
        2022,
        2022,
        tmp_path / "data",
        10,
        client=FakeClient(),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert progress == [(0, 2), (1, 2), (2, 2)]


def test_update_dividends_saves_when_checkpoint_stops(tmp_path):
    root = tmp_path / "data"
    checkpoint = StopAfterFirstRequest()
    lock_path = init_data_storage(root).lock_path

    result = update_dividends(
        ["sh.600000"],
        2022,
        2023,
        root,
        10,
        client=FakeClient(),
        checkpoint=checkpoint,
    )

    assert result["year"].tolist() == [2022]
    assert read_database(root, "SELECT COUNT(*) FROM core.dividend") == [(1,)]
    assert read_database(
        root,
        "SELECT query_year, field_set_id FROM meta.dividend_coverage",
    ) == [(2022, 2)]
    assert not lock_path.exists()


def test_update_profit_saves_when_checkpoint_stops(tmp_path):
    root = tmp_path / "data"
    checkpoint = StopAfterFirstRequest()
    lock_path = init_data_storage(root).lock_path

    result = update_profit_quarterly(
        ["sh.600000"],
        "2022-01-01",
        "2022-12-31",
        root,
        10,
        client=FakeClient(),
        checkpoint=checkpoint,
    )

    assert result["quarter"].tolist() == [1]
    assert read_database(
        root, "SELECT COUNT(*) FROM core.share_capital_quarterly"
    ) == [(1,)]
    assert read_database(
        root,
        "SELECT report_year, report_quarter FROM meta.share_capital_coverage",
    ) == [(2022, 1)]
    assert not lock_path.exists()


def test_request_log_counts_today(tmp_path):
    log_path = tmp_path / "request_log.csv"
    append_request_log(
        log_path, "endpoint", "sh.600000", "d", "2024-01-02", "2024-01-03"
    )

    assert request_count_today(log_path, date.today()) == 1
    assert "rows" not in pd.read_csv(log_path).columns
    assert request_count_today(log_path, date(1999, 1, 1)) == 0


def test_history_request_log_records_requests_before_runtime_error(tmp_path):
    root = tmp_path / "data"
    paths = init_data_storage(root)

    with pytest.raises(RuntimeError, match="simulated source failure"):
        update_history_dataset(
            "stock",
            "d",
            ["sh.600000", "sz.000001", "sz.000002"],
            "2024-01-02",
            "2024-01-02",
            root,
            10,
            client=FailingHistoryClient(),
        )

    log = pd.read_csv(paths.request_log_path, keep_default_na=False)
    assert log.columns.tolist() == [
        "date",
        "time",
        "endpoint",
        "code",
        "frequency",
        "start_date",
        "end_date",
    ]
    assert request_count_today(paths.request_log_path) == 3
    assert read_database(
        root,
        """
        SELECT s.symbol
        FROM meta.stock_daily_coverage AS c
        JOIN ref.security AS s USING (security_id)
        ORDER BY s.symbol
        """,
    ) == [("000001.SZ",), ("600000.SH",)]

    append_request_log(
        paths.request_log_path,
        "query_history_k_data_plus",
        "sh.600519",
        "d",
        "2024-01-02",
        "2024-01-02",
    )

    assert request_count_today(paths.request_log_path) == 4
    assert pd.read_csv(paths.request_log_path, keep_default_na=False).iloc[-1][
        "endpoint"
    ] == "query_history_k_data_plus"


def test_resolve_pool_requests_are_logged_with_fake_client(tmp_path):
    paths = init_data_storage(tmp_path / "data")

    assert resolve_baostock_codes(
        "hs300", "2024-01-03", FakeClient(), paths.request_log_path
    ) == ["sh.600000", "sz.000001"]

    log = pd.read_csv(paths.request_log_path, keep_default_na=False)
    assert log["endpoint"].tolist() == [
        "query_trade_dates",
        "query_hs300_stocks",
    ]


def test_validate_request_limit_rejects_values_above_hard_limit():
    assert validate_request_limit(BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY) == 49_000
    with pytest.raises(ValueError, match="hard limit"):
        validate_request_limit(BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY + 1)


def test_resolve_hs300_and_all_a_codes():
    client = FakeClient()

    assert resolve_baostock_codes("hs300", "2024-01-03", client) == [
        "sh.600000",
        "sz.000001",
    ]
    assert resolve_baostock_codes("all", "2024-01-03", client) == [
        "sh.600000",
        "sz.000001",
    ]


def test_update_dataset_dispatches_dividend_dates_and_limits_tasks(tmp_path):
    client = FakeClient()

    out = update_dataset(
        "dividend",
        start="2022-02-01",
        end="2023-07-01",
        pool=["600000.SH", "600000.SH"],
        max_tasks=1,
        client=client,
        data_root=tmp_path / "data",
    )

    assert out[["year", "status"]].values.tolist() == [[2022, "success"]]
    assert client.calls == [("sh.600000", "2022", "operate")]


def test_update_dataset_validates_options_before_requests(tmp_path):
    client = FakeClient()

    with pytest.raises(ValueError, match="No security codes were selected"):
        update_dataset(
            "stock_daily",
            start="2024-01-01",
            pool=[],
            client=client,
            data_root=tmp_path / "data",
        )

    assert client.calls == []


def test_update_dataset_preserves_worker_error(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("local storage failed")

    monkeypatch.setattr("pyquant._data_update.update_history_dataset", fail)

    with pytest.raises(OSError, match="local storage failed"):
        update_dataset(
            "stock_daily",
            start="2024-01-01",
            end="2024-01-02",
            pool=["sh.600000"],
            client=FakeClient(),
            data_root=tmp_path / "data",
        )


def test_update_index_accepts_code_pool_but_rejects_named_pool(tmp_path):
    client = FakeClient()

    out = update_dataset(
        "index_daily",
        start="2024-01-02",
        end="2024-01-02",
        pool=["sh.000300"],
        client=client,
        data_root=tmp_path / "data",
    )

    assert out["status"].tolist() == ["success"]
    assert read_database(
        tmp_path / "data",
        "SELECT symbol, close FROM api.index_daily",
    ) == [("sh.000300", 10.5)]
    assert not (tmp_path / "data" / "raw" / "index_daily").exists()
    cached = update_dataset(
        "index_daily",
        start="2024-01-02",
        end="2024-01-02",
        pool=["sh.000300"],
        client=client,
        data_root=tmp_path / "data",
    )
    assert cached.empty
    assert len(client.calls) == 1
    with pytest.raises(ValueError, match="does not support named pools"):
        update_dataset(
            "index_daily",
            start="2024-01-02",
            pool="all",
            client=client,
            data_root=tmp_path / "other-data",
        )


def test_clean_csindex_history_selects_documented_fields():
    data = clean_csindex_history(
        pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "指数代码": ["H30269"],
                "收盘": ["1234.5"],
                "指数中文全称": ["ignored"],
            }
        ),
        "H30269",
    )

    assert data.columns.tolist() == ["date", "symbol", "close"]
    assert pd.api.types.is_datetime64_any_dtype(data["date"])
    assert data["symbol"].tolist() == ["H30269"]
    assert data["close"].tolist() == [1234.5]


def test_update_csindex_daily_formats_dates_and_writes_duckdb(tmp_path):
    client = FakeAkshare()

    out = update_csindex_daily(
        ["H30269", "H20269"],
        "2024-01-02",
        "2024-01-03",
        data_root=tmp_path / "data",
        client=client,
    )

    assert client.calls == [
        ("H30269", "20240102", "20240103"),
        ("H20269", "20240102", "20240103"),
    ]
    assert out["status"].tolist() == ["success", "success"]
    rows = read_database(
        tmp_path / "data",
        "SELECT symbol, close FROM api.index_daily ORDER BY symbol",
    )
    assert rows == [("H20269", 1_234.5), ("H30269", 1_234.5)]
    assert not (tmp_path / "data" / "raw" / "csindex_daily").exists()
    cached = update_csindex_daily(
        ["H30269", "H20269"],
        "2024-01-02",
        "2024-01-03",
        data_root=tmp_path / "data",
        client=client,
    )
    assert cached.empty
    assert len(client.calls) == 2


def test_update_csindex_daily_rejects_unknown_code_before_request(tmp_path):
    client = FakeAkshare()

    with pytest.raises(ValueError, match="Unsupported CSI index codes"):
        update_csindex_daily(
            ["H00000"],
            "2024-01-02",
            "2024-01-03",
            data_root=tmp_path / "data",
            client=client,
        )

    assert client.calls == []


def test_update_csindex_daily_rejects_malformed_source_response(tmp_path):
    class MissingCloseAkshare:
        def stock_zh_index_hist_csindex(self, symbol, start_date, end_date):
            return pd.DataFrame({"日期": ["2024-01-02"], "指数代码": [symbol]})

    out = update_csindex_daily(
        ["H30269"],
        "2024-01-02",
        "2024-01-03",
        data_root=tmp_path / "data",
        client=MissingCloseAkshare(),
    )

    assert out["status"].tolist() == ["failed"]
    assert "missing required columns" in out.loc[0, "error"]

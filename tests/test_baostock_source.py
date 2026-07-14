from datetime import date
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

from pyquant.baostock_source import (
    BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY,
    BAOSTOCK_SOCKET_TIMEOUT_SECONDS,
    BaostockClient,
    append_request_log,
    atomic_write_parquet,
    build_baostock_slices,
    clean_baostock_data,
    clean_baostock_dividends,
    clean_baostock_profit,
    daily_target_path,
    init_baostock_storage,
    minute_5_target_path,
    request_count_today,
    resolve_baostock_codes,
    run_baostock_slices,
    update_baostock_dataset,
    update_baostock_dividends,
    update_baostock_profit_quarterly,
    validate_request_limit,
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
    socket = SimpleNamespace(timeout=None, settimeout=lambda value: setattr(socket, "timeout", value))
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

    def query_history_k_data_plus(self, code, fields, start_date, end_date, frequency, adjustflag):
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
        rows = [] if year == "2023" else [[code, "2022-05-01", "2022-05-10", "2022-05-11", "2022-05-20", "0.25"]]
        return FakeResult(
            [
                "code",
                "dividPlanAnnounceDate",
                "dividRegistDate",
                "dividOperateDate",
                "dividPayDate",
                "dividCashPsAfterTax",
            ],
            rows,
        )

    def query_profit_data(self, code, year, quarter):
        self.calls.append((code, year, quarter))
        rows = [] if quarter == "2" else [[code, "2022-04-30", "2022-03-31", "123456789"]]
        return FakeResult(["code", "pubDate", "statDate", "totalShare"], rows)


class StopAfterFirstRequest:
    quit_requested = True

    def __init__(self):
        self.messages = []

    def before_request(self):
        return True

    def after_request(self):
        return False

    def output(self, message):
        self.messages.append(message)


def test_init_baostock_storage_has_no_task_state(tmp_path):
    paths = init_baostock_storage(tmp_path / "baostock")

    assert paths.daily_stock_dir.exists()
    assert paths.request_log_path.exists()
    assert not (paths.state_dir / "tasks.parquet").exists()


def test_build_slices_uses_existing_daily_data(tmp_path):
    target = daily_target_path("sh.600000", "stock", tmp_path / "baostock")
    atomic_write_parquet(pd.DataFrame({"date": [date(2024, 1, 3)]}), target)

    slices = build_baostock_slices(
        "stock",
        "d",
        ["sh.600000", "sz.000001"],
        "2024-01-02",
        "2024-01-05",
        tmp_path / "baostock",
    )

    assert slices[["code", "start_date"]].values.tolist() == [
        ["sh.600000", "2024-01-02"],
        ["sh.600000", "2024-01-04"],
        ["sz.000001", "2024-01-02"],
    ]
    assert Path(slices.loc[0, "target_path"]).parent.name == "none"


def test_build_slices_adds_only_the_earlier_missing_range(tmp_path):
    target = daily_target_path("sh.600000", "stock", tmp_path / "baostock")
    atomic_write_parquet(
        pd.DataFrame({"date": [date(2014, 1, 2), date(2014, 1, 3)]}), target
    )

    slices = build_baostock_slices(
        "stock",
        "d",
        ["sh.600000"],
        "2013-01-01",
        "2014-01-03",
        tmp_path / "baostock",
    )

    assert slices[["start_date", "end_date"]].values.tolist() == [["2013-01-01", "2014-01-02"]]


def test_minute_slices_are_partitioned_by_year(tmp_path):
    slices = build_baostock_slices(
        "stock",
        "5",
        ["sh.600000"],
        "2023-12-29",
        "2024-01-03",
        tmp_path / "baostock",
        "forward",
    )

    assert slices["target_path"].tolist() == [
        str(minute_5_target_path("sh.600000", 2023, tmp_path / "baostock", "forward")),
        str(minute_5_target_path("sh.600000", 2024, tmp_path / "baostock", "forward")),
    ]


def test_update_merges_data_and_skips_covered_range(tmp_path):
    client = FakeClient()
    first = update_baostock_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2024-01-02",
        "2024-01-02",
        tmp_path / "baostock",
        10,
        client=client,
    )
    second = update_baostock_dataset(
        "stock",
        "d",
        ["sh.600000"],
        "2024-01-02",
        "2024-01-02",
        tmp_path / "baostock",
        10,
        client=client,
    )

    assert first["status"].tolist() == ["success"]
    assert second.empty
    assert len(client.calls) == 1


def test_update_reports_completed_stock_count(tmp_path):
    progress = []

    update_baostock_dataset(
        "stock",
        "d",
        ["sh.600000", "sz.000001"],
        "2024-01-02",
        "2024-01-02",
        tmp_path / "baostock",
        10,
        client=FakeClient(),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert progress == [(0, 2), (1, 2), (2, 2)]


def test_run_slices_respects_request_limit_and_pause(tmp_path):
    paths = init_baostock_storage(tmp_path / "baostock")
    slices = build_baostock_slices(
        "stock", "d", ["sh.600000", "sz.000001"], "2024-01-02", "2024-01-02", paths.raw_root
    )
    append_request_log(paths.request_log_path, "endpoint", "sh.600519", "d", "2024-01-02", "2024-01-02", "success", 1)

    limited = run_baostock_slices(slices, paths, "d", "3", 2, FakeClient())
    paused = run_baostock_slices(slices, paths, "d", "3", 10, FakeClient(), StopAfterFirstRequest())

    assert limited["status"].tolist() == ["success"]
    assert paused["status"].tolist() == ["success"]
    assert request_count_today(paths.request_log_path) == 3


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
    assert isinstance(out.loc[0, "date"], date)
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
                "dividCashPsAfterTax": ["0.25"],
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
        "cash_dividend_after_tax",
    ]
    assert out.loc[0, "operate_date"] == date(2022, 5, 11)
    assert out.loc[0, "cash_dividend_after_tax"] == pytest.approx(0.25)


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
    assert out.loc[0, "report_date"] == date(2022, 3, 31)
    assert out.loc[0, "total_shares"] == pytest.approx(123456789)


def test_update_baostock_dividends_skips_saved_and_empty_code_years(tmp_path):
    client = FakeClient()
    first = update_baostock_dividends(
        ["sh.600000"], 2022, 2023, tmp_path / "baostock", 10, client=client
    )
    second = update_baostock_dividends(
        ["sh.600000"], 2022, 2023, tmp_path / "baostock", 10, client=client
    )

    assert first["status"].tolist() == ["success", "success"]
    assert second.empty
    assert client.calls == [
        ("sh.600000", "2022", "operate"),
        ("sh.600000", "2023", "operate"),
    ]
    dividend_path = tmp_path / "baostock" / "dividend.parquet"
    query_cache_path = tmp_path / "baostock" / "state" / "dividend_queries.parquet"
    assert len(pd.read_parquet(dividend_path)) == 1
    assert pd.read_parquet(query_cache_path).values.tolist() == [
        ["sh.600000", 2022],
        ["sh.600000", 2023],
    ]


def test_update_baostock_profit_skips_saved_and_empty_code_quarters(tmp_path):
    client = FakeClient()
    first = update_baostock_profit_quarterly(
        ["sh.600000"], "2022-01-01", "2022-12-31", tmp_path / "baostock", 10, client=client
    )
    second = update_baostock_profit_quarterly(
        ["sh.600000"], "2022-01-01", "2022-12-31", tmp_path / "baostock", 10, client=client
    )

    assert first["status"].tolist() == ["success"] * 4
    assert second.empty
    assert client.calls == [
        ("sh.600000", "2022", "1"),
        ("sh.600000", "2022", "2"),
        ("sh.600000", "2022", "3"),
        ("sh.600000", "2022", "4"),
    ]
    profit_path = tmp_path / "baostock" / "stock_profit_quarterly.parquet"
    query_cache_path = tmp_path / "baostock" / "state" / "stock_profit_quarterly_queries.parquet"
    assert len(pd.read_parquet(profit_path)) == 3
    assert pd.read_parquet(query_cache_path).values.tolist() == [
        ["sh.600000", 2022, 1],
        ["sh.600000", 2022, 2],
        ["sh.600000", 2022, 3],
        ["sh.600000", 2022, 4],
    ]


def test_update_baostock_profit_infers_quarters_from_dates(tmp_path):
    client = FakeClient()

    update_baostock_profit_quarterly(
        ["sh.600000"], "2022-02-01", "2022-07-01", tmp_path / "baostock", 10, client=client
    )

    assert client.calls == [
        ("sh.600000", "2022", "1"),
        ("sh.600000", "2022", "2"),
        ("sh.600000", "2022", "3"),
    ]


def test_update_baostock_dividends_reports_completed_stock_count(tmp_path):
    progress = []

    update_baostock_dividends(
        ["sh.600000", "sz.000001"],
        2022,
        2022,
        tmp_path / "baostock",
        10,
        client=FakeClient(),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert progress == [(0, 2), (1, 2), (2, 2)]


def test_update_baostock_dividends_saves_when_control_stops(tmp_path):
    root = tmp_path / "baostock"
    control = StopAfterFirstRequest()

    result = update_baostock_dividends(
        ["sh.600000"],
        2022,
        2023,
        root,
        10,
        client=FakeClient(),
        control=control,
    )

    assert result["year"].tolist() == [2022]
    assert len(pd.read_parquet(root / "dividend.parquet")) == 1
    assert pd.read_parquet(root / "state" / "dividend_queries.parquet").values.tolist() == [
        ["sh.600000", 2022]
    ]
    assert control.messages == ["Downloaded data has been saved."]


def test_request_log_counts_today(tmp_path):
    log_path = tmp_path / "request_log.csv"
    append_request_log(log_path, "endpoint", "sh.600000", "d", "2024-01-02", "2024-01-03", "success", 1)

    assert request_count_today(log_path, date.today()) == 1
    assert request_count_today(log_path, "1999-01-01") == 0


def test_validate_request_limit_rejects_values_above_hard_limit():
    assert validate_request_limit(BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY) == 49_000
    with pytest.raises(ValueError, match="hard limit"):
        validate_request_limit(BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY + 1)


def test_resolve_hs300_and_all_a_codes():
    client = FakeClient()

    assert resolve_baostock_codes("hs300", "2024-01-03", client) == ["sh.600000", "sz.000001"]
    assert resolve_baostock_codes("all", "2024-01-03", client) == ["sh.600000", "sz.000001"]

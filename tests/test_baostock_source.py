from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pyquant.baostock_source import (
    BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY,
    append_request_log,
    atomic_write_parquet,
    build_baostock_slices,
    clean_baostock_data,
    daily_target_path,
    init_baostock_storage,
    minute_5_target_path,
    request_count_today,
    resolve_baostock_codes,
    run_baostock_slices,
    update_baostock_dataset,
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


class FakeClient:
    def __init__(self):
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

    def query_all_stock(self, trade_date):
        return FakeResult(["code"], [["sh.600000"], ["sh.600000"], ["sz.000001"]])

    def query_trade_dates(self, start_date, end_date):
        return FakeResult(["calendar_date", "is_trading_day"], [[end_date, "1"]])


class StopAfterFirstRequest:
    def before_request(self):
        return True

    def after_request(self):
        return False


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
        ["sh.600000", "2024-01-04"],
        ["sz.000001", "2024-01-02"],
    ]
    assert Path(slices.loc[0, "target_path"]).parent.name == "none"


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

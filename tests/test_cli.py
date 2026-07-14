import argparse
from datetime import date

import pytest

from pyquant.cli import (
    load_baostock_download_config,
    run_baostock_dividend_download,
    run_baostock_download,
    run_baostock_profit_download,
)


class FakeClientContext:
    entered = 0

    def __init__(self):
        self.bs = self

    def __enter__(self):
        type(self).entered += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def query_stock_basic(self):
        class Result:
            fields = ["code", "type"]
            error_code = "0"
            error_msg = ""

            def __init__(self):
                self.rows = [["sh.600000", "1"], ["sh.510050", "2"]]
                self.idx = -1

            def next(self):
                self.idx += 1
                return self.idx < len(self.rows)

            def get_row_data(self):
                return self.rows[self.idx]

        return Result()

    def query_trade_dates(self, start_date, end_date):
        class Result:
            fields = ["calendar_date", "is_trading_day"]
            error_code = "0"
            error_msg = ""

            def __init__(self):
                self.rows = [[end_date, "1"]]
                self.idx = -1

            def next(self):
                self.idx += 1
                return self.idx < len(self.rows)

            def get_row_data(self):
                return self.rows[self.idx]

        return Result()


def _fake_result():
    class Result:
        def __getitem__(self, key):
            if key == "status":
                return self
            raise KeyError(key)

        def value_counts(self):
            return self

        def to_dict(self):
            return {"pending": 1}

    return Result()


def test_load_baostock_download_config_reads_limits_and_path():
    assert load_baostock_download_config("configs/baostock_download.yaml") == {
        "raw_root": "data/raw/baostock",
        "hard_max_requests_per_day": 50_000,
        "safe_max_requests_per_day": 49_000,
    }


def test_cli_index_download_uses_frequency_and_config(monkeypatch):
    captured = {}

    def fake_update(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    FakeClientContext.entered = 0
    monkeypatch.setattr("pyquant.cli.BaostockClient", FakeClientContext)
    monkeypatch.setattr("pyquant.cli.update_baostock_dataset", fake_update)
    args = argparse.Namespace(
        frequency="d",
        adjustflag=None,
        start_date="2024-01-02",
        end_date="2024-01-03",
        index="sh.000300",
        pool=None,
        pool_date=None,
    )

    assert run_baostock_download(args) is None
    assert captured["dataset"] == "index"
    assert captured["frequency"] == "d"
    assert captured["adjustflag"] is None
    assert captured["codes"] == ["sh.000300"]
    assert captured["raw_root"] == "data/raw/baostock"
    assert captured["max_requests_per_day"] == 49_000
    assert FakeClientContext.entered == 1


def test_cli_uses_today_when_end_date_is_missing(monkeypatch):
    captured = {}

    def fake_update(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr("pyquant.cli.BaostockClient", FakeClientContext)
    monkeypatch.setattr("pyquant.cli.update_baostock_dataset", fake_update)
    args = argparse.Namespace(
        frequency="d",
        adjustflag=None,
        start_date="2024-01-02",
        end_date=None,
        index="sh.000300",
        pool=None,
        pool_date=None,
    )

    assert run_baostock_download(args) is None
    assert captured["end_date"] == date.today().isoformat()


def test_cli_rejects_index_minute_frequency():
    args = argparse.Namespace(
        frequency="5",
        adjustflag=None,
        start_date="2024-01-02",
        end_date="2024-01-03",
        index="sh.000300",
        pool=None,
        pool_date=None,
    )

    with pytest.raises(ValueError, match="index minute"):
        run_baostock_download(args)


def test_cli_all_pool_means_all_a(monkeypatch):
    captured = {}

    def fake_update(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    FakeClientContext.entered = 0
    monkeypatch.setattr("pyquant.cli.BaostockClient", FakeClientContext)
    monkeypatch.setattr("pyquant.cli.update_baostock_dataset", fake_update)
    args = argparse.Namespace(
        frequency="d",
        adjustflag=None,
        start_date="2024-01-02",
        end_date="2024-01-03",
        index=None,
        pool="all",
        pool_date=None,
    )

    assert run_baostock_download(args) is None
    assert captured["dataset"] == "stock"
    assert captured["frequency"] == "d"
    assert captured["codes"] == ["sh.600000"]
    assert FakeClientContext.entered == 1


def test_cli_dividend_download_uses_pool_and_config(monkeypatch):
    captured = {}

    def fake_update(*args, **kwargs):
        captured["codes"] = args[0]
        captured["start_year"] = args[1]
        captured["end_year"] = args[2]
        captured.update(kwargs)
        return _fake_result()

    FakeClientContext.entered = 0
    monkeypatch.setattr("pyquant.cli.BaostockClient", FakeClientContext)
    monkeypatch.setattr("pyquant.cli.update_baostock_dividends", fake_update)
    args = argparse.Namespace(
        start_year=2022,
        end_year=2023,
        pool="all",
        pool_date="2024-01-03",
    )

    assert run_baostock_dividend_download(args) is None
    assert captured["codes"] == ["sh.600000"]
    assert captured["start_year"] == 2022
    assert captured["end_year"] == 2023
    assert captured["raw_root"] == "data/raw/baostock"
    assert captured["max_requests_per_day"] == 49_000
    assert FakeClientContext.entered == 1


def test_cli_profit_download_uses_pool_and_config(monkeypatch):
    captured = {}

    def fake_update(*args, **kwargs):
        captured["codes"] = args[0]
        captured["start_date"] = args[1]
        captured["end_date"] = args[2]
        captured.update(kwargs)
        return _fake_result()

    FakeClientContext.entered = 0
    monkeypatch.setattr("pyquant.cli.BaostockClient", FakeClientContext)
    monkeypatch.setattr("pyquant.cli.update_baostock_profit_quarterly", fake_update)
    args = argparse.Namespace(
        start_date="2022-01-01",
        end_date="2023-12-31",
        pool="all",
        pool_date="2024-01-03",
    )

    assert run_baostock_profit_download(args) is None
    assert captured["codes"] == ["sh.600000"]
    assert captured["start_date"] == "2022-01-01"
    assert captured["end_date"] == "2023-12-31"
    assert captured["raw_root"] == "data/raw/baostock"
    assert captured["max_requests_per_day"] == 49_000
    assert FakeClientContext.entered == 1

from contextvars import ContextVar
from threading import Event

import IPython
import IPython.display
import pandas as pd
import pytest

from pyquant import data as data_module
from pyquant import (
    DatasetUpdate,
    load_dataset,
    load_price,
    standardize_price,
    update_dataset,
)


def test_standardize_price_renames_required_fields():
    data = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"],
            "ticker": [1],
            "close": [10.0],
            "vol": [100],
        }
    )

    out = standardize_price(data)

    assert list(out.columns) == ["date", "symbol", "close", "volume"]
    assert out.loc[0, "symbol"] == "1"
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_load_price_csv(tmp_path):
    path = tmp_path / "price.csv"
    pd.DataFrame(
        {"date": ["2024-01-02"], "symbol": ["000001"], "close": [10.0]}
    ).to_csv(path, index=False)

    out = load_price(path)

    assert out.loc[0, "close"] == 10.0


def test_load_partitioned_dataset_uses_catalog_and_canonical_fields(
    tmp_path, monkeypatch
):
    path = tmp_path / "raw" / "stock_daily" / "sh.600000.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "close": [10.0, 11.0, 12.0],
            "amount": [100.0, 110.0, 120.0],
            "peTTM": [8.0, 9.0, 10.0],
        }
    ).to_parquet(path, index=False)
    catalog = {
        "version": 1,
        "datasets": {
            "stock_daily": {
                "source": "test",
                "storage": {
                    "kind": "symbol_files",
                    "path": str(
                        tmp_path
                        / "raw"
                        / "stock_daily"
                        / "{symbol}.parquet"
                    ),
                    "symbol_from": "stem",
                },
                "columns": ["date", "symbol", "close", "amount", "pe_ttm"],
                "required": ["date", "symbol", "close", "amount", "pe_ttm"],
                "primary_key": ["date", "symbol"],
                "date_column": "date",
                "date_columns": ["date"],
                "numeric_columns": ["close", "amount", "pe_ttm"],
                "field_map": {"peTTM": "pe_ttm"},
            }
        },
    }
    monkeypatch.setattr(data_module, "DATASET_CATALOG", catalog)

    out = load_dataset(
        "stock_daily",
        start="2024-01-03",
        end="2024-01-04",
        symbols=["sh.600000"],
    )

    assert out.columns.tolist() == ["date", "symbol", "close", "amount", "pe_ttm"]
    assert out["date"].tolist() == [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]
    assert out["symbol"].unique().tolist() == ["sh.600000"]
    assert out["pe_ttm"].tolist() == [9.0, 10.0]


def test_load_dataset_requires_explicit_partition_dates(monkeypatch):
    catalog = {
        "version": 1,
        "datasets": {
            "stock_daily": {
                "source": "test",
                "storage": {"kind": "symbol_files", "path": "missing/{symbol}.parquet"},
                "columns": [],
                "required": [],
                "field_map": {},
            }
        },
    }
    monkeypatch.setattr(data_module, "DATASET_CATALOG", catalog)

    with pytest.raises(ValueError, match="requires explicit start and end"):
        load_dataset("stock_daily")


def test_dataset_update_pauses_resumes_and_reports_progress(monkeypatch, capsys):
    first_started = Event()
    release_first = Event()
    second_started = Event()

    def fake_update(name, *, checkpoint, progress, **options):
        assert name == "stock_daily"
        assert options == {
            "start": "2024-01-01",
            "end": None,
            "pool": "all",
            "pool_date": None,
            "max_tasks": None,
        }
        progress(0, 2)
        assert checkpoint()
        first_started.set()
        assert release_first.wait(1)
        progress(1, 2)
        if not checkpoint():
            return pd.DataFrame({"status": ["success"]})
        second_started.set()
        progress(2, 2)
        return pd.DataFrame({"status": ["success", "success"]})

    monkeypatch.setattr("pyquant._data_update.update_dataset", fake_update)

    job = update_dataset("stock_daily", start="2024-01-01", pool="all")

    assert isinstance(job, DatasetUpdate)
    assert first_started.wait(1)
    assert job.state == "running"
    job.pause()
    assert job.state == "paused"
    release_first.set()
    assert not second_started.wait(0.05)
    job.resume()
    assert second_started.wait(1)
    result = job.wait()

    assert result["status"].tolist() == ["success", "success"]
    assert job.state == "completed"
    assert (job.completed, job.total, job.error) == (2, 2, None)
    assert capsys.readouterr().out == (
        "\rUpdated 0/2\rUpdated 1/2\rUpdated 2/2\rUpdated 2/2\n"
    )


def test_dataset_update_uses_ipython_display_for_progress(monkeypatch):
    records = []

    class FakeDisplayHandle:
        def display(self, data, *, raw):
            records.append(("display", data, raw))

        def update(self, data, *, raw):
            records.append(("update", data, raw))

    def fake_update(name, *, checkpoint, progress, **options):
        progress(1, 1)
        return pd.DataFrame()

    monkeypatch.setattr(IPython, "get_ipython", lambda: object())
    monkeypatch.setattr(IPython.display, "DisplayHandle", FakeDisplayHandle)
    monkeypatch.setattr("pyquant._data_update.update_dataset", fake_update)

    job = update_dataset("stock_daily", start="2024-01-01", pool=["sh.600000"])

    assert job.wait().empty
    assert records == [
        ("display", {"text/plain": "Updated 0/0"}, True),
        ("update", {"text/plain": "Updated 1/1"}, True),
        ("update", {"text/plain": "Updated 1/1"}, True),
    ]


def test_dataset_update_inherits_context():
    marker = ContextVar("marker", default="missing")
    marker.set("notebook-cell")
    seen = []

    def worker(checkpoint, progress):
        seen.append(marker.get())
        return pd.DataFrame()

    job = DatasetUpdate(worker)
    marker.set("caller-changed")

    assert job.wait().empty
    assert seen == ["notebook-cell"]


def test_dataset_update_stops_while_paused(monkeypatch):
    first_started = Event()
    release_first = Event()
    second_started = Event()

    def fake_update(name, *, checkpoint, progress, **options):
        progress(0, 2)
        assert checkpoint()
        first_started.set()
        assert release_first.wait(1)
        result = pd.DataFrame({"status": ["success"]})
        progress(1, 2)
        if not checkpoint():
            return result
        second_started.set()
        return pd.concat([result, result], ignore_index=True)

    monkeypatch.setattr("pyquant._data_update.update_dataset", fake_update)
    job = update_dataset("stock_daily", start="2024-01-01", pool="all")

    assert first_started.wait(1)
    job.pause()
    release_first.set()
    job.stop()
    assert job.state == "stopping"
    result = job.wait()

    assert result["status"].tolist() == ["success"]
    assert not second_started.is_set()
    assert job.state == "completed"
    assert (job.completed, job.total) == (1, 2)
    job.pause()
    job.resume()
    job.stop()
    assert job.state == "completed"


def test_dataset_update_reraises_background_error(monkeypatch):
    error = RuntimeError("download failed")

    def fake_update(name, *, checkpoint, progress, **options):
        raise error

    monkeypatch.setattr("pyquant._data_update.update_dataset", fake_update)
    job = update_dataset("stock_daily", start="2024-01-01", pool="all")

    with pytest.raises(RuntimeError, match="download failed"):
        job.wait()

    assert job.state == "failed"
    assert job.error is error

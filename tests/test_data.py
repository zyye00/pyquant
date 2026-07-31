from contextvars import ContextVar
from threading import Event

import IPython
import IPython.display
import pandas as pd
import pytest

from pyquant import data as data_module
from pyquant import (
    DatasetUpdate,
    get_dataset,
    get_period_end_dates,
    load_dataset,
    load_price,
    normalize_query_years,
    standardize_price,
    update_dataset,
)
from pyquant.database import (
    CSINDEX_DAILY_FIELD_SET_ID,
    connect_database,
    initialize_database,
    write_index_daily_request,
    write_stock_daily_request,
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


def test_five_minute_dataset_is_not_available():
    with pytest.raises(ValueError, match="Unknown dataset 'stock_5m'"):
        get_dataset("stock_5m")


def test_get_period_end_dates_returns_last_available_date():
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-28"]
    )

    out = get_period_end_dates(dates)

    assert out.tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-28"),
    ]


def test_normalize_query_years_expands_ranges():
    queries = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "start": pd.to_datetime(["2022-03-01", "2023-01-01"]),
            "end": pd.to_datetime(["2024-02-01", "2023-12-31"]),
        }
    )

    out = normalize_query_years(queries)

    assert out.to_dict("records") == [
        {"symbol": "A", "year": 2022},
        {"symbol": "A", "year": 2023},
        {"symbol": "A", "year": 2024},
        {"symbol": "B", "year": 2023},
    ]


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


def test_load_partitioned_dataset_excludes_query_metadata(tmp_path, monkeypatch):
    data_dir = tmp_path / "raw" / "stock_daily"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "close": [10.0],
            "amount": [100.0],
            "peTTM": [8.0],
        }
    ).to_parquet(data_dir / "sh.600000.parquet", index=False)
    query_path = data_dir / "queries.parquet"
    pd.DataFrame(
        {
            "code": ["sh.600000", "sh.600001"],
            "start": ["2024-01-01", "2024-01-01"],
            "end": ["2024-01-02", "2024-01-02"],
        }
    ).to_parquet(query_path, index=False)
    catalog = {
        "version": 1,
        "datasets": {
            "stock_daily": {
                "source": "test",
                "storage": {
                    "kind": "symbol_files",
                    "path": str(data_dir / "{symbol}.parquet"),
                    "query_path": str(query_path),
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

    out = load_dataset("stock_daily", start="2024-01-01", end="2024-01-02")

    assert out[["date", "symbol"]].to_dict("records") == [
        {"date": pd.Timestamp("2024-01-02"), "symbol": "sh.600000"}
    ]


def test_load_duckdb_dataset_filters_dates_and_normalizes_symbols(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)
    with connect_database(database_path) as connection:
        write_stock_daily_request(
            connection,
            "sh.600000",
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "close": [10.0, 11.0],
                    "amount": [100.0, 110.0],
                    "peTTM": [8.0, 9.0],
                }
            ),
            "2024-01-02",
            "2024-01-03",
        )
    catalog = {
        "version": 1,
        "datasets": {
            "stock_daily": {
                "source": "test",
                "storage": {
                    "kind": "duckdb",
                    "path": str(database_path),
                    "relation": "api.stock_daily",
                    "requires_dates": True,
                },
                "columns": ["date", "symbol", "close", "amount", "pe_ttm"],
                "required": ["date", "symbol", "close", "amount", "pe_ttm"],
                "primary_key": ["date", "symbol"],
                "date_column": "date",
                "date_columns": ["date"],
                "numeric_columns": ["close", "amount", "pe_ttm"],
                "field_map": {},
            }
        },
    }
    monkeypatch.setattr(data_module, "DATASET_CATALOG", catalog)

    out = load_dataset(
        "stock_daily",
        start="2024-01-03",
        end="2024-01-03",
        symbols=["sh.600000"],
    )

    assert out.to_dict("records") == [
        {
            "date": pd.Timestamp("2024-01-03"),
            "symbol": "600000.SH",
            "close": 11.0,
            "amount": 110.0,
            "pe_ttm": 9.0,
        }
    ]


def test_load_duckdb_index_dataset_preserves_index_codes(tmp_path, monkeypatch):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)
    with connect_database(database_path) as connection:
        write_index_daily_request(
            connection,
            "H30269",
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02"]),
                    "close": [1_234.5],
                }
            ),
            "2024-01-02",
            "2024-01-02",
            CSINDEX_DAILY_FIELD_SET_ID,
        )
    catalog = {
        "version": 1,
        "datasets": {
            "csindex_daily": {
                "source": "test",
                "storage": {
                    "kind": "duckdb",
                    "path": str(database_path),
                    "relation": "api.index_daily",
                    "requires_dates": True,
                    "normalize_symbols": False,
                    "allowed_symbols": ["H30269"],
                },
                "columns": ["date", "symbol", "close"],
                "required": ["date", "symbol", "close"],
                "primary_key": ["date", "symbol"],
                "date_column": "date",
                "date_columns": ["date"],
                "numeric_columns": ["close"],
                "field_map": {},
            }
        },
    }
    monkeypatch.setattr(data_module, "DATASET_CATALOG", catalog)

    out = load_dataset(
        "csindex_daily",
        start="2024-01-02",
        end="2024-01-02",
        symbols=["H30269"],
    )

    assert out.to_dict("records") == [
        {
            "date": pd.Timestamp("2024-01-02"),
            "symbol": "H30269",
            "close": 1_234.5,
        }
    ]


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

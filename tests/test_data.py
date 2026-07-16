import pandas as pd
import pytest

from pyquant import data as data_module
from pyquant import load_dataset, load_price, standardize_price
from pyquant._data_migration import migrate_legacy_data_layout


def test_dataset_catalog_defines_complete_canonical_contracts():
    catalog = data_module._load_dataset_catalog()

    assert {
        "stock_daily",
        "index_daily",
        "stock_5m",
        "other_daily",
        "dividend",
        "dividend_queries",
        "stock_profit_quarterly",
        "stock_profit_quarterly_queries",
    } == set(catalog["datasets"])
    for name, dataset in catalog["datasets"].items():
        data_module._get_dataset(catalog, name)
        assert set(dataset["required"]) <= set(dataset["columns"])
        assert set(dataset.get("primary_key", [])) <= set(dataset["columns"])
        if dataset["source"] != "generated":
            assert dataset["source"] in catalog["sources"]


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
    path = tmp_path / "raw" / "stock_daily" / "none" / "sh.600000.parquet"
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
                        / "{adjustment}"
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
    monkeypatch.setattr(data_module, "_load_dataset_catalog", lambda: catalog)

    out = load_dataset(
        "stock_daily",
        start="2024-01-03",
        end="2024-01-04",
        symbols=["sh.600000"],
    )

    assert out.columns.tolist() == ["date", "symbol", "close", "amount", "pe_ttm"]
    assert out["date"].tolist() == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
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
    monkeypatch.setattr(data_module, "_load_dataset_catalog", lambda: catalog)

    with pytest.raises(ValueError, match="requires explicit start and end"):
        load_dataset("stock_daily")


def test_migrate_legacy_layout_dry_run_and_move(tmp_path):
    legacy = tmp_path / "data" / "raw" / "baostock"
    stock = legacy / "daily" / "stock" / "none" / "sh.600000.parquet"
    index = legacy / "daily" / "index" / "sh.000300.parquet"
    dividend = legacy / "dividend.parquet"
    query = legacy / "state" / "dividend_queries.parquet"
    for path in [stock, index, dividend, query]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    planned = migrate_legacy_data_layout(legacy, tmp_path / "data", dry_run=True)

    assert len(planned) == 4
    assert stock.exists()
    moved = migrate_legacy_data_layout(legacy, tmp_path / "data")
    assert moved == planned
    assert not legacy.exists()
    assert (
        tmp_path / "data" / "raw" / "stock_daily" / "none" / "sh.600000.parquet"
    ).read_bytes() == b"sh.600000.parquet"
    assert (
        tmp_path / "data" / "raw" / "index_daily" / "none" / "sh.000300.parquet"
    ).exists()
    assert (tmp_path / "data" / "raw" / "dividend" / "data.parquet").exists()


def test_migrate_legacy_layout_stops_before_destination_collision(tmp_path):
    legacy = tmp_path / "data" / "raw" / "baostock"
    source = legacy / "dividend.parquet"
    destination = tmp_path / "data" / "raw" / "dividend" / "data.parquet"
    source.parent.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    destination.write_bytes(b"destination")

    with pytest.raises(FileExistsError, match="destinations already exist"):
        migrate_legacy_data_layout(legacy, tmp_path / "data")

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"destination"

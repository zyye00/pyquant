from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pyquant.data.catalog import (
    DatasetSpec,
    dataset_spec_from_mapping,
    get_dataset_spec,
)


def make_dataset(**changes):
    values = {
        "source": "generated",
        "storage": {
            "kind": "duckdb",
            "path": "data/pyquant.duckdb",
            "relation": "api.sample",
            "requires_dates": False,
        },
        "columns": ["date", "symbol"],
        "required": ["symbol"],
        "primary_key": ["date", "symbol"],
        "date_column": "date",
        "date_columns": ["date"],
    }
    values.update(changes)
    return values


def test_builtin_catalog_returns_immutable_dataset_spec():
    dataset = get_dataset_spec("stock_daily")

    assert isinstance(dataset, DatasetSpec)
    assert dataset.storage.resolve_path(Path("/tmp/data")) == Path(
        "/tmp/data/pyquant.duckdb"
    )
    with pytest.raises(FrozenInstanceError):
        dataset.name = "changed"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source": "unknown"}, "unknown source"),
        ({"columns": []}, "columns must be unique and non-empty"),
        ({"required": ["missing"]}, "required fields are not columns"),
        ({"primary_key": ["missing"]}, "primary_key fields are not columns"),
        ({"date_column": "missing"}, "date_column is not a column"),
        (
            {"numeric_columns": ["missing"]},
            "numeric_columns are not columns",
        ),
        (
            {"storage": {"kind": "duckdb", "path": "data/db.duckdb"}},
            "requires relation",
        ),
        (
            {"storage": {"kind": "unknown", "path": "data/file"}},
            "unknown storage kind",
        ),
    ],
)
def test_catalog_rejects_invalid_dataset_definitions(changes, message):
    with pytest.raises(ValueError, match=message):
        dataset_spec_from_mapping("sample", make_dataset(**changes))

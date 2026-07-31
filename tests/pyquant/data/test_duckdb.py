import pandas as pd

from pyquant.data.duckdb import initialize_database, load_relation


def test_initialize_database_exposes_empty_api_relation(tmp_path):
    database_path = tmp_path / "pyquant.duckdb"
    initialize_database(database_path)

    out = load_relation(
        "api.stock_daily",
        ["date", "symbol", "close"],
        database_path=database_path,
        date_column="date",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-31"),
    )

    assert out.empty
    assert out.columns.tolist() == ["date", "symbol", "close"]

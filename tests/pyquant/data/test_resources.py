from types import MappingProxyType

import pytest

from pyquant.data.resources import load_schema_sql, load_source_protocols


def test_load_source_protocols_returns_cached_immutable_configuration():
    protocols = load_source_protocols()

    assert protocols is load_source_protocols()
    assert set(protocols) >= {"baostock", "csindex", "rqdata", "price"}
    assert isinstance(protocols["baostock"], MappingProxyType)
    assert isinstance(protocols["baostock"]["history"]["daily"], tuple)
    with pytest.raises(TypeError):
        protocols["price"] = {}
    with pytest.raises(TypeError):
        protocols["baostock"]["history"]["daily"][0] = "changed"


def test_load_schema_sql_returns_packaged_duckdb_schema():
    schema = load_schema_sql()

    assert "CREATE SCHEMA IF NOT EXISTS api" in schema
    assert "CREATE OR REPLACE VIEW api.stock_daily" in schema

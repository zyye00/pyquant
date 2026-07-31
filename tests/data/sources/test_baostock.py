import pytest

from pyquant.data.sources.baostock import (
    normalize_baostock_code,
    validate_request_limit,
)


def test_normalize_baostock_code_uses_source_format():
    assert normalize_baostock_code("600000.SH") == "sh.600000"


def test_request_limit_rejects_non_positive_values():
    with pytest.raises(ValueError, match="must be positive"):
        validate_request_limit(0)

import pytest

from pyquant.data.identifiers import normalize_index_code, normalize_security_symbol


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("sh.600000", "600000.SH"),
        ("600000.SH", "600000.SH"),
        ("sz.000001", "000001.SZ"),
    ],
)
def test_normalize_security_symbol(source, expected):
    assert normalize_security_symbol(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("H30269", "H30269"),
        ("000300.SH", "sh.000300"),
        ("sh.000300", "sh.000300"),
    ],
)
def test_normalize_index_code(source, expected):
    assert normalize_index_code(source) == expected


def test_identifiers_reject_unsupported_values():
    with pytest.raises(ValueError, match="Unsupported security symbol"):
        normalize_security_symbol("600000")
    with pytest.raises(ValueError, match="Unsupported market index code"):
        normalize_index_code("CSI300")

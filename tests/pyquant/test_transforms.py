import pandas as pd

from pyquant import transform_factor


def test_transform_factor_by_date_preserves_index():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B", "C"]],
        names=["date", "symbol"],
    )
    factor = pd.Series([1, 2, 100, 2, 4, 6], index=idx)

    out = transform_factor(factor)

    assert out.index.equals(factor.index)
    assert abs(out.groupby(level=0).mean()).max() < 1e-12

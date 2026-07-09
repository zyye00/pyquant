import pandas as pd

from pyquant import build_universe


def test_build_universe_filters_dates_and_symbols():
    price = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "symbol": ["A", "B", "A"],
            "close": [1, 2, 3],
        }
    )

    out = build_universe(price, symbols=["A"], start="2024-01-03")

    assert out.index.names == ["date", "symbol"]
    assert list(out.index.get_level_values("symbol")) == ["A"]
    assert out["in_universe"].all()

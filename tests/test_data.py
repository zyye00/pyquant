import pandas as pd

from pyquant import load_price, standardize_price


def test_standardize_price_renames_required_fields():
    df = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"],
            "ticker": [1],
            "close": [10.0],
            "vol": [100],
        }
    )

    out = standardize_price(df)

    assert list(out.columns) == ["date", "symbol", "close", "volume"]
    assert out.loc[0, "symbol"] == "1"
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_load_price_csv(tmp_path):
    path = tmp_path / "price.csv"
    pd.DataFrame({"date": ["2024-01-02"], "symbol": ["000001"], "close": [10.0]}).to_csv(
        path, index=False
    )

    out = load_price(path)

    assert out.loc[0, "close"] == 10.0

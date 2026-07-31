import pandas as pd

from pyquant.data.sources.csindex import clean_csindex_history


def test_clean_csindex_history_uses_documented_fields():
    out = clean_csindex_history(
        pd.DataFrame(
            {"日期": ["2024-01-02"], "指数代码": ["H30269"], "收盘": ["123.4"]}
        ),
        "H30269",
    )

    assert out.to_dict("records") == [
        {
            "date": pd.Timestamp("2024-01-02"),
            "symbol": "H30269",
            "close": 123.4,
        }
    ]

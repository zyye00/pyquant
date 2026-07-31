import pandas as pd

from pyquant.data.sources.rqdata import extract_changed_snapshots


def test_changed_snapshots_skip_unchanged_daily_sets():
    out = extract_changed_snapshots(
        {
            pd.Timestamp("2024-01-02"): ["600000.XSHG"],
            pd.Timestamp("2024-01-03"): ["600000.XSHG"],
        },
        "H30269",
    )

    assert out.to_dict("records") == [
        {
            "effective_date": pd.Timestamp("2024-01-02"),
            "index_code": "H30269",
            "symbol": "600000.SH",
        }
    ]

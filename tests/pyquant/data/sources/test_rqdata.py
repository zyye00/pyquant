from types import SimpleNamespace

import pandas as pd

from pyquant.data.sources.rqdata import (
    extract_changed_snapshots,
    project_symbol_to_rqdata,
    query_rqdata_quota_remaining,
    query_rqdata_trading_dates,
    query_stock_minute_1m,
)


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


class FakeRQData:
    def __init__(self):
        self.calls = []
        self.user = SimpleNamespace(
            get_quota=lambda: {"bytes_limit": 1_000, "bytes_used": 250}
        )

    def init(self):
        self.calls.append(("init",))

    def get_trading_dates(self, start_date, end_date, market):
        self.calls.append(("calendar", start_date, end_date, market))
        return [pd.Timestamp(start_date), pd.Timestamp(end_date)]

    def get_price(self, **kwargs):
        self.calls.append(("price", kwargs))
        index = pd.MultiIndex.from_tuples(
            [
                ("600000.XSHG", pd.Timestamp("2024-01-02 09:31")),
                ("600000.XSHG", pd.Timestamp("2024-01-02 09:32")),
            ],
            names=["order_book_id", "datetime"],
        )
        return pd.DataFrame(
            {
                "open": [10.0, 10.1],
                "high": [10.2, 10.2],
                "low": [9.9, 10.0],
                "close": [10.1, 10.2],
                "volume": [100.0, 200.0],
                "total_turnover": [1_000.0, 2_000.0],
            },
            index=index,
        )


def test_minute_query_uses_unadjusted_skip_suspended_contract():
    client = FakeRQData()

    out = query_stock_minute_1m(
        "600000.SH",
        "2024-01-02",
        "2024-01-02",
        client=client,
    )

    _, arguments = client.calls[-1]
    assert project_symbol_to_rqdata("600000.SH") == "600000.XSHG"
    assert arguments["frequency"] == "1m"
    assert arguments["adjust_type"] == "none"
    assert arguments["skip_suspended"] is True
    assert out.columns.tolist() == [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "total_turnover",
    ]


def test_calendar_and_quota_are_normalized():
    client = FakeRQData()

    dates = query_rqdata_trading_dates(
        "2024-01-02",
        "2024-01-03",
        client=client,
    )

    assert dates.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert query_rqdata_quota_remaining(client=client) == 750
    assert client.calls.count(("init",)) == 1


def test_minute_rqdata_client_is_initialized_once_across_requests():
    client = FakeRQData()

    query_rqdata_trading_dates("2024-01-02", "2024-01-03", client=client)
    query_rqdata_quota_remaining(client=client)
    query_stock_minute_1m(
        "600000.SH",
        "2024-01-02",
        "2024-01-02",
        client=client,
    )

    assert client.calls.count(("init",)) == 1

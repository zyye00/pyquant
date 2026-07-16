import argparse

import pandas as pd
import pytest

from pyquant.cli import main, run_data_update


def test_data_update_cli_forwards_dataset_arguments(monkeypatch):
    captured = {}

    def fake_update(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return pd.DataFrame({"status": ["success"]})

    monkeypatch.setattr("pyquant.cli.update_dataset", fake_update)
    args = argparse.Namespace(
        dataset="stock_daily",
        start_date="2024-01-02",
        end_date="2024-01-03",
        pool="all",
        symbols=None,
        pool_date=None,
        adjustment="none",
        max_tasks=2,
    )

    assert run_data_update(args) is None
    assert captured["name"] == "stock_daily"
    assert captured["start"] == "2024-01-02"
    assert captured["end"] == "2024-01-03"
    assert captured["pool"] == "all"
    assert captured["adjustment"] == "none"
    assert captured["max_tasks"] == 2


def test_data_update_cli_accepts_explicit_symbols(monkeypatch):
    captured = {}

    def fake_update(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return pd.DataFrame({"status": []})

    monkeypatch.setattr("pyquant.cli.update_dataset", fake_update)

    main(
        [
            "data-update",
            "index_daily",
            "--start-date",
            "2024-01-02",
            "--symbols",
            "sh.000300",
        ]
    )

    assert captured["name"] == "index_daily"
    assert captured["symbols"] == ["sh.000300"]
    assert captured["end"] is None


@pytest.mark.parametrize(
    "command",
    ["baostock-download", "baostock-dividend-download", "baostock-profit-download"],
)
def test_source_specific_commands_are_removed(command):
    with pytest.raises(SystemExit):
        main([command])

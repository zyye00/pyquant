import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).parents[3] / "notebooks" / "dividend_low_vol"


def load_notebooks(*names: str) -> dict[str, dict]:
    notebooks = {
        name: json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
        for name in names
    }
    for notebook in notebooks.values():
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "notebook_cell", "exec")
    return notebooks


def test_strategy_1_notebooks_split_downloads_from_calculation():
    notebooks = load_notebooks("download.ipynb", "strategy_1_monthly_rebalance.ipynb")

    assert "update_dataset" in str(notebooks["download.ipynb"])
    strategy_notebook = str(notebooks["strategy_1_monthly_rebalance.ipynb"])
    assert "update_dataset" not in strategy_notebook
    assert "csindex_daily" in str(notebooks["download.ipynb"])
    assert "csindex_daily" in strategy_notebook
    assert "calculate_dividend_low_vol_monthly_rebalanced_index" in strategy_notebook
    assert "official_index_job.wait()" in str(notebooks["download.ipynb"])
    assert "update_minute_data" in str(notebooks["download.ipynb"])
    assert "build_intraday_minute_requests" in str(notebooks["download.ipynb"])
    assert "minute_job.wait()" in str(notebooks["download.ipynb"])


def test_strategy_3_notebook_and_download_entry_are_separated():
    notebooks = load_notebooks(
        "download.ipynb", "strategy_3_valuation_spread_timing.ipynb"
    )

    download_notebook = str(notebooks["download.ipynb"])
    strategy_notebook = str(notebooks["strategy_3_valuation_spread_timing.ipynb"])
    assert "index_constituents" in download_notebook
    assert "constituent_job.wait()" in download_notebook
    assert "conda run" not in download_notebook
    assert "subprocess" not in download_notebook
    assert "rqdatac" not in strategy_notebook
    assert "update_dataset" not in strategy_notebook
    assert "calculate_bp_spread" in strategy_notebook
    assert "backtest_valuation_spread_timing" in strategy_notebook
    assert "估值差BP_spread与对应下个月红利低波全收益指数月度收益" in strategy_notebook
    assert "twinx" in strategy_notebook

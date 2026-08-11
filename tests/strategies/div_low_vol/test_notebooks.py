import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).parents[3] / "notebooks" / "div_low_vol"


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
    notebooks = load_notebooks("download.ipynb", "1_rebalance.ipynb")

    assert "update_dataset" in str(notebooks["download.ipynb"])
    strategy_notebook = str(notebooks["1_rebalance.ipynb"])
    assert "update_dataset" not in strategy_notebook
    assert "csindex_daily" in str(notebooks["download.ipynb"])
    assert "csindex_daily" in strategy_notebook
    assert "calculate_div_low_vol_monthly_rebalanced_index" in strategy_notebook
    assert "calculate_traditional_volatility_group_indices" in strategy_notebook
    assert "传统波动率因子五分组及多空对冲净值走势对比（全A）" in strategy_notebook
    assert "传统波动率因子五分组及多空对冲净值走势对比（红利股票池）" in strategy_notebook
    assert "group_labels = [\"分组1\", \"分组2\", \"分组3\", \"分组4\", \"分组5\"]" in strategy_notebook
    assert "traditional-volatility-summary-table" in strategy_notebook
    for metric in ["IC均值", "年化ICIR", "Rank IC均值", "年化Rank ICIR"]:
        assert metric in strategy_notebook
    assert "stock_adjust_factor" in strategy_notebook
    assert "gross_reinvested_index" not in strategy_notebook
    assert "月调组合（无费率、分红立即再投资）" not in strategy_notebook
    assert "official_index_downloads = current_job.wait()" in str(
        notebooks["download.ipynb"]
    )
    assert "update_minute_data" in str(notebooks["download.ipynb"])
    assert "build_intraday_minute_requests" in str(notebooks["download.ipynb"])
    assert "minute_downloads = current_job.wait()" in str(
        notebooks["download.ipynb"]
    )


def test_download_notebook_groups_sources_and_controls_current_job():
    notebook = load_notebooks("download.ipynb")["download.ipynb"]
    cells = notebook["cells"]
    sources = ["".join(cell.get("source", [])) for cell in cells]
    code_sources = [
        source
        for cell, source in zip(cells, sources)
        if cell["cell_type"] == "code"
    ]

    def index_of(fragment: str) -> int:
        return next(index for index, source in enumerate(sources) if fragment in source)

    assert "active_source" not in str(notebook)
    assert "download_minute_data" not in str(notebook)
    assert any('"current_job" not in globals()' in source for source in code_sources)
    assert index_of("## 下载控制") < index_of('current_job = update_dataset(')
    assert all(
        "if active_source" not in source
        for source in code_sources
    )

    baostock_daily = index_of('"stock_daily",')
    baostock_adjust = index_of('"stock_adjust_factor",')
    pool_calculation = index_of("pool = select_div_low_vol_download_symbols")
    baostock_dividend = index_of('"dividend",')
    baostock_shares = index_of('"stock_profit_quarterly",')
    rqdata_constituents = index_of('"index_constituents",')
    rqdata_pb = index_of('"stock_pb_daily",')
    minute_requests = index_of("minute_requests = []")
    rqdata_minute = index_of("update_minute_data(")
    akshare_index = index_of('"csindex_daily",')
    assert (
        baostock_daily
        < baostock_adjust
        < pool_calculation
        < baostock_dividend
        < baostock_shares
        < rqdata_constituents
        < rqdata_pb
        < minute_requests
        < rqdata_minute
        < akshare_index
    )
    assert "stock_adjust_factor" not in sources[pool_calculation]
    assert "update_dataset" not in sources[minute_requests]

    start_indices = [
        index
        for index, source in enumerate(sources)
        if "current_job = update_" in source
    ]
    assert len(start_indices) == 8
    for start_index in start_indices:
        next_code = next(
            "".join(cell.get("source", []))
            for cell in cells[start_index + 1 :]
            if cell["cell_type"] == "code"
        )
        assert ".wait()" in next_code


def test_strategy_3_notebook_and_download_entry_are_separated():
    notebooks = load_notebooks(
        "download.ipynb", "3_timing.ipynb"
    )

    download_notebook = str(notebooks["download.ipynb"])
    strategy_notebook = str(notebooks["3_timing.ipynb"])
    assert "index_constituents" in download_notebook
    assert "constituent_snapshots = current_job.wait()" in download_notebook
    assert "stock_pb_daily" in download_notebook
    assert "pb_downloads = current_job.wait()" in download_notebook
    assert 'pool="all"' in download_notebook
    assert "valuation_symbols" not in download_notebook
    cells = notebooks["download.ipynb"]["cells"]
    pb_index = next(
        index
        for index, cell in enumerate(cells)
        if '"stock_pb_daily",' in "".join(cell.get("source", []))
    )
    minute_request_index = next(
        index
        for index, cell in enumerate(cells)
        if "minute_requests = []" in "".join(cell.get("source", []))
    )
    assert pb_index < minute_request_index
    assert "conda run" not in download_notebook
    assert "subprocess" not in download_notebook
    assert "rqdatac" not in strategy_notebook
    assert "update_dataset" not in strategy_notebook
    assert "calculate_bp_spread" in strategy_notebook
    assert "stock_pb_daily" in strategy_notebook
    assert "backtest_valuation_spread_timing" in strategy_notebook
    assert "vectorbt" in strategy_notebook
    assert "H20269" in strategy_notebook
    assert "估值差BP_spread与对应下个月红利低波全收益指数月度收益" in strategy_notebook
    assert "twinx" in strategy_notebook


def test_strategy_2_notebook_uses_public_interfaces_and_candidate_pool_scope():
    notebooks = load_notebooks("download.ipynb", "2_high_frequency.ipynb")
    download_notebook = str(notebooks["download.ipynb"])
    strategy_notebook = str(notebooks["2_high_frequency.ipynb"])

    assert "config[\"strategy_2\"]" in download_notebook
    assert "update_dataset" not in strategy_notebook
    assert "load_dataset(\"intraday_volatility_daily\"" in strategy_notebook
    assert "calculate_high_frequency_div_low_vol_monthly_rebalanced_index" in strategy_notebook
    assert "calculate_high_frequency_volatility_candidate_group_indices" in strategy_notebook
    assert "候选池" in strategy_notebook
    assert "不代表研报的全 A 或完整红利股票池结果" in strategy_notebook
    assert "Rank IC均值" in strategy_notebook

# pyquant

`pyquant` 是一个用于复刻量化研报策略的轻量研究项目。

项目边界：

- notebook 作为策略主程序和结果展示层；
- `src/pyquant/` 只沉淀稳定、通用、跨策略复用的 API；
- 具体因子、择时指标、组合规则优先留在对应策略目录；
- 不构建完整量化平台，不接入实盘交易。

## 项目结构

通用框架位于 `src/pyquant/`，具体策略实现位于 `src/strategies/{strategy_name}/`，notebook
统一位于 `notebooks/{strategy_name}/`。以红利低波策略为例：

```text
src/strategies/dividend_low_vol/
├─ config.yaml
├─ components.py
└─ timing.py

notebooks/dividend_low_vol/
├─ download.ipynb
├─ strategy_1_monthly_rebalance.ipynb
└─ strategy_3_valuation_spread_timing.ipynb
```

从仓库根目录启动 notebook。它会将 `src/` 加入导入路径，并从 `strategies` 与 `pyquant` 包读取
策略实现和数据接口；策略配置在 notebook 中通过 `yaml.safe_load` 读取。

## 数据目录

基础数据保存在 `data/pyquant.duckdb`。`configs/datasets.yaml` 是数据集目录，说明每个数据集的：

- 标准字段、必需字段、主键和日期字段；
- DuckDB 视图或文件路径；
- 上游数据源及字段映射；
- 是否支持更新、股票池选择和源代码映射。

当前可更新的数据集为：

```text
stock_daily
index_daily
csindex_daily
index_constituents
dividend
stock_profit_quarterly
```

`dividend_queries` 和 `stock_profit_quarterly_queries` 是 DuckDB 下载覆盖视图；
`other_daily` 是保留的只读文件数据集。

数据层按职责拆分：

```text
pyquant.data.catalog       数据集定义与校验
pyquant.data.loader        规范化读取
pyquant.data.updater       更新编排与 UpdateJob
pyquant.data.duckdb        连接、schema 和可信关系查询
pyquant.data.store         事实表与覆盖事务
pyquant.data.migration     离线迁移与验收
pyquant.data.sources       BaoStock、AKShare、RQData 适配器
```

## 读取数据

```python
from pyquant import load_dataset

price = load_dataset(
    "stock_daily",
    start="2023-01-01",
    end="2024-12-31",
)
dividends = load_dataset("dividend")
dividend_queries = load_dataset("dividend_queries")
shares = load_dataset("stock_profit_quarterly")
constituents = load_dataset("index_constituents")
```

日行情必须显式提供起止日期，避免意外读取全部历史数据。日期过滤包含起止日。
测试或独立数据根目录可通过 `data_root=Path(...)` 指定，不需要修改全局 catalog。

## 更新数据

在 notebook 中调用统一的数据集更新接口：

```python
from pyquant import update_dataset

job = update_dataset(
    "stock_daily",
    start="2024-01-02",
    end="2024-01-03",
    pool="all",
)
```

`update_dataset()` 会立即返回后台任务，notebook 可以继续执行其他单元格。任务状态和
股票下载进度会在启动单元格中自动覆盖同一行显示，例如 `Updated 120/5231`。状态也可
直接读取：

```python
job.state
job.completed, job.total
```

下载任务支持暂停、继续和正常停止：

```python
job.pause()
job.resume()
job.stop()
result = job.wait()
```

`stop()` 会在当前网络请求结束后停止，并执行待写数据保存、下载锁清理和数据源登出；
它不会强制杀死线程。`wait()` 等待任务结束并返回本次结果，后台下载失败时会重新抛出
原异常。强制终止 notebook 内核无法保证尚在内存中的数据保存。

也可以显式指定证券，分红和季度总股本同样使用日期范围：

```python
index_job = update_dataset(
    "index_daily",
    start="2024-01-02",
    pool=["sh.000300"],
)
official_index_job = update_dataset(
    "csindex_daily",
    start="2014-01-01",
    end="2023-06-30",
    pool=["H30269", "H20269"],
)
constituent_job = update_dataset(
    "index_constituents",
    start="2014-01-01",
    end="2023-06-30",
    pool=["H30269"],
)
dividend_job = update_dataset("dividend", start="2021-01-01", pool="all")
shares_job = update_dataset(
    "stock_profit_quarterly",
    start="2021-01-01",
    pool="all",
)
```

`end` 默认使用当天。`pool` 可以是 `all`、`sz50`、`hs300`、`zz500`，也可以是
BaoStock 证券代码的可迭代对象。代码列表会去重并保持原顺序，适合逐级筛选后只更新剩余
证券。不支持命名股票池的数据集必须传显式代码集合；`index_constituents` 的项目指数代码
会由 catalog 映射为 RQData 代码。`max_tasks` 可限制本次最多执行的远端请求任务数。
下载器依据 DuckDB 覆盖表补齐缺口，并区分“空结果”和“尚未查询”。

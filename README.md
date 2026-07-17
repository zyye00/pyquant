# pyquant

`pyquant` 是一个用于复刻量化研报策略的轻量研究项目。

项目边界：

- notebook 作为策略主程序和结果展示层；
- `src/pyquant/` 只沉淀稳定、通用、跨策略复用的 API；
- 具体因子、择时指标、组合规则优先留在对应策略目录；
- 不构建完整量化平台，不接入实盘交易。

## 数据目录

项目按数据集组织本地数据。`configs/datasets.yaml` 是唯一可执行的数据目录，说明每个数据集的：

- 标准字段、必需字段、主键和日期字段；
- 本地路径、分区方式和股票代码来源；
- 当前上游数据源及其字段映射；
- 是否支持更新、股票池选择和复权参数。

当前可更新的数据集为：

```text
stock_daily
index_daily
stock_5m
dividend
stock_profit_quarterly
```

`other_daily`、`dividend_queries` 和 `stock_profit_quarterly_queries` 是只读数据集。

数据落盘位置不包含数据源名称：

```text
data/
├─ raw/
│  ├─ stock_daily/{symbol}.parquet
│  ├─ index_daily/{symbol}.parquet
│  ├─ stock_5m/{symbol}/{year}.parquet
│  ├─ other_daily/{symbol}.parquet
│  ├─ dividend/{data,queries}.parquet
│  └─ stock_profit_quarterly/{data,queries}.parquet
└─ state/
   ├─ request_log.csv
   └─ download.lock
```

## 读取数据

```python
from pyquant import load_dataset

price = load_dataset(
    "stock_daily",
    start="2023-01-01",
    end="2024-12-31",
    adjustment="none",
)
dividends = load_dataset("dividend", start="2021-01-01", end="2024-12-31")
dividend_queries = load_dataset("dividend_queries")
shares = load_dataset(
    "stock_profit_quarterly",
    start="2021-01-01",
    end="2024-12-31",
)
```

分区行情必须显式提供起止日期，避免意外读取全部历史数据。日期过滤包含起止日。

## 更新数据

在 notebook 中调用统一的数据集更新接口：

```python
from pyquant import update_dataset

job = update_dataset(
    "stock_daily",
    start="2024-01-02",
    end="2024-01-03",
    pool="all",
    adjustment="none",
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
dividend_job = update_dataset("dividend", start="2021-01-01", pool="all")
shares_job = update_dataset(
    "stock_profit_quarterly",
    start="2021-01-01",
    pool="all",
)
```

`end` 默认使用当天。`pool` 可以是 `all`、`sz50`、`hs300`、`zz500`，也可以是
BaoStock 证券代码的可迭代对象。代码列表会去重并保持原顺序，适合逐级筛选后只更新剩余
证券。`max_tasks` 可限制本次最多执行的远端请求任务数。下载器根据每只证券已有 parquet
的日期范围分别补齐前向和后向缺口，并通过查询记录区分“空结果”和“尚未查询”。

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
│  ├─ stock_daily/{adjustment}/{symbol}.parquet
│  ├─ index_daily/{adjustment}/{symbol}.parquet
│  ├─ stock_5m/{adjustment}/{symbol}/{year}.parquet
│  ├─ other_daily/{adjustment}/{symbol}.parquet
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

所有下载使用统一的数据集命令：

```bash
pyquant data-update stock_daily \
  --pool all \
  --start-date 2024-01-02 \
  --end-date 2024-01-03 \
  --adjustment none
```

也可以显式指定证券：

```bash
pyquant data-update index_daily \
  --symbols sh.000300 \
  --start-date 2024-01-02 \
  --end-date 2024-01-03
```

分红和季度总股本同样使用日期范围：

```bash
pyquant data-update dividend --pool all --start-date 2021-01-01
pyquant data-update stock_profit_quarterly --pool all --start-date 2021-01-01
```

`--end-date` 默认使用当天。`--pool` 支持 `all`、`sz50`、`hs300`、`zz500`，并与
`--symbols` 二选一。`--max-tasks` 可限制本次最多执行的远端请求任务数。

下载过程中可使用 `p` 暂停、`c` 继续、`q` 保存并退出。下载器根据已有 parquet 的日期
范围补齐缺口，并通过查询记录区分“空结果”和“尚未查询”。

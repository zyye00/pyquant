# 本地基础数据库

本项目的长期基础数据统一保存在 `data/pyquant.duckdb`，包括股票与指数日行情、指数成分股、分红、季度总股本及其下载覆盖。

## 数据层级

| Schema | 用途 |
| --- | --- |
| `ref` | 股票、指数维表及稳定的整数 ID |
| `core` | 清洗后的基础事实表 |
| `meta` | 已完成的下载覆盖 |
| `api` | 面向 `pyquant.load_dataset()` 和策略代码的视图 |
| `feature` | 可跨策略复用的高成本衍生数据 |

证券代码在 `api` 层统一为 `600000.SH`、`000001.SZ`、`920001.BJ`。BaoStock 的 `sh.600000` 只在数据源适配层和旧备份中出现。

指数使用独立的 `ref.market_index`，不会混入股票维表。指数代码保留数据源格式，例如中证官方代码 `H30269` 和 BaoStock 代码 `sh.000300`。

## 核心事实表

### `core.stock_daily`

主键语义为 `security_id + trade_date`，但不建立数据库唯一索引。迁移和下载写入时负责去重。

| 字段 | DuckDB 类型 | 说明 |
| --- | --- | --- |
| `security_id` | `UINTEGER` | `ref.security` 的内部证券 ID |
| `trade_date` | `DATE` | 交易日期 |
| `open`、`high`、`low`、`close`、`preclose` | `FLOAT` | 价格字段 |
| `volume` | `BIGINT` | 成交量 |
| `amount` | `DOUBLE` | 成交额，单位为元 |
| `turn` | `FLOAT` | 换手率 |
| `pe_ttm`、`ps_ttm`、`pcf_ncf_ttm` | `FLOAT` | BaoStock 日行情估值字段 |
| `is_st` | `BOOLEAN` | 是否为 ST 股票 |

`pct_chg` 不在核心表重复保存，由 `api.stock_daily` 使用 `100 × (close / preclose - 1)` 动态计算，单位为百分比。

### `core.stock_pb_daily`

主键语义为 `security_id + trade_date`，六列 PB 均来自 RQData `get_factor`，保存为 DuckDB `DOUBLE`：

| 字段 | 说明 |
| --- | --- |
| `pb_ratio_lf`、`pb_ratio_lyr`、`pb_ratio_ttm` | 归属母公司股东权益分别按 LF、LYR、TTM 口径计算的市净率 |
| `pb_ratio_1_lf`、`pb_ratio_1_lyr`、`pb_ratio_1_ttm` | 剔除其他权益工具后的对应市净率 |

策略 3 默认使用 `pb_ratio_lf`，可通过 `src/strategies/dividend_low_vol/config.yaml` 的
`strategy_3.pb_factor` 切换到其他五列。`book_to_market_ratio_*` 是 PB 的倒数，不属于本数据集。

### `core.index_daily`

业务键为 `index_id + trade_date`，写入时覆盖同一指数日期。

| 字段 | DuckDB 类型 | 说明 |
| --- | --- | --- |
| `index_id` | `USMALLINT` | `ref.market_index` 的内部指数 ID |
| `trade_date` | `DATE` | 交易日期 |
| `open`、`high`、`low`、`preclose` | `FLOAT` | BaoStock 指数价格字段；中证官方数据为 `NULL` |
| `close` | `DOUBLE` | 收盘点位；保留中证官方数据精度 |
| `volume` | `BIGINT` | 成交量 |
| `amount` | `DOUBLE` | 成交额 |
| `turn` | `FLOAT` | 换手率 |
| `pe_ttm`、`ps_ttm`、`pcf_ncf_ttm` | `FLOAT` | 指数估值字段 |
| `is_st` | `BOOLEAN` | 源字段；指数通常为 `NULL` |

`api.index_daily` 动态计算 `pct_chg = close / preclose - 1`。`csindex_daily` 目前只提供 `date, symbol, close`，其他行情字段保持空值。

### `core.index_constituent`

| 字段 | DuckDB 类型 | 说明 |
| --- | --- | --- |
| `index_id` | `USMALLINT` | 内部指数 ID |
| `effective_date` | `DATE` | 当前成分集合开始生效的日期 |
| `security_id` | `UINTEGER` | `ref.security` 的内部股票 ID |

唯一键为 `index_id + effective_date + security_id`。只保存成分集合发生变化后的快照，不展开为逐日记录。`api.index_constituents` 返回 `effective_date, index_code, symbol`，其中股票代码为 `600000.SH` 等统一格式。

### `core.dividend`

| 字段 | DuckDB 类型 | 说明 |
| --- | --- | --- |
| `security_id` | `UINTEGER` | 内部证券 ID |
| `announce_date` | `DATE` | 公告日 |
| `record_date` | `DATE` | 股权登记日 |
| `ex_date` | `DATE` | 除权除息日 |
| `payment_date` | `DATE` | 派息日 |
| `cash_dividend_before_tax` | `FLOAT` | 每股税前现金分红 |

该表只保存 BaoStock 的原始税前字段 `dividCashPsBeforeTax` 转换后的浮点数。数据层不保存
按投资者税率区分的税后字符串，也不在存储阶段推导税后值。红利低波策略统一以
`cash_dividend_before_tax × 0.9` 近似税后现金分红；该固定比例仅用于策略计算，不会写回
DuckDB。

### `core.share_capital_quarterly`

| 字段 | DuckDB 类型 | 说明 |
| --- | --- | --- |
| `security_id` | `UINTEGER` | 内部证券 ID |
| `report_date` | `DATE` | 报告期末 |
| `publish_date` | `DATE` | 公告日 |
| `total_shares` | `BIGINT` | 总股本，缺失源值保存为 `NULL` |

`year` 和 `quarter` 可由 `report_date` 推导，不在事实表重复保存。

## 下载覆盖

| 表 | 覆盖键 |
| --- | --- |
| `meta.stock_daily_coverage` | `security_id + start_date + end_date + field_set_id` |
| `meta.stock_pb_daily_coverage` | `security_id + start_date + end_date + field_set_id` |
| `meta.index_daily_coverage` | `index_id + start_date + end_date + field_set_id` |
| `meta.dividend_coverage` | `security_id + query_year + field_set_id` |
| `meta.share_capital_coverage` | `security_id + report_year + report_quarter` |

日行情的重叠或相邻区间会合并。分红 `field_set_id=1` 只表示旧税后字段曾经查询过；正式下载器只把 `field_set_id=2` 视为税前字段已完成查询。空结果也会写入覆盖表。

指数行情的字段集互不替代：

- `field_set_id=1`：BaoStock 完整指数日行情；
- `field_set_id=2`：中证官方收盘点位。

下载成功但返回空结果时仍记录对应字段集的请求覆盖。

## Python 访问

```python
from pyquant import load_dataset

price = load_dataset(
    "stock_daily",
    start="2024-01-01",
    end="2024-12-31",
)
pb = load_dataset(
    "stock_pb_daily",
    start="2014-01-01",
    end="2023-06-30",
)
dividends = load_dataset("dividend")
shares = load_dataset("stock_profit_quarterly")
index_price = load_dataset(
    "csindex_daily",
    start="2014-01-01",
    end="2023-06-30",
)
constituents = load_dataset("index_constituents")
```

`load_dataset()` 查询 `api` 视图。`stock_daily` 仍要求显式给出开始和结束日期；当前分红未重新下载时，`load_dataset("dividend")` 返回具有正式字段但没有数据行的 DataFrame。

# pyquant

`pyquant` 是一个用于复刻量化研报策略的轻量研究项目。

项目边界：

- notebook 作为策略主程序和结果展示层；
- `src/pyquant/` 只沉淀稳定、通用、跨策略复用的 API；
- 具体因子、择时指标、组合规则优先留在对应策略目录；
- 不构建完整量化平台，不接入实盘交易。

当前最小结构：

```text
src/pyquant/
├─ data.py
├─ universe.py
├─ transforms.py
├─ backtest.py
├─ metrics.py
└─ io.py

strategies/
├─ cross_sectional/
│  └─ dividend_low_vol/
└─ timing/
```

## 下载数据

项目使用 BaoStock 下载原始数据。下载参数主要通过 CLI 指定；路径和请求限制写在
`configs/baostock_download.yaml`：

```yaml
baostock_limits:
  hard_max_requests_per_day: 50000
  safe_max_requests_per_day: 49000

paths:
  raw_root: data/raw/baostock
```

查看命令帮助：

```bash
pyquant baostock-download --help
```

下载指数日频行情：

```bash
pyquant baostock-download \
  --frequency d \
  --index sh.000300 \
  --start-date 2024-01-02 \
  --end-date 2024-01-03
```

下载某个股票池的股票行情，例如沪深 300 成分股：

```bash
pyquant baostock-download \
  --frequency d \
  --pool hs300 \
  --start-date 2024-01-02 \
  --end-date 2024-01-03
```

`--pool` 支持 `all`、`sz50`、`hs300`、`zz500`，且必须与 `--index` 二选一。下载
全 A 股票池时指定 `--pool all`：

```bash
pyquant baostock-download \
  --frequency d \
  --pool all \
  --start-date 2024-01-02 \
  --end-date 2024-01-03
```

下载 5 分钟股票行情：

```bash
pyquant baostock-download \
  --frequency 5 \
  --pool hs300 \
  --start-date 2024-01-02 \
  --end-date 2024-01-03
```

下载季度总股本数据：

```bash
pyquant baostock-profit-download \
  --pool all \
  --start-date 2013-01-01
```

`--end-date` 可以省略，省略时使用当天。下载器会查询起止日期覆盖到的所有季度。数据保存为
`data/raw/baostock/stock_profit_quarterly.parquet`，字段为 `code`、`year`、`quarter`、
`publish_date`、`report_date`、`total_shares`。已查询的股票、年份和季度组合（包括空结果）
记录在 `data/raw/baostock/state/stock_profit_quarterly_queries.parquet`，重新运行时会跳过。

`--end-date` 可以省略，省略时使用当天日期。BaoStock 不提供指数分钟线，因此
`--index` 只能与 `--frequency d` 一起使用。

股票下载可用 `--adjustflag forward` 指定前复权，`--adjustflag backward` 指定后复权；
不指定时为不复权，数据保存到 `stock/none/`。三种数据分别保存到
`stock/forward/`、`stock/backward/`、`stock/none/`。

如果当前环境尚未生成 `pyquant` 命令，也可以使用模块方式运行：

```bash
python -m pyquant.cli baostock-download --help
```

下载过程中终端会提示：

```text
p 暂停
c 继续
q 保存并退出
```

下载器根据现有 parquet 的最大日期计算每只证券需要补齐的时间范围。每次成功请求都会
合并写回目标文件，因此重新运行会从已有数据后的日期继续。请求日志只保留当天记录，
用于执行每日请求限制。

数据落盘位置：

```text
data/raw/baostock/
├─ daily/
│  ├─ stock/
│  └─ index/
├─ minute_5/
│  └─ stock/
└─ state/
```

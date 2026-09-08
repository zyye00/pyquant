# pyquant

`pyquant` 是一个复刻量化研报策略的轻量研究项目，围绕红利低波策略，研究调仓频率、波动率因子和估值择时如何影响组合表现。

项目将数据更新与存储、策略实现、实验展示分开：通用研究能力放在 `pyquant`，具体策略放在 `strategies`，notebook 负责流程编排、图表和结论。不构建完整量化平台，不接入实盘交易。

## 研究内容与阅读入口

研究参考国盛证券 2023-08-16《“薪火”量化分析系列研究（三）：红利低波的增强方案》。策略定义、研报结果和项目口径见[策略说明](docs/strategies/div_low_vol.md)。

| 入口 | 内容 |
|---|---|
| [0_download.ipynb](notebooks/div_low_vol/0_download.ipynb) | 按数据源更新行情、分红、复权因子、历史成份、估值、市值和分钟数据 |
| [1_rebalance.ipynb](notebooks/div_low_vol/1_rebalance.ipynb) | 红利低波月调组合，以及传统波动率分组和 IC 检验 |
| [2_high_frequency.ipynb](notebooks/div_low_vol/2_high_frequency.ipynb) | 高频波动率增强组合，以及候选池内的因子分组和 IC 检验 |
| [3_timing.ipynb](notebooks/div_low_vol/3_timing.ipynb) | 估值差择时，比较持有、空仓和理论卖空情景 |

建议先阅读策略说明，再按 1、2、3 查看研究。实际运行前，按需执行下载 notebook 中的对应任务，并等待任务完成。

高频因子的分组和 IC 检验仅覆盖每月股息率前 150 名候选池，不等同于研报的全 A 或完整红利股票池检验。

**结果状态：** notebook 保留的历史输出尚未按当前调仓与指标口径重新验证，不能作为当前实现的正式绩效；策略说明中的研报数字属于引用结果。

## 研究口径

- 月调组合和择时均在每月最后一个交易日按收盘价调仓，可用信息截止前一个交易日。
- 前一交易日按输入行情的全市场交易日期确定，输入需包含完整交易日期及足够的历史回看数据。
- 组合回测输入月末价格与目标权重，只展示月末净值；波动率和最大回撤不包含月中波动。
- 比较基准为官方红利低波全收益指数 `H20269`；历史成份使用 `H30269`，不对全收益指数额外应用股票复权因子。
- 月调组合默认交易成本为零；择时的卖空情景不计融券费、保证金和卖空可行性约束，仅作理论研究。
- 月收益按每年 12 期年化，波动率采用总体标准差（`ddof=0`）。Sharpe 的无风险收益率默认设为零，信息比率以 H20269 为基准；年化收益／年化波动单列为研报对照比值。
- 最大回撤包含初始净值，统一用非负损失幅度表示，无回撤时为零。

参数以[策略配置](src/strategies/div_low_vol/config.yaml)为准。估值择时默认使用 `pb_mrq`；选择 `pb_ratio_*` 时改用 RQData 的对应 PB 口径。

## 项目结构

```text
configs/
├─ dataset_catalog.yaml       数据集合同、存储位置和更新能力
└─ source_protocols.yaml      数据源协议、请求字段和字段映射
src/
├─ pyquant/                  通用数据、股票池、回测、指标和输出能力
└─ strategies/div_low_vol/    选样、因子、组合、择时和参数
notebooks/div_low_vol/        下载与研究入口
tests/                      按框架与策略职责组织的测试
docs/                       策略说明
```

依赖方向为 `notebooks → strategies → pyquant`，notebook 也可直接调用 `pyquant`。数据源适配器不写数据库，读取接口不触发下载，策略与 notebook 不直接写 DuckDB 物理表。

| 数据层模块 | 职责 |
|---|---|
| `catalog`、`resources` | 数据集校验和静态资源读取 |
| `loader`、`identifiers` | 标准化读取和证券代码处理 |
| `updater` | 下载编排、后台任务与覆盖检查 |
| `store`、`duckdb`、`sql/schema.sql` | 数据写入、覆盖事务、数据库连接和表／视图定义 |
| `sources` | BaoStock、RQData 和中证指数行情适配 |
| `adjustments`、`intraday` | 复权价格和日内波动率计算 |

## 使用与验证

Python 版本要求及依赖分组见 [pyproject.toml](pyproject.toml)。数据源依赖按需选用；运行 notebook 需要 notebook 依赖，运行测试需要 test 依赖。RQData 任务需要自行配置具备相应数据权限的账户；分钟行情请求还受账户配额限制。

从仓库根目录启动 notebook，其初始化单元会定位项目并加载 `src` 中的代码。先运行配置单元，再执行所需的数据更新和研究单元。数据默认存放在 `data/pyquant.duckdb`，不随仓库分发。

以下指标示例可在完成 notebook 初始化后运行，不需要下载行情：

```python
import pandas as pd
from pyquant import calc_metrics

metrics = calc_metrics(
    pd.Series([-0.10, 0.0]),
    periods_per_year=12,
)
assert round(metrics["max_drawdown"], 10) == 0.1
```

在已配置项目依赖的 Python 环境中，从仓库根目录运行：

```bash
pytest -q
```

测试覆盖数据层、策略逻辑及合成数据下的关键 notebook 单元；测试通过不代表真实数据研究流程已完整重跑。

## 数据接口

外部调用优先使用 `from pyquant import ...`。数据集的必需字段、主键、日期要求与存储视图见 [dataset_catalog.yaml](configs/dataset_catalog.yaml)，源字段和请求协议见 [source_protocols.yaml](configs/source_protocols.yaml)。

| 数据集 | 来源／用途 | 更新入口 |
|---|---|---|
| `stock_daily` | BaoStock 日行情及 `pb_mrq` 等估值字段 | `update_dataset` |
| `stock_adjust_factor` | BaoStock 复权因子事件 | `update_dataset` |
| `dividend` | BaoStock 分红 | `update_dataset` |
| `stock_profit_quarterly` | BaoStock 季度总股本，供审计使用 | `update_dataset` |
| `stock_pb_daily` | RQData 六种 PB 口径 | `update_dataset` |
| `stock_market_cap_daily` | RQData `market_cap_3` PIT 总市值 | `update_dataset` |
| `index_constituents` | RQData 官方历史成份快照 | `update_dataset` |
| `index_daily` | BaoStock 指数日行情 | `update_dataset` |
| `csindex_daily` | 经 AKShare 获取的中证指数行情 | `update_dataset` |
| `stock_minute_1m` | RQData 未复权分钟行情 | `update_minute_data` |
| `intraday_volatility_daily` | 分钟行情生成的日内波动率 | 随分钟更新生成 |

`dividend_queries`、`stock_profit_quarterly_queries` 和 `stock_adjust_factor_coverage` 提供下载覆盖信息，用于区分已查询但无数据与尚未查询。PIT 总市值用于策略市值计算，季度总股本不参与该计算。

### 读取

读取前需完成对应数据更新。日频和分钟行情等标记 `requires_dates` 的数据集必须显式提供起止日期；日期过滤包含起止日。

```python
from pyquant import load_dataset

price = load_dataset(
    "stock_daily",
    start="2024-01-02",
    end="2024-01-31",
    symbols=["600000.SH", "000001.SZ"],
)
dividends = load_dataset("dividend")
constituents = load_dataset("index_constituents")
```

`stock_daily.pb_mrq` 与 `stock_pb_daily` 的六种 PB 字段分别保存，不会互相覆盖。读写接口可通过 `data_root` 指定独立数据目录。

### 更新

`update_dataset()` 返回后台 `UpdateJob`。示例逐个等待任务完成，下载失败时 `wait()` 抛出异常，后续任务不会继续启动：

```python
from pyquant import update_dataset

job = update_dataset(
    "stock_daily",
    start="2024-01-02",
    end="2024-01-03",
    pool=["sh.600000", "sz.000001"],
)
price_result = job.wait()

job = update_dataset(
    "stock_pb_daily",
    start="2024-01-02",
    end="2024-01-03",
    pool=["600000.SH", "000001.SZ"],
)
pb_result = job.wait()
```

上例仅演示接口，不足以满足策略的多年回看要求。完整依赖顺序和分钟数据候选请求见下载 notebook。

`pool` 的支持范围取决于数据集：

| 数据集 | `pool` |
|---|---|
| BaoStock 股票行情、分红、复权因子、季度总股本 | `all`、`sz50`、`hs300`、`zz500`，或显式证券列表 |
| RQData PB、PIT 市值 | `all`，或显式证券列表 |
| BaoStock 指数行情 | 显式指数列表，如 `["sh.000300"]` |
| 中证指数行情、历史成份 | 显式指数列表，如 `["H30269"]`；全收益行情可使用 `H20269` |

`end` 默认当天，正式研究建议显式指定截止日期。`max_tasks` 可限制支持该参数的更新任务请求数。命名股票池的日期规则随数据源不同，不能将单日池自动视为整个回测区间的历史股票池。

任务运行时可查看 `job.state`、`job.completed` 和 `job.total`，或按需调用 `pause()`、`resume()`、`stop()`。`stop()` 是协作式停止，会等待当前请求结束；随后调用 `wait()` 确认结束或接收异常。失败前已成功落库的内容保留，强制终止内核无法保证内存中的待写数据保存。

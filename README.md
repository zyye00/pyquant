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

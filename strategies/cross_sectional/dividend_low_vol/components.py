"""红利低波策略专属组件占位。

具体因子和组合规则将在接入真实数据口径后实现。
"""

import pandas as pd


def calc_factor(price: pd.DataFrame, universe: pd.DataFrame, config: dict) -> pd.Series:
    """计算当前策略因子。"""
    raise NotImplementedError("dividend_low_vol factor logic is not implemented yet")

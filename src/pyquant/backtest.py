"""Minimal weight return backtest."""

import pandas as pd
from typing import Union


def run_backtest(
    returns: Union[pd.Series, pd.DataFrame],
    weights: Union[pd.Series, pd.DataFrame],
    fee_rate: float = 0.0,
) -> pd.Series:
    """用 t 日权重持有下一期收益，返回组合收益序列。"""
    aligned_returns, aligned_weights = returns.align(weights, join="inner", axis=None)
    held_weights = aligned_weights.shift(1).fillna(0)
    weighted_returns = held_weights * aligned_returns
    portfolio_returns = (
        weighted_returns.sum(axis=1)
        if isinstance(weighted_returns, pd.DataFrame)
        else weighted_returns
    )

    turnover_raw = aligned_weights.fillna(0).diff().abs()
    turnover = (
        turnover_raw.sum(axis=1) if isinstance(turnover_raw, pd.DataFrame) else turnover_raw
    ).fillna(0)
    cost = turnover * fee_rate
    return (portfolio_returns - cost).rename("portfolio_return")

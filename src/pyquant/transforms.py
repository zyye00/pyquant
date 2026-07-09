"""Common factor transforms."""

import pandas as pd
from typing import Optional


def transform_factor(
    factor: pd.Series,
    winsor_n: Optional[float] = 3.0,
    zscore: bool = True,
) -> pd.Series:
    """按日期做横截面去极值和标准化，输入输出索引保持一致。"""
    if not isinstance(factor.index, pd.MultiIndex) or factor.index.nlevels < 2:
        raise ValueError("factor must use a MultiIndex with date and symbol levels")

    def transform_one_date(x: pd.Series) -> pd.Series:
        out = x.astype(float).copy()
        if winsor_n is not None:
            median = out.median(skipna=True)
            mad = (out - median).abs().median(skipna=True)
            if pd.notna(mad) and mad > 0:
                out = out.clip(median - winsor_n * mad, median + winsor_n * mad)
        if zscore:
            std = out.std(skipna=True, ddof=0)
            if pd.notna(std) and std > 0:
                out = (out - out.mean(skipna=True)) / std
            else:
                out = out * 0
        return out

    return factor.groupby(level=0, group_keys=False).apply(transform_one_date)

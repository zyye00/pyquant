"""Universe construction helpers."""

import pandas as pd
from typing import Optional


def build_universe(
    price: pd.DataFrame,
    symbols: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """从标准行情长表生成逐日股票池。"""
    required = {"date", "symbol"}
    missing = required - set(price.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    df = price.loc[:, ["date", "symbol"]].copy()

    if symbols is not None:
        df = df[df["symbol"].isin([str(symbol) for symbol in symbols])]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    out = df.drop_duplicates().sort_values(["date", "symbol"])
    out["in_universe"] = True
    return out.set_index(["date", "symbol"])

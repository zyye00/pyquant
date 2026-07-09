"""Data loading and field standardization."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
DEFAULT_PRICE_FIELD_MAP = {
    "trade_date": "date",
    "datetime": "date",
    "ticker": "symbol",
    "code": "symbol",
    "sec_code": "symbol",
    "vol": "volume",
    "money": "amount",
    "turnover": "amount",
}


def load_price(
    path: Union[str, Path],
    field_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """读取标准行情数据并返回统一字段。"""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported price file type: {path.suffix}")
    return standardize_price(df, field_map=field_map)


def standardize_price(
    df: pd.DataFrame,
    field_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """统一行情字段名、日期类型和股票代码格式。"""
    rename_map = DEFAULT_PRICE_FIELD_MAP | (field_map or {})
    out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()

    missing = [col for col in ["date", "symbol", "close"] if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)

    ordered = [col for col in PRICE_COLUMNS if col in out.columns]
    extras = [col for col in out.columns if col not in ordered]
    return out[ordered + extras].sort_values(["date", "symbol"]).reset_index(drop=True)

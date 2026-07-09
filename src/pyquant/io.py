"""Configuration and output helpers."""

from pathlib import Path
from typing import Any, Union

import pandas as pd
import yaml


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """读取 YAML 配置。"""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在并返回 Path。"""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_output(
    data: Union[pd.DataFrame, pd.Series],
    path: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """按扩展名保存结果，默认不覆盖已有文件。"""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    ensure_dir(path.parent)

    if path.suffix == ".csv":
        data.to_csv(path)
    elif path.suffix in {".parquet", ".pq"}:
        data.to_frame().to_parquet(path) if isinstance(data, pd.Series) else data.to_parquet(path)
    else:
        raise ValueError(f"Unsupported output file type: {path.suffix}")
    return path

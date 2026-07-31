"""CSI index history through AKShare."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pyquant.data.resources import load_source_protocols


_CSINDEX = load_source_protocols()["csindex"]


def clean_csindex_history(data: pd.DataFrame, code: str) -> pd.DataFrame:
    """Select documented CSI fields and convert them to catalog columns."""
    missing = sorted(set(_CSINDEX["source_fields"]) - set(data))
    if missing:
        raise ValueError(f"AKShare CSI result missing required columns: {missing}")
    out = (
        data.loc[:, _CSINDEX["source_fields"]]
        .rename(columns=_CSINDEX["field_map"])
        .copy()
    )
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["symbol"] = out["symbol"].fillna(code).astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out


def query_csindex_history(
    code: str,
    start_date: str,
    end_date: str,
    client: Any | None = None,
) -> pd.DataFrame:
    """Download CSI daily history from AKShare."""
    if client is None:
        try:
            import akshare as client
        except ImportError as exc:
            raise ImportError("AKShare download requires package 'akshare'.") from exc
    return client.stock_zh_index_hist_csindex(
        symbol=code,
        start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
    )

"""Valuation-spread timing for the dividend low-volatility index."""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd
import yaml

from pyquant import get_period_end_dates


config_file = files("strategies.dividend_low_vol").joinpath("config.yaml")
with config_file.open(encoding="utf-8") as _config_stream:
    _OUTPUT_COLUMNS = yaml.safe_load(_config_stream)["output_columns"]

SPREAD_COLUMNS = _OUTPUT_COLUMNS["timing_spread"]
BACKTEST_COLUMNS = _OUTPUT_COLUMNS["timing_backtest"]


def calculate_bp_spread(
    price: pd.DataFrame,
    constituents: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Calculate monthly constituent/non-constituent BP spread and lower band."""
    strategy = _validate_strategy_config(config)
    price_data = _prepare_price(price, strategy["pb_factor"])
    constituent_data = _prepare_constituents(
        constituents,
        strategy["index_code"],
    )
    price_data["month"] = price_data["date"].dt.to_period("M")
    period_ends = get_period_end_dates(price_data["date"])
    valuation_dates = pd.Series(period_ends, index=period_ends.to_period("M"))
    price_data = price_data[
        np.isfinite(price_data["pb"]) & price_data["pb"].gt(0)
    ]
    price_data = (
        price_data.sort_values(["date", "symbol"])
        .groupby(["month", "symbol"], as_index=False)
        .tail(1)
    )
    snapshots = {
        effective_date: frozenset(snapshot["symbol"])
        for effective_date, snapshot in constituent_data.groupby(
            "effective_date",
            sort=True,
        )
    }
    snapshot_dates = pd.DatetimeIndex(snapshots)
    rows = []
    for month, valuation_date in valuation_dates.items():
        position = snapshot_dates.searchsorted(valuation_date, side="right") - 1
        if position < 0:
            raise ValueError(
                f"No constituent snapshot is available at {valuation_date.date()}"
            )
        effective_date = snapshot_dates[position]
        monthly = price_data[price_data["month"].eq(month)].copy()
        monthly["bp"] = 1.0 / monthly["pb"]
        is_constituent = monthly["symbol"].isin(snapshots[effective_date])
        low = monthly.loc[is_constituent, ["symbol", "bp"]]
        high = monthly.loc[~is_constituent, ["symbol", "bp"]]
        bp_low, low_trimmed_count = _trimmed_mean(
            low,
            strategy["trim_ratio"],
            "constituent",
            valuation_date,
        )
        bp_high, high_trimmed_count = _trimmed_mean(
            high,
            strategy["trim_ratio"],
            "non-constituent",
            valuation_date,
        )
        rows.append(
            {
                "date": valuation_date,
                "constituent_effective_date": effective_date,
                "constituent_count": len(low),
                "non_constituent_count": len(high),
                "constituent_trimmed_count": low_trimmed_count,
                "non_constituent_trimmed_count": high_trimmed_count,
                "bp_low": bp_low,
                "bp_high": bp_high,
                "bp_spread": bp_low - bp_high,
            }
        )

    out = pd.DataFrame(rows).set_index("date")
    rolling = out["bp_spread"].rolling(strategy["band_window_months"])
    out["lower_band"] = rolling.mean() - strategy["band_std_multiplier"] * rolling.std(
        ddof=0
    )
    out["bearish_signal"] = out["lower_band"].notna() & out["bp_spread"].lt(
        out["lower_band"]
    )
    return out[SPREAD_COLUMNS]


def backtest_valuation_spread_timing(
    signal: pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Apply each month-end bearish signal to the following month's return."""
    signal_data = _prepare_signal(signal)
    benchmark_data = _prepare_benchmark(benchmark)
    signal_months = signal_data.index.to_period("M")
    benchmark_dates = get_period_end_dates(benchmark_data.index)
    benchmark_monthly = benchmark_data.reindex(benchmark_dates)
    benchmark_monthly.index = benchmark_dates.to_period("M")
    missing = signal_months[~signal_months.isin(benchmark_monthly.index)]
    if len(missing):
        raise ValueError(
            "Benchmark is missing signal months: "
            f"{sorted(map(str, missing.unique()))[:5]}"
        )

    benchmark_close = benchmark_monthly.reindex(signal_months)
    benchmark_close.index = signal_data.index
    benchmark_return = benchmark_close.pct_change(fill_method=None)
    bearish_position = signal_data["bearish_signal"].shift(
        1,
        fill_value=False,
    )
    cash_return = benchmark_return.mask(bearish_position, 0.0)
    short_return = benchmark_return.where(
        ~bearish_position,
        -benchmark_return,
    )
    signal_dates = pd.Series(
        signal_data.index,
        index=signal_data.index,
        dtype="datetime64[ns]",
    ).shift(1)
    out = pd.DataFrame(
        {
            "signal_date": signal_dates,
            "benchmark_return": benchmark_return,
            "bearish_position": bearish_position,
            "cash_timing_return": cash_return,
            "short_timing_return": short_return,
            "benchmark_nav": (1.0 + benchmark_return.fillna(0.0)).cumprod(),
            "cash_timing_nav": (1.0 + cash_return.fillna(0.0)).cumprod(),
            "short_timing_nav": (1.0 + short_return.fillna(0.0)).cumprod(),
        }
    )
    return out[BACKTEST_COLUMNS]


def _validate_strategy_config(config: dict) -> dict:
    try:
        strategy = config["strategy_3"]
        index_code = str(strategy["index_code"])
        trim_ratio = float(strategy["trim_ratio"])
        band_window = int(strategy["band_window_months"])
        band_multiplier = float(strategy["band_std_multiplier"])
        pb_factor = str(strategy["pb_factor"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid strategy_3 configuration") from exc
    if not index_code:
        raise ValueError("index_code must not be empty")
    if not 0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    if band_window <= 0:
        raise ValueError("band_window_months must be positive")
    if band_multiplier < 0:
        raise ValueError("band_std_multiplier must not be negative")
    if not pb_factor:
        raise ValueError("pb_factor must not be empty")
    return {
        "index_code": index_code,
        "trim_ratio": trim_ratio,
        "band_window_months": band_window,
        "band_std_multiplier": band_multiplier,
        "pb_factor": pb_factor,
    }


def _prepare_price(price: pd.DataFrame, pb_factor: str) -> pd.DataFrame:
    required = {"date", "symbol", pb_factor}
    _require_columns(price, required, "price")
    out = price.loc[:, ["date", "symbol", pb_factor]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("price date must not contain invalid values")
    out["symbol"] = out["symbol"].astype(str)
    out = out.rename(columns={pb_factor: "pb"})
    out["pb"] = pd.to_numeric(out["pb"], errors="coerce")
    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("price contains duplicate (date, symbol) rows")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _prepare_constituents(
    constituents: pd.DataFrame,
    index_code: str,
) -> pd.DataFrame:
    required = {"effective_date", "index_code", "symbol"}
    _require_columns(constituents, required, "constituents")
    out = constituents.loc[:, sorted(required)].copy()
    out["effective_date"] = pd.to_datetime(
        out["effective_date"],
        errors="coerce",
    )
    if out["effective_date"].isna().any():
        raise ValueError("constituent effective_date must not contain invalid values")
    out["index_code"] = out["index_code"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    key = ["effective_date", "index_code", "symbol"]
    if out.duplicated(key).any():
        raise ValueError(f"constituents contain duplicate keys: {key}")
    out = out[out["index_code"].eq(index_code)]
    if out.empty:
        raise ValueError(f"No constituent snapshots found for {index_code}")
    return out.sort_values(["effective_date", "symbol"]).reset_index(drop=True)


def _prepare_signal(signal: pd.DataFrame) -> pd.DataFrame:
    _require_columns(signal, {"bearish_signal"}, "signal")
    out = signal[["bearish_signal"]].copy()
    if "date" in signal:
        out.index = pd.to_datetime(signal["date"], errors="coerce")
    else:
        out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.hasnans:
        raise ValueError("signal dates must not contain invalid values")
    if out.index.has_duplicates:
        raise ValueError("signal dates must be unique")
    months = out.index.to_period("M")
    if months.has_duplicates:
        raise ValueError("signal must contain at most one row per month")
    if out["bearish_signal"].isna().any():
        raise ValueError("bearish_signal must not contain missing values")
    out["bearish_signal"] = out["bearish_signal"].astype(bool)
    out.index.name = "date"
    return out.sort_index()


def _prepare_benchmark(benchmark: pd.Series) -> pd.Series:
    out = benchmark.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.hasnans:
        raise ValueError("benchmark dates must not contain invalid values")
    if out.index.has_duplicates:
        raise ValueError("benchmark dates must be unique")
    out = pd.to_numeric(out, errors="coerce").dropna().sort_index()
    if out.empty or not out.gt(0).all():
        raise ValueError("benchmark must contain positive values")
    return out


def _trimmed_mean(
    values: pd.DataFrame,
    trim_ratio: float,
    group_name: str,
    valuation_date: pd.Timestamp,
) -> tuple[float, int]:
    values = values.sort_values(["bp", "symbol"]).reset_index(drop=True)
    trim_count = int(np.floor(len(values) * trim_ratio))
    trimmed = values.iloc[trim_count : len(values) - trim_count or None]
    if trimmed.empty:
        raise ValueError(f"No {group_name} BP values remain at {valuation_date.date()}")
    return float(trimmed["bp"].mean()), len(trimmed)


def _require_columns(
    data: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

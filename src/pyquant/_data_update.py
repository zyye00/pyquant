"""Private dataset-update implementation for the configured upstream source."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from collections.abc import Collection
from typing import Any, Callable, Iterable

import pandas as pd

from pyquant.data import DATASET_CATALOG, get_dataset

_baostock = DATASET_CATALOG["sources"]["baostock"]
_datasets = DATASET_CATALOG["datasets"]
_fields = _baostock["fields"]

BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY = _baostock["hard_max_requests_per_day"]
BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY = _baostock["safe_max_requests_per_day"]
BAOSTOCK_SOCKET_TIMEOUT_SECONDS = _baostock["socket_timeout_seconds"]
BAOSTOCK_STOCK_POOL_QUERIES = _baostock["stock_pool_queries"]
ADJUSTMENT_FLAGS = _baostock["adjustment_flags"]
ADJUSTMENT_DIRS = _baostock["adjustment_dirs"]


@dataclass(frozen=True)
class DataPaths:
    data_root: Path

    @property
    def raw_root(self) -> Path:
        return _configured_path("data/raw", self.data_root)

    @property
    def daily_stock_dir(self) -> Path:
        return _dataset_path(
            "stock_daily", self.data_root, adjustment="none", symbol="_"
        ).parents[1]

    @property
    def daily_index_dir(self) -> Path:
        return _dataset_path(
            "index_daily", self.data_root, adjustment="none", symbol="_"
        ).parents[1]

    @property
    def minute_5_stock_dir(self) -> Path:
        return _dataset_path(
            "stock_5m", self.data_root, adjustment="none", symbol="_", year=2000
        ).parents[2]

    @property
    def state_dir(self) -> Path:
        return _configured_path(DATASET_CATALOG["state"]["root"], self.data_root)

    @property
    def dividend_path(self) -> Path:
        return _dataset_path("dividend", self.data_root)

    @property
    def dividend_queries_path(self) -> Path:
        return _dataset_path("dividend_queries", self.data_root)

    @property
    def profit_path(self) -> Path:
        return _dataset_path("stock_profit_quarterly", self.data_root)

    @property
    def profit_queries_path(self) -> Path:
        return _dataset_path("stock_profit_quarterly_queries", self.data_root)

    @property
    def request_log_path(self) -> Path:
        return _configured_path(DATASET_CATALOG["state"]["request_log"], self.data_root)

    @property
    def lock_path(self) -> Path:
        return _configured_path(DATASET_CATALOG["state"]["lock"], self.data_root)


def _configured_path(template: str, data_root: str | Path, **values: object) -> Path:
    return Path(data_root) / Path(template.format(**values)).relative_to("data")


def _dataset_path(name: str, data_root: str | Path, **values: object) -> Path:
    return _configured_path(_datasets[name]["storage"]["path"], data_root, **values)


class BaostockClient:
    """Thin BaoStock client wrapper with lazy import."""

    def __init__(self) -> None:
        try:
            import baostock as bs
        except ImportError as exc:
            raise ImportError("BaoStock download requires package 'baostock'.") from exc
        self.bs = bs

    def __enter__(self) -> "BaostockClient":
        result = self.bs.login()
        if getattr(result, "error_code", "0") != "0":
            msg = f"BaoStock login failed: {result.error_code} {result.error_msg}"
            raise RuntimeError(msg)
        self.bs.common.context.default_socket.settimeout(
            BAOSTOCK_SOCKET_TIMEOUT_SECONDS
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.bs.logout()


def init_data_storage(data_root: str | Path = "data") -> DataPaths:
    """Create the dataset-oriented raw-data directory skeleton."""
    paths = DataPaths(Path(data_root))
    for path in [
        paths.daily_stock_dir,
        *(paths.daily_stock_dir / name for name in ADJUSTMENT_DIRS.values()),
        paths.daily_index_dir,
        paths.minute_5_stock_dir,
        *(paths.minute_5_stock_dir / name for name in ADJUSTMENT_DIRS.values()),
        paths.state_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    reset_request_log(paths.request_log_path)
    return paths


def daily_target_path(
    code: str,
    dataset: str,
    data_root: str | Path = "data",
    adjustflag: str | None = None,
) -> Path:
    return _dataset_path(
        {"stock": "stock_daily", "index": "index_daily"}[dataset],
        data_root,
        adjustment=adjustment_dir(adjustflag),
        symbol=code,
    )


def minute_5_target_path(
    code: str,
    year: int,
    data_root: str | Path = "data",
    adjustflag: str | None = None,
) -> Path:
    return _dataset_path(
        "stock_5m",
        data_root,
        adjustment=adjustment_dir(adjustflag),
        symbol=code,
        year=year,
    )


def adjustment_dir(adjustflag: str | None) -> str:
    return ADJUSTMENT_DIRS[baostock_adjustflag(adjustflag)]


def baostock_adjustflag(adjustflag: str | None) -> str:
    if adjustflag is None:
        return ADJUSTMENT_FLAGS["none"]
    if adjustflag in ADJUSTMENT_DIRS:
        return adjustflag
    try:
        return ADJUSTMENT_FLAGS[adjustflag]
    except KeyError as exc:
        raise ValueError(f"Unsupported BaoStock adjustment: {adjustflag}") from exc


def clean_baostock_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert BaoStock strings to compact types and remove source-only fields."""
    out = data.copy()
    if "tradestatus" in out:
        out = out.loc[out["tradestatus"].astype(str) == "1"].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    for column in set(_fields["history"]["float32"]) & set(out.columns):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float32")
    if "amount" in out:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    if "isST" in out:
        out["isST"] = pd.to_numeric(out["isST"], errors="coerce").astype("boolean")
    if "volume" in out:
        column = "volume"
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    return out.drop(columns=["code", "tradestatus", "adjustflag"], errors="ignore")


def clean_baostock_dividends(
    data: pd.DataFrame,
    code: str,
    year: int,
) -> pd.DataFrame:
    """Keep dividend fields needed for point-in-time yield calculations."""
    fields = _fields["dividend"]
    out = data.rename(
        columns={
            key: value
            for key, value in _datasets["dividend"]["field_map"].items()
            if key != "code"
        }
    ).copy()
    for column in fields["data"]:
        if column not in out:
            out[column] = pd.NA
    out["code"] = out["code"].fillna(code).astype(str)
    out["year"] = year
    for column in ["announce_date", "record_date", "operate_date", "payment_date"]:
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.date
    for column in fields["float32"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float32")
    return out[fields["data"]]


def clean_baostock_profit(
    data: pd.DataFrame,
    code: str,
    year: int,
    quarter: int,
) -> pd.DataFrame:
    """Keep quarterly publication dates and total shares."""
    fields = _fields["profit_quarterly"]
    out = data.rename(
        columns={
            key: value
            for key, value in _datasets["stock_profit_quarterly"]["field_map"].items()
            if key != "code"
        }
    ).copy()
    for column in fields["data"]:
        if column not in out:
            out[column] = pd.NA
    out["code"] = out["code"].fillna(code).astype(str)
    out["year"] = year
    out["quarter"] = quarter
    for column in ["publish_date", "report_date"]:
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.date
    for column in fields["numeric"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[fields["data"]]


def missing_baostock_ranges(
    target_path: str | Path,
    start_date: str,
    end_date: str,
    date_column: str = "date",
) -> list[tuple[str, str]]:
    """Return missing ranges before and after the locally covered dates."""
    path = Path(target_path)
    if not path.exists():
        return [(start_date, end_date)]
    existing = pd.read_parquet(path, columns=[date_column])
    if existing.empty:
        return [(start_date, end_date)]
    first_date = pd.to_datetime(existing[date_column]).min()
    last_date = pd.to_datetime(existing[date_column]).max()
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    ranges = []
    if requested_start < first_date:
        previous_date = first_date - pd.Timedelta(days=1)
        ranges.append(
            (start_date, min(requested_end, previous_date).strftime("%Y-%m-%d"))
        )
    next_date = last_date + pd.Timedelta(days=1)
    if next_date <= requested_end:
        ranges.append((max(requested_start, next_date).strftime("%Y-%m-%d"), end_date))
    return ranges


def atomic_write_parquet(
    data: pd.DataFrame, target_path: str | Path, overwrite: bool = False
) -> Path:
    """Write a parquet file through a temporary path, then atomically replace."""
    path = Path(target_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    data.to_parquet(tmp_path, index=False, compression="zstd")
    os.replace(tmp_path, path)
    return path


def request_count_today(
    request_log_path: str | Path,
    today: date | str | None = None,
) -> int:
    reset_request_log(request_log_path, today)
    path = Path(request_log_path)
    if not path.exists():
        return 0
    log = pd.read_csv(path)
    return 0 if log.empty else len(log)


def reset_request_log(
    request_log_path: str | Path,
    today: date | str | None = None,
) -> None:
    path = Path(request_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    today_str = (
        (today or date.today()).isoformat() if not isinstance(today, str) else today
    )
    if not path.exists():
        pd.DataFrame(columns=_fields["request_log"]).to_csv(path, index=False)
        return
    log = pd.read_csv(path)
    if log.empty or "date" not in log.columns:
        pd.DataFrame(columns=_fields["request_log"]).to_csv(path, index=False)
        return
    today_log = log[log["date"].astype(str) == today_str]
    if len(today_log) != len(log):
        today_log.to_csv(path, index=False)


def append_request_log(
    request_log_path: str | Path,
    endpoint: str,
    code: str,
    frequency: str,
    start_date: str,
    end_date: str,
    status: str,
    error_code: str = "",
    error_msg: str = "",
) -> None:
    path = Path(request_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    row = pd.DataFrame(
        [
            {
                "date": now.date().isoformat(),
                "time": now.strftime("%H:%M:%S"),
                "endpoint": endpoint,
                "code": code,
                "frequency": frequency,
                "start_date": start_date,
                "end_date": end_date,
                "status": status,
                "error_code": error_code,
                "error_msg": error_msg,
            }
        ]
    )
    header = not path.exists()
    row.to_csv(path, mode="a", header=header, index=False)


def create_download_lock(data_root: str | Path = "data") -> Path:
    paths = init_data_storage(data_root)
    if paths.lock_path.exists():
        message = f"BaoStock download lock exists: {paths.lock_path}"
        try:
            owner_pid = int(paths.lock_path.read_text(encoding="utf-8"))
        except ValueError:
            raise RuntimeError(message) from None
        if owner_pid <= 0:
            raise RuntimeError(message)
        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            paths.lock_path.unlink()
        except PermissionError:
            raise RuntimeError(message) from None
        else:
            raise RuntimeError(message)
    paths.lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return paths.lock_path


def remove_download_lock(data_root: str | Path = "data") -> None:
    lock_path = DataPaths(Path(data_root)).lock_path
    if lock_path.exists():
        lock_path.unlink()


def merge_history_data(data: pd.DataFrame, target_path: str | Path) -> pd.DataFrame:
    path = Path(target_path)
    if not path.exists():
        return data
    out = pd.concat([pd.read_parquet(path), data], ignore_index=True)
    keys = ["date"] + (["time"] if "time" in out else [])
    return (
        out.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    )


def update_dividends(
    codes: Iterable[str],
    start_year: int,
    end_year: int,
    data_root: str | Path = "data",
    max_requests_per_day: int = BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Download BaoStock dividends by operating year, skipping queried code-years."""
    fields = _fields["dividend"]
    codes = list(codes)
    paths = init_data_storage(data_root)
    dividend_path = paths.dividend_path
    query_cache_path = paths.dividend_queries_path
    dividends = pd.read_parquet(dividend_path) if dividend_path.exists() else None
    query_cache = (
        pd.read_parquet(query_cache_path)
        if query_cache_path.exists()
        else pd.DataFrame(columns=fields["query"])
    )
    queried = set(query_cache.itertuples(index=False, name=None))
    effective_limit = validate_request_limit(max_requests_per_day)
    context = None if client is not None else BaostockClient()
    active_client = client if client is not None else context.__enter__()
    results = []
    pending_dividends = []
    pending_queries = []
    remaining = {
        code: sum(
            (code, year) not in queried for year in range(start_year, end_year + 1)
        )
        for code in codes
    }
    completed = sum(count == 0 for count in remaining.values())
    if progress is not None:
        progress(completed, len(codes))
    create_download_lock(data_root)
    try:
        for code in codes:
            for year in range(start_year, end_year + 1):
                if (code, year) in queried:
                    continue
                if max_tasks is not None and len(results) >= max_tasks:
                    return pd.DataFrame(results, columns=fields["result"])
                if request_count_today(paths.request_log_path) >= effective_limit:
                    return pd.DataFrame(results, columns=fields["result"])
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
                try:
                    data = query_baostock_dividends(code, year, active_client)
                    data = clean_baostock_dividends(data, code, year)
                    if not data.empty:
                        pending_dividends.append(data)
                    pending_queries.append((code, year))
                    queried.add((code, year))
                    append_request_log(
                        paths.request_log_path,
                        "query_dividend_data",
                        code,
                        "dividend",
                        str(year),
                        str(year),
                        "success",
                    )
                    results.append((code, year, "success", len(data), ""))
                    remaining[code] -= 1
                    if remaining[code] == 0:
                        completed += 1
                        if progress is not None:
                            progress(completed, len(codes))
                except Exception as exc:
                    append_request_log(
                        paths.request_log_path,
                        "query_dividend_data",
                        code,
                        "dividend",
                        str(year),
                        str(year),
                        "failed",
                        exc.__class__.__name__,
                        str(exc),
                    )
                    results.append((code, year, "failed", 0, str(exc)))
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
            if pending_dividends:
                new_data = pd.concat(pending_dividends, ignore_index=True)
                dividends = (
                    new_data if dividends is None else pd.concat([dividends, new_data])
                )
                dividends = (
                    dividends.drop_duplicates()
                    .sort_values(["code", "year"])
                    .reset_index(drop=True)
                )
                atomic_write_parquet(dividends, dividend_path, overwrite=True)
                pending_dividends.clear()
            if pending_queries:
                query_cache = pd.concat(
                    [
                        query_cache,
                        pd.DataFrame(pending_queries, columns=fields["query"]),
                    ],
                    ignore_index=True,
                ).drop_duplicates()
                atomic_write_parquet(query_cache, query_cache_path, overwrite=True)
                pending_queries.clear()
        return pd.DataFrame(results, columns=fields["result"])
    finally:
        if pending_dividends:
            new_data = pd.concat(pending_dividends, ignore_index=True)
            dividends = (
                new_data if dividends is None else pd.concat([dividends, new_data])
            )
            dividends = (
                dividends.drop_duplicates()
                .sort_values(["code", "year"])
                .reset_index(drop=True)
            )
            atomic_write_parquet(dividends, dividend_path, overwrite=True)
        if pending_queries:
            query_cache = pd.concat(
                [
                    query_cache,
                    pd.DataFrame(pending_queries, columns=fields["query"]),
                ],
                ignore_index=True,
            ).drop_duplicates()
            atomic_write_parquet(query_cache, query_cache_path, overwrite=True)
        remove_download_lock(data_root)
        if context is not None:
            context.__exit__(None, None, None)


def update_profit_quarterly(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    data_root: str | Path = "data",
    max_requests_per_day: int = BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Download total shares for every quarter overlapping a date range."""
    fields = _fields["profit_quarterly"]
    codes = list(codes)
    paths = init_data_storage(data_root)
    profit_path = paths.profit_path
    query_cache_path = paths.profit_queries_path
    profits = pd.read_parquet(profit_path) if profit_path.exists() else None
    query_cache = (
        pd.read_parquet(query_cache_path)
        if query_cache_path.exists()
        else pd.DataFrame(columns=fields["query"])
    )
    queried = set(query_cache.itertuples(index=False, name=None))
    effective_limit = validate_request_limit(max_requests_per_day)
    context = None if client is not None else BaostockClient()
    active_client = client if client is not None else context.__enter__()
    results = []
    pending_profits = []
    pending_queries = []
    periods = [
        (period.year, period.quarter)
        for period in pd.period_range(start_date, end_date, freq="Q")
    ]
    remaining = {
        code: sum((code, *period) not in queried for period in periods)
        for code in codes
    }
    completed = sum(count == 0 for count in remaining.values())
    if progress is not None:
        progress(completed, len(codes))
    create_download_lock(data_root)
    try:
        for code in codes:
            for year, quarter in periods:
                if (code, year, quarter) in queried:
                    continue
                if max_tasks is not None and len(results) >= max_tasks:
                    return pd.DataFrame(results, columns=fields["result"])
                if request_count_today(paths.request_log_path) >= effective_limit:
                    return pd.DataFrame(results, columns=fields["result"])
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
                try:
                    data = clean_baostock_profit(
                        query_baostock_profit(code, year, quarter, active_client),
                        code,
                        year,
                        quarter,
                    )
                    if not data.empty:
                        pending_profits.append(data)
                    pending_queries.append((code, year, quarter))
                    queried.add((code, year, quarter))
                    append_request_log(
                        paths.request_log_path,
                        "query_profit_data",
                        code,
                        "profit_quarterly",
                        str(year),
                        str(quarter),
                        "success",
                    )
                    results.append((code, year, quarter, "success", len(data), ""))
                    remaining[code] -= 1
                    if remaining[code] == 0:
                        completed += 1
                        if progress is not None:
                            progress(completed, len(codes))
                except Exception as exc:
                    append_request_log(
                        paths.request_log_path,
                        "query_profit_data",
                        code,
                        "profit_quarterly",
                        str(year),
                        str(quarter),
                        "failed",
                        exc.__class__.__name__,
                        str(exc),
                    )
                    results.append((code, year, quarter, "failed", 0, str(exc)))
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
            if pending_profits:
                new_data = pd.concat(pending_profits, ignore_index=True)
                profits = (
                    new_data if profits is None else pd.concat([profits, new_data])
                )
                profits = profits.drop_duplicates(
                    ["code", "year", "quarter"], keep="last"
                )
                profits = profits.sort_values(["code", "year", "quarter"]).reset_index(
                    drop=True
                )
                atomic_write_parquet(profits, profit_path, overwrite=True)
                pending_profits.clear()
            if pending_queries:
                query_cache = pd.concat(
                    [
                        query_cache,
                        pd.DataFrame(pending_queries, columns=fields["query"]),
                    ],
                    ignore_index=True,
                ).drop_duplicates()
                atomic_write_parquet(query_cache, query_cache_path, overwrite=True)
                pending_queries.clear()
        return pd.DataFrame(results, columns=fields["result"])
    finally:
        if pending_profits:
            new_data = pd.concat(pending_profits, ignore_index=True)
            profits = new_data if profits is None else pd.concat([profits, new_data])
            profits = profits.drop_duplicates(["code", "year", "quarter"], keep="last")
            profits = profits.sort_values(["code", "year", "quarter"]).reset_index(
                drop=True
            )
            atomic_write_parquet(profits, profit_path, overwrite=True)
        if pending_queries:
            query_cache = pd.concat(
                [
                    query_cache,
                    pd.DataFrame(pending_queries, columns=fields["query"]),
                ],
                ignore_index=True,
            ).drop_duplicates()
            atomic_write_parquet(query_cache, query_cache_path, overwrite=True)
        remove_download_lock(data_root)
        if context is not None:
            context.__exit__(None, None, None)


def update_history_dataset(
    dataset: str,
    frequency: str,
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    data_root: str | Path = "data",
    max_requests_per_day: int = BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    adjustflag: str | None = None,
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Check and update each security's locally missing date ranges."""
    fields = _fields["history"]
    codes = list(codes)
    paths = init_data_storage(data_root)
    adjustment = baostock_adjustflag(adjustflag)
    effective_limit = validate_request_limit(max_requests_per_day)
    context = None if client is not None else BaostockClient()
    active_client = client if client is not None else context.__enter__()
    results = []
    completed = 0
    tasks = 0
    if progress is not None:
        progress(0, len(codes))
    create_download_lock(data_root)
    try:
        for code in codes:
            if checkpoint is not None and not checkpoint():
                return pd.DataFrame(results, columns=fields["result"])
            slices = []
            if frequency == "d":
                target = daily_target_path(code, dataset, data_root, adjustment)
                slices.extend(
                    (first, last, str(target))
                    for first, last in missing_baostock_ranges(
                        target, start_date, end_date
                    )
                )
            else:
                for year in range(
                    pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1
                ):
                    first = max(
                        pd.Timestamp(start_date), pd.Timestamp(f"{year}-01-01")
                    ).strftime("%Y-%m-%d")
                    last = min(
                        pd.Timestamp(end_date), pd.Timestamp(f"{year}-12-31")
                    ).strftime("%Y-%m-%d")
                    target = minute_5_target_path(
                        code, year, data_root, adjustment
                    )
                    slices.extend(
                        (range_start, range_end, str(target))
                        for range_start, range_end in missing_baostock_ranges(
                            target, first, last
                        )
                    )

            code_complete = True
            for range_start, range_end, target_path in slices:
                if max_tasks is not None and tasks >= max_tasks:
                    return pd.DataFrame(results, columns=fields["result"])
                if request_count_today(paths.request_log_path) >= effective_limit:
                    return pd.DataFrame(results, columns=fields["result"])
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
                tasks += 1
                try:
                    data = query_baostock_history(
                        code,
                        range_start,
                        range_end,
                        fields["daily"] if frequency == "d" else fields["minute_5"],
                        frequency,
                        adjustment,
                        active_client,
                    )
                    data = merge_history_data(clean_baostock_data(data), target_path)
                    atomic_write_parquet(data, target_path, overwrite=True)
                    append_request_log(
                        paths.request_log_path,
                        "query_history_k_data_plus",
                        code,
                        frequency,
                        range_start,
                        range_end,
                        "success",
                    )
                    results.append(
                        (
                            code,
                            range_start,
                            range_end,
                            target_path,
                            "success",
                            len(data),
                            "",
                        )
                    )
                except Exception as exc:
                    code_complete = False
                    append_request_log(
                        paths.request_log_path,
                        "query_history_k_data_plus",
                        code,
                        frequency,
                        range_start,
                        range_end,
                        "failed",
                        exc.__class__.__name__,
                        str(exc),
                    )
                    results.append(
                        (
                            code,
                            range_start,
                            range_end,
                            target_path,
                            "failed",
                            0,
                            str(exc),
                        )
                    )
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
            if code_complete:
                completed += 1
                if progress is not None:
                    progress(completed, len(codes))
        return pd.DataFrame(results, columns=fields["result"])
    finally:
        remove_download_lock(data_root)
        if context is not None:
            context.__exit__(None, None, None)


def validate_request_limit(max_requests_per_day: int) -> int:
    """Validate the user safety threshold against BaoStock's hard limit."""
    if max_requests_per_day <= 0:
        raise ValueError("max_requests_per_day must be positive")
    if max_requests_per_day > BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY:
        raise ValueError(
            "max_requests_per_day exceeds BaoStock hard limit "
            f"{BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY}: {max_requests_per_day}"
        )
    return min(max_requests_per_day, BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY)


def update_dataset(
    name: str,
    *,
    start: str,
    pool: str | Collection[str],
    end: str | None = None,
    pool_date: str | None = None,
    adjustment: str | None = None,
    max_tasks: int | None = None,
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    data_root: str | Path = "data",
) -> pd.DataFrame:
    """Update a named catalog dataset through its current source."""
    dataset = get_dataset(name)
    update = dataset.get("update")
    if update is None:
        raise ValueError(f"Dataset {name!r} is read-only")
    end_date = end or date.today().isoformat()
    if pd.Timestamp(start) > pd.Timestamp(end_date):
        raise ValueError("start must not be after end")
    if isinstance(pool, str) and not update["pool"]:
        raise ValueError(f"Dataset {name!r} does not support named pools")
    if adjustment is not None and not update["adjustment"]:
        raise ValueError(f"Dataset {name!r} does not support adjustment")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    if checkpoint is not None and not checkpoint():
        return pd.DataFrame(columns=_fields[update["kind"]]["result"])

    context = None if client is not None else BaostockClient()
    client = client if client is not None else context.__enter__()
    try:
        codes = (
            resolve_baostock_codes(pool, pool_date or end_date, client)
            if isinstance(pool, str)
            else list(dict.fromkeys(pool))
        )
        if not codes:
            raise ValueError("No security codes were selected")
        common = {"client": client}
        if Path(data_root) != Path("data"):
            common["data_root"] = data_root
        if checkpoint is not None:
            common["checkpoint"] = checkpoint
        if progress is not None:
            common["progress"] = progress
        if max_tasks is not None:
            common["max_tasks"] = max_tasks
        if update["kind"] == "history":
            if adjustment is not None:
                common["adjustflag"] = adjustment
            return update_history_dataset(
                update["target"],
                update["frequency"],
                codes,
                start,
                end_date,
                **common,
            )
        if update["kind"] == "dividend":
            return update_dividends(
                codes,
                pd.Timestamp(start).year,
                pd.Timestamp(end_date).year,
                **common,
            )
        return update_profit_quarterly(codes, start, end_date, **common)
    finally:
        if context is not None:
            context.__exit__(None, None, None)


def resolve_baostock_codes(
    pool: str,
    trade_date: str,
    client: Any,
) -> list[str]:
    """Resolve a pool on its latest available trading day."""
    if pool == "all":
        data = baostock_result_to_frame(client.bs.query_stock_basic())
        if not {"code", "type"}.issubset(data.columns):
            raise ValueError(
                f"BaoStock stock-basic result has unexpected columns: {list(data.columns)}"
            )
        return (
            data.loc[data["type"].astype(str) == "1", "code"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
    if pool not in BAOSTOCK_STOCK_POOL_QUERIES:
        raise ValueError(f"Unsupported BaoStock stock pool: {pool}")
    query = getattr(client.bs, BAOSTOCK_STOCK_POOL_QUERIES[pool])

    end = pd.Timestamp(trade_date)
    calendar = baostock_result_to_frame(
        client.bs.query_trade_dates(
            (end - pd.Timedelta(days=14)).strftime("%Y-%m-%d"), trade_date
        )
    )
    if not {"calendar_date", "is_trading_day"}.issubset(calendar.columns):
        raise ValueError(
            f"BaoStock trade calendar has unexpected columns: {list(calendar.columns)}"
        )
    for day in calendar.loc[
        calendar["is_trading_day"].astype(str) == "1", "calendar_date"
    ].iloc[::-1]:
        data = baostock_result_to_frame(query(str(day)))
        if "code" not in data.columns:
            raise ValueError(
                f"BaoStock pool result has no code column: {list(data.columns)}"
            )
        codes = data["code"].dropna().astype(str).drop_duplicates().tolist()
        if codes:
            return codes
    return []


def baostock_result_to_frame(result: Any) -> pd.DataFrame:
    if getattr(result, "error_code", "0") != "0":
        raise RuntimeError(
            f"BaoStock query failed: {result.error_code} {result.error_msg}"
        )
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def query_baostock_history(
    code: str,
    start_date: str,
    end_date: str,
    fields: list[str],
    frequency: str,
    adjustflag: str,
    client: Any,
) -> pd.DataFrame:
    """Query BaoStock history and convert its cursor-like result to DataFrame."""
    result = client.bs.query_history_k_data_plus(
        code,
        ",".join(fields),
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        adjustflag=adjustflag,
    )
    if getattr(result, "error_code", "0") != "0":
        raise RuntimeError(
            f"BaoStock query failed: {result.error_code} {result.error_msg}"
        )

    return baostock_result_to_frame(result)


def query_baostock_dividends(code: str, year: int, client: Any) -> pd.DataFrame:
    """Query BaoStock dividends by operating year."""
    return baostock_result_to_frame(
        client.bs.query_dividend_data(code, str(year), yearType="operate")
    )


def query_baostock_profit(
    code: str,
    year: int,
    quarter: int,
    client: Any,
) -> pd.DataFrame:
    """Query BaoStock quarterly profit data."""
    return baostock_result_to_frame(
        client.bs.query_profit_data(code, str(year), str(quarter))
    )

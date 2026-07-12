"""BaoStock download helpers.

This module keeps BaoStock-specific behavior at the data-source boundary.
Tests should use a fake client and must not access the network.
"""

from __future__ import annotations

import os
import select
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd


DAILY_FIELDS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "turn",
    "tradestatus",
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "isST",
]
MINUTE_5_FIELDS = [
    "date",
    "time",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adjustflag",
]
SLICE_COLUMNS = ["code", "start_date", "end_date", "target_path"]
RESULT_COLUMNS = SLICE_COLUMNS + ["status", "row_count", "error"]
REQUEST_LOG_COLUMNS = [
    "date",
    "time",
    "endpoint",
    "code",
    "frequency",
    "start_date",
    "end_date",
    "status",
    "rows",
    "error_code",
    "error_msg",
]
BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY = 50_000
BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY = 49_000
BAOSTOCK_STOCK_POOL_QUERIES = {
    "sz50": "query_sz50_stocks",
    "hs300": "query_hs300_stocks",
    "zz500": "query_zz500_stocks",
}
ADJUSTMENT_FLAGS = {"forward": "2", "backward": "1", "none": "3"}
ADJUSTMENT_DIRS = {value: key for key, value in ADJUSTMENT_FLAGS.items()}
FLOAT32_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "turn",
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
}


@dataclass(frozen=True)
class BaostockPaths:
    raw_root: Path

    @property
    def daily_stock_dir(self) -> Path:
        return self.raw_root / "daily" / "stock"

    @property
    def daily_index_dir(self) -> Path:
        return self.raw_root / "daily" / "index"

    @property
    def minute_5_stock_dir(self) -> Path:
        return self.raw_root / "minute_5" / "stock"

    @property
    def state_dir(self) -> Path:
        return self.raw_root / "state"

    @property
    def request_log_path(self) -> Path:
        return self.state_dir / "request_log.csv"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "download.lock"


class BaostockClient:
    """Thin BaoStock client wrapper with lazy import."""

    def __init__(self) -> None:
        try:
            import baostock as bs
        except ImportError as exc:
            raise ImportError("BaoStock download requires package 'baostock'.") from exc
        self.bs = bs
        self.logged_in = False

    def __enter__(self) -> "BaostockClient":
        result = self.bs.login()
        if getattr(result, "error_code", "0") != "0":
            raise RuntimeError(
                f"BaoStock login failed: {result.error_code} {result.error_msg}"
            )
        self.logged_in = True
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.logged_in:
            self.bs.logout()
            self.logged_in = False

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        start_date: str,
        end_date: str,
        frequency: str,
        adjustflag: str,
    ) -> Any:
        return self.bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjustflag,
        )

    def query_hs300_stocks(self, trade_date: str) -> Any:
        return self.bs.query_hs300_stocks(trade_date)

    def query_sz50_stocks(self, trade_date: str) -> Any:
        return self.bs.query_sz50_stocks(trade_date)

    def query_zz500_stocks(self, trade_date: str) -> Any:
        return self.bs.query_zz500_stocks(trade_date)

    def query_all_stock(self, trade_date: str) -> Any:
        return self.bs.query_all_stock(trade_date)

    def query_trade_dates(self, start_date: str, end_date: str) -> Any:
        return self.bs.query_trade_dates(start_date, end_date)


class StdinDownloadControl:
    """Keyboard control using p=pause, c=continue, q=quit between requests."""

    def __init__(
        self,
        pause_key: str = "p",
        resume_key: str = "c",
        quit_key: str = "q",
        poll_timeout: float = 0.0,
        sleep_seconds: float = 0.2,
        output: Callable[[str], None] = print,
    ) -> None:
        self.pause_key = pause_key
        self.resume_key = resume_key
        self.quit_key = quit_key
        self.poll_timeout = poll_timeout
        self.sleep_seconds = sleep_seconds
        self.output = output
        self.paused = False

    def before_request(self) -> bool:
        return self._handle_command(block_when_paused=True)

    def after_request(self) -> bool:
        return self._handle_command(block_when_paused=False)

    def _handle_command(
        self,
        block_when_paused: bool,
    ) -> bool:
        command = self._read_command(block=False)
        if command == self.pause_key:
            self.paused = True
            self.output(f"Paused. Press '{self.resume_key}' to resume or '{self.quit_key}' to quit.")
        if command == self.quit_key:
            self.output("Quit requested. Downloaded data has been saved.")
            return False

        while self.paused and block_when_paused:
            command = self._read_command(block=True)
            if command == self.resume_key:
                self.paused = False
                self.output("Resumed.")
                return True
            if command == self.quit_key:
                self.output("Quit requested. Downloaded data has been saved.")
                return False
            time.sleep(self.sleep_seconds)
        return True

    def _read_command(self, block: bool) -> Optional[str]:
        if block:
            line = sys.stdin.readline()
            return line.strip()[:1] if line else None
        ready, _, _ = select.select([sys.stdin], [], [], self.poll_timeout)
        if not ready:
            return None
        line = sys.stdin.readline()
        return line.strip()[:1] if line else None


def baostock_paths(raw_root: str | Path = "data/raw/baostock") -> BaostockPaths:
    return BaostockPaths(Path(raw_root))


def init_baostock_storage(raw_root: str | Path = "data/raw/baostock") -> BaostockPaths:
    """Create the BaoStock raw-data directory skeleton."""
    paths = baostock_paths(raw_root)
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
    raw_root: str | Path = "data/raw/baostock",
    adjustflag: str | None = None,
) -> Path:
    paths = baostock_paths(raw_root)
    if dataset == "stock":
        return paths.daily_stock_dir / adjustment_dir(adjustflag) / f"{code}.parquet"
    if dataset == "index":
        return paths.daily_index_dir / f"{code}.parquet"
    raise ValueError(f"Unsupported daily dataset: {dataset}")


def minute_5_target_path(
    code: str,
    year: int,
    raw_root: str | Path = "data/raw/baostock",
    adjustflag: str | None = None,
) -> Path:
    paths = baostock_paths(raw_root)
    return paths.minute_5_stock_dir / adjustment_dir(adjustflag) / code / f"{year}.parquet"


def adjustment_dir(adjustflag: str | None) -> str:
    adjustflag = baostock_adjustflag(adjustflag)
    try:
        return ADJUSTMENT_DIRS[adjustflag]
    except KeyError as exc:
        raise ValueError(f"Unsupported BaoStock adjustflag: {adjustflag}") from exc


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
    for column in FLOAT32_COLUMNS & set(out.columns):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float32")
    if "amount" in out:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    if "isST" in out:
        out["isST"] = pd.to_numeric(out["isST"], errors="coerce").astype("boolean")
    if "volume" in out:
        column = "volume"
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    return out.drop(columns=["code", "tradestatus", "adjustflag"], errors="ignore")


def next_start_date(
    target_path: str | Path,
    default_start_date: str,
    date_column: str = "date",
) -> Optional[str]:
    """Return the next calendar date after an existing file's max date."""
    path = Path(target_path)
    if not path.exists():
        return default_start_date
    existing = pd.read_parquet(path, columns=[date_column])
    if existing.empty:
        return default_start_date
    last_date = pd.to_datetime(existing[date_column]).max()
    return (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


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
    today_str = (today or date.today()).isoformat() if not isinstance(today, str) else today
    if not path.exists():
        pd.DataFrame(columns=REQUEST_LOG_COLUMNS).to_csv(path, index=False)
        return
    log = pd.read_csv(path)
    if log.empty or "date" not in log.columns:
        pd.DataFrame(columns=REQUEST_LOG_COLUMNS).to_csv(path, index=False)
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
    rows: int,
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
                "rows": rows,
                "error_code": error_code,
                "error_msg": error_msg,
            }
        ]
    )
    header = not path.exists()
    row.to_csv(path, mode="a", header=header, index=False)


def create_download_lock(raw_root: str | Path = "data/raw/baostock") -> Path:
    paths = init_baostock_storage(raw_root)
    if paths.lock_path.exists():
        raise RuntimeError(f"BaoStock download lock exists: {paths.lock_path}")
    paths.lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return paths.lock_path


def remove_download_lock(raw_root: str | Path = "data/raw/baostock") -> None:
    lock_path = baostock_paths(raw_root).lock_path
    if lock_path.exists():
        lock_path.unlink()


def build_baostock_slices(
    dataset: str,
    frequency: str,
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    raw_root: str | Path = "data/raw/baostock",
    adjustflag: str | None = None,
) -> pd.DataFrame:
    """Build only the date ranges not already covered by local files."""
    adjustment = baostock_adjustflag(adjustflag)
    rows = []
    for code in codes:
        if dataset in {"stock", "index"} and frequency == "d":
            target = daily_target_path(code, dataset, raw_root, adjustment)
            next_date = next_start_date(target, start_date)
            if next_date and next_date <= end_date:
                rows.append((code, next_date, end_date, str(target)))
        elif dataset == "stock" and frequency == "5":
            for year in range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1):
                first = max(pd.Timestamp(start_date), pd.Timestamp(f"{year}-01-01")).strftime("%Y-%m-%d")
                last = min(pd.Timestamp(end_date), pd.Timestamp(f"{year}-12-31")).strftime("%Y-%m-%d")
                target = minute_5_target_path(code, year, raw_root, adjustment)
                next_date = next_start_date(target, first)
                if next_date and next_date <= last:
                    rows.append((code, next_date, last, str(target)))
        else:
            raise ValueError(f"Unsupported BaoStock dataset/frequency: {dataset}/{frequency}")
    return pd.DataFrame(rows, columns=SLICE_COLUMNS)


def merge_baostock_data(data: pd.DataFrame, target_path: str | Path) -> pd.DataFrame:
    path = Path(target_path)
    if not path.exists():
        return data
    out = pd.concat([pd.read_parquet(path), data], ignore_index=True)
    keys = ["date"] + (["time"] if "time" in out else [])
    return out.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def update_baostock_dataset(
    dataset: str,
    frequency: str,
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    raw_root: str | Path = "data/raw/baostock",
    max_requests_per_day: int = BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    adjustflag: str | None = None,
    client: Optional[Any] = None,
    control: StdinDownloadControl | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Download only data not already covered by local parquet files."""
    codes = list(codes)
    paths = init_baostock_storage(raw_root)
    slices = build_baostock_slices(
        dataset, frequency, codes, start_date, end_date, raw_root, adjustflag
    )
    create_download_lock(raw_root)
    try:
        return run_baostock_slices(
            slices,
            paths,
            frequency,
            baostock_adjustflag(adjustflag),
            max_requests_per_day,
            client,
            control,
            total_codes=len(codes),
            progress=progress,
        )
    finally:
        remove_download_lock(raw_root)


def run_baostock_slices(
    slices: pd.DataFrame,
    paths: BaostockPaths,
    frequency: str,
    adjustflag: str,
    max_requests_per_day: int = BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    client: Optional[Any] = None,
    control: StdinDownloadControl | None = None,
    total_codes: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Download local-data gaps serially and return this run's results."""
    effective_limit = validate_request_limit(max_requests_per_day)
    active_client = client
    context = None if client is not None else BaostockClient()
    if context is not None:
        active_client = context.__enter__()
    results = []
    remaining = slices["code"].value_counts().to_dict()
    total = total_codes if total_codes is not None else len(remaining)
    completed = total - len(remaining)
    if progress is not None:
        progress(completed, total)
    try:
        for item in slices.itertuples(index=False):
            if request_count_today(paths.request_log_path) >= effective_limit:
                break
            if control is not None and not control.before_request():
                break
            try:
                data = query_baostock_history(
                    item.code,
                    item.start_date,
                    item.end_date,
                    DAILY_FIELDS if frequency == "d" else MINUTE_5_FIELDS,
                    frequency,
                    adjustflag,
                    active_client,
                )
                data = merge_baostock_data(clean_baostock_data(data), item.target_path)
                atomic_write_parquet(data, item.target_path, overwrite=True)
                append_request_log(
                    paths.request_log_path,
                    "query_history_k_data_plus",
                    item.code,
                    frequency,
                    item.start_date,
                    item.end_date,
                    "success",
                    len(data),
                )
                results.append((*item, "success", len(data), ""))
                remaining[item.code] -= 1
                if remaining[item.code] == 0:
                    completed += 1
                    if progress is not None:
                        progress(completed, total)
            except Exception as exc:
                append_request_log(
                    paths.request_log_path,
                    "query_history_k_data_plus",
                    item.code,
                    frequency,
                    item.start_date,
                    item.end_date,
                    "failed",
                    0,
                    exc.__class__.__name__,
                    str(exc),
                )
                results.append((*item, "failed", 0, str(exc)))
            if control is not None and not control.after_request():
                break
        return pd.DataFrame(results, columns=RESULT_COLUMNS)
    finally:
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


def resolve_baostock_codes(
    pool: Optional[str],
    trade_date: str,
    client: Any,
) -> list[str]:
    """Resolve a pool on its latest available trading day."""
    if pool == "all":
        query = client.query_all_stock
    elif pool in BAOSTOCK_STOCK_POOL_QUERIES:
        query = getattr(client, BAOSTOCK_STOCK_POOL_QUERIES[pool])
    else:
        raise ValueError(f"Unsupported BaoStock stock pool: {pool}")

    end = pd.Timestamp(trade_date)
    calendar = baostock_result_to_frame(
        client.query_trade_dates((end - pd.Timedelta(days=14)).strftime("%Y-%m-%d"), trade_date)
    )
    if not {"calendar_date", "is_trading_day"}.issubset(calendar.columns):
        raise ValueError(f"BaoStock trade calendar has unexpected columns: {list(calendar.columns)}")
    for day in calendar.loc[calendar["is_trading_day"].astype(str) == "1", "calendar_date"].iloc[::-1]:
        data = baostock_result_to_frame(query(str(day)))
        if "code" not in data.columns:
            raise ValueError(f"BaoStock pool result has no code column: {list(data.columns)}")
        codes = data["code"].dropna().astype(str).drop_duplicates().tolist()
        if codes:
            return codes
    return []


def baostock_result_to_frame(result: Any) -> pd.DataFrame:
    if getattr(result, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock query failed: {result.error_code} {result.error_msg}")
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
    result = client.query_history_k_data_plus(
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

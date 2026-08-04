"""Dataset update orchestration."""

from __future__ import annotations

import os
from collections.abc import Callable, Collection, Iterable, Mapping
from contextvars import copy_context
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import Condition, Thread
from typing import Any

import pandas as pd

from pyquant.data.catalog import get_dataset_spec
from pyquant.data.duckdb import connect_database, get_database_path, initialize_database
from pyquant.data.identifiers import normalize_security_symbol
from pyquant.data.intraday import (
    MINUTE_DAY_INCOMPLETE,
    MINUTE_DAY_NO_DATA_CONFIRMED,
    MINUTE_DAY_VALID,
    calculate_daily_intraday_volatility,
    normalize_minute_bars,
)
from pyquant.data.resources import load_source_protocols
from pyquant.data.sources.baostock import (
    BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    BaostockClient,
    append_request_log,
    clean_baostock_data,
    clean_baostock_dividends,
    clean_baostock_profit,
    normalize_baostock_code,
    query_baostock_dividends,
    query_baostock_history,
    query_baostock_profit,
    request_count_today,
    reset_request_log,
    resolve_baostock_codes,
    validate_request_limit,
)
from pyquant.data.sources.csindex import (
    clean_csindex_history,
    query_csindex_history,
)
from pyquant.data.sources.rqdata import (
    query_index_constituents,
    query_rqdata_quota_remaining,
    query_rqdata_stock_symbols,
    query_rqdata_trading_dates,
    query_stock_pb_daily,
    query_stock_minute_1m,
)
from pyquant.data.store import (
    BAOSTOCK_INDEX_DAILY_FIELD_SET_ID,
    CSINDEX_DAILY_FIELD_SET_ID,
    MINUTE_TASK_RUNNING,
    completed_minute_days,
    create_minute_download_task,
    dividend_coverage,
    index_daily_coverage,
    share_capital_coverage,
    stock_pb_coverage,
    stock_daily_coverage,
    recover_minute_download_tasks,
    update_minute_download_task,
    write_dividend_request,
    write_index_constituents,
    write_index_daily_request,
    write_share_capital_request,
    write_minute_request,
    write_minute_request_failure,
    write_stock_daily_request,
    write_stock_pb_request,
)

_fields = load_source_protocols()["baostock"]
_csindex = load_source_protocols()["csindex"]
_rqdata_minute = load_source_protocols()["rqdata"]["stock_minute_1m"]
_rqdata_pb = load_source_protocols()["rqdata"]["stock_pb_daily"]
_normalize_baostock_code = normalize_baostock_code


class UpdateJob:
    """A controllable dataset update running in a background thread."""

    def __init__(
        self,
        worker: Callable[
            [Callable[[], bool], Callable[[int, int], None]], pd.DataFrame
        ],
    ) -> None:
        self._condition = Condition()
        self._state = "running"
        self._completed = 0
        self._total = 0
        self._progress_printed = False
        self._progress_handle: Any | None = None
        self._error: Exception | None = None
        self._result: pd.DataFrame
        try:
            from IPython import get_ipython
            from IPython.display import DisplayHandle
        except ImportError:
            pass
        else:
            if get_ipython() is not None:
                self._progress_handle = DisplayHandle()
                self._progress_handle.display({"text/plain": "Updated 0/0"}, raw=True)
        context = copy_context()
        self._thread = Thread(target=context.run, args=(self._run, worker))
        self._thread.start()

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    @property
    def completed(self) -> int:
        with self._condition:
            return self._completed

    @property
    def total(self) -> int:
        with self._condition:
            return self._total

    @property
    def error(self) -> Exception | None:
        with self._condition:
            return self._error

    def pause(self) -> None:
        """Pause before the next remote request."""
        with self._condition:
            if self._state == "running":
                self._state = "paused"

    def resume(self) -> None:
        """Resume a paused update."""
        with self._condition:
            if self._state == "paused":
                self._state = "running"
                self._condition.notify_all()

    def stop(self) -> None:
        """Stop gracefully after the current remote request."""
        with self._condition:
            if self._state not in {"completed", "failed"}:
                self._state = "stopping"
                self._condition.notify_all()

    def wait(self) -> pd.DataFrame:
        """Wait for completion and return results or raise the worker error."""
        self._thread.join()
        with self._condition:
            if self._error is not None:
                raise self._error
            return self._result

    def _run(
        self,
        worker: Callable[
            [Callable[[], bool], Callable[[int, int], None]], pd.DataFrame
        ],
    ) -> None:
        def checkpoint() -> bool:
            with self._condition:
                while self._state == "paused":
                    self._condition.wait()
                return self._state != "stopping"

        def progress(completed: int, total: int) -> None:
            with self._condition:
                self._completed = completed
                self._total = total
            self._show_progress(completed, total)
            self._progress_printed = True

        try:
            result = worker(checkpoint, progress)
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._state = "failed"
                self._condition.notify_all()
        else:
            with self._condition:
                self._result = result
                self._state = "completed"
                self._condition.notify_all()
        finally:
            if self._progress_printed:
                self._show_progress(self._completed, self._total, final=True)

    def _show_progress(self, completed: int, total: int, final: bool = False) -> None:
        message = f"Updated {completed}/{total}"
        if self._progress_handle is not None:
            self._progress_handle.update({"text/plain": message}, raw=True)
        else:
            print(f"\r{message}", end="\n" if final else "", flush=True)


@dataclass(frozen=True)
class DataPaths:
    data_root: Path

    @property
    def database_path(self) -> Path:
        return get_database_path(self.data_root)

    @property
    def state_dir(self) -> Path:
        return self.data_root / "state"

    @property
    def request_log_path(self) -> Path:
        return self.state_dir / "request_log.csv"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "download.lock"


def init_data_storage(data_root: Path = Path("data")) -> DataPaths:
    """Create DuckDB and source-specific state storage."""
    paths = DataPaths(data_root)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    initialize_database(paths.database_path)
    reset_request_log(paths.request_log_path)
    return paths


@dataclass(frozen=True)
class MinuteRequest:
    """One strategy-generated need for a security's minute bars."""

    symbol: str
    start_date: date
    end_date: date
    retain_raw: bool = True

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.start_date).date()
        end = pd.Timestamp(self.end_date).date()
        if start > end:
            raise ValueError("start_date must not be after end_date")
        object.__setattr__(self, "symbol", normalize_security_symbol(self.symbol))
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)


def plan_missing_minute_requests(
    requests: Iterable[MinuteRequest],
    trading_dates: Collection[object],
    completed: Collection[tuple[str, object]] = (),
) -> list[MinuteRequest]:
    """Subtract completed days and merge adjacent trading-day requirements."""
    requests = list(requests)
    if not requests:
        return []
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_dates), errors="raise"))
    if calendar.hasnans:
        raise ValueError("trading_dates must not contain invalid values")
    calendar = calendar.normalize().drop_duplicates().sort_values()
    positions = {value: index for index, value in enumerate(calendar)}
    completed = {
        (normalize_security_symbol(symbol), pd.Timestamp(trade_date).normalize())
        for symbol, trade_date in completed
    }
    needed: dict[str, dict[pd.Timestamp, bool]] = {}
    for request in requests:
        dates = calendar[
            (calendar >= pd.Timestamp(request.start_date))
            & (calendar <= pd.Timestamp(request.end_date))
        ]
        symbol_dates = needed.setdefault(request.symbol, {})
        for trade_date in dates:
            if (request.symbol, trade_date) not in completed:
                symbol_dates[trade_date] = (
                    symbol_dates.get(trade_date, False) or request.retain_raw
                )
    planned = []
    for symbol, values in sorted(needed.items()):
        current: list[object] | None = None
        for trade_date, retain_raw in sorted(values.items()):
            if (
                current is None
                or retain_raw != current[2]
                or positions[trade_date] != positions[current[1]] + 1
            ):
                if current is not None:
                    planned.append(
                        MinuteRequest(symbol, current[0], current[1], current[2])
                    )
                current = [trade_date, trade_date, retain_raw]
            else:
                current[1] = trade_date
        if current is not None:
            planned.append(MinuteRequest(symbol, current[0], current[1], current[2]))
    return planned


def _run_minute_update(
    requests: Iterable[MinuteRequest],
    *,
    data_root: Path = Path("data"),
    client: Any | None = None,
    max_attempts: int = 3,
    quota_reserve_bytes: int = 80_000_000,
    min_bars_per_day: int | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Synchronously fulfill strategy-generated one-minute data requirements."""
    requests = list(requests)
    if not requests:
        raise ValueError("No minute requests were provided")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if quota_reserve_bytes < 0:
        raise ValueError("quota_reserve_bytes must not be negative")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    start = min(request.start_date for request in requests)
    end = max(request.end_date for request in requests)
    calendar = query_rqdata_trading_dates(
        str(start),
        str(end),
        client=client,
    )
    paths = init_data_storage(data_root)
    results = []
    exit_reason = None
    total_tasks = 0
    with connect_database(paths.database_path) as connection:
        recover_minute_download_tasks(connection, max_attempts)
        completed = completed_minute_days(
            connection,
            [request.symbol for request in requests],
        )
        planned = plan_missing_minute_requests(requests, calendar, completed)
        if max_tasks is not None:
            planned = planned[:max_tasks]
        tasks = [
            (
                request,
                create_minute_download_task(
                    connection,
                    request.symbol,
                    str(request.start_date),
                    str(request.end_date),
                    request.retain_raw,
                ),
            )
            for request in planned
        ]
        total_tasks = len(tasks)
        if progress is not None:
            progress(0, len(tasks))
        for completed_tasks, (request, task_id) in enumerate(tasks):
            if checkpoint is not None and not checkpoint():
                exit_reason = "user"
                break
            if quota_reserve_bytes:
                remaining = query_rqdata_quota_remaining(client=client)
                if remaining is not None and remaining < quota_reserve_bytes:
                    exit_reason = (
                        f"quota reserve ({remaining} bytes remaining; "
                        f"reserve: {quota_reserve_bytes})"
                    )
                    break
            task_calendar = calendar[
                (calendar >= pd.Timestamp(request.start_date))
                & (calendar <= pd.Timestamp(request.end_date))
            ]
            attempts = connection.execute(
                """
                SELECT attempts
                FROM meta.minute_download_task
                WHERE task_id = ?
                """,
                [task_id],
            ).fetchone()[0]
            for attempt in range(attempts, max_attempts):
                update_minute_download_task(
                    connection,
                    task_id,
                    MINUTE_TASK_RUNNING,
                    increment_attempts=True,
                )
                try:
                    minute = normalize_minute_bars(
                        query_stock_minute_1m(
                            request.symbol,
                            str(request.start_date),
                            str(request.end_date),
                            client=client,
                        ),
                        request.symbol,
                        str(request.start_date),
                        str(request.end_date),
                    )
                    daily = calculate_daily_intraday_volatility(
                        minute,
                        request.symbol,
                        task_calendar,
                        min_bars_per_day=min_bars_per_day,
                    )
                    write_minute_request(
                        connection,
                        task_id,
                        request.symbol,
                        minute,
                        daily,
                        str(request.start_date),
                        str(request.end_date),
                        retain_raw=request.retain_raw,
                    )
                except Exception as exc:
                    if attempt + 1 < max_attempts:
                        continue
                    write_minute_request_failure(
                        connection,
                        task_id,
                        request.symbol,
                        task_calendar,
                        exc,
                    )
                    results.append(
                        (
                            request.symbol,
                            str(request.start_date),
                            str(request.end_date),
                            "failed",
                            0,
                            0,
                            str(exc),
                        )
                    )
                else:
                    statuses = set(daily["status"])
                    status = (
                        "partial"
                        if MINUTE_DAY_INCOMPLETE in statuses
                        else "no_data"
                        if statuses == {MINUTE_DAY_NO_DATA_CONFIRMED}
                        else "success"
                    )
                    results.append(
                        (
                            request.symbol,
                            str(request.start_date),
                            str(request.end_date),
                            status,
                            len(minute),
                            int(daily["status"].eq(MINUTE_DAY_VALID).sum()),
                            "",
                        )
                    )
                break
            if progress is not None:
                progress(completed_tasks + 1, len(tasks))
    if exit_reason is None:
        print(
            "Minute-data update completed: "
            f"{len(results)}/{total_tasks} tasks processed."
        )
    else:
        print(
            f"Minute-data update stopped by {exit_reason}: "
            f"{len(results)}/{total_tasks} tasks processed."
        )
    return pd.DataFrame(results, columns=_rqdata_minute["result"])


def update_minute_data(
    requests: Iterable[MinuteRequest],
    *,
    data_root: Path = Path("data"),
    max_attempts: int = 3,
    quota_reserve_bytes: int = 80_000_000,
    min_bars_per_day: int | None = None,
    max_tasks: int | None = None,
) -> UpdateJob:
    """Start a controllable background update for one-minute data requests."""
    requests = list(requests)

    def run(
        checkpoint: Callable[[], bool],
        progress: Callable[[int, int], None],
    ) -> pd.DataFrame:
        return _run_minute_update(
            requests,
            data_root=data_root,
            max_attempts=max_attempts,
            quota_reserve_bytes=quota_reserve_bytes,
            min_bars_per_day=min_bars_per_day,
            checkpoint=checkpoint,
            progress=progress,
            max_tasks=max_tasks,
        )

    return UpdateJob(run)


def missing_baostock_ranges(
    start_date: str,
    end_date: str,
    queried_ranges: Iterable[tuple[str, str]] = (),
) -> list[tuple[str, str]]:
    """Return ranges not covered by completed source queries."""
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    cursor = requested_start
    missing = []
    for first, last in sorted(
        (pd.Timestamp(first), pd.Timestamp(last)) for first, last in queried_ranges
    ):
        if last < cursor:
            continue
        if first > requested_end:
            break
        if first > cursor:
            missing.append(
                (
                    cursor.strftime("%Y-%m-%d"),
                    (first - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                )
            )
        cursor = max(cursor, last + pd.Timedelta(days=1))
    if cursor <= requested_end:
        missing.append((cursor.strftime("%Y-%m-%d"), end_date))
    return missing


def create_download_lock(data_root: Path = Path("data")) -> Path:
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


def remove_download_lock(data_root: Path = Path("data")) -> None:
    lock_path = DataPaths(data_root).lock_path
    if lock_path.exists():
        lock_path.unlink()


def update_csindex_daily(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    data_root: Path = Path("data"),
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Download official CSI histories directly into DuckDB."""
    codes = list(codes)
    allowed_codes = get_dataset_spec("csindex_daily").storage.allowed_symbols
    unsupported = sorted(set(codes) - set(allowed_codes))
    if unsupported:
        raise ValueError(f"Unsupported CSI index codes: {unsupported}")
    if progress is not None:
        progress(0, len(codes))
    paths = init_data_storage(data_root)
    connection = connect_database(paths.database_path)
    results = []
    completed = 0
    tasks = 0
    try:
        for code in codes:
            if checkpoint is not None and not checkpoint():
                break
            queried = index_daily_coverage(
                connection,
                code,
                CSINDEX_DAILY_FIELD_SET_ID,
            )
            for range_start, range_end in missing_baostock_ranges(
                start_date,
                end_date,
                queried_ranges=queried,
            ):
                if max_tasks is not None and tasks >= max_tasks:
                    return pd.DataFrame(results, columns=_csindex["result_columns"])
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=_csindex["result_columns"])
                tasks += 1
                try:
                    data = clean_csindex_history(
                        query_csindex_history(
                            code,
                            range_start,
                            range_end,
                            client,
                        ),
                        code,
                    )
                    write_index_daily_request(
                        connection,
                        code,
                        data,
                        range_start,
                        range_end,
                        CSINDEX_DAILY_FIELD_SET_ID,
                    )
                    results.append(
                        (
                            code,
                            range_start,
                            range_end,
                            str(paths.database_path),
                            "success",
                            len(data),
                            "",
                        )
                    )
                except Exception as exc:
                    results.append(
                        (
                            code,
                            range_start,
                            range_end,
                            str(paths.database_path),
                            "failed",
                            0,
                            str(exc),
                        )
                    )
            completed += 1
            if progress is not None:
                progress(completed, len(codes))
        return pd.DataFrame(results, columns=_csindex["result_columns"])
    finally:
        connection.close()


def _split_rqdata_pb_range(
    start_date: str,
    end_date: str,
    months: int,
) -> list[tuple[str, str]]:
    if months <= 0:
        raise ValueError("RQData PB request_months must be positive")
    start_at = pd.Timestamp(start_date)
    end_at = pd.Timestamp(end_date)
    if start_at > end_at:
        raise ValueError("start_date must not be after end_date")
    ranges = []
    cursor = start_at
    while cursor <= end_at:
        range_end = min(cursor + pd.DateOffset(months=months) - pd.Timedelta(days=1), end_at)
        ranges.append((cursor.strftime("%Y-%m-%d"), range_end.strftime("%Y-%m-%d")))
        cursor = range_end + pd.Timedelta(days=1)
    return ranges


def _update_stock_pb(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    data_root: Path = Path("data"),
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Download all configured RQData PB factors into DuckDB."""
    symbols = sorted({normalize_security_symbol(symbol) for symbol in symbols})
    if not symbols:
        raise ValueError("No security symbols were selected")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    paths = init_data_storage(data_root)
    tasks_by_range: dict[tuple[str, str], list[str]] = {}
    with connect_database(paths.database_path) as connection:
        for symbol in symbols:
            missing = missing_baostock_ranges(
                start_date,
                end_date,
                stock_pb_coverage(connection, symbol),
            )
            for missing_start, missing_end in missing:
                for range_start, range_end in _split_rqdata_pb_range(
                    missing_start,
                    missing_end,
                    int(_rqdata_pb["request_months"]),
                ):
                    tasks_by_range.setdefault((range_start, range_end), []).append(symbol)
        tasks = [
            (range_start, range_end, sorted(batch_symbols))
            for (range_start, range_end), batch_symbols in sorted(tasks_by_range.items())
        ]
        if max_tasks is not None:
            tasks = tasks[:max_tasks]
        if progress is not None:
            progress(0, len(tasks))
        results = []
        for completed, (range_start, range_end, batch_symbols) in enumerate(tasks, start=1):
            if checkpoint is not None and not checkpoint():
                break
            data = query_stock_pb_daily(
                batch_symbols,
                range_start,
                range_end,
                client=client,
            )
            write_stock_pb_request(
                connection,
                batch_symbols,
                data,
                range_start,
                range_end,
                int(_rqdata_pb["field_set_id"]),
            )
            results.append(
                (
                    range_start,
                    range_end,
                    "success",
                    len(batch_symbols),
                    len(data),
                    "",
                )
            )
            if progress is not None:
                progress(completed, len(tasks))
    return pd.DataFrame(results, columns=_rqdata_pb["result"])


def update_dividends(
    codes: Iterable[str],
    start_year: int,
    end_year: int,
    data_root: Path = Path("data"),
    max_requests_per_day: int = BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Download before-tax dividends by year into DuckDB."""
    fields = _fields["dividend"]
    codes = [_normalize_baostock_code(code) for code in codes]
    paths = init_data_storage(data_root)
    effective_limit = validate_request_limit(max_requests_per_day)
    context = None if client is not None else BaostockClient()
    active_client = client if client is not None else context.__enter__()
    connection = connect_database(paths.database_path)
    queried = dividend_coverage(connection)
    results = []
    remaining = {
        code: sum(
            (normalize_security_symbol(code), year) not in queried
            for year in range(start_year, end_year + 1)
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
                query_key = normalize_security_symbol(code), year
                if query_key in queried:
                    continue
                if max_tasks is not None and len(results) >= max_tasks:
                    return pd.DataFrame(results, columns=fields["result"])
                if request_count_today(paths.request_log_path) >= effective_limit:
                    return pd.DataFrame(results, columns=fields["result"])
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
                append_request_log(
                    paths.request_log_path,
                    "query_dividend_data",
                    code,
                    "dividend",
                    str(year),
                    str(year),
                )
                data = query_baostock_dividends(code, year, active_client)
                data = clean_baostock_dividends(data, code, year)
                write_dividend_request(connection, code, year, data)
                queried.add(query_key)
                results.append((code, year, "success", len(data), ""))
                remaining[code] -= 1
                if remaining[code] == 0:
                    completed += 1
                    if progress is not None:
                        progress(completed, len(codes))
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
        return pd.DataFrame(results, columns=fields["result"])
    finally:
        connection.close()
        remove_download_lock(data_root)
        if context is not None:
            context.__exit__(None, None, None)


def update_profit_quarterly(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    data_root: Path = Path("data"),
    max_requests_per_day: int = BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Download quarterly total shares into DuckDB."""
    fields = _fields["profit_quarterly"]
    codes = [_normalize_baostock_code(code) for code in codes]
    paths = init_data_storage(data_root)
    effective_limit = validate_request_limit(max_requests_per_day)
    context = None if client is not None else BaostockClient()
    active_client = client if client is not None else context.__enter__()
    connection = connect_database(paths.database_path)
    queried = share_capital_coverage(connection)
    results = []
    periods = [
        (period.year, period.quarter)
        for period in pd.period_range(start_date, end_date, freq="Q")
    ]
    remaining = {
        code: sum(
            (normalize_security_symbol(code), *period) not in queried
            for period in periods
        )
        for code in codes
    }
    completed = sum(count == 0 for count in remaining.values())
    if progress is not None:
        progress(completed, len(codes))
    create_download_lock(data_root)
    try:
        for code in codes:
            for year, quarter in periods:
                query_key = normalize_security_symbol(code), year, quarter
                if query_key in queried:
                    continue
                if max_tasks is not None and len(results) >= max_tasks:
                    return pd.DataFrame(results, columns=fields["result"])
                if request_count_today(paths.request_log_path) >= effective_limit:
                    return pd.DataFrame(results, columns=fields["result"])
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
                append_request_log(
                    paths.request_log_path,
                    "query_profit_data",
                    code,
                    "profit_quarterly",
                    str(year),
                    str(quarter),
                )
                data = clean_baostock_profit(
                    query_baostock_profit(code, year, quarter, active_client),
                    code,
                    year,
                    quarter,
                )
                write_share_capital_request(connection, code, year, quarter, data)
                queried.add(query_key)
                results.append((code, year, quarter, "success", len(data), ""))
                remaining[code] -= 1
                if remaining[code] == 0:
                    completed += 1
                    if progress is not None:
                        progress(completed, len(codes))
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
        return pd.DataFrame(results, columns=fields["result"])
    finally:
        connection.close()
        remove_download_lock(data_root)
        if context is not None:
            context.__exit__(None, None, None)


def update_history_dataset(
    dataset: str,
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    data_root: Path = Path("data"),
    max_requests_per_day: int = BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Check and update each security's locally missing daily ranges."""
    fields = _fields["history"]
    codes = [_normalize_baostock_code(code) for code in codes]
    paths = init_data_storage(data_root)
    connection = connect_database(paths.database_path)
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
            queried = (
                stock_daily_coverage(connection, code)
                if dataset == "stock"
                else index_daily_coverage(
                    connection,
                    code,
                    BAOSTOCK_INDEX_DAILY_FIELD_SET_ID,
                )
            )
            for range_start, range_end in missing_baostock_ranges(
                start_date,
                end_date,
                queried_ranges=queried,
            ):
                if max_tasks is not None and tasks >= max_tasks:
                    return pd.DataFrame(results, columns=fields["result"])
                if request_count_today(paths.request_log_path) >= effective_limit:
                    return pd.DataFrame(results, columns=fields["result"])
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
                tasks += 1
                append_request_log(
                    paths.request_log_path,
                    "query_history_k_data_plus",
                    code,
                    "d",
                    range_start,
                    range_end,
                )
                data = query_baostock_history(
                    code,
                    range_start,
                    range_end,
                    fields["daily"],
                    "d",
                    active_client,
                )
                data = clean_baostock_data(data)
                if dataset == "stock":
                    write_stock_daily_request(
                        connection,
                        code,
                        data,
                        range_start,
                        range_end,
                    )
                else:
                    write_index_daily_request(
                        connection,
                        code,
                        data,
                        range_start,
                        range_end,
                        BAOSTOCK_INDEX_DAILY_FIELD_SET_ID,
                    )
                results.append(
                    (
                        code,
                        range_start,
                        range_end,
                        str(paths.database_path),
                        "success",
                        len(data),
                        "",
                    )
                )
                if checkpoint is not None and not checkpoint():
                    return pd.DataFrame(results, columns=fields["result"])
            completed += 1
            if progress is not None:
                progress(completed, len(codes))
        return pd.DataFrame(results, columns=fields["result"])
    finally:
        connection.close()
        remove_download_lock(data_root)
        if context is not None:
            context.__exit__(None, None, None)


def update_index_constituents(
    codes: Collection[str],
    start_date: str,
    end_date: str,
    source_codes: Mapping[str, str],
    data_root: Path = Path("data"),
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Download and replace RQData constituent snapshots."""
    codes = list(dict.fromkeys(codes))
    if not codes:
        raise ValueError("No security codes were selected")
    unsupported = sorted(set(codes) - set(source_codes))
    if unsupported:
        raise ValueError(f"Unsupported RQData index codes: {unsupported}")
    if max_tasks is not None:
        codes = codes[:max_tasks]
    paths = init_data_storage(data_root)
    results = []
    if progress is not None:
        progress(0, len(codes))
    with connect_database(paths.database_path) as connection:
        for completed, code in enumerate(codes, start=1):
            if checkpoint is not None and not checkpoint():
                break
            data = query_index_constituents(
                start_date,
                end_date,
                source_codes[code],
                client=client,
            )
            write_index_constituents(connection, code, data)
            results.append(
                (
                    code,
                    start_date,
                    end_date,
                    str(paths.database_path),
                    "success",
                    len(data),
                    "",
                )
            )
            if progress is not None:
                progress(completed, len(codes))
    return pd.DataFrame(results, columns=_csindex["result_columns"])


def _run_update_dataset(
    name: str,
    *,
    start: str,
    pool: str | Collection[str],
    end: str | None = None,
    pool_date: str | None = None,
    max_tasks: int | None = None,
    client: Any | None = None,
    checkpoint: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    data_root: Path = Path("data"),
) -> pd.DataFrame:
    """Update a named catalog dataset through its current source."""
    dataset = get_dataset_spec(name)
    update = dataset.update
    if update is None:
        raise ValueError(f"Dataset {name!r} is read-only")
    end_date = end or date.today().isoformat()
    if pd.Timestamp(start) > pd.Timestamp(end_date):
        raise ValueError("start must not be after end")
    if isinstance(pool, str) and not update.pool:
        raise ValueError(f"Dataset {name!r} does not support named pools")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    if dataset.source == "akshare":
        if isinstance(pool, str):
            raise ValueError(f"Dataset {name!r} does not support named pools")
        codes = list(dict.fromkeys(pool))
        if not codes:
            raise ValueError("No security codes were selected")
        return update_csindex_daily(
            codes,
            start,
            end_date,
            data_root,
            client,
            checkpoint,
            progress,
            max_tasks,
        )
    if dataset.source == "rqdata":
        if update.kind == "stock_pb":
            if isinstance(pool, str):
                if pool != "all":
                    raise ValueError(
                        f"Dataset {name!r} only supports the named pool 'all'"
                    )
                symbols = query_rqdata_stock_symbols(
                    start,
                    end_date,
                    client=client,
                )
            else:
                symbols = pool
            return _update_stock_pb(
                symbols,
                start,
                end_date,
                data_root,
                client,
                checkpoint,
                progress,
                max_tasks,
            )
        if isinstance(pool, str):
            raise ValueError(f"Dataset {name!r} does not support named pools")
        return update_index_constituents(
            pool,
            start,
            end_date,
            update.source_codes,
            data_root,
            client,
            checkpoint,
            progress,
            max_tasks,
        )
    if dataset.source != "baostock":
        raise ValueError(f"Dataset {name!r} has unsupported source {dataset.source!r}")
    if checkpoint is not None and not checkpoint():
        return pd.DataFrame(columns=_fields[update.kind]["result"])

    context = None if client is not None else BaostockClient()
    client = client if client is not None else context.__enter__()
    try:
        codes = (
            resolve_baostock_codes(
                pool,
                pool_date or end_date,
                client,
                request_log_path=DataPaths(data_root).request_log_path,
            )
            if isinstance(pool, str)
            else list(dict.fromkeys(pool))
        )
        if not codes:
            raise ValueError("No security codes were selected")
        common = {"client": client}
        if data_root != Path("data"):
            common["data_root"] = data_root
        if checkpoint is not None:
            common["checkpoint"] = checkpoint
        if progress is not None:
            common["progress"] = progress
        if max_tasks is not None:
            common["max_tasks"] = max_tasks
        if update.kind == "history":
            assert update.target is not None
            return update_history_dataset(
                update.target,
                codes,
                start,
                end_date,
                **common,
            )
        if update.kind == "dividend":
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


def update_dataset(
    name: str,
    *,
    start: str,
    pool: str | Collection[str],
    end: str | None = None,
    pool_date: str | None = None,
    max_tasks: int | None = None,
    data_root: Path = Path("data"),
) -> UpdateJob:
    """Start a background update for a named pool or code collection."""
    parameters = locals()

    def run(
        checkpoint: Callable[[], bool],
        progress: Callable[[int, int], None],
    ) -> pd.DataFrame:
        return _run_update_dataset(
            **parameters,
            checkpoint=checkpoint,
            progress=progress,
        )

    return UpdateJob(run)

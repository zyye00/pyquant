"""BaoStock client, source queries, and field cleaning."""

from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pyquant.data.identifiers import normalize_security_symbol
from pyquant.data.resources import load_source_protocols

BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY = 50_000
BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY = 49_000
BAOSTOCK_SOCKET_TIMEOUT_SECONDS = 30
_BAOSTOCK = load_source_protocols()["baostock"]


def normalize_baostock_code(code: object) -> str:
    symbol = normalize_security_symbol(code)
    security_code, exchange = symbol.split(".")
    return f"{exchange.lower()}.{security_code}"


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


def clean_baostock_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert BaoStock strings to compact types and remove source-only fields."""
    out = data.copy()
    if "tradestatus" in out:
        out = out.loc[out["tradestatus"].astype(str) == "1"].copy()
    out["date"] = pd.to_datetime(out["date"])
    for column in set(_BAOSTOCK["history"]["float32"]) & set(out.columns):
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
    fields = _BAOSTOCK["dividend"]
    out = data.rename(
        columns={
            key: value for key, value in fields["field_map"].items() if key != "code"
        }
    ).copy()
    for column in fields["data"]:
        if column not in out:
            out[column] = pd.NA
    out["code"] = out["code"].fillna(code).astype(str)
    out["year"] = year
    for column in ["announce_date", "record_date", "operate_date", "payment_date"]:
        out[column] = pd.to_datetime(out[column], errors="coerce")
    raw_cash = out["cash_dividend_before_tax"]
    parsed_cash = pd.to_numeric(raw_cash, errors="coerce")
    invalid = (
        raw_cash.notna() & raw_cash.astype(str).str.strip().ne("") & parsed_cash.isna()
    )
    if invalid.any():
        examples = raw_cash.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"Invalid dividCashPsBeforeTax values; examples: {examples}")
    out["cash_dividend_before_tax"] = parsed_cash.astype("float32")
    return out[list(fields["data"])]


def clean_baostock_adjust_factors(
    data: pd.DataFrame,
    code: str,
) -> pd.DataFrame:
    """Normalize BaoStock cumulative adjustment-factor events."""
    fields = _BAOSTOCK["adjust_factor"]
    out = data.rename(
        columns={
            key: value for key, value in fields["field_map"].items() if key != "code"
        }
    ).copy()
    for column in fields["data"]:
        if column not in out:
            out[column] = pd.NA
    out["code"] = out["code"].fillna(code).astype(str)
    out["operate_date"] = pd.to_datetime(out["operate_date"], errors="coerce")
    if out["operate_date"].isna().any():
        raise ValueError("Adjustment-factor operate_date must not be missing")
    for column in fields["numeric"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any() or (~out[column].gt(0)).any():
            raise ValueError(f"{column} must contain positive numbers")
    if out.duplicated("operate_date").any():
        raise ValueError("Adjustment factors contain duplicate operate dates")
    return out[list(fields["data"])].sort_values("operate_date").reset_index(drop=True)


def clean_baostock_profit(
    data: pd.DataFrame,
    code: str,
    year: int,
    quarter: int,
) -> pd.DataFrame:
    """Keep quarterly publication dates and total shares."""
    fields = _BAOSTOCK["profit_quarterly"]
    out = data.rename(
        columns={
            key: value for key, value in fields["field_map"].items() if key != "code"
        }
    ).copy()
    for column in fields["data"]:
        if column not in out:
            out[column] = pd.NA
    out["code"] = out["code"].fillna(code).astype(str)
    out["year"] = year
    out["quarter"] = quarter
    for column in ["publish_date", "report_date"]:
        out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in fields["numeric"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[list(fields["data"])]


def request_count_today(
    request_log_path: Path,
    today: date | None = None,
) -> int:
    reset_request_log(request_log_path, today)
    with request_log_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader)
        return sum(bool(row) for row in reader)


def _reset_request_log(request_log_path: Path) -> None:
    request_log_path.parent.mkdir(parents=True, exist_ok=True)
    with request_log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_BAOSTOCK["request_log_fields"])
        writer.writeheader()


def reset_request_log(
    request_log_path: Path,
    today: date | None = None,
) -> None:
    today_str = (today or date.today()).isoformat()
    if not request_log_path.exists():
        _reset_request_log(request_log_path)
        return
    with request_log_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        first_row = next(reader, [])
    if header != list(_BAOSTOCK["request_log_fields"]) or (
        first_row and first_row[0] != today_str
    ):
        _reset_request_log(request_log_path)


def append_request_log(
    request_log_path: Path,
    endpoint: str,
    code: str,
    frequency: str,
    start_date: str,
    end_date: str,
) -> None:
    """Append an outgoing BaoStock request before it is sent."""
    reset_request_log(request_log_path)
    now = datetime.now()
    with request_log_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_BAOSTOCK["request_log_fields"])
        writer.writerow(
            {
                "date": now.date().isoformat(),
                "time": now.strftime("%H:%M:%S"),
                "endpoint": endpoint,
                "code": code,
                "frequency": frequency,
                "start_date": start_date,
                "end_date": end_date,
            }
        )


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
    pool: str,
    trade_date: str,
    client: Any,
    request_log_path: Path | None = None,
    max_requests_per_day: int = BAOSTOCK_DEFAULT_SAFE_REQUEST_LIMIT_PER_DAY,
) -> list[str]:
    """Resolve a pool on its latest available trading day."""
    if pool == "all":
        data = _query_with_request_log(
            request_log_path,
            "query_stock_basic",
            pool,
            "pool",
            trade_date,
            trade_date,
            lambda: baostock_result_to_frame(client.bs.query_stock_basic()),
            max_requests_per_day,
        )
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
    if pool not in _BAOSTOCK["stock_pool_queries"]:
        raise ValueError(f"Unsupported BaoStock stock pool: {pool}")
    query = getattr(client.bs, _BAOSTOCK["stock_pool_queries"][pool])

    end = pd.Timestamp(trade_date)
    calendar_start = (end - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    calendar = _query_with_request_log(
        request_log_path,
        "query_trade_dates",
        pool,
        "calendar",
        calendar_start,
        trade_date,
        lambda: baostock_result_to_frame(
            client.bs.query_trade_dates(calendar_start, trade_date)
        ),
        max_requests_per_day,
    )
    if not {"calendar_date", "is_trading_day"}.issubset(calendar.columns):
        raise ValueError(
            f"BaoStock trade calendar has unexpected columns: {list(calendar.columns)}"
        )
    for day in calendar.loc[
        calendar["is_trading_day"].astype(str) == "1", "calendar_date"
    ].iloc[::-1]:
        data = _query_with_request_log(
            request_log_path,
            _BAOSTOCK["stock_pool_queries"][pool],
            pool,
            "pool",
            str(day),
            str(day),
            lambda: baostock_result_to_frame(query(str(day))),
            max_requests_per_day,
        )
        if "code" not in data.columns:
            raise ValueError(
                f"BaoStock pool result has no code column: {list(data.columns)}"
            )
        codes = data["code"].dropna().astype(str).drop_duplicates().tolist()
        if codes:
            return codes
    return []


def _query_with_request_log(
    request_log_path: Path | None,
    endpoint: str,
    code: str,
    frequency: str,
    start_date: str,
    end_date: str,
    query: Callable[[], pd.DataFrame],
    max_requests_per_day: int,
) -> pd.DataFrame:
    """Run one source query after durably recording it in the request log."""
    if request_log_path is None:
        return query()
    if request_count_today(request_log_path) >= validate_request_limit(
        max_requests_per_day
    ):
        raise RuntimeError("BaoStock request limit reached while resolving stock pool")
    append_request_log(
        request_log_path, endpoint, code, frequency, start_date, end_date
    )
    try:
        return query()
    except Exception as exc:
        raise RuntimeError(f"BaoStock request failed: {exc}") from exc


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
    client: Any,
) -> pd.DataFrame:
    """Query BaoStock history and convert its cursor-like result to DataFrame."""
    try:
        result = client.bs.query_history_k_data_plus(
            code,
            ",".join(fields),
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag="3",
        )
        if getattr(result, "error_code", "0") != "0":
            raise RuntimeError(
                f"BaoStock query failed: {result.error_code} {result.error_msg}"
            )
        return baostock_result_to_frame(result)
    except Exception as exc:
        raise RuntimeError(f"BaoStock request failed: {exc}") from exc


def query_baostock_dividends(code: str, year: int, client: Any) -> pd.DataFrame:
    """Query BaoStock dividends by operating year."""
    try:
        return baostock_result_to_frame(
            client.bs.query_dividend_data(code, str(year), yearType="operate")
        )
    except Exception as exc:
        raise RuntimeError(f"BaoStock request failed: {exc}") from exc


def query_baostock_adjust_factors(
    code: str,
    start_date: str,
    end_date: str,
    client: Any,
) -> pd.DataFrame:
    """Query adjustment-factor events for one security."""
    try:
        return baostock_result_to_frame(
            client.bs.query_adjust_factor(
                code,
                start_date=start_date,
                end_date=end_date,
            )
        )
    except Exception as exc:
        raise RuntimeError(f"BaoStock request failed: {exc}") from exc


def query_baostock_profit(
    code: str,
    year: int,
    quarter: int,
    client: Any,
) -> pd.DataFrame:
    """Query BaoStock quarterly profit data."""
    try:
        return baostock_result_to_frame(
            client.bs.query_profit_data(code, str(year), str(quarter))
        )
    except Exception as exc:
        raise RuntimeError(f"BaoStock request failed: {exc}") from exc

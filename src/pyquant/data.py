"""Dataset catalog, loading, and field standardization."""

from __future__ import annotations

from collections.abc import Callable, Collection
from contextvars import copy_context
from glob import glob
from pathlib import Path
from threading import Condition, Thread
from typing import Any

import pandas as pd

from pyquant.io import load_config


DEFAULT_DATASET_CATALOG = Path(__file__).parents[2] / "configs/datasets.yaml"
DATASET_CATALOG = load_config(DEFAULT_DATASET_CATALOG)
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


class DatasetUpdate:
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


def load_dataset(
    name: str,
    *,
    start: str | None = None,
    end: str | None = None,
    symbols: Collection[str] | None = None,
) -> pd.DataFrame:
    """Load a catalog dataset with canonical columns."""
    dataset = get_dataset(name)
    storage = dataset["storage"]
    kind = storage["kind"]
    if kind != "table" and (start is None or end is None):
        raise ValueError(f"Dataset {name!r} requires explicit start and end dates")
    start_at = pd.Timestamp(start) if start is not None else None
    end_at = pd.Timestamp(end) if end is not None else None
    if start_at is not None and end_at is not None and start_at > end_at:
        raise ValueError("start must not be after end")
    paths = _dataset_paths(storage, symbols, start_at, end_at)
    if not paths:
        raise FileNotFoundError(f"No files found for dataset {name!r}")
    frames = [
        _read_dataset_file(path, dataset, storage, start_at, end_at) for path in paths
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=dataset["columns"])
    out = pd.concat(frames, ignore_index=True)
    out = _canonicalize_dataset(out, dataset)
    if symbols is not None:
        out = out[out["symbol"].isin({str(symbol) for symbol in symbols})]
    key = dataset.get("primary_key", [])
    if key and out.duplicated(key).any():
        raise ValueError(f"Dataset {name!r} contains duplicate primary keys: {key}")
    columns = [column for column in dataset["columns"] if column in out]
    return out[columns].sort_values(key or columns[:1]).reset_index(drop=True)


def update_dataset(
    name: str,
    *,
    start: str,
    end: str | None = None,
    pool: str | Collection[str],
    pool_date: str | None = None,
    max_tasks: int | None = None,
) -> DatasetUpdate:
    """Start a background update for a named pool or security-code collection."""
    parameters = locals()
    from pyquant._data_update import update_dataset as update_source_dataset

    def run(
        checkpoint: Callable[[], bool],
        progress: Callable[[int, int], None],
    ) -> pd.DataFrame:
        return update_source_dataset(
            **parameters,
            checkpoint=checkpoint,
            progress=progress,
        )

    return DatasetUpdate(run)


def load_price(
    path: str | Path,
    field_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a standalone price file and standardize its fields."""
    path = Path(path)
    if path.suffix == ".csv":
        data = pd.read_csv(path)
    elif path.suffix in {".parquet", ".pq"}:
        data = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported price file type: {path.suffix}")
    return standardize_price(data, field_map=field_map)


def standardize_price(
    data: pd.DataFrame,
    field_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Standardize price field names, dates, and symbols."""
    rename_map = DEFAULT_PRICE_FIELD_MAP | (field_map or {})
    out = data.rename(
        columns={key: value for key, value in rename_map.items() if key in data.columns}
    ).copy()
    missing = [column for column in ["date", "symbol", "close"] if column not in out]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    ordered = [column for column in PRICE_COLUMNS if column in out]
    extras = [column for column in out if column not in ordered]
    return out[ordered + extras].sort_values(["date", "symbol"]).reset_index(drop=True)


def get_dataset(name: str) -> dict[str, Any]:
    """Return one built-in dataset definition by name."""
    try:
        return DATASET_CATALOG["datasets"][name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_CATALOG["datasets"]))
        raise ValueError(f"Unknown dataset {name!r}; available: {available}") from exc


def _dataset_paths(
    storage: dict[str, str],
    symbols: Collection[str] | None,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> list[Path]:
    template = storage["path"]
    if storage["kind"] == "table":
        path = Path(template)
        return [path] if path.exists() else []
    symbol_values = [str(symbol) for symbol in symbols] if symbols else ["*"]
    years: list[int | str] = ["*"]
    if "{year}" in template and start is not None and end is not None:
        years = list(range(start.year, end.year + 1))
    query_path = storage.get("query_path")
    paths = {
        Path(path)
        for symbol in symbol_values
        for year in years
        for path in glob(
            template.format(symbol=symbol, year=year)
        )
        if query_path is None or Path(path) != Path(query_path)
    }
    return sorted(paths)


def _read_dataset_file(
    path: Path,
    dataset: dict[str, Any],
    storage: dict[str, str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    data = pd.read_parquet(path)
    if storage.get("symbol_from") == "stem":
        data["symbol"] = path.stem
    elif storage.get("symbol_from") == "parent":
        data["symbol"] = path.parent.name
    date_column = dataset.get("date_column")
    source_date = next(
        (
            source
            for source, target in dataset["field_map"].items()
            if target == date_column
        ),
        date_column,
    )
    if source_date in data:
        values = data[source_date]
        if not pd.api.types.is_datetime64_any_dtype(values):
            values = pd.to_datetime(values, errors="coerce")
        if start is not None:
            data = data[values >= start]
            values = values.loc[data.index]
        if end is not None:
            data = data[values <= end]
    return data


def _canonicalize_dataset(data: pd.DataFrame, dataset: dict[str, Any]) -> pd.DataFrame:
    rename_map = {
        source: target
        for source, target in dataset["field_map"].items()
        if source in data and target not in data
    }
    out = data.rename(columns=rename_map).copy()
    missing = sorted(set(dataset["required"]) - set(out))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    if "symbol" in out:
        if out["symbol"].isna().any():
            raise ValueError("Dataset symbol must not contain missing values")
        out["symbol"] = out["symbol"].astype(str)
    for column in dataset.get("date_columns", []):
        if column in out and not pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in dataset.get("numeric_columns", []):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out

"""One-time migration from the legacy source-named raw-data layout."""

from pathlib import Path
from shutil import move


ADJUSTMENT_NAMES = {"backward", "forward", "none"}


def migrate_legacy_data_layout(
    legacy_root: str | Path,
    data_root: str | Path,
    *,
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """Move legacy BaoStock files into dataset-named directories without overwrite."""
    legacy = Path(legacy_root)
    destination_root = Path(data_root)
    if not legacy.exists():
        return []

    planned = [
        (source, _migration_target(source.relative_to(legacy), destination_root))
        for source in sorted(path for path in legacy.rglob("*") if path.is_file())
    ]
    collisions = [destination for _, destination in planned if destination.exists()]
    if collisions:
        examples = [str(path) for path in collisions[:5]]
        raise FileExistsError(
            f"Migration destinations already exist ({len(collisions)}); examples: {examples}"
        )
    if dry_run:
        return planned

    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        move(source, destination)
    for directory in sorted(
        (path for path in legacy.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.rmdir()
    legacy.rmdir()
    return planned


def _migration_target(relative: Path, data_root: Path) -> Path:
    parts = relative.parts
    if parts[:2] == ("daily", "stock"):
        return data_root / "raw" / "stock_daily" / Path(*parts[2:])
    if parts[:2] == ("daily", "index"):
        tail = parts[2:]
        if not tail or tail[0] not in ADJUSTMENT_NAMES:
            tail = ("none", *tail)
        return data_root / "raw" / "index_daily" / Path(*tail)
    if parts[:2] == ("daily", "other"):
        tail = parts[2:]
        if not tail or tail[0] not in ADJUSTMENT_NAMES:
            tail = ("none", *tail)
        return data_root / "raw" / "other_daily" / Path(*tail)
    if parts[:2] == ("minute_5", "stock"):
        return data_root / "raw" / "stock_5m" / Path(*parts[2:])

    table_targets = {
        ("dividend.parquet",): ("raw", "dividend", "data.parquet"),
        ("stock_profit_quarterly.parquet",): (
            "raw",
            "stock_profit_quarterly",
            "data.parquet",
        ),
        ("state", "dividend_queries.parquet"): (
            "raw",
            "dividend",
            "queries.parquet",
        ),
        ("state", "stock_profit_quarterly_queries.parquet"): (
            "raw",
            "stock_profit_quarterly",
            "queries.parquet",
        ),
        ("state", "request_log.csv"): ("state", "request_log.csv"),
        ("state", "download.lock"): ("state", "download.lock"),
    }
    try:
        target_parts = table_targets[parts]
    except KeyError as exc:
        raise ValueError(f"Unsupported legacy data path: {relative}") from exc
    return data_root / Path(*target_parts)

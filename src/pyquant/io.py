"""Physical table input and output helpers."""

from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory when needed and return its path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_output(
    data: pd.DataFrame | pd.Series,
    path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Save a CSV table without overwriting by default."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    ensure_dir(path.parent)

    if path.suffix == ".csv":
        data.to_csv(path)
    else:
        raise ValueError(f"Unsupported output file type: {path.suffix}")
    return path

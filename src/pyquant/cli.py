"""Command-line entry points for pyquant."""

from __future__ import annotations

import argparse
import sys

from pyquant._data_update import StdinDownloadControl, update_dataset
from pyquant.data import _load_dataset_catalog


def main(argv: list[str] | None = None) -> None:
    """Run the pyquant command line."""
    parser = argparse.ArgumentParser(prog="pyquant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_data_update_parser(subparsers)
    args = parser.parse_args(argv)
    if args.command == "data-update":
        run_data_update(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


def _add_data_update_parser(subparsers: argparse._SubParsersAction) -> None:
    catalog = _load_dataset_catalog()
    datasets = sorted(
        name for name, config in catalog["datasets"].items() if "update" in config
    )
    parser = subparsers.add_parser(
        "data-update",
        help="Update a configured dataset without exposing its source implementation.",
    )
    parser.add_argument("dataset", choices=datasets)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pool", choices=["all", "sz50", "hs300", "zz500"])
    target.add_argument("--symbols", nargs="+")
    parser.add_argument("--pool-date")
    parser.add_argument("--adjustment", choices=["forward", "backward", "none"])
    parser.add_argument("--max-tasks", type=int)


def run_data_update(args: argparse.Namespace) -> None:
    """Update one catalog dataset and print request progress."""
    control = StdinDownloadControl()
    progress_printed = False

    def show_progress(completed: int, total: int) -> None:
        nonlocal progress_printed
        if progress_printed:
            sys.stdout.write("\r\033[1A\033[2K")
        progress_printed = True
        sys.stdout.write(f"Updated {completed}/{total}\n> ")
        sys.stdout.flush()

    print("Keyboard control: 'p' pause, 'c' continue, 'q' save and quit.")
    result = update_dataset(
        args.dataset,
        start=args.start_date,
        end=args.end_date,
        symbols=args.symbols,
        pool=args.pool,
        pool_date=args.pool_date,
        adjustment=args.adjustment,
        max_tasks=args.max_tasks,
        _control=control,
        _progress=show_progress,
    )
    print(f"Update status: {result['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

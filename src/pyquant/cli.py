"""Command line entry points for pyquant."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from pyquant.baostock_source import (
    BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY,
    BaostockClient,
    StdinDownloadControl,
    resolve_baostock_codes,
    update_baostock_dataset,
)
from pyquant.io import load_config


DEFAULT_BAOSTOCK_CONFIG = Path("configs/baostock_download.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyquant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_baostock_download_parser(subparsers)
    args = parser.parse_args(argv)
    if args.command == "baostock-download":
        return run_baostock_download(args)
    raise ValueError(f"Unsupported command: {args.command}")


def _add_baostock_download_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "baostock-download",
        help="Download BaoStock data serially with request-boundary pause/continue.",
    )
    parser.add_argument("--frequency", choices=["d", "5"], required=True)
    parser.add_argument("--adjustflag", choices=["forward", "backward", "none"])
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--index",
        help="Index code for index daily data, e.g. sh.000300.",
    )
    target.add_argument(
        "--pool",
        choices=["all", "sz50", "hs300", "zz500"],
        help="Stock pool.",
    )
    parser.add_argument(
        "--pool-date",
        help="Date used to resolve a stock pool. Defaults to --end-date.",
    )


def run_baostock_download(args: argparse.Namespace) -> int:
    cfg = load_baostock_download_config(DEFAULT_BAOSTOCK_CONFIG)
    raw_root = cfg["raw_root"]
    max_requests_per_day = cfg["safe_max_requests_per_day"]
    end_date = args.end_date or today_text()

    if args.index and args.frequency != "d":
        raise ValueError("BaoStock does not provide index minute data; use --frequency d with --index")

    dataset = "index" if args.index else "stock"
    control = StdinDownloadControl()
    progress_printed = False

    def show_progress(completed: int, total: int) -> None:
        nonlocal progress_printed
        if progress_printed:
            sys.stdout.write("\r\033[1A\033[2K")
        progress_printed = True
        sys.stdout.write(f"Downloaded {completed}/{total}\n> ")
        sys.stdout.flush()

    print("Keyboard control: 'p' pause, 'c' continue, 'q' save and quit.")
    with BaostockClient() as client:
        if args.index:
            codes = [args.index]
        else:
            codes = resolve_baostock_codes(args.pool, args.pool_date or end_date, client)

        if not codes:
            raise ValueError("No BaoStock codes were selected")

        pool_name = args.index if args.index else args.pool
        print(f"Stock pool: {pool_name} ({len(codes)} securities)")
        result = update_baostock_dataset(
            dataset=dataset,
            frequency=args.frequency,
            codes=codes,
            start_date=args.start_date,
            end_date=end_date,
            raw_root=raw_root,
            max_requests_per_day=max_requests_per_day,
            adjustflag=args.adjustflag,
            client=client,
            control=control,
            progress=show_progress,
        )
    if progress_printed:
        print()
    status_counts = result["status"].value_counts().to_dict()
    print(f"Selected codes: {len(codes)}")
    print(f"Download status: {status_counts}")
    return 0


def today_text() -> str:
    return date.today().isoformat()


def load_baostock_download_config(path: str | Path) -> dict:
    config = load_config(path)
    limits = config.get("baostock_limits", {})
    paths = config.get("paths", {})
    raw_root = paths.get("raw_root")
    hard_limit = limits.get("hard_max_requests_per_day")
    safe_limit = limits.get("safe_max_requests_per_day")
    if raw_root is None:
        raise ValueError("Missing config value: paths.raw_root")
    if hard_limit is None:
        raise ValueError("Missing config value: baostock_limits.hard_max_requests_per_day")
    if safe_limit is None:
        raise ValueError("Missing config value: baostock_limits.safe_max_requests_per_day")
    if hard_limit != BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY:
        raise ValueError(f"BaoStock hard request limit must be {BAOSTOCK_HARD_REQUEST_LIMIT_PER_DAY}")
    if safe_limit > hard_limit:
        raise ValueError("safe_max_requests_per_day cannot exceed hard_max_requests_per_day")
    return {
        "raw_root": raw_root,
        "hard_max_requests_per_day": int(hard_limit),
        "safe_max_requests_per_day": int(safe_limit),
    }


if __name__ == "__main__":
    raise SystemExit(main())

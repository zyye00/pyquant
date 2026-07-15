# Dataset-Centric Data Layer Design

## Goal

Make datasets, rather than BaoStock, the public organizing concept for local data, loading, and updates. Strategies consume canonical dataset fields and do not know which upstream service supplied them.

The first catalog covers the datasets already supported by the repository: `stock_daily`, `index_daily`, `stock_5m`, `dividend`, and `stock_profit_quarterly`. BaoStock remains the initial upstream service internally, but it is no longer a public module, command family, directory level, or strategy dependency.

## Data Catalog

Add `configs/datasets.yaml` as the executable catalog. Each dataset entry declares:

- a stable dataset name and description;
- its canonical columns, required columns, primary key, and date column;
- its storage format, path template, and optional symbol derivation rule;
- its current source and explicit source-to-canonical field map;
- update parameters that are valid for the dataset.

The catalog also contains source connection limits shared by all datasets currently backed by BaoStock. The old `configs/baostock_download.yaml` is removed after all settings have moved into the catalog.

Markdown documentation may explain the catalog, but YAML is the source of truth because the loader and updater must validate and use it directly.

## Local Data Layout

Remove the source-name directory level and organize raw data by dataset:

```text
data/
├── raw/
│   ├── stock_daily/{adjustment}/{symbol}.parquet
│   ├── index_daily/{adjustment}/{symbol}.parquet
│   ├── stock_5m/{adjustment}/{symbol}/{year}.parquet
│   ├── dividend/data.parquet
│   ├── dividend/queries.parquet
│   ├── stock_profit_quarterly/data.parquet
│   └── stock_profit_quarterly/queries.parquet
└── state/
    ├── request_log.csv
    └── download.lock
```

Provide a one-time migration script that moves the existing `data/raw/baostock` files into these catalog paths without rewriting parquet contents. It fails before moving anything if a destination exists, supports `--dry-run`, and removes empty legacy directories only after a successful migration. The implementation does not redownload data and never overwrites a raw file.

## Public Interfaces

`pyquant.data` exposes two dataset-centric operations:

```python
load_dataset(
    name: str,
    *,
    start: str | None = None,
    end: str | None = None,
    symbols: Collection[str] | None = None,
    adjustment: str | None = None,
) -> pandas.DataFrame

update_dataset(
    name: str,
    *,
    start: str,
    end: str | None = None,
    symbols: Collection[str] | None = None,
    pool: str | None = None,
    pool_date: str | None = None,
    adjustment: str | None = None,
    max_tasks: int | None = None,
) -> pandas.DataFrame
```

`load_dataset` reads only the path pattern and fields declared for the named dataset, derives `symbol` from file names where configured, applies canonical renames and types, filters dates and symbols, and validates required columns. Date filters are inclusive. Partitioned price datasets require explicit `start` and `end`, preventing accidental full-history loads; table datasets may omit them.

`update_dataset` validates arguments against the dataset catalog and internally executes the current source-specific request logic. There is no provider protocol, registry, or public BaoStock adapter until a second source actually exists.

The public command becomes:

```text
pyquant data-update DATASET --start-date DATE [--end-date DATE]
                           (--pool POOL | --symbols SYMBOL [SYMBOL ...])
                           [--pool-date DATE] [--adjustment ADJUSTMENT]
                           [--max-tasks N]
```

All datasets use date ranges. For dividend data, the updater converts the inclusive range to the operating years required by the current upstream API. Dataset validation rejects irrelevant options, such as `--adjustment` for dividends. Existing `baostock-download`, `baostock-dividend-download`, and `baostock-profit-download` commands are removed rather than retained as aliases.

`pyquant.__init__` exports `load_dataset` and `update_dataset`. It no longer exports functions named after BaoStock. Existing generic `load_price` and `standardize_price` remain for callers loading standalone files.

## Internal Structure and Data Flow

- `data.py` owns catalog loading, dataset validation, canonical field conversion, and local reads.
- A private update module owns request scheduling, locking, query coverage, merge behavior, and the current BaoStock client calls. It is reached only through `update_dataset` and is not exported as a source layer.
- `cli.py` translates generic command arguments into `update_dataset`; it contains no source-specific command handlers.
- Dataset source fields are renamed before storage when practical. The loader still validates and applies the catalog mapping so existing files and later source changes produce the same canonical contract.

The dividend-low-volatility strategy consumes the canonical schemas:

- `stock_daily`: `date`, `symbol`, `close`, `amount`, `pe_ttm`;
- `dividend`: `symbol`, `year`, `announce_date`, `cash_dividend_after_tax`;
- dividend query coverage: `symbol`, `year` from `dividend/queries.parquet`;
- `stock_profit_quarterly`: `symbol`, `publish_date`, `total_shares`.

Its strategy configuration keeps only strategy and backtest parameters. Dataset locations and sources belong exclusively to the catalog. The caller loads the required datasets with explicit date bounds that include the 240-observation warm-up, then passes canonical frames to `build_dividend_universe`.

## Errors and Compatibility

- Unknown dataset names list the available catalog names.
- Missing catalog keys, unsupported options, missing files, schema mismatches, duplicate primary keys, and invalid date ranges raise clear `ValueError` or `FileNotFoundError` exceptions.
- Update coverage tables distinguish a completed empty upstream query from a query that was never run.
- The one-time layout migration is the compatibility boundary. Code does not silently fall back to `data/raw/baostock`, because fallback would preserve the source-centric hierarchy and hide incomplete migration.
- Existing raw data remains byte-for-byte unchanged apart from file moves. Request history and lock state move to the shared state directory.

## Testing and Acceptance

- Catalog tests validate every entry, path template, canonical schema, and field map.
- Loader tests cover single-table and partitioned parquet data, symbol derivation, inclusive date filtering, adjustment selection, missing files, and schema failures.
- Update tests preserve current pause, request-limit, merge, query-cache, and empty-result behavior behind `update_dataset`, plus dataset-option validation and dividend date-to-year conversion.
- CLI tests cover each dataset class through `data-update` and confirm old BaoStock commands are absent.
- Migration tests cover dry-run output, successful moves, destination collision, and unchanged parquet bytes.
- Strategy tests use only canonical fields and continue to verify point-in-time announcement-date, share-publication, coverage, ranking, and tie behavior.
- Acceptance requires focused tests, the full `quant`-environment pytest suite, Ruff checks, import validation, and `git diff --check` to pass. No network download is part of validation.

## Explicit Defaults

- `configs/datasets.yaml` is the single executable data-source description; no parallel per-source config remains.
- Existing data is migrated to dataset directories instead of supported through a legacy fallback.
- `end` defaults to the current local date for updates; loaders use exactly the requested inclusive bounds.
- `adjustment` defaults to `none` for price datasets.
- BaoStock support remains an internal implementation detail until another real source is added; no speculative provider abstraction is introduced.

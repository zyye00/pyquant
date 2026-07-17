# Notebook Download Progress Design

## Goal

Make notebook progress updates remain attached to the cell that starts a dataset
update, count historical-price progress directly as each security is checked and
processed, and remove the unused `rows` field from request logging.

## Chosen Approach

Process historical-price updates one security at a time. For each security, inspect
local date coverage, download its zero to several missing ranges, save each successful
range atomically, and then advance the completed-security count. This replaces the
current full-pool slice DataFrame and the indirect `value_counts()` progress logic.

Alternatives rejected:

- Keeping the full-pool slice table and adding a separate scan counter would preserve
  duplicate state and retain the incorrect `max_tasks` completion calculation.
- Adding a persistent history-query cache would make empty-range coverage durable but
  would reintroduce task metadata that the project deliberately removed. Empty source
  ranges may therefore be queried again on a later run.

## Notebook Display Context

`DatasetUpdate` will capture the current `contextvars` context before starting its
background thread. The thread will run through that copied context so ipykernel keeps
the original cell parent header for every `DisplayHandle.update()` message. Public
pause, resume, stop, wait, state, and progress properties remain unchanged.

## Historical Progress And Download Flow

After resolving the security pool, report `0 / total` immediately. Then process codes
in pool order:

1. Compute only that security's missing date ranges from its local Parquet file.
2. If no range is missing, mark the security complete immediately.
3. Otherwise download its ranges in order, atomically merging and saving each success.
4. Mark the security complete only when all its required ranges succeed.
5. Emit progress immediately after a security becomes complete.

The daily request limit, graceful stop checks, and `max_tasks` limit remain request-
slice boundaries. If any limit stops work partway through a security, that security is
not counted complete. A failed range also leaves the security incomplete.

The full-pool `build_download_slices()` and `run_download_slices()` staging functions
will be removed because their responsibilities move into `update_history_dataset()`.
The reusable `missing_baostock_ranges()` and target-path functions remain.

Dividend and quarterly-profit progress already counts completed securities from their
query caches and is not changed.

## Request Log

Remove `rows` from the configured request-log schema, `append_request_log()` signature,
and every caller. Request limits continue to count CSV records, so this does not change
safety behavior. Existing log records retain all other fields.

Add `scripts/remove_request_log_rows.py` as a one-time migration. It will read
`data/state/request_log.csv` with the standard-library CSV module, remove only the
`rows` column, write a temporary sibling file, and atomically replace the original.
Running it again will be harmless when the column is already absent.

## Compatibility And Errors

No public API or stored market-data schema changes. Historical result DataFrames keep
their existing columns. The request-log CSV intentionally loses one unused column.
Existing raw market data is never rewritten by this migration.

## Testing

- Verify the update thread inherits a context variable captured in the starting cell.
- Verify historical progress starts at zero and advances immediately for covered and
  successfully downloaded securities.
- Verify multiple ranges advance progress only once per security.
- Verify failures, stop, request limit, and `max_tasks` do not mark partial securities
  complete.
- Verify request logs use the new schema and request limits still count records.
- Run the migration against a temporary representative CSV before applying it to the
  real log, then confirm row count and every retained field are unchanged.
- Run Ruff, focused tests, the complete pytest suite, import checks, and
  `git diff --check` in the `quant` environment.

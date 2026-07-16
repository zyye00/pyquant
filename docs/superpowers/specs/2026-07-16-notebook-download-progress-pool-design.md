# Notebook Download Progress and Pool Design

## Goal

Restore automatic download progress in notebooks, simplify target selection to one
`pool` parameter, preserve controllable background jobs, and verify incremental
downloads for both securities and date ranges.

## Public API

`update_dataset()` keeps returning `DatasetUpdate` and replaces the separate
`symbols` and `pool` parameters with:

```python
pool: str | Iterable[str]
```

- A string selects a configured BaoStock pool: `all`, `sz50`, `hs300`, or `zz500`.
- Any non-string iterable supplies BaoStock security codes directly.
- Code iterables are materialized once, deduplicated while preserving order, and
  rejected when empty.
- Existing date, adjustment, and task-limit arguments remain unchanged.

## Progress and Control

The background job prints `Updated {completed}/{total}` automatically. Each update
uses a carriage return so the notebook output stays on one line. Completion,
graceful stop, and failure terminate the progress line with a newline.

`DatasetUpdate.state`, `completed`, `total`, and `error` remain available. Notebook
controls appear in this order so related actions are adjacent:

1. Status
2. Pause
3. Resume
4. Stop and save
5. Wait for normal completion

Stopping remains cooperative. `stop()` requests termination and `wait()` confirms
that pending data was saved and cleanup completed.

## Incremental Behavior

History downloads continue deriving work from local parquet coverage:

- A code without a local file receives the complete requested range.
- A covered code receives only an earlier missing range, a later missing range, or
  both ranges when the request expands in both directions.
- A fully covered code generates no request and counts as completed progress.

Dividend and quarterly-profit downloads continue using their query coverage files,
but accept the same unified `pool` argument.

## Validation

Focused tests cover automatic progress output, iterable pool forwarding and
deduplication, named pools, empty iterables, existing/new security coverage, and
earlier/later history slices. Ruff, notebook code compilation, the full test suite,
and `git diff --check` must pass without making live BaoStock requests.

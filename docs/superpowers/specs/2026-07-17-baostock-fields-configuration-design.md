# BaoStock Field Configuration Design

## Goal

Move every static field list from `src/pyquant/_data_update.py` into the existing
`configs/datasets.yaml` catalog. Keep one configuration entry point, reduce the
module-level constant block, and preserve all current download and storage behavior.

## Configuration Structure

Group source-facing and update-process fields by BaoStock operation under
`sources.baostock.fields`:

```yaml
sources:
  baostock:
    fields:
      history:
        daily: [...]
        minute_5: [...]
        slice: [...]
        result: [...]
      dividend:
        data: [...]
        query: [...]
        result: [...]
        float32: [...]
      profit_quarterly:
        data: [...]
        query: [...]
        result: [...]
      request_log: [...]
```

Dataset-level canonical schemas, primary keys, numeric columns, and field maps remain
in their existing `datasets.*` entries. Request limits, timeout values, pool queries,
and adjustment mappings remain under `sources.baostock` because they are technical
source configuration rather than field schemas.

## Python Structure

`src/pyquant/_data_update.py` will read the grouped field configuration through one
private `_fields` reference. It will remove the static `*_FIELDS`, `*_COLUMNS`,
`*_QUERY_COLUMNS`, `*_RESULT_COLUMNS`, and numeric-column list constants.

The redundant `_config = DATASET_CATALOG` alias will be removed. Dataset field maps
remain the sole mapping definitions; dividend and quarterly-profit cleaners will read
them directly and omit the `code` mapping because raw update storage continues to use
`code`.

Technical constants used in function defaults or repeatedly throughout the module,
such as request limits and socket timeout, remain named Python constants backed by the
catalog. They are not field definitions and the names improve call signatures.

## Data Flow

1. `data.py` loads `configs/datasets.yaml` once into `DATASET_CATALOG`.
2. `_data_update.py` obtains the BaoStock source configuration and grouped fields.
3. Query, cleaning, result-frame, cache-frame, and request-log operations read the
   relevant field list from the grouped configuration.
4. Existing Parquet schemas and request log columns remain unchanged.

## Compatibility And Errors

This refactor does not change public functions, download ranges, incremental behavior,
request limits, pause or stop behavior, or stored column names. The built-in catalog
is trusted, so no new configuration validation is added. Missing or malformed built-in
configuration should fail naturally during import or lookup.

## Testing

Update focused tests only where they reference moved field configuration. Run Ruff and
format checks on the changed Python files, the data and update test modules, the full
pytest suite in the `quant` environment, and `git diff --check`. No config-only schema
test will be added.

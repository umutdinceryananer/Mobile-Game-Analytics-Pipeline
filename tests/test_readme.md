# Pytest Quick Run Summary

Running `pytest -q` triggers the `tests/test_synthetic.py` suite (six checks) and they all fail right now because the environment lacks a Parquet engine. Pandas raises `ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'` when `pd.read_parquet('data/processed/events.parquet')` is called.

## Tests involved
- `test_required_columns` verifies that the synthetic processing step produced `acquisition_channel`, `CAC`, `revenue`, `ROI`, `country`, and `platform` columns.
- `test_determinism` ensures that user-level assignments (channel/country/platform) are deterministic across reloads of the processed file.
- `test_country_distribution` checks the country mix against the weights in `data/config/synthetic.yaml`.
- `test_platform_distribution` validates the install split between App Store and Google Play.
- `test_platform_revenue_share` asserts revenue weighting by platform (26M vs 9M target ratio).
- `test_no_negative_values` confirms that CAC/revenue are non-negative and ROI has finite values.

## How to fix the current failures
Install either `pyarrow` or `fastparquet` (both are listed in `requirements.txt`). For example:

```
pip install -r requirements.txt  # or pip install pyarrow
```

After installing a parquet backend, re-run `pytest -q`; the tests should execute against `data/processed/events.parquet` without raising the import error.

The tests must be executed in main directory.

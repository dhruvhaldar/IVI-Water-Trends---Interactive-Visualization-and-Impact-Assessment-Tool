## 2025-02-28 - [Categorical GroupBy Optimization]
**Learning:** Grouping by categorical columns (`category` dtype) is significantly faster (observed ~30-40% speedup) than grouping by string/object columns, especially for high-cardinality columns like IDs. The cost of one-time conversion to category (e.g., during data loading) is amortized if the dataframe is used for multiple operations or even a single heavy groupby.
**Action:** When loading data that will be used for grouping/aggregation, convert string columns with repeated values (like IDs, categories) to `category` dtype early in the pipeline (e.g., in cleaning step).

## 2025-03-01 - [Categorical Merge & Sort Optimization]
**Learning:** Merging and sorting DataFrames on `category` columns is significantly faster (~20-25%) than on `object` (string) columns. Converting to `category` *before* sorting (e.g. in `sort_values`) improves the sort performance as well.
**Action:** Ensure string columns used for joining or sorting are converted to `category` dtype upstream (e.g. in data cleaning/loading methods) to benefit downstream operations.

## 2025-03-01 - [Optimized Sort by Categorical Conversion]
**Learning:** Sorting a DataFrame by columns that are already converted to `category` dtype is significantly faster (observed ~50-75% speedup) than sorting by object/string columns, especially for high-cardinality columns.
**Action:** In data cleaning pipelines, always convert string columns (with repeated values) to `category` dtype *before* performing sort operations (`sort_values`).

## 2025-03-01 - [Categorical GroupBy on dynamic columns]
**Learning:** Functions that aggregate over dynamic or variable user-provided columns (e.g. `location_level` in `create_seasonal_summary`) should still explicitly convert those grouping columns to `category` dtype right before the `groupby()` call. This ensures the ~35-45% speedup is realized even if upstream processes haven't converted them (or converted a different column).
**Action:** Always verify the dtype of dynamically selected grouping columns and safely convert them to `category` prior to performing grouped aggregations on large datasets.

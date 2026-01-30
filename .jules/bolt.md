## 2025-02-28 - [Categorical GroupBy Optimization]
**Learning:** Grouping by categorical columns (`category` dtype) is significantly faster (observed ~30-40% speedup) than grouping by string/object columns, especially for high-cardinality columns like IDs. The cost of one-time conversion to category (e.g., during data loading) is amortized if the dataframe is used for multiple operations or even a single heavy groupby.
**Action:** When loading data that will be used for grouping/aggregation, convert string columns with repeated values (like IDs, categories) to `category` dtype early in the pipeline (e.g., in cleaning step).

## 2025-03-01 - [Categorical Merge & Sort Optimization]
**Learning:** Merging and sorting DataFrames on `category` columns is significantly faster (~20-25%) than on `object` (string) columns. Converting to `category` *before* sorting (e.g. in `sort_values`) improves the sort performance as well.
**Action:** Ensure string columns used for joining or sorting are converted to `category` dtype upstream (e.g. in data cleaning/loading methods) to benefit downstream operations.

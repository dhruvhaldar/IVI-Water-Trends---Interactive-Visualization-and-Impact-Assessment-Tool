## 2024-05-23 - [Pandas Loop Optimization]
**Learning:** Iterating over `df.groupby()` is significantly slower if you perform sorting inside the loop. Sorting the entire DataFrame once before grouping (using a stable sort like `mergesort`) and then grouping with `sort=False` avoids the overhead of repeated dataframe slicing and sorting, leading to massive speedups (observed ~30% faster in benchmarks for this specific case, potentially more for larger groups).
**Action:** Always check for repeated expensive operations inside loops. For Pandas, "Sort Once, Group Later" is a powerful pattern. Also, remember to handle both list and string inputs for `group_by` arguments to maintain robustness.

## 2025-02-19 - [Pandas isna() optimization]
**Learning:** `df[['col1', 'col2', ...]].isna().any(axis=1)` creates an intermediate DataFrame subset which involves allocation and copying, and then reduces it. For a small number of columns, explicit boolean OR operations on Series (`df['col1'].isna() | df['col2'].isna() | ...`) are significantly faster (3-4x) because they operate directly on the Series without creating a new DataFrame structure.
**Action:** When filtering by missing values on a few known columns, prefer explicit boolean operators over `subset.any(axis=1)`.

## 2025-02-19 - [Pandas Inplace Optimization]
**Learning:** Using `inplace` modification to avoid `df.copy()` at the start of a processing pipeline saves memory and time on the copy operation itself (~45% faster). However, if the subsequent pipeline involves heavy filtering or type conversions that inevitably create new objects, the overall function speedup may be modest (2-5%).
**Action:** Use `inplace` optimizations primarily for memory efficiency on large datasets, but do not expect dramatic overall CPU speedups if the pipeline is dominated by other expensive operations.

## 2025-02-21 - [Single GroupBy vs Double GroupBy]
**Learning:** When calculating complex statistics on filtered data alongside counts on unfiltered data, performing a single `groupby` on a masked DataFrame (using NaN for invalid values) is significantly faster (40-80%) than performing two separate `groupby` operations (one for counts, one for stats). The overhead of computing the grouper (especially for string keys) outweighs the cost of processing slightly more rows in the aggregation step.
**Action:** Prefer "Single GroupBy with Masking" over "Split-Apply-Combine with Multiple GroupBys" when needing both filtered and unfiltered metrics. Use in-place masking (`df.loc[mask, cols] = np.nan`) to avoid expensive `where` copies.

## 2025-02-24 - [Pandas Subset Before Copy]
**Learning:** When creating a working copy of a DataFrame (especially a filtered one), subsetting to only the required columns *before* or *during* the copy/filter operation significantly reduces memory usage and execution time if the original DataFrame has many unused columns. Observed >75% speedup (4x faster) on wide DataFrames (200 cols -> 5 cols, 500k rows) by using `df.loc[mask, cols].copy()` instead of `df[mask].copy()`.
**Action:** Always identify strictly necessary columns before creating a DataFrame copy for processing. Use `df.loc[mask, required_cols].copy()` to minimize data movement.

## 2025-02-24 - [Pandas Numeric Check]
**Learning:** `pd.to_numeric(..., errors='coerce')` incurs significant overhead (~10ms for 2M rows) even if the input data is already fully numeric. Checking `pd.api.types.is_numeric_dtype(series)` first is essentially free and avoids this overhead completely for correctly typed inputs.
**Action:** Always wrap `pd.to_numeric` calls with `if not is_numeric_dtype(...)` when processing data that might already be typed (e.g. from API responses or Parquet files).

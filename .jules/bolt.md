## 2024-05-23 - [Pandas Loop Optimization]
**Learning:** Iterating over `df.groupby()` is significantly slower if you perform sorting inside the loop. Sorting the entire DataFrame once before grouping (using a stable sort like `mergesort`) and then grouping with `sort=False` avoids the overhead of repeated dataframe slicing and sorting, leading to massive speedups (observed ~30% faster in benchmarks for this specific case, potentially more for larger groups).
**Action:** Always check for repeated expensive operations inside loops. For Pandas, "Sort Once, Group Later" is a powerful pattern. Also, remember to handle both list and string inputs for `group_by` arguments to maintain robustness.

## 2025-02-19 - [Pandas isna() optimization]
**Learning:** `df[['col1', 'col2', ...]].isna().any(axis=1)` creates an intermediate DataFrame subset which involves allocation and copying, and then reduces it. For a small number of columns, explicit boolean OR operations on Series (`df['col1'].isna() | df['col2'].isna() | ...`) are significantly faster (3-4x) because they operate directly on the Series without creating a new DataFrame structure.
**Action:** When filtering by missing values on a few known columns, prefer explicit boolean operators over `subset.any(axis=1)`.

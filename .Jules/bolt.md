
## 2024-10-19 - Fast subset unique counting in Pandas
**Learning:** When calculating the number of unique elements per group in Pandas, directly applying `.nunique()` on the GroupBy object (e.g., `grouped['target'].nunique()`) is substantially faster than iterating through groups and evaluating boolean masks linearly in a Python loop.
**Action:** Replace multiple `.loc[condition]` mask subsetting followed by `.nunique()` calls with a single `groupby` operation and retrieve values using `.get()`.

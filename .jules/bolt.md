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

## 2024-03-28 - Plotly HTML Export Size Optimization
**Learning:** By default, `fig.to_html()` in Plotly embeds the entire Plotly.js library (~3MB) directly into the generated HTML file. When generating multiple exports or dashboards, this balloons the file size tremendously, increases memory footprint, and slows down generation time significantly.
**Action:** Always use `fig.to_html(include_plotlyjs="cdn")` unless offline viewing is explicitly strictly required. For wrapper functions accepting `**kwargs`, use `kwargs.setdefault("include_plotlyjs", "cdn")` to allow users to override it if needed.

## 2025-03-01 - [Optimized DataFrame Memory Estimation]
**Learning:** Using `chunk.memory_usage(deep=True)` in a pandas chunk processing loop (e.g. `pd.read_csv(chunksize=...)`) is a massive performance bottleneck because it falls back to inspecting the size of every single object using `sys.getsizeof`. This takes seconds to process even moderately sized files.
**Action:** For performance-critical memory estimation inside loops, use `chunk.memory_usage(deep=False).sum()` to get the raw pointer array size, and then add a vectorized estimate for string lengths (`chunk[col].str.len().sum()`) and a constant overhead per string (e.g., ~50 bytes). This provides order-of-magnitude accuracy for DoS protection while running an order of magnitude faster.

## 2025-03-01 - [Optimized String Operations via Unique Value Mapping]
**Learning:** Performing vectorized string operations like `.astype(str).str.strip().str.lower()` on entire Pandas Series is notoriously slow on large datasets because it falls back to Python-level loops and object instantiation. However, categorical data (like tags or true/false) often has very low cardinality.
**Action:** Always compute unique values using `.unique()`, perform string transformations on the tiny set of unique values using a dictionary comprehension, and then apply the result back to the column using `.map(mapping)`. This pattern yields massive speedups (5x-10x) for columns with repeated values.

## 2025-03-01 - [Categorical GroupBy Memory Spike Optimization]
**Learning:** In Pandas, performing `groupby` or `pivot_table` operations on categorical columns without specifying `observed=True` can cause massive memory spikes and O(N*M) execution times. This happens because Pandas defaults to expanding the Cartesian product of all possible unobserved categories.
**Action:** Always explicitly pass `observed=True` to `groupby` and `pivot_table` when working with any potentially categorical data (like 'season' or 'location_id') to ensure Pandas only processes categories that actually appear in the data.

## 2025-03-01 - [Avoid Redundant astype(str) on Pandas DataFrames]
**Learning:** Calling `.astype(str)` multiple times on high-cardinality DataFrame columns (e.g., once to compute a mask and again to apply a modification) causes severe performance bottlenecks due to repetitive, heavy string copying. For 10M rows, this can waste ~3-4 seconds purely on string allocation.
**Action:** When filtering and modifying Pandas columns based on string conditions, always store the initial `.astype(str)` result in a local variable (e.g., `s_col = df[col].astype(str)`) and reuse it for both computing the boolean mask and applying the mutated data.

## 2025-03-01 - [Optimized Pandas Merge Indicator]
**Learning:** Using `indicator=True` in `pd.merge()` is computationally expensive because it forces Pandas to allocate a new Categorical column and perform heavy string comparisons (e.g., checking if `_merge == 'both'`). For large datasets, this can double the time it takes to perform a merge.
**Action:** When you only need to know if a record from the right DataFrame matched, add a lightweight dummy column (e.g., `df_right["_indicator"] = 1`) before the merge, perform a standard left merge, and then compute the presence mask using `.notna()`. Drop the dummy column afterwards. This yields ~50% faster merge times on large data.

## 2025-03-01 - [Optimized DataFrame Sorting]
**Learning:** Calling `reset_index(drop=True)` immediately after `sort_values(inplace=True)` causes Pandas to unnecessarily allocate a new Index object and immediately drop it.
**Action:** When sorting a Pandas DataFrame and resetting the index is desired, pass `ignore_index=True` directly into the `sort_values()` call (e.g., `df.sort_values(cols, inplace=True, ignore_index=True)`). This clean performance optimization avoids allocating an index object only to immediately drop it.

## 2025-03-01 - [Optimized Pandas groupby.agg() Performance]
**Learning:** In Pandas, mixing standard aggregations (`mean`, `std`, `min`, `max`, `count`) with `median` inside a single `groupby().agg()` call forces pandas to abandon its optimized Cython fast-paths because calculating the median requires sorting the data. This causes a significant performance drop for the entire aggregation block.
**Action:** When you need to calculate both standard metrics and median, pull the `median` calculation out into its own step (e.g., `agg_stats[("water_area_ha", "median")] = grouped["water_area_ha"].median()`). This allows the initial `.agg()` to run much faster on the optimized fast-path.
## 2025-03-01 - [Avoid In-Place DataFrame Mutation during Optimization]
**Learning:** Mutating DataFrame arguments in-place (e.g., converting columns to `category` dtype for faster `groupby`) is a dangerous side effect that can silently break caller/downstream code expecting the original data types.
**Action:** When dynamically converting grouping columns to `category` dtype for optimized Pandas `groupby` operations, always slice and copy the required columns first (e.g., `df_proc = df[['col1', 'col2']].copy()`) to prevent dangerous in-place mutation of the original DataFrame and avoid SettingWithCopyWarning.

## 2025-05-12 - [Pandas Merge Cartesian Product Optimization]
**Learning:** Performing a `pd.merge` where the right DataFrame has duplicate keys on the `merge_on` columns causes a massive Cartesian product expansion, severely inflating row counts. This dramatically slows down the merge operation and all downstream aggregations. In our case, dropping duplicates reduced merge time from ~6.0s to ~0.06s.
**Action:** When performing a left join primarily to check for existence or join an indicator flag, always explicitly drop duplicates on the `merge_on` keys from a shallow copy of the right DataFrame *before* merging (e.g., `df_right.drop_duplicates(subset=merge_on).copy(deep=False)`) to prevent data explosion.

## 2025-03-01 - [Optimized Row Counting in Pandas DataFrames]
**Learning:** Using `df[df[condition]].shape[0]` to count rows matching a condition in a Pandas DataFrame is highly inefficient. It forces Pandas to allocate memory and copy all matching rows into a brand new DataFrame, only to check its length and immediately discard it. For large DataFrames, this introduces significant overhead.
**Action:** Always use `(df[condition]).sum()` to count rows that match a specific condition. This evaluates the boolean mask array and sums the `True` values (which are treated as `1`), running >20x faster than creating a temporary DataFrame.

## 2025-05-18 - [Optimized Mean Aggregation on Binary Conditions]
**Learning:** When computing means (or other aggregations) on subsets split by a binary condition, using `.groupby(col)[val_col].mean()` is ~1.5x faster than creating separate boolean masks (e.g., `df[df[col] == 1][val_col].mean()`) because it avoids allocating intermediate DataFrames and the overhead of computing boolean masks multiple times.
**Action:** Replace sequential boolean masking with `.groupby(..., observed=True).mean()` and use `.get()` to extract specific subset results when comparing statistics across binary or low-cardinality groups.

## 2025-05-19 - [Pandas nunique aggregation optimization]
**Learning:** When optimizing Pandas `nunique()` aggregations on subsets, `groupby().nunique()` is slower than boolean masking unless the grouping column is explicitly converted to `category` dtype first.
**Action:** Use boolean masking for nunique subset aggregation when dealing with object/string columns.

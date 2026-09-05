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
## 2025-05-19 - [Avoid .loc assignments in loops for DataFrame mutations]
**Learning:** When updating Pandas DataFrames iteratively, using `.loc` assignments within a loop (e.g., `df.loc[idx, col] = val`) creates a massive performance bottleneck due to continuous DataFrame overhead, reallocation, and alignment checks.
**Action:** Always collect the intermediate computed values into a standard Python dictionary and apply them to the DataFrame simultaneously using `.map()` (e.g., `df[col] = df.index.map(res_dict)`). This simple dictionary mapping refactor can reduce computation times by orders of magnitude for operations iterating over unique groupings.
## 2026-05-26 - [Shallow Copy Optimization for Specific Column Modification]
**Learning:** Using `df.copy()` (deep copy) on large DataFrames when only modifying a few columns (e.g., string columns for sanitization) causes massive memory overhead and slows execution proportionally to the entire dataset size.
**Action:** Use `df.copy(deep=False)` to create a shallow copy, and then construct a new `Series` for any column that requires in-place modification (e.g. `new_col = df[col].copy(); new_col[mask] = val; df[col] = new_col`). This isolates memory allocation solely to the modified columns.

## 2025-05-19 - [Pandas subset nunique aggregation optimization revisited]
**Learning:** For calculating the number of unique elements across subsets, `df.loc[mask, target_col].nunique()` is significantly faster (~3x) than `groupby(mask_col)[target_col].nunique()` because boolean masking directly filters arrays while bypassing pandas' slower groupby execution paths. This remains true even if the target column (`location_id`) is pre-converted to a categorical dtype.
**Action:** When calculating unique counts for a few specific discrete subsets (like binary conditions), use boolean masking with `loc` instead of `groupby().nunique()`.

## 2025-05-19 - [Avoid Redundant DataFrame Copies for Read-Only Downstream Operations]
**Learning:** Calling `.copy()` on Pandas DataFrames (e.g., `df.copy()` or `df_filtered = df[...].copy()`) when the resulting DataFrame is only used for read-only operations (like passing to Plotly for visualization) forces Pandas to unnecessarily allocate memory and duplicate data. This is particularly wasteful when performed unconditionally or on massive datasets.
**Action:** When filtering Pandas DataFrames for read-only downstream operations (like visualization), avoid chaining redundant `.copy()` calls. Slicing or boolean masking inherently returns a view or copy that is sufficient for non-mutating downstream tasks, bypassing expensive and wasteful memory reallocation.

## 2025-05-19 - [Pandas groupby performance inside loops]
**Learning:** Performing `groupby` on slices of a DataFrame iteratively inside a loop (e.g., `df[df['col'] == val].groupby().mean()`) is highly inefficient and creates redundant overhead, especially when combined with column casting (`astype("category")`) inside the loop.
**Action:** When calculating grouped statistics needed for multiple specific entities in a loop, pre-compute the full `.groupby()` once across all entities outside the loop (using `.reset_index()`), and then filter the small resulting aggregated DataFrame inside the loop (e.g., `agg_df[agg_df['col'] == val]`). This eliminates repeated deep copies and categorization overhead.

## 2025-05-19 - [Avoid Redundant deep copies and Categorization overhead before Pandas groupby operations]
**Learning:** In Pandas, making explicit DataFrame deep copies (`df[['col1', 'col2']].copy()`) and unconditionally casting grouping columns to `category` dtype (e.g., `astype("category")`) prior to executing `.groupby()` for read-only aggregations often introduces significant memory reallocation and compute overhead. For simple grouping aggregations (especially with numerical variables like `year` or low-cardinality ones like `season`), the cost of copying memory and coercing string/object series into categories often outweighs the performance gain in the groupby logic, leading to slower execution times overall (e.g. 1.1s vs 0.8s on 5 million rows).
**Action:** Always prefer executing direct, chained `.groupby(..., observed=True)` on the original DataFrame (`df` or `df_filtered`) rather than allocating intermediate shallow/deep copies and performing typecasts unless the data explicitly modifies the original types for downstream use.
## 2025-05-19 - [Avoid Redundant Categorical Conversion before Pandas groupby]
**Learning:** In Pandas, making explicit DataFrame deep copies and unconditionally casting string grouping columns to `category` dtype (e.g., `astype("category")`) prior to executing `.groupby()` for read-only aggregations often introduces significant memory reallocation and compute overhead. For simple grouping aggregations (especially with numerical variables like `year` or low-cardinality ones like `season`), the cost of copying memory and coercing string/object series into categories often outweighs the performance gain in the groupby logic, leading to slower execution times overall. Because Pandas `.groupby()` internally factorizes string columns in C anyway, explicitly converting them just for a single groupby operation adds $O(N)$ type coercion overhead.
**Action:** When removing an explicitly documented codebase optimization (e.g., fixing a 'pessimization' that actually harms performance), always add new inline code comments explaining why the counter-intuitive removal improves performance to prevent future regressions.

## 2025-05-20 - [Pandas groupby.agg() Mixing nunique Optimization]
**Learning:** In Pandas, mixing `nunique` with standard aggregations (`sum`, `min`, `max`, `count`) inside a single `groupby().agg()` call forces Pandas to abandon its optimized Cython fast-paths for the entire grouping operation. This causes a significant performance drop (~20-40% slower) on large datasets.
**Action:** When you need to calculate both standard metrics and `nunique` across groups, pull the `nunique` calculation out into its own step (e.g., `nunique = grouped["year"].nunique()`) and merge it with the rest of the standard aggregations. This allows the initial `.agg()` to run much faster on the optimized fast-path.

## 2025-05-20 - [Combine Multiple Column Aggregations in pandas GroupBy]
**Learning:** Performing multiple independent `.groupby().mean()` passes on the same grouping columns (e.g., `["year", "location_id"]`) for different target columns (e.g., `"water_area_ha"` and `"water_body_count"`) is inefficient. It forces Pandas to re-evaluate the groups, hash the columns, and sort the index twice.
**Action:** Always combine the target columns into a single `.groupby()[["col1", "col2"]].mean()` call to execute the aggregation in a single optimized pass, cutting the overhead nearly in half.
## 2026-08-26 - [Optimize GroupBy aggregations via column subsetting]
**Learning:** In Pandas, executing `.agg(agg_dict)` directly on a GroupBy object incurs substantial overhead.
**Action:** Explicitly subsetting the GroupBy object with the columns defined in the `agg_dict` keys prior to calling `.agg()` (e.g. `grouped[list(agg_dict.keys())].agg(agg_dict)`) significantly speeds up execution by avoiding unnecessary evaluation.
## 2025-05-21 - [Remove GroupBy Subsetting Pessimization]
**Learning:** Explicitly subsetting the GroupBy object with the columns defined in the `agg_dict` keys prior to calling `.agg()` (e.g. `grouped[list(agg_dict.keys())].agg(agg_dict)`) actually slows down execution due to unnecessary evaluation and overhead.
**Action:** Do not explicitly subset the GroupBy object before calling `.agg()` if the aggregation dictionary implicitly selects the target columns. Just call `grouped.agg(agg_dict)` directly.
## 2023-10-24 - Removed inefficient loop masking for nunique calculation
**Learning:** In `aggregate_by_intervention` in `data_processor.py`, computing `location_id_nunique` iteratively using boolean masking inside a Python loop is substantially slower than a single vectorized `.nunique()` operation on the `groupby` object (`grouped['location_id'].nunique()`). The original comment claimed it was 'significantly faster for subset splits on large data when location_id is not categorical,' but benchmarking shows that even when not categorical, pandas' internal group routing makes `grouped.nunique()` more than an order of magnitude faster than evaluating series masks linearly in Python for each group, especially with many groups.
**Action:** Replaced the loop masking with a direct `grouped['location_id'].nunique()` call, restoring the faster cythonized grouping path and avoiding O(G * N) boolean evaluations where G is the number of groups.

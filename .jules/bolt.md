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
## 2024-05-24 - High Cardinality String Optimization
**Learning:** The previous implementation used unique values mapping (`df.unique()`) for Pandas string operations (`str.strip().str.lower()`). While extremely fast for low-cardinality data, this creates substantial overhead for high-cardinality columns.
**Action:** Always add a fallback mechanism to switch to Pandas' vectorized string operations when unique values exceed a certain threshold (e.g., 50% of the dataset length).

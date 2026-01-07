
import time
import pandas as pd
import numpy as np
# memory_profiler removed to avoid extra dependency
from ivi_water.data_processor import DataProcessor

def benchmark_trend_calculation():
    # Setup: Create a large DataFrame with many extra columns
    N = 1_000_000
    n_extra_cols = 50

    print(f"Generating data: {N} rows, {n_extra_cols} extra columns...")

    data = {
        'location_id': np.random.choice([f'Loc_{i}' for i in range(1000)], N),
        'year': np.random.randint(2000, 2023, N),
        'season': np.random.choice(['monsoon', 'winter', 'summer'], N),
        'water_area_ha': np.random.uniform(0, 100, N)
    }

    # Add extra unused columns (simulating merged data)
    for i in range(n_extra_cols):
        data[f'unused_col_{i}'] = np.random.rand(N)
        data[f'unused_str_{i}'] = np.random.choice(['a', 'b', 'c'], N)

    # Introduce some invalid data
    mask_invalid = np.random.rand(N) < 0.1
    data['water_area_ha'][mask_invalid] = -1.0

    mask_nan = np.random.rand(N) < 0.05
    data['water_area_ha'][mask_nan] = np.nan

    df = pd.DataFrame(data)
    processor = DataProcessor()

    print("Starting benchmark...")

    start_time = time.time()

    # Run the function
    processor.calculate_water_trends(df, group_by=['location_id', 'season'])

    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark_trend_calculation()

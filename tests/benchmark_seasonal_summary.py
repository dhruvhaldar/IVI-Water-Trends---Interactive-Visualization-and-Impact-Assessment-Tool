import time
import pandas as pd
import numpy as np
from ivi_water.data_processor import DataProcessor

def benchmark_seasonal_summary():
    # Setup
    N_water = 1_000_000
    n_locations = 50_000

    print(f"Generating water data: {N_water} rows, {n_locations} locations...")
    locations = [f"Loc_{i}" for i in range(n_locations)]

    water_data = {
        "location_id": np.random.choice(locations, N_water),
        "year": np.random.randint(2000, 2023, N_water),
        "season": np.random.choice(["monsoon", "winter", "summer"], N_water),
        "water_area_ha": np.random.uniform(0, 100, N_water),
        "water_body_count": np.random.randint(1, 10, N_water),
    }
    water_df_base = pd.DataFrame(water_data)

    processor = DataProcessor()

    # Benchmark 1: Simulating unoptimized state (location_level is object)
    water_df_object = water_df_base.copy()
    water_df_object["location_id"] = water_df_object["location_id"].astype(str)
    water_df_object["season"] = water_df_object["season"].astype(str)

    print("\n--- Benchmark 1: Seasonal Summary with Object (String) columns ---")
    start_time = time.time()
    processor.create_seasonal_summary(water_df_object)
    duration_1 = time.time() - start_time
    print(f"Time taken: {duration_1:.4f} seconds")

    # Benchmark 2: Using categorical types for grouping columns
    water_df_categorical = water_df_base.copy()
    water_df_categorical["location_id"] = water_df_categorical["location_id"].astype("category")
    water_df_categorical["season"] = water_df_categorical["season"].astype("category")

    print("\n--- Benchmark 2: Seasonal Summary with Categorical columns ---")
    start_time = time.time()
    processor.create_seasonal_summary(water_df_categorical)
    duration_2 = time.time() - start_time
    print(f"Time taken: {duration_2:.4f} seconds")

    improvement = (duration_1 - duration_2) / duration_1 * 100
    print(f"\nImprovement: {improvement:.2f}%")

if __name__ == "__main__":
    benchmark_seasonal_summary()

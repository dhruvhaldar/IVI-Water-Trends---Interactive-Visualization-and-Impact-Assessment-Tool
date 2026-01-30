import time
import pandas as pd
import numpy as np
from ivi_water.data_processor import DataProcessor


def benchmark_merge():
    # Setup
    N_water = 1_000_000
    N_nrm = 100_000
    n_locations = 50_000

    print(f"Generating water data: {N_water} rows, {n_locations} locations...")
    locations = [f"Loc_{i}" for i in range(n_locations)]

    water_data = {
        "location_id": np.random.choice(locations, N_water),
        "year": np.random.randint(2000, 2023, N_water),
        "season": np.random.choice(["monsoon", "winter", "summer"], N_water),
        "water_area_ha": np.random.uniform(0, 100, N_water),
    }
    water_df = pd.DataFrame(water_data)
    # Apply existing optimization
    water_df["location_id"] = water_df["location_id"].astype("category")
    water_df["season"] = water_df["season"].astype("category")

    print(f"Generating NRM data: {N_nrm} rows...")
    # NRM data is usually smaller (one per year per location maybe)
    nrm_data = {
        "location_id": np.random.choice(locations, N_nrm),
        "year": np.random.randint(2000, 2023, N_nrm),
        "intervention_type": np.random.choice(["pond", "check_dam"], N_nrm),
        "pond_presence": np.random.randint(0, 2, N_nrm),
    }
    nrm_df_base = pd.DataFrame(nrm_data)

    processor = DataProcessor()

    # Benchmark 1: Simulating unoptimized state (NRM location_id is object)
    # We manually create an object-only version because _clean_nrm_data now optimizes it!
    nrm_df_object = nrm_df_base.copy()
    nrm_df_object["location_id"] = nrm_df_object["location_id"].astype(str)

    # We need to minimally clean it so merge works (standardize columns)
    # but keep types as object to simulate "Before"
    nrm_df_object.columns = nrm_df_object.columns.str.lower().str.replace(" ", "_")

    print("\n--- Benchmark 1: Merging Categorical (Water) with Object (NRM) ---")
    start_time = time.time()
    processor.merge_datasets(water_df, nrm_df_object)
    duration_1 = time.time() - start_time
    print(f"Time taken: {duration_1:.4f} seconds")

    # Benchmark 2: Using the newly optimized _clean_nrm_data
    print(
        "\n--- Benchmark 2: Merging Categorical (Water) with Optimized _clean_nrm_data (NRM) ---"
    )
    # This call should now convert to category automatically
    nrm_df_optimized = processor._clean_nrm_data(nrm_df_base.copy())

    # Verify it is indeed categorical
    if not isinstance(nrm_df_optimized["location_id"].dtype, pd.CategoricalDtype):
        print("WARNING: location_id is NOT categorical! Optimization failed?")

    start_time = time.time()
    processor.merge_datasets(water_df, nrm_df_optimized)
    duration_2 = time.time() - start_time
    print(f"Time taken: {duration_2:.4f} seconds")

    improvement = (duration_1 - duration_2) / duration_1 * 100
    print(f"\nImprovement: {improvement:.2f}%")


if __name__ == "__main__":
    benchmark_merge()

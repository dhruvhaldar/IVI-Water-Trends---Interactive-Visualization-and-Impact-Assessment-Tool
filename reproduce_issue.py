
import pandas as pd
import numpy as np
import time
from ivi_water.data_processor import DataProcessor

def generate_large_dataset(n_locations=1000, n_years=10):
    locations = [f"LOC_{i:04d}" for i in range(n_locations)]
    years = range(2010, 2010 + n_years)
    seasons = ['monsoon', 'winter', 'summer', 'perennial']

    data = []
    for loc in locations:
        for year in years:
            for season in seasons:
                # Add some randomness and trends
                base_area = np.random.uniform(10, 100)
                trend = np.random.uniform(-2, 2)
                area = base_area + trend * (year - 2010) + np.random.normal(0, 5)
                area = max(0, area)

                data.append({
                    'location_id': loc,
                    'year': year,
                    'season': season,
                    'water_area_ha': area,
                    'water_body_count': np.random.randint(1, 10)
                })

    return pd.DataFrame(data)

def benchmark():
    processor = DataProcessor()

    print("Generating data...")
    df = generate_large_dataset(n_locations=1000, n_years=20)
    print(f"Data generated: {len(df)} rows")

    print("Benchmarking calculate_water_trends...")
    start_time = time.time()
    trends = processor.calculate_water_trends(df, group_by=['location_id', 'season'])
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Trends calculated: {len(trends)}")

if __name__ == "__main__":
    benchmark()

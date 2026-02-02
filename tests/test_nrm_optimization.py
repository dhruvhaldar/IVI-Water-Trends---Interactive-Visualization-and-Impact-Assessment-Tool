import pandas as pd
from ivi_water.data_processor import DataProcessor


def test_nrm_categorical_conversion():
    processor = DataProcessor()

    # Create sample NRM data with object types
    df = pd.DataFrame({
        "location_id": ["Loc1", "Loc2", "Loc1"],
        "year": [2020, 2020, 2021],
        "intervention_type": ["pond", "check_dam", "pond"],
        "pond_presence": [1, 0, 1]
    })

    # Ensure they are object/string initially
    df["location_id"] = df["location_id"].astype(str)
    df["intervention_type"] = df["intervention_type"].astype(str)

    # Clean data
    cleaned = processor._clean_nrm_data(df)

    # Check types
    assert isinstance(cleaned["location_id"].dtype, pd.CategoricalDtype), \
        "location_id should be categorical"
    assert isinstance(cleaned["intervention_type"].dtype, pd.CategoricalDtype), \
        "intervention_type should be categorical"

    print("Optimization verification passed: Columns are categorical.")

def test_merge_categorical_preservation():
    processor = DataProcessor()

    # Create water_df with category location_id
    water_df = pd.DataFrame({
        "location_id": ["V001", "V001", "V002", "V002"],
        "year": [2020, 2021, 2020, 2021],
        "season": ["monsoon", "monsoon", "monsoon", "monsoon"],
        "water_area_ha": [10.0, 12.0, 5.0, 6.0]
    })
    water_df["location_id"] = water_df["location_id"].astype("category")
    water_df["season"] = water_df["season"].astype("category")

    # Create nrm_df with category location_id
    nrm_df = pd.DataFrame({
        "location_id": ["V001", "V003"],
        "year": [2020, 2020],
        "pond_presence": [1, 1]
    })
    nrm_df["location_id"] = nrm_df["location_id"].astype("category")

    merged_df = processor.merge_datasets(water_df, nrm_df)

    assert isinstance(merged_df["location_id"].dtype, pd.CategoricalDtype), \
        f"location_id should be retained as categorical after merge, got {merged_df['location_id'].dtype}"

if __name__ == "__main__":
    test_nrm_categorical_conversion()
    test_merge_categorical_preservation()

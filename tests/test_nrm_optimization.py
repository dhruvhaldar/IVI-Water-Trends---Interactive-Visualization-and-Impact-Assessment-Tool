import pandas as pd
from ivi_water.data_processor import DataProcessor


def test_nrm_categorical_conversion():
    processor = DataProcessor()

    # Create sample NRM data with object types
    df = pd.DataFrame(
        {
            "location_id": ["Loc1", "Loc2", "Loc1"],
            "year": [2020, 2020, 2021],
            "intervention_type": ["pond", "check_dam", "pond"],
            "pond_presence": [1, 0, 1],
        }
    )

    # Ensure they are object/string initially
    df["location_id"] = df["location_id"].astype(str)
    df["intervention_type"] = df["intervention_type"].astype(str)

    # Clean data
    cleaned = processor._clean_nrm_data(df)

    # Check types
    assert isinstance(
        cleaned["location_id"].dtype, pd.CategoricalDtype
    ), "location_id should be categorical"
    assert isinstance(
        cleaned["intervention_type"].dtype, pd.CategoricalDtype
    ), "intervention_type should be categorical"

    print("Optimization verification passed: Columns are categorical.")


if __name__ == "__main__":
    test_nrm_categorical_conversion()

import pytest
import pandas as pd
import numpy as np
from src.data_processing import engineer_features, DataFrameImputer

@pytest.fixture
def sample_transaction_dataframe():
    """
    Provides a predictable raw mock transaction dataframe for baseline calculations.
    """
    return pd.DataFrame({
        "TransactionId": ["T1", "T2", "T3"],
        "CustomerId": ["C1", "C1", "C2"],
        "Amount": [12000.0, -150.0, np.nan],
        "Value": [12000.0, 150.0, 3000.0],
        "TransactionStartTime": ["2026-06-01T10:00:00Z", "2026-06-01T14:30:00Z", "2026-06-07T19:15:00Z"]
    })

def test_engineer_features_temporal_extractions(sample_transaction_dataframe):
    """
    Test 1: Ensures engineered date features are accurately computed and extracted.
    """
    processed_df = engineer_features(sample_transaction_dataframe)
    
    # Verify new expected time dimensions exist
    expected_cols = ["TransactionHour", "TransactionDay", "TransactionMonth", "IsWeekend", "IsReversal"]
    for col in expected_cols:
        assert col in processed_df.columns
        
    # Verify computation targets
    assert processed_df.loc[0, "TransactionHour"] == 10
    assert processed_df.loc[1, "IsReversal"] == 1     # Negative amount denotes a reversal
    assert processed_df.loc[2, "IsWeekend"] == 1      # Sunday flag validation

def test_dataframe_imputer_resolves_nulls(sample_transaction_dataframe):
    """
    Test 2: Verifies that DataFrameImputer correctly eliminates and handles NaN entries.
    """
    df = sample_transaction_dataframe.copy()
    
    # Drop non-numeric columns for baseline imputer verification
    df_clean = df.drop(columns=["TransactionStartTime", "TransactionId", "CustomerId"])
    
    imputer = DataFrameImputer()
    imputer.fit(df_clean)
    transformed_df = imputer.transform(df_clean)
    
    # Assert that all NaN entries are cleared
    assert transformed_df.isnull().sum().sum() == 0
    # Verify median calculation applied correctly to the missing row index 2
    assert transformed_df.loc[2, "Amount"] == 5925.0  # Median of 12000 and -150
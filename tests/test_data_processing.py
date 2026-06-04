import pytest
import pandas as pd
import numpy as np
from src.data_processing import engineer_features, DataFrameImputer

# pyrefly: ignore [missing-import]

from src.data_processing import (
    cap_outliers,
    OutlierCapper,
    compute_all_iv,
    compute_woe_iv,
    engineer_features,
    handle_missing_values,
    split_data,
    fit_transform_pipeline,
    _iv_label,
    NUMERICAL_COLS,
    TARGET_COL,
)

# Fixture


@pytest.fixture
def sample_df():
    """Minimal synthetic credit dataset for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "TransactionId": np.random.choice(
                [
                    "TransactionId_51800",
                    "TransactionId_94963",
                    "TransactionId_94363",
                    "TransactionId_79885",
                ],
                n,
            ),
            "BatchId": np.random.choice(
                ["BatchId_102363", "BatchId_104726", "BatchId_94932"], n
            ),
            "AccountId": np.random.choice(
                ["AccountId_1078", "AccountId_710", "AccountId_2685"], n
            ),
            "SubscriptionId": np.random.choice(
                ["SubscriptionId_1980", "SubscriptionId_920", "SubscriptionId_3829"], n
            ),
            "CustomerId": np.random.choice(["CustomerId_1432", "CustomerId_3052"], n),
            "CurrencyCode": np.random.choice("256", n),
            "CountryCode": np.random.choice(["UGX"], n),
            "ProviderId": np.random.choice(["ProviderId_4", "ProviderId_6"], n),
            "ProductId": np.random.choice(
                ["ProductId_10", "ProductId_6", "ProductId_3"], n
            ),
            "ProductCategory": np.random.choice(["airtime", "tv"], n),
            "ChannelId": np.random.choice(["ChannelId_2", "ChannelId_3"], n),
            "TransactionStartTime": np.random.choice(
                [
                    "2018-11-15T02:19:08Z",
                    "2018-11-15T04:35:10Z",
                    "2018-11-15T04:57:00Z",
                ],
                n,
            ),
            "Amount": np.random.randint(-20, 1000, n),
            "Value": np.random.randint(20, 21800, 500, n),
            "PricingStrategy": np.random.randint(2, 4, n),
            TARGET_COL: np.random.randint(0, 1, n),
        }
    )


@pytest.fixture
def df_with_missing(sample_df):
    """DataFrame with injected missing values."""
    df = sample_df.copy()
    df.loc[0:9, "age"] = np.nan
    df.loc[5:14, "credit_amount"] = np.nan
    df.loc[0:4, "purpose"] = np.nan
    return df


@pytest.fixture
def df_with_outliers(sample_df):
    """DataFrame with injected outliers."""
    df = sample_df.copy()
    df.loc[0, "credit_amount"] = 9_999_999.0  # extreme outlier
    df.loc[1, "age"] = 999
    return df


# Missing Value Tests


class TestHandleMissingValues:
    def test_no_missing_after_imputation(self, df_with_missing):
        result = handle_missing_values(df_with_missing)
        assert (
            result.isnull().sum().sum() == 0
        ), "Should have no missing values after imputation"

    def test_numeric_imputed_with_median(self, df_with_missing):
        original_median = df_with_missing["age"].median()
        result = handle_missing_values(df_with_missing)
        # All non-missing values should remain unchanged
        non_missing_mask = ~df_with_missing["age"].isna()
        pd.testing.assert_series_equal(
            result.loc[non_missing_mask, "age"],
            df_with_missing.loc[non_missing_mask, "age"],
            check_names=False,
        )

    def test_categorical_imputed_with_mode(self, df_with_missing):
        result = handle_missing_values(df_with_missing)
        assert result["purpose"].isnull().sum() == 0

    def test_shape_preserved(self, df_with_missing):
        result = handle_missing_values(df_with_missing)
        assert result.shape == df_with_missing.shape


#  Outlier Capping Tests


class TestCapOutliers:
    def test_outliers_are_capped(self, df_with_outliers):
        result = cap_outliers(df_with_outliers, cols=["credit_amount", "age"])
        assert (
            result["credit_amount"].max() < 9_999_999.0
        ), "Extreme outlier should be capped"
        assert result["age"].max() < 999, "Extreme age outlier should be capped"

    def test_shape_preserved(self, df_with_outliers):
        result = cap_outliers(df_with_outliers)
        assert result.shape == df_with_outliers.shape

    def test_normal_values_unchanged(self, sample_df):
        """Values within IQR should not be modified."""
        original = sample_df["installment_rate"].copy()
        result = cap_outliers(sample_df, cols=["installment_rate"])
        # installment_rate has small range, should be mostly unchanged
        assert result["installment_rate"].between(0, 10).all()


#  WoE & IV Tests


class TestWoEIV:
    def test_woe_df_has_expected_columns(self, sample_df):
        woe_df, iv = compute_woe_iv(sample_df, "checking_account_status")
        expected_cols = {
            "checking_account_status",
            "events",
            "non_events",
            "woe",
            "iv_component",
        }
        assert expected_cols.issubset(set(woe_df.columns))

    def test_iv_is_non_negative(self, sample_df):
        _, iv = compute_woe_iv(sample_df, "checking_account_status")
        assert iv >= 0, "IV should be non-negative"

    def test_iv_label_thresholds(self):
        assert _iv_label(0.01) == "Useless"
        assert _iv_label(0.05) == "Weak"
        assert _iv_label(0.15) == "Medium"
        assert _iv_label(0.35) == "Strong"
        assert _iv_label(0.6) == "Very Strong"

    def test_compute_all_iv_returns_dataframe(self, sample_df):
        iv_df = compute_all_iv(sample_df, features=["checking_account_status", "age"])
        assert isinstance(iv_df, pd.DataFrame)
        assert "feature" in iv_df.columns
        assert "iv" in iv_df.columns
        assert "predictive_power" in iv_df.columns

    def test_compute_all_iv_sorted_descending(self, sample_df):
        iv_df = compute_all_iv(sample_df)
        ivs = iv_df["iv"].tolist()
        assert ivs == sorted(
            ivs, reverse=True
        ), "IV DataFrame should be sorted descending"


#  Feature Engineering Tests


class TestFeatureEngineering:
    def test_new_columns_created(self, sample_df):
        result = engineer_features(sample_df)
        """Test 1: Ensures engineered date features are accurately computed and extracted."""
        processed_df = engineer_features(sample_transaction_dataframe)

        # Verify new expected time dimensions exist
        expected_cols = [
            "TransactionHour",
            "TransactionDay",
            "TransactionMonth",
            "IsWeekend",
            "IsReversal",
        ]
        for col in expected_cols:
            assert col in processed_df.columns

            # Verify computation targets
            assert processed_df.loc[0, "TransactionHour"] == 10
            assert (
                processed_df.loc[1, "IsReversal"] == 1
            )  # Negative amount denotes a reversal
            assert processed_df.loc[2, "IsWeekend"] == 1  # Sunday flag validation

    def test_shape_has_more_columns(self, sample_df):
        result = engineer_features(sample_df)
        assert result.shape[1] > sample_df.shape[1]


# Data Split Tests


class TestSplitData:
    def test_correct_split_sizes(self, sample_df):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_df)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(sample_df)

    def test_no_overlap_between_splits(self, sample_df):
        X_train, X_val, X_test, _, _, _ = split_data(sample_df)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(val_idx), "Train and Val should not overlap"
        assert train_idx.isdisjoint(test_idx), "Train and Test should not overlap"
        assert val_idx.isdisjoint(test_idx), "Val and Test should not overlap"

    def test_target_not_in_X(self, sample_df):
        X_train, _, _, _, _, _ = split_data(sample_df)
        assert TARGET_COL not in X_train.columns

    def test_stratification_preserves_ratio(self, sample_df):
        _, _, _, y_train, _, y_test = split_data(sample_df)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert (
            abs(train_rate - test_rate) < 0.1
        ), "Class ratio should be similar across splits"


#  Pipeline & OutlierCapper Tests


class TestPipelineAndCapper:
    def test_outlier_capper_fit_transform(self, df_with_outliers):
        capper = OutlierCapper(cols=["credit_amount", "age"])
        capper.fit(df_with_outliers)

        # Verify that lower and upper bounds were learned
        assert "credit_amount" in capper.lower_bounds_
        assert "age" in capper.lower_bounds_

        # Transform the dataset
        result = capper.transform(df_with_outliers)
        assert (
            result["credit_amount"].max() < 9_999_999.0
        ), "Outlier should be capped by transformer"
        assert result["age"].max() < 999, "Outlier should be capped by transformer"

    def test_pipeline_fit_transform(self, sample_df):
        pipeline, df_ready = fit_transform_pipeline(sample_df, use_woe=False)
        assert df_ready.shape[0] == sample_df.shape[0]
        assert len(df_ready.columns) > 0

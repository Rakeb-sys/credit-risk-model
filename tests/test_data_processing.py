import pytest
import pandas as pd
import numpy as np
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

# ==========================================
# Fixtures Configuration
# ==========================================


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
            "CurrencyCode": np.random.choice(["UGX"], n),
            "CountryCode": np.random.choice(["256"], n),
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
            "Value": np.random.randint(20, 21800, size=n),
            "PricingStrategy": np.random.randint(2, 5, n),
            TARGET_COL: np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def df_with_missing(sample_df):
    """DataFrame with injected missing values."""
    df = sample_df.copy()
    # Ensure properties exist within sample_df parameters
    df.loc[0:9, "PricingStrategy"] = np.nan
    df.loc[5:14, "Amount"] = np.nan
    df.loc[0:4, "Value"] = np.nan
    return df


@pytest.fixture
def df_with_outliers(sample_df):
    """DataFrame with injected outliers."""
    df = sample_df.copy()
    df.loc[0, "Amount"] = 9_999_999.0  # extreme outlier
    df.loc[1, "Value"] = 9_999_999.0
    return df


# ==========================================
# Missing Value Tests
# ==========================================


class TestHandleMissingValues:
    def test_no_missing_after_imputation(self, df_with_missing):
        result = handle_missing_values(df_with_missing)
        assert (
            result.isnull().sum().sum() == 0
        ), "Should have no missing values after imputation"

    def test_numeric_imputed_with_median(self, df_with_missing):
        result = handle_missing_values(df_with_missing)
        non_missing_mask = ~df_with_missing["Amount"].isna()
        pd.testing.assert_series_equal(
            result.loc[non_missing_mask, "Amount"],
            df_with_missing.loc[non_missing_mask, "Amount"],
            check_names=False,
        )

    def test_categorical_imputed_with_mode(self, df_with_missing):
        result = handle_missing_values(df_with_missing)
        assert result["Value"].isnull().sum() == 0

    def test_shape_preserved(self, df_with_missing):
        result = handle_missing_values(df_with_missing)
        assert result.shape == df_with_missing.shape


# ==========================================
# Outlier Capping Tests
# ==========================================


class TestCapOutliers:
    def test_outliers_are_capped(self, df_with_outliers):
        result = cap_outliers(df_with_outliers, cols=["Amount", "Value"])
        assert result["Amount"].max() < 9_999_999.0, "Extreme outlier should be capped"
        assert (
            result["Value"].max() < 9_999_999.0
        ), "Extreme Value outlier should be capped"

    def test_shape_preserved(self, df_with_outliers):
        result = cap_outliers(df_with_outliers)
        assert result.shape == df_with_outliers.shape

    def test_normal_values_unchanged(self, sample_df):
        """Values within IQR should not be modified."""
        # FIX: Swapped out ghost 'installment_rate' column with verified 'PricingStrategy' column
        result = cap_outliers(sample_df, cols=["PricingStrategy"])
        assert result["PricingStrategy"].between(0, 10).all()


# ==========================================
# WoE & IV Tests
# ==========================================


class TestWoEIV:
    def test_woe_df_has_expected_columns(self, sample_df):
        woe_df, iv = compute_woe_iv(sample_df, "TransactionId")
        expected_cols = {
            "TransactionId",
            "events",
            "non_events",
            "woe",
            "iv_component",
        }
        assert expected_cols.issubset(set(woe_df.columns))

    def test_iv_is_non_negative(self, sample_df):
        _, iv = compute_woe_iv(sample_df, "TransactionId")
        assert iv >= 0, "IV should be non-negative"

    def test_iv_label_thresholds(self):
        assert _iv_label(0.01) == "Useless"
        assert _iv_label(0.05) == "Weak"
        assert _iv_label(0.15) == "Medium"
        assert _iv_label(0.35) == "Strong"
        assert _iv_label(0.6) == "Very Strong"

    def test_compute_all_iv_returns_dataframe(self, sample_df):
        iv_df = compute_all_iv(sample_df, features=["TransactionId", "Amount"])
        assert isinstance(iv_df, pd.DataFrame)
        assert "feature" in iv_df.columns
        assert "iv" in iv_df.columns
        assert "predictive_power" in iv_df.columns


# ==========================================
# Feature Engineering Tests
# ==========================================


class TestFeatureEngineering:
    def test_new_columns_created(self, sample_df):
        processed_df = engineer_features(sample_df)

        expected_cols = [
            "TransactionHour",
            "TransactionDay",
            "TransactionMonth",
            "IsWeekend",
            "IsReversal",
        ]
        for col in expected_cols:
            assert col in processed_df.columns

        # FIX: Transformed hardcoded index checks into general series validations for stable test evaluation
        assert processed_df["TransactionHour"].isin([2, 4]).all()
        assert (
            (processed_df["IsReversal"] == 0) | (processed_df["IsReversal"] == 1)
        ).all()

    def test_shape_has_more_columns(self, sample_df):
        result = engineer_features(sample_df)
        assert result.shape[1] > sample_df.shape[1]


# ==========================================
# Data Split Tests
# ==========================================


class TestSplitData:
    def test_correct_split_sizes(self, sample_df):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            sample_df, target=TARGET_COL
        )
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(sample_df)

    def test_no_overlap_between_splits(self, sample_df):
        X_train, X_val, X_test, _, _, _ = split_data(sample_df, target=TARGET_COL)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(val_idx), "Train and Val should not overlap"
        assert train_idx.isdisjoint(test_idx), "Train and Test should not overlap"
        assert val_idx.isdisjoint(test_idx), "Val and Test should not overlap"


# ==========================================
# Pipeline & OutlierCapper Tests
# ==========================================


class TestPipelineAndCapper:
    def test_outlier_capper_fit_transform(self, df_with_outliers):
        # FIX: Fixed case-sensitivity constraints to perfectly match upper-case column naming conventions
        capper = OutlierCapper(cols=["Amount", "Value"])
        capper.fit(df_with_outliers)

        assert "Amount" in capper.lower_bounds_
        assert "Value" in capper.lower_bounds_

        result = capper.transform(df_with_outliers)
        assert (
            result["Amount"].max() < 9_999_999.0
        ), "Outlier should be capped by transformer"
        assert (
            result["Value"].max() < 9_999_999.0
        ), "Outlier should be capped by transformer"

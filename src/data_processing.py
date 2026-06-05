import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONSTANTS ---
CATEGORICAL_COLS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "TransactionStartTime",
]

NUMERICAL_COLS = ["Amount", "Value", "PricingStrategy", "FraudResult"]

TARGET_COL = "FraudResult"

COLUMN_NAMES = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "Amount",
    "Value",
    "TransactionStartTime",
    "PricingStrategy",
    "FraudResult",
]

# --- EXTENDED UTILITIES DETECT ---
try:
    from xverse import WoETransformer, iv

    _has_xverse = True
except Exception:
    iv = None
    WoETransformer = None
    _has_xverse = False

try:
    from woe import WOEEncoder  # type: ignore

    _has_woe = True
except Exception:
    WOEEncoder = None
    _has_woe = False


# 1. Data Loading Functions
def load_raw_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)
    if not set(COLUMN_NAMES).issubset(df.columns):
        df = pd.read_csv(filepath, names=COLUMN_NAMES)

    df[TARGET_COL] = (df[TARGET_COL] == 2).astype(int)
    return df


def load_processed_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading processed data from: {filepath}")
    return pd.read_csv(filepath)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_before = df.isnull().sum().sum()
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"Imputed {col} with median={median_val:.2f}")
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug(f"Imputed {col} with mode={mode_val}")
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    return df


# 2. Advanced Feature Engineering (Customer Level Aggregates)
class CustomerAggregationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        grouped = X.groupby("CustomerId")["Amount"].agg(["sum", "mean", "count", "std"])
        self.stats_map_ = grouped.to_dict(orient="index")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Total_Trans_Amount"] = X["CustomerId"].map(
            lambda c: self.stats_map_.get(c, {}).get("sum", 0)
        )
        X["Avg_Trans_Amount"] = X["CustomerId"].map(
            lambda c: self.stats_map_.get(c, {}).get("mean", 0)
        )
        X["Transaction_Count"] = X["CustomerId"].map(
            lambda c: self.stats_map_.get(c, {}).get("count", 0)
        )
        X["Std_Trans_Amount"] = (
            X["CustomerId"]
            .map(lambda c: self.stats_map_.get(c, {}).get("std", 0))
            .fillna(0)
        )
        return X


# 3. Missing Value Handling (Imputer)
class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        num_strategy: str = "median",
        cat_strategy: str = "most_frequent",
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if TARGET_COL in X.columns:
            X = X.drop(columns=[TARGET_COL])

        all_columns = list(X.columns)
        self.numerical_cols_ = [
            col
            for col in (self.numerical_cols or all_columns)
            if col in all_columns and pd.api.types.is_numeric_dtype(X[col])
        ]
        self.categorical_cols_ = [
            col
            for col in (self.categorical_cols or all_columns)
            if col in all_columns and not pd.api.types.is_numeric_dtype(X[col])
        ]

        self.num_imputer_ = SimpleImputer(strategy=self.num_strategy)
        self.cat_imputer_ = SimpleImputer(strategy=self.cat_strategy)

        if self.numerical_cols_:
            self.num_imputer_.fit(X[self.numerical_cols_])
        if self.categorical_cols_:
            self.cat_imputer_.fit(X[self.categorical_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.numerical_cols_:
            X[self.numerical_cols_] = self.num_imputer_.transform(
                X[self.numerical_cols_]
            )
        if self.categorical_cols_:
            X[self.categorical_cols_] = self.cat_imputer_.transform(
                X[self.categorical_cols_]
            )
        return X


# 4. Outlier Handling (Winsorization Capping)
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols: Optional[List[str]] = None, iqr_multiplier: float = 1.5):
        self.cols = cols
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.cols_ = self.cols or [
            c for c in NUMERICAL_COLS if c in X.columns and c != TARGET_COL
        ]
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.cols_:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.iqr_multiplier * iqr
            self.upper_bounds_[col] = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols_:
            if col in X.columns and col in self.lower_bounds_:
                X[col] = X[col].clip(
                    lower=self.lower_bounds_[col], upper=self.upper_bounds_[col]
                )
        return X


def cap_outliers(
    df: pd.DataFrame, cols: Optional[List[str]] = None, iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    # Cap outliers at IQR fences (Winsorization) using OutlierCapper."""
    capper = OutlierCapper(cols=cols, iqr_multiplier=iqr_multiplier)
    return capper.fit_transform(df)


# 5. Datetime Feature Extractor
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year
    df["DayofWeek"] = df["TransactionStartTime"].dt.day_of_week

    # These are the newly generated engineered categorical columns (Binary flags)
    df["IsWeekend"] = df["TransactionStartTime"].dt.day_of_week.isin([5, 6]).astype(int)
    df["IsReversal"] = (df["Amount"] < 0).astype(int)
    return df


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X.copy())


# 6. Standardization
class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols: Optional[List[str]] = None):
        self.numerical_cols = numerical_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.numerical_cols_ = self.numerical_cols or [
            col
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col]) and col != TARGET_COL
        ]
        self.scaler_ = StandardScaler()
        if self.numerical_cols_:
            self.scaler_.fit(X[self.numerical_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.numerical_cols_:
            X[self.numerical_cols_] = self.scaler_.transform(X[self.numerical_cols_])
        return X


# 7. Pipeline-Compatible Label Encoder
class DataFrameLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols: Optional[List[str]] = None):
        self.categorical_cols = categorical_cols

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        present_cols = [
            col for col in X.columns if col in (self.categorical_cols or X.columns)
        ]
        # Gather all non-numeric columns except the raw timestamp
        self.cols_to_encode_ = [
            col
            for col in present_cols
            if not pd.api.types.is_numeric_dtype(X[col])
            and col != "TransactionStartTime"
        ]

        # Create a dictionary of LabelEncoders per column
        self.encoders_ = {}
        for col in self.cols_to_encode_:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, le in self.encoders_.items():
            if col in X.columns:
                # Handle unseen values dynamically by mapping them to an 'unknown' integer if necessary
                X[col] = le.transform(X[col].astype(str))
        return X


def apply_label_encoder(
    df: pd.DataFrame, cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Safely applies Label Encoding to the specified columns of a DataFrame.
    """
    df = df.copy()  # Create a copy to avoid setting values on a copy slice warning

    # Fallback to predefined CATEGORICAL_COLS if no specific columns are passed
    columns_to_encode = (
        cols
        if cols is not None
        else [
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "CustomerId",
            "CurrencyCode",
            "CountryCode",
            "ProviderId",
            "ProductId",
            "ProductCategory",
            "ChannelId",
        ]
    )

    le = LabelEncoder()

    for col in columns_to_encode:
        if col in df.columns:
            # Convert to string first to handle any mixed data types or NaNs safely
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])

    print(df[columns_to_encode].head())
    return df


# 8. Weight of Evidence (WoE) & Information Value (IV) Engineering
def compute_woe_iv(
    df: pd.DataFrame, feature: str, target: str = TARGET_COL, epsilon: float = 1e-6
) -> Tuple[pd.DataFrame, float]:
    total_events = (df[target] == 1).sum()
    total_non_events = (df[target] == 0).sum()

    stats = (
        df.groupby(feature, observed=False)[target]
        .agg(events=lambda x: (x == 1).sum(), non_events=lambda x: (x == 0).sum())
        .reset_index()
    )

    stats["dist_events"] = (stats["events"] + epsilon) / (total_events + epsilon)
    stats["dist_non_events"] = (stats["non_events"] + epsilon) / (
        total_non_events + epsilon
    )
    stats["woe"] = np.log(stats["dist_events"] / stats["dist_non_events"])
    stats["iv_component"] = (stats["dist_events"] - stats["dist_non_events"]) * stats[
        "woe"
    ]
    iv_total = stats["iv_component"].sum()
    return stats, iv_total


def _iv_label(iv: float) -> str:
    if iv < 0.02:
        return "Useless"
    elif iv < 0.1:
        return "Weak"
    elif iv < 0.3:
        return "Medium"
    elif iv < 0.5:
        return "Strong"
    else:
        return "Very Strong"


def compute_all_iv(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: str = TARGET_COL,
) -> pd.DataFrame:
    features = features or CATEGORICAL_COLS + NUMERICAL_COLS
    results = []
    for feat in features:
        if feat not in df.columns or feat == target:
            continue
        col = df[feat]
        if feat in NUMERICAL_COLS:
            try:
                col = pd.qcut(df[feat], q=10, duplicates="drop")
            except Exception:
                col = pd.cut(df[feat], bins=5)
        temp_df = df.copy()
        temp_df[feat] = col
        if iv is not None:
            iv_value = iv(temp_df[feat], temp_df[target])
        else:
            _, iv_value = compute_woe_iv(temp_df, feat, target)
        results.append({"feature": feat, "iv": iv_value})
    iv_df = (
        pd.DataFrame(results).sort_values("iv", ascending=False).reset_index(drop=True)
    )
    iv_df["predictive_power"] = iv_df["iv"].apply(_iv_label)
    return iv_df


def encode_woe(
    df: pd.DataFrame,
    features: List[str],
    target: str = TARGET_COL,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    try:
        if WOEEncoder is None:
            raise ImportError("`woe` library is not installed.")
        encoder = WOEEncoder(cols=features, target=target)
        transformed = encoder.fit_transform(df.copy())
        woe_maps: Dict[str, Dict] = {}
        for col in features:
            woe_maps[col] = encoder.mapping_[col]
        return transformed, woe_maps
    except Exception:
        logger.warning("Falling back to custom WoE encoding due to error")
        df_copy = df.copy()
        woe_maps: Dict[str, Dict] = {}
        for feat in features:
            if feat not in df_copy.columns:
                continue
            woe_df, _ = compute_woe_iv(df_copy, feat, target)
            woe_map = dict(zip(woe_df[feat], woe_df["woe"]))
            df_copy[f"{feat}_woe"] = df_copy[feat].map(woe_map)
            woe_maps[feat] = woe_map
        return df_copy, woe_maps


class WoEFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, categorical_cols: Optional[List[str]] = None, target_col: str = TARGET_COL
    ):
        self.categorical_cols = categorical_cols
        self.target_col = target_col

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if y is not None:
            X[self.target_col] = y

        self.categorical_cols_ = self.categorical_cols or [
            col
            for col in X.columns
            if col not in [self.target_col, "TransactionStartTime"]
            and not pd.api.types.is_numeric_dtype(X[col])
        ]
        self.woe_maps_ = {}
        for col in self.categorical_cols_:
            if col not in X.columns:
                continue
            stats_df, _ = compute_woe_iv(
                X[[col, self.target_col]], col, target=self.target_col
            )
            self.woe_maps_[col] = dict(zip(stats_df[col], stats_df["woe"]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.woe_maps_.items():
            if col in X.columns:
                X[f"{col}_woe"] = X[col].map(mapping).fillna(0.0)
        return X


# --- 9. Dynamic Pipeline Construction ---
def build_feature_pipeline(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    use_woe: bool = False,
) -> Pipeline:

    # Injecting the newly extracted/engineered categorical features here!
    categorical_cols = categorical_cols or CATEGORICAL_COLS + [
        "IsWeekend",
        "IsReversal",
    ]

    steps = [
        ("aggregations", CustomerAggregationTransformer()),
        (
            "imputer",
            DataFrameImputer(
                numerical_cols=numerical_cols, categorical_cols=categorical_cols
            ),
        ),
        ("capper", OutlierCapper()),
        ("engineer", FeatureEngineerTransformer()),
        ("scaler", FeatureScaler(numerical_cols=numerical_cols)),
    ]

    if use_woe:
        steps.append(("woe", WoEFeatureTransformer(categorical_cols=categorical_cols)))
    else:
        steps.append(
            ("one_hot", DataFrameLabelEncoder(categorical_cols=categorical_cols))
        )

    return Pipeline(steps)


def fit_transform_pipeline(
    df: pd.DataFrame,
    use_woe: bool = False,
) -> Tuple[Pipeline, pd.DataFrame]:
    pipeline = build_feature_pipeline(use_woe=use_woe)
    transformed = pipeline.fit_transform(df)
    return pipeline, transformed


class ProxyTargetEngineer(BaseEstimator, TransformerMixin):
    """
    Computes RFM profiles, segments customers using K-Means,
    and engineers an 'is_high_risk' proxy target column.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self.high_risk_cluster_id_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"])

        # 1. Define Snapshot Date consistently (1 day after the latest transaction)
        snapshot_date = X["TransactionStartTime"].max() + pd.Timedelta(days=1)

        # 2. Calculate RFM Metrics
        rfm = (
            X.groupby("CustomerId")
            .agg(
                {
                    "TransactionStartTime": lambda x: (
                        snapshot_date - x.max()
                    ).days,  # Recency
                    "TransactionId": "count",  # Frequency
                    "Amount": "sum",  # Monetary
                }
            )
            .rename(
                columns={
                    "TransactionStartTime": "Recency",
                    "TransactionId": "Frequency",
                    "Amount": "Monetary",
                }
            )
        )

        # Handle negative or zero monetary balances safely before log transforming
        rfm["Monetary"] = rfm["Monetary"].clip(lower=0.1)

        # 3. Log Transform to handle skewness, then Scale
        rfm_log = np.log1p(rfm)
        rfm_scaled = self.scaler.fit_transform(rfm_log)

        # 4. Run K-Means
        self.kmeans.fit(rfm_scaled)
        rfm["Cluster"] = self.kmeans.labels_

        # 5. Automatically identify the "High-Risk" Cluster
        # (Lowest average frequency and lowest average monetary spend)
        cluster_profiles = rfm.groupby("Cluster")[["Frequency", "Monetary"]].mean()

        # High risk = rank by combined low metrics
        # We find the cluster where the sum of normalized ranks for Frequency and Monetary is lowest
        rank_sum = (
            cluster_profiles["Frequency"].rank() + cluster_profiles["Monetary"].rank()
        )
        self.high_risk_cluster_id_ = rank_sum.idxmin()

        logger.info(
            f"Identified Cluster {self.high_risk_cluster_id_} as the High-Risk Proxy."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"])
        snapshot_date = X["TransactionStartTime"].max() + pd.Timedelta(days=1)

        # Recalculate customer-level labels to safely map back to transaction entries
        rfm = (
            X.groupby("CustomerId")
            .agg(
                {
                    "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
                    "TransactionId": "count",
                    "Amount": "sum",
                }
            )
            .rename(
                columns={
                    "TransactionStartTime": "Recency",
                    "TransactionId": "Frequency",
                    "Amount": "Monetary",
                }
            )
        )

        rfm["Monetary"] = rfm["Monetary"].clip(lower=0.1)
        rfm_log = np.log1p(rfm)
        rfm_scaled = self.scaler.transform(rfm_log)

        # Predict clusters
        clusters = self.kmeans.predict(rfm_scaled)
        rfm["Cluster"] = clusters

        # Assign Target column: 1 if high-risk cluster, else 0
        rfm["is_high_risk"] = (rfm["Cluster"] == self.high_risk_cluster_id_).astype(int)

        # Map target back to the transaction-level dataframe
        target_map = rfm["is_high_risk"].to_dict()
        X["is_high_risk"] = X["CustomerId"].map(target_map)

        return X


def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train, transform all splits using FeatureScaler."""
    scaler_transformer = FeatureScaler(numerical_cols=num_cols)
    scaler_transformer.fit(X_train)
    X_train_scaled = scaler_transformer.transform(X_train)
    X_val_scaled = scaler_transformer.transform(X_val)
    X_test_scaled = scaler_transformer.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler_transformer.scaler_

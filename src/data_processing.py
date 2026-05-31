import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from constants import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL, COLUMN_NAMES


# 3. Outlier Capping

class OutlierCapper(BaseEstimator, TransformerMixin):

    def __init__(self, cols: Optional[List[str]] = None, iqr_multiplier: float = 1.5):
        self.cols = cols
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.cols_ = self.cols or [c for c in NUMERICAL_COLS if c in X.columns]
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.cols_:
            if col not in X.columns:
                continue
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.iqr_multiplier * iqr
            self.upper_bounds_[col] = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols_:
            if col not in X.columns or col not in self.lower_bounds_:
                continue
            X[col] = X[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
        return X


def cap_outliers(df: pd.DataFrame, cols: Optional[List[str]] = None, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    #Cap outliers at IQR fences (Winsorization) using OutlierCapper."""
    capper = OutlierCapper(cols=cols, iqr_multiplier=iqr_multiplier)
    return capper.fit_transform(df)
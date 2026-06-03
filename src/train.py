import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.data_processing import (
    load_processed_data, split_data, scale_features,
    encode_woe, CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_OUTPUT_DIR = "data/processed"
MODEL_FILENAME = "best_model.joblib"


# Evaluation

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, split_name: str = "Test") -> dict:
    """Compute and log all evaluation metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
    }

    logger.info(f"\n{'='*50}\n{split_name} Results\n{'='*50}")
    for k, v in metrics.items():
        if k != "split":
            logger.info(f"  {k:12s}: {v:.4f}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=['Good Credit', 'Default'])}")
    return metrics


# Models 

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train Logistic Regression with cross-validation and regularization tuning."""
    logger.info("Training Logistic Regression...")
    param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs"], "max_iter": [1000]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        LogisticRegression(class_weight="balanced", random_state=42),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    logger.info(f"Best LR params: {gs.best_params_} | CV AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train Random Forest with hyperparameter tuning."""
    logger.info("Training Random Forest...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    logger.info(f"Best RF params: {gs.best_params_} | CV AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """Train XGBoost with scale_pos_weight for class imbalance."""
    logger.info("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
        ),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    logger.info(f"Best XGB params: {gs.best_params_} | CV AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_


# Main Training Loop

def main():
    # ── Load & prepare data ──
    processed_path = "data/processed/german_credit_processed.csv"
    df = load_processed_data(processed_path)

    # WoE encode categoricals
    df, woe_maps = encode_woe(df, CATEGORICAL_COLS)
    woe_cols = [f"{c}_woe" for c in CATEGORICAL_COLS if f"{c}_woe" in df.columns]
    feature_cols = NUMERICAL_COLS + woe_cols
    df_model = df[feature_cols + [TARGET_COL]].dropna()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_model)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test, num_cols=NUMERICAL_COLS)

    # ── Train models ──
    models = {
        "LogisticRegression": train_logistic_regression(X_train, y_train),
        "RandomForest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }

    # ── Evaluate on validation set ──
    val_results = []
    for name, model in models.items():
        logger.info(f"\n--- {name} ---")
        metrics = evaluate_model(model, X_val, y_val, split_name=f"{name} [Val]")
        metrics["model_name"] = name
        val_results.append(metrics)

    # ── Select best model by ROC-AUC ──
    best = max(val_results, key=lambda x: x["roc_auc"])
    best_model = models[best["model_name"]]
    logger.info(f"\n Best model: {best['model_name']} (Val AUC={best['roc_auc']:.4f})")

    # ── Final evaluation on test set ──
    evaluate_model(best_model, X_test, y_test, split_name="FINAL TEST")

    # ── Save artifacts ──
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    artifact = {
        "model": best_model,
        "scaler": scaler,
        "woe_maps": woe_maps,
        "feature_cols": feature_cols,
        "model_name": best["model_name"],
    }
    save_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)
    joblib.dump(artifact, save_path)
    logger.info(f"Model artifact saved to: {save_path}")

    return artifact


if __name__ == "__main__":
    main()

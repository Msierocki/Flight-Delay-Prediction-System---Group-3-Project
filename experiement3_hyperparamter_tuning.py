#used BearcatGPT ai to add/clean up comments and add some additional validation checks.
import json
import hashlib
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# File paths
PROCESSED_DIR = Path(r"C:\Users\audre\OneDrive\Desktop\ML Model Code\Data\Processed")
MODEL_DIR = Path(r"C:\Users\audre\OneDrive\Desktop\ML Model Code\Models")
DATA_PATH = PROCESSED_DIR / "flights_model_ready_v1.csv"

MLFLOW_TRACKING_URI = "sqlite:///" + str(
    Path(r"C:\Users\audre\OneDrive\Desktop\ML Model Code") / "mlflow.db"
)  

# Number of hyperparameter combinations to try per model.
# Higher = better chance of finding the optimum, but slower.
# 20 is a solid balance for a dataset this size.
N_ITER = 20

# Cross-validation folds. 3 is used instead of 5 to keep runtime manageable
# at 4.8M training rows. Each fold trains on ~3.2M rows.
CV_FOLDS = 3

# Scoring metric used to rank hyperparameter combinations.
# F1 on the delayed class (label=1) is the right choice here because:
#   - Accuracy is misleading at 78/22 class imbalance
#   - We care about catching delays (recall) without too many false alarms (precision)
#   - F1 balances both
SCORING = "f1"

FEATURE_CANDIDATES = [
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN_STATE_NM",
    "DEST_STATE_NM",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "MONTH",
    "DAY_OF_WEEK",
    "DAY",
    "DAY_OF_MONTH",
    "IS_WEEKEND",
    "SEASON",
    "CRS_DEP_HOUR",
    "CRS_DEP_MINUTE",
    "CRS_ARR_HOUR",
    "CRS_ARR_MINUTE",
    "DEP_HOUR",
    "DEP_TIME_BUCKET",
    "ROUTE",
]

# Random Forest hyperparameter search space.
# Each key maps to a list of values RandomizedSearchCV will sample from.
RF_PARAM_DIST = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [10, 15, 20, 25, None],
    "classifier__min_samples_split": [5, 10, 20],
    "classifier__min_samples_leaf": [2, 5, 10],
    "classifier__max_features": ["sqrt", "log2", 0.3],
    "classifier__class_weight": ["balanced", "balanced_subsample"],
}

# Logistic Regression hyperparameter search space.
# C controls regularization: small C = strong regularization (simpler model),
# large C = weak regularization (fits training data more closely).
LR_PARAM_DIST = {
    "classifier__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "classifier__max_iter": [500, 1000, 2000],
    "classifier__solver": ["saga"],
    "classifier__class_weight": ["balanced"],
}


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 checksum of a file for data versioning."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load model-ready dataset and run basic integrity checks."""
    print(f"Loading dataset: {path}")
    dtype_overrides = {"OP_UNIQUE_CARRIER": str, "TAIL_NUM": str}
    df = pd.read_csv(path, low_memory=False, dtype=dtype_overrides)
    print(f"Shape: {df.shape}")

    if "FL_DATE" not in df.columns:
        raise ValueError("FL_DATE column is required for time-based splitting.")
    if "TARGET_DELAYED" not in df.columns:
        raise ValueError("TARGET_DELAYED column is required as the target variable.")
    if df["TARGET_DELAYED"].isna().any():
        raise ValueError("TARGET_DELAYED contains nulls.")
    if not df["TARGET_DELAYED"].isin([0, 1]).all():
        raise ValueError("TARGET_DELAYED contains values outside {0, 1}.")

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df[df["FL_DATE"].notna()].copy()
    df = df.sort_values("FL_DATE").reset_index(drop=True)

    print(f"Target distribution:\n{df['TARGET_DELAYED'].value_counts(dropna=False).to_string()}")
    print(f"Delay rate: {df['TARGET_DELAYED'].mean():.2%}")
    print(f"Date range: {df['FL_DATE'].min().date()} to {df['FL_DATE'].max().date()}")
    return df


def build_splits(df: pd.DataFrame, features: list):
    """Time-based 70/15/15 split — no data leakage from future months."""
    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    X = df[features].copy()
    y = df["TARGET_DELAYED"].astype(int).copy()

    splits = {
        "train": (X.iloc[:train_end], y.iloc[:train_end]),
        "valid": (X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]),
        "test":  (X.iloc[valid_end:], y.iloc[valid_end:]),
    }

    print(f"\nTime-based split (70 / 15 / 15):")
    for name, (X_s, _) in splits.items():
        idx = X_s.index
        print(f"  {name.capitalize():>10}: {len(X_s):>9,} rows  |  "
              f"{df.loc[idx, 'FL_DATE'].min().date()} to {df.loc[idx, 'FL_DATE'].max().date()}")

    return splits


def build_preprocessor(numeric_features: list, categorical_features: list,
                        scale: bool = False) -> ColumnTransformer:
    """
    Build the preprocessing ColumnTransformer.

    scale=True adds StandardScaler for numeric features.
    Logistic Regression needs scaling for gradient convergence.
    Random Forest does not need scaling — trees split on thresholds,
    not distances, so feature magnitude does not affect them.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def evaluate(model, X, y, split_label: str) -> dict:
    """Run predictions and return all metrics for a given split."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "split": split_label,
        "accuracy":  round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y, y_prob)), 4),
    }

    print(f"\n{split_label.upper()} METRICS")
    for k, v in metrics.items():
        if k != "split":
            print(f"  {k.capitalize():<12}: {v:.4f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y, y_pred)}")
    print(f"\n  Classification Report:\n{classification_report(y, y_pred, zero_division=0)}")

    return metrics


def tune_model(pipeline, param_dist, X_train, y_train, model_label: str):
    """
    Run RandomizedSearchCV over the given pipeline and parameter distribution.

    RandomizedSearchCV works by:
      1. Randomly sampling N_ITER combinations from param_dist
      2. For each combination, running CV_FOLDS-fold cross-validation
      3. Scoring each fold using SCORING (f1)
      4. Returning the combination with the highest mean CV score

    This is much faster than GridSearchCV which tries every combination,
    while still reliably finding near-optimal hyperparameters.
    """
    print(f"\nTuning {model_label} with {N_ITER} iterations x {CV_FOLDS}-fold CV...")
    print(f"Total fits: {N_ITER * CV_FOLDS}")

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
        random_state=42,
        verbose=2,
        refit=True,
    )

    search.fit(X_train, y_train)

    print(f"\nBest CV {SCORING} score: {search.best_score_:.4f}")
    print(f"Best parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    return search


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_validate(DATA_PATH)
    data_checksum = file_md5(DATA_PATH)
    print(f"\nDataset MD5: {data_checksum}")

    features = [col for col in FEATURE_CANDIDATES if col in df.columns and col != "TAIL_NUM"]
    missing = [col for col in FEATURE_CANDIDATES if col not in df.columns]
    print(f"\nFeatures used ({len(features)}): {features}")
    if missing:
        print(f"Skipped (not in dataset): {missing}")
    if not features:
        raise ValueError("No modeling features found.")

    X_sample = df[features]
    numeric_features = X_sample.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_sample.select_dtypes(exclude=["number"]).columns.tolist()
    print(f"\nNumeric  ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical ({len(categorical_features)}): {categorical_features}")

    splits = build_splits(df, features)
    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]
    X_test,  y_test  = splits["test"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Flight Delay Prediction")

    all_results = []

    # Random Forest tuning
    print("\n" + "=" * 60)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 60)

    rf_preprocessor = build_preprocessor(numeric_features, categorical_features, scale=False)
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", rf_preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    rf_search = tune_model(rf_pipeline, RF_PARAM_DIST, X_train, y_train, "Random Forest")
    best_rf = rf_search.best_estimator_

    rf_model_path = MODEL_DIR / "v1_random_forest_tuned.joblib"
    joblib.dump(best_rf, rf_model_path)
    print(f"\nTuned RF model saved: {rf_model_path}")

    with mlflow.start_run(run_name="RF_Tuned_v1"):
        mlflow.set_tags({
            "model_type": "RandomForest",
            "tuning": "RandomizedSearchCV",
            "dataset_version": "v1",
            "dataset_md5": data_checksum,
        })
        mlflow.log_params({k.replace("classifier__", ""): v
                           for k, v in rf_search.best_params_.items()})
        mlflow.log_param("best_cv_f1", round(rf_search.best_score_, 4))

        rf_valid_metrics = evaluate(best_rf, X_valid, y_valid, "Validation")
        rf_test_metrics  = evaluate(best_rf, X_test,  y_test,  "Test")

        for m in rf_valid_metrics:
            if m != "split":
                mlflow.log_metric(f"valid_{m}", rf_valid_metrics[m])
        for m in rf_test_metrics:
            if m != "split":
                mlflow.log_metric(f"test_{m}", rf_test_metrics[m])

        mlflow.log_artifact(str(rf_model_path))
        rf_run_id = mlflow.active_run().info.run_id

    rf_result = {
        "model": "RandomForest_Tuned",
        "best_params": rf_search.best_params_,
        "best_cv_f1": round(rf_search.best_score_, 4),
        "validation": rf_valid_metrics,
        "test": rf_test_metrics,
        "mlflow_run_id": rf_run_id,
        "model_path": str(rf_model_path),
        "timestamp": datetime.now().isoformat(),
        "dataset_md5": data_checksum,
    }
    all_results.append(rf_result)

    # Logistic Regression tuning
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION HYPERPARAMETER TUNING")
    print("=" * 60)

    lr_preprocessor = build_preprocessor(numeric_features, categorical_features, scale=True)
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", lr_preprocessor),
        ("classifier", LogisticRegression(random_state=42))
    ])

    lr_search = tune_model(lr_pipeline, LR_PARAM_DIST, X_train, y_train, "Logistic Regression")
    best_lr = lr_search.best_estimator_

    lr_model_path = MODEL_DIR / "v1_baseline_logistic_tuned.joblib"
    joblib.dump(best_lr, lr_model_path)
    print(f"\nTuned LR model saved: {lr_model_path}")

    with mlflow.start_run(run_name="LR_Tuned_v1"):
        mlflow.set_tags({
            "model_type": "LogisticRegression",
            "tuning": "RandomizedSearchCV",
            "dataset_version": "v1",
            "dataset_md5": data_checksum,
        })
        mlflow.log_params({k.replace("classifier__", ""): v
                           for k, v in lr_search.best_params_.items()})
        mlflow.log_param("best_cv_f1", round(lr_search.best_score_, 4))

        lr_valid_metrics = evaluate(best_lr, X_valid, y_valid, "Validation")
        lr_test_metrics  = evaluate(best_lr, X_test,  y_test,  "Test")

        for m in lr_valid_metrics:
            if m != "split":
                mlflow.log_metric(f"valid_{m}", lr_valid_metrics[m])
        for m in lr_test_metrics:
            if m != "split":
                mlflow.log_metric(f"test_{m}", lr_test_metrics[m])

        mlflow.log_artifact(str(lr_model_path))
        lr_run_id = mlflow.active_run().info.run_id

    lr_result = {
        "model": "LogisticRegression_Tuned",
        "best_params": lr_search.best_params_,
        "best_cv_f1": round(lr_search.best_score_, 4),
        "validation": lr_valid_metrics,
        "test": lr_test_metrics,
        "mlflow_run_id": lr_run_id,
        "model_path": str(lr_model_path),
        "timestamp": datetime.now().isoformat(),
        "dataset_md5": data_checksum,
    }
    all_results.append(lr_result)

    # Save combined results
    results_path = MODEL_DIR / "v1_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTuning results saved: {results_path}")

    # Summary comparison
    print("TUNING SUMMARY")
    for result in all_results:
        print(f"\n{result['model']}")
        print(f"  Best CV F1      : {result['best_cv_f1']:.4f}")
        print(f"  Validation F1   : {result['validation']['f1']:.4f}")
        print(f"  Validation AUC  : {result['validation']['roc_auc']:.4f}")
        print(f"  Test F1         : {result['test']['f1']:.4f}")
        print(f"  Test AUC        : {result['test']['roc_auc']:.4f}")
        print(f"  Best params     : {result['best_params']}")

    print("\nHyperparameter tuning complete.")


if __name__ == "__main__":
    main()


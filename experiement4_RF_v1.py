#used BearcatGPT ai to add/clean up comments and add some additional validation checks.
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# File paths
PROCESSED_DIR = Path(r"C:\Users\audre\OneDrive\Desktop\ML Model Code\Data\Processed")
MODEL_DIR = Path(r"C:\Users\audre\OneDrive\Desktop\ML Model Code\Models")
DATA_PATH = PROCESSED_DIR / "flights_model_ready_v1.csv"
MODEL_PATH = MODEL_DIR / "v1_random_forest_tuned.joblib"
RESULTS_PATH = MODEL_DIR / "v1_experiment4_threshold_results.json"

MLFLOW_TRACKING_URI = "sqlite:///" + str(
    Path(r"C:\Users\audre\OneDrive\Desktop\ML Model Code") / "mlflow.db"
)

# Thresholds to evaluate.
# Default sklearn threshold is 0.50. We sweep below and above it to find
# the best tradeoff between precision and recall for this use case.
# Lower threshold = model flags more flights as delayed (higher recall, lower precision)
# Higher threshold = model is more conservative (lower recall, higher precision)
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

FEATURE_CANDIDATES = [
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "TAIL_NUM",
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


def load_and_split(path: Path, features: list):
    """Load the model-ready dataset and return the test split only.
    
    Experiment 4 only needs the test set — the model is already trained
    and saved. We are purely evaluating how the prediction threshold
    affects what the model calls delayed vs not delayed.
    Threshold changes do not affect the underlying probabilities,
    only where we draw the line between the two classes.
    """
    print(f"Loading dataset: {path}")
    dtype_overrides = {"OP_UNIQUE_CARRIER": str, "TAIL_NUM": str}
    df = pd.read_csv(path, low_memory=False, dtype=dtype_overrides)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df[df["FL_DATE"].notna()].copy()
    df = df.sort_values("FL_DATE").reset_index(drop=True)

    available = [col for col in features if col in df.columns]
    X = df[available].copy()
    y = df["TARGET_DELAYED"].astype(int).copy()

    n = len(df)
    valid_end = int(n * 0.85)

    X_test = X.iloc[valid_end:].copy()
    y_test = y.iloc[valid_end:].copy()

    print(f"Test set: {len(X_test):,} rows  |  "
          f"{df.iloc[valid_end:]['FL_DATE'].min().date()} to "
          f"{df.iloc[valid_end:]['FL_DATE'].max().date()}")
    print(f"Test delay rate: {y_test.mean():.2%}")
    return X_test, y_test


def evaluate_threshold(y_true, y_prob, threshold: float) -> dict:
    """Apply a classification threshold and compute all metrics.
    
    Instead of using model.predict() which always uses 0.50,
    we take the raw predicted probabilities (y_prob) and manually
    apply our chosen threshold:
        predicted = 1 if probability >= threshold else 0
    This lets us control the precision/recall tradeoff without
    retraining the model.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_threshold_results(results: list):
    """Print a formatted comparison table across all thresholds."""
    print("\nTHRESHOLD SWEEP RESULTS")
    print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 62)
    for r in results:
        marker = " <-- default" if r["threshold"] == 0.50 else ""
        print(f"  {r['threshold']:>8.2f} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}{marker}")


def find_best_threshold(results: list) -> dict:
    """Identify the best threshold by F1, and flag the best precision/recall tradeoff.
    
    Best F1 = best overall balance between catching delays and avoiding false alarms.
    We also flag the threshold that keeps recall >= 0.65 with the highest precision,
    because for this use case (passenger-facing delay warnings) missing a real delay
    is more costly than a false alarm, but precision still matters for trust.
    """
    best_f1 = max(results, key=lambda x: x["f1"])

    high_recall = [r for r in results if r["recall"] >= 0.65]
    best_precision_at_recall = max(high_recall, key=lambda x: x["precision"]) if high_recall else None

    return {
        "best_f1_threshold": best_f1,
        "best_precision_at_recall_65": best_precision_at_recall,
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tuned Random Forest model...")
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    X_test, y_test = load_and_split(DATA_PATH, FEATURE_CANDIDATES)

    print("\nGenerating predicted probabilities...")
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"Probability range: {y_prob.min():.4f} to {y_prob.max():.4f}")
    print(f"Mean predicted probability: {y_prob.mean():.4f}")

    print(f"\nSweeping {len(THRESHOLDS)} thresholds: {THRESHOLDS}")
    results = []
    for threshold in THRESHOLDS:
        r = evaluate_threshold(y_test, y_prob, threshold)
        results.append(r)

    print_threshold_results(results)

    best = find_best_threshold(results)

    print("\nBEST THRESHOLD BY F1:")
    bf = best["best_f1_threshold"]
    print(f"  Threshold : {bf['threshold']}")
    print(f"  Accuracy  : {bf['accuracy']:.4f}")
    print(f"  Precision : {bf['precision']:.4f}")
    print(f"  Recall    : {bf['recall']:.4f}")
    print(f"  F1        : {bf['f1']:.4f}")
    print(f"  ROC-AUC   : {bf['roc_auc']:.4f}")

    if best["best_precision_at_recall_65"]:
        bp = best["best_precision_at_recall_65"]
        print(f"\nBEST PRECISION WHILE KEEPING RECALL >= 0.65:")
        print(f"  Threshold : {bp['threshold']}")
        print(f"  Precision : {bp['precision']:.4f}")
        print(f"  Recall    : {bp['recall']:.4f}")
        print(f"  F1        : {bp['f1']:.4f}")

    print(f"\nDetailed confusion matrices:")
    for r in results:
        cm = np.array(r["confusion_matrix"])
        print(f"\n  Threshold {r['threshold']:.2f}:")
        print(f"    True Negatives  (on-time correctly predicted): {cm[0][0]:>10,}")
        print(f"    False Positives (on-time wrongly flagged):      {cm[0][1]:>10,}")
        print(f"    False Negatives (delays missed):                {cm[1][0]:>10,}")
        print(f"    True Positives  (delays correctly caught):      {cm[1][1]:>10,}")

    output = {
        "experiment": "Experiment 4 — Classification Threshold Sweep",
        "model": str(MODEL_PATH),
        "run_timestamp": datetime.now().isoformat(),
        "thresholds_tested": THRESHOLDS,
        "results": results,
        "best_by_f1": best["best_f1_threshold"],
        "best_precision_at_recall_65": best["best_precision_at_recall_65"],
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {RESULTS_PATH}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Flight Delay Prediction")

    with mlflow.start_run(run_name="Experiment4_ThresholdSweep"):
        mlflow.set_tags({
            "experiment": "4",
            "stage": "threshold_tuning",
            "model": "RandomForest_Tuned",
            "dataset_version": "v1",
            "run_date": datetime.now().strftime("%Y-%m-%d"),
        })

        for r in results:
            t = r["threshold"]
            mlflow.log_metrics({
                f"threshold_{t}_accuracy":  r["accuracy"],
                f"threshold_{t}_precision": r["precision"],
                f"threshold_{t}_recall":    r["recall"],
                f"threshold_{t}_f1":        r["f1"],
                f"threshold_{t}_roc_auc":   r["roc_auc"],
            })

        bf = best["best_f1_threshold"]
        mlflow.log_metrics({
            "best_threshold":           bf["threshold"],
            "best_threshold_accuracy":  bf["accuracy"],
            "best_threshold_precision": bf["precision"],
            "best_threshold_recall":    bf["recall"],
            "best_threshold_f1":        bf["f1"],
            "best_threshold_roc_auc":   bf["roc_auc"],
        })

        mlflow.log_artifact(str(RESULTS_PATH))
        run_id = mlflow.active_run().info.run_id

    print(f"\nMLflow run ID: {run_id}")
    print("\nExperiment 4 complete.")


if __name__ == "__main__":
    main()


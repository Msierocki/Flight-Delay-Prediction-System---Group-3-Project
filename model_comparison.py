#used BearcatGPT ai to add/clean up comments and add some additional validation checks.

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================
# File paths
# =========================
DATA_PATH = Path(r"C:\Users\audre\Documents\ML-Airport-Delay-Prediction\data\processed\flights_model_ready_v1.csv")
RESULTS_DIR = Path(r"C:\Users\audre\Documents\ML-Airport-Delay-Prediction\results")
RESULTS_PATH = RESULTS_DIR / "v1_model_comparison.csv"


# =========================
# Load data
# =========================
print("Loading model-ready dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

if "FL_DATE" not in df.columns:
    raise ValueError("FL_DATE column is required for time-based splitting.")

if "TARGET_DELAYED" not in df.columns:
    raise ValueError("TARGET_DELAYED column is required as the target variable.")

# Parse and sort by time
# Using chronological evaluation aligns better with the proposal
# than a random split.
df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
df = df[df["FL_DATE"].notna()].copy()
df = df.sort_values("FL_DATE").reset_index(drop=True)


# =========================
# Feature selection
# Version 1 = pre-departure / scheduled information only
# Combined from the proposal and strongest notebook choices
# =========================
feature_candidates = [
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "ORIGIN_STATE_NM",
    "DEST",
    "DEST_STATE_NM",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "MONTH",
    "DAY_OF_WEEK",
    "DAY",
    "DAY_OF_MONTH",
    "IS_WEEKEND",
    "CRS_DEP_HOUR",
    "CRS_DEP_MINUTE",
    "CRS_ARR_HOUR",
    "CRS_ARR_MINUTE",
    "DEP_HOUR",
    "ROUTE"
]

features = [col for col in feature_candidates if col in df.columns]
if not features:
    raise ValueError("No modeling features were found in the dataset.")

X = df[features].copy()
y = df["TARGET_DELAYED"].astype(int).copy()


# =========================
# Time-based split
# 70% train, 15% validation, 15% test
# =========================
train_end = int(len(df) * 0.70)
valid_end = int(len(df) * 0.85)

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_valid = X.iloc[train_end:valid_end].copy()
y_valid = y.iloc[train_end:valid_end].copy()

X_test = X.iloc[valid_end:].copy()
y_test = y.iloc[valid_end:].copy()

print("\nTime-based split summary:")
print(f"Train rows: {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")
print(f"Test rows: {len(X_test):,}")
print(f"Train date range: {df.iloc[:train_end]['FL_DATE'].min()} to {df.iloc[:train_end]['FL_DATE'].max()}")
print(f"Validation date range: {df.iloc[train_end:valid_end]['FL_DATE'].min()} to {df.iloc[train_end:valid_end]['FL_DATE'].max()}")
print(f"Test date range: {df.iloc[valid_end:]['FL_DATE'].min()} to {df.iloc[valid_end:]['FL_DATE'].max()}")


# =========================
# Identify numeric and categorical features
# =========================
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [col for col in X_train.columns if col not in numeric_features]


# =========================
# Shared preprocessors
# =========================
logistic_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

rf_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric_features),
    ("cat", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])


# =========================
# Models aligned to proposal
# =========================
models = {
    "Balanced Logistic Regression": Pipeline(steps=[
        ("preprocessor", logistic_preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]),
    "Random Forest": Pipeline(steps=[
        ("preprocessor", rf_preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}


# =========================
# Evaluation helper
# =========================
def evaluate_split(model, X_split, y_split):
    y_pred = model.predict(X_split)
    y_prob = model.predict_proba(X_split)[:, 1]
    return {
        "Accuracy": accuracy_score(y_split, y_pred),
        "Precision": precision_score(y_split, y_pred, zero_division=0),
        "Recall": recall_score(y_split, y_pred, zero_division=0),
        "F1 Score": f1_score(y_split, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_split, y_prob)
    }


# =========================
# Train and compare
# =========================
rows = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    valid_metrics = evaluate_split(model, X_valid, y_valid)
    test_metrics = evaluate_split(model, X_test, y_test)

    rows.append({
        "Model": model_name,
        "Validation Accuracy": valid_metrics["Accuracy"],
        "Validation Precision": valid_metrics["Precision"],
        "Validation Recall": valid_metrics["Recall"],
        "Validation F1 Score": valid_metrics["F1 Score"],
        "Validation ROC-AUC": valid_metrics["ROC-AUC"],
        "Test Accuracy": test_metrics["Accuracy"],
        "Test Precision": test_metrics["Precision"],
        "Test Recall": test_metrics["Recall"],
        "Test F1 Score": test_metrics["F1 Score"],
        "Test ROC-AUC": test_metrics["ROC-AUC"]
    })

comparison_df = pd.DataFrame(rows)
comparison_df = comparison_df.sort_values(by=["Test F1 Score", "Test ROC-AUC"], ascending=False).reset_index(drop=True)


# =========================
# Save results
# =========================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(RESULTS_PATH, index=False)

print("\n================ MODEL COMPARISON ================")
print(comparison_df.round(4))
print(f"\nSaved model comparison table to: {RESULTS_PATH}")


# =========================
# Proposal-aligned interpretation notes
# =========================
print("\nProposal-aligned guidance:")
print("- Accuracy is included, but should not be the main decision metric because the classes are imbalanced.")
print("- Precision shows how often a predicted delay is truly delayed.")
print("- Recall shows how many true delays the model successfully catches.")
print("- F1 Score balances precision and recall.")
print("- ROC-AUC helps compare ranking performance across thresholds.")
print("- For this project, prefer the model with stronger Recall, F1 Score, and ROC-AUC unless false alarms become too costly.")

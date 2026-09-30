#used BearcatGPT ai to add/clean up comments and add some additional validation checks.

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
from pathlib import Path

# =========================
# File paths
# =========================
DATA_PATH = Path(r"C:\Users\audre\Documents\ML-Airport-Delay-Prediction\data\processed\flights_model_ready_v1.csv")
MODEL_DIR = Path(r"C:\Users\audre\OneDrive\Desktop\ML Model Code\ML-Airport-Delay-Prediction\models")
MODEL_PATH = MODEL_DIR / "v1_random_forest.joblib"

# =========================
# Load data
# =========================
print("Loading model-ready dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())


# =========================
# Basic cleanup
# =========================
if "FL_DATE" not in df.columns:
    raise ValueError("FL_DATE column is required for time-based splitting.")

if "TARGET_DELAYED" not in df.columns:
    raise ValueError("TARGET_DELAYED column is required as the target variable.")

df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
df = df[df["FL_DATE"].notna()].copy()
df = df.sort_values("FL_DATE").reset_index(drop=True)

print("\nTarget distribution:")
print(df["TARGET_DELAYED"].value_counts(dropna=False))
print(f"Delay rate: {df['TARGET_DELAYED'].mean():.2%}")


# =========================
# Feature selection
# Combined strongest Version 1 features from the proposal and notebook
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
missing_feature_candidates = [col for col in feature_candidates if col not in df.columns]

print("\nUsing features:")
print(features)

if missing_feature_candidates:
    print("\nFeature candidates not found in dataset:")
    print(missing_feature_candidates)

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

print("\nNumeric features:")
print(numeric_features)
print("\nCategorical features:")
print(categorical_features)


# =========================
# Preprocessing pipelines
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# =========================
# Random Forest model
# Uses a stronger but still conservative Version 1 setup inspired by the notebook
# =========================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
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

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nSaved trained model to: {MODEL_PATH}")

# =========================
# Validation evaluation
# =========================
y_valid_pred = model.predict(X_valid)
y_valid_prob = model.predict_proba(X_valid)[:, 1]

print("\n================ VALIDATION METRICS ================")
print(f"Accuracy:  {accuracy_score(y_valid, y_valid_pred):.4f}")
print(f"Precision: {precision_score(y_valid, y_valid_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_valid, y_valid_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_valid, y_valid_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_valid, y_valid_prob):.4f}")
print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_valid, y_valid_pred))
print("\nValidation Classification Report:")
print(classification_report(y_valid, y_valid_pred, zero_division=0))


# =========================
# Test evaluation
# =========================
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

print("\n==================== TEST METRICS ===================")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_prob):.4f}")
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))

print("\nRandom Forest training complete.")

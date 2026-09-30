#used BearcatGPT ai to add/clean up comments and add some additional validation checks.

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# File paths
# =========================
RAW_DATA_PATH = Path(r"C:\Users\audre\Documents\ML-Airport-Delay-Prediction\data\raw\T_ONTIME_REPORTING.csv")
CLEAN_FULL_PATH = Path(r"C:\Users\audre\Documents\ML-Airport-Delay-Prediction\data\processed\flights_clean_full.csv")
MODEL_READY_PATH = Path(r"C:\Users\audre\Documents\ML-Airport-Delay-Prediction\data\processed\flights_model_ready_v1.csv")


# =========================
# Helper functions
# =========================
def parse_hhmm(value):
    """Convert BTS HHMM time format into hour and minute."""
    if pd.isna(value):
        return pd.Series([np.nan, np.nan])

    try:
        value = int(float(value))
        hhmm = f"{value:04d}"
        hour = int(hhmm[:2])
        minute = int(hhmm[2:])

        if hour == 24 and minute == 0:
            hour = 0
        elif hour > 23 or minute > 59:
            return pd.Series([np.nan, np.nan])

        return pd.Series([hour, minute])
    except Exception:
        return pd.Series([np.nan, np.nan])


# =========================
# Load data
# =========================
print("Loading raw data...")
df = pd.read_csv(RAW_DATA_PATH)

print("\nInitial dataset loaded successfully.")
print(f"Initial shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())


# =========================
# Standardize columns
# =========================
df.columns = [col.strip().upper() for col in df.columns]


# =========================
# Drop duplicate rows
# =========================
initial_rows = len(df)
df = df.drop_duplicates()
rows_after_dedup = len(df)
print(f"\nDropped {initial_rows - rows_after_dedup:,} duplicate rows.")


# =========================
# Parse date
# =========================
df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")


# =========================
# Basic validation checks
# =========================
print("\nBasic validation checks:")
key_precheck_cols = [
    "FL_DATE", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME",
    "CRS_ARR_TIME", "ARR_DEL15", "CANCELLED", "DIVERTED", "CRS_ELAPSED_TIME"
]
for col in key_precheck_cols:
    if col in df.columns:
        print(f"Missing {col}: {df[col].isna().sum():,}")


# =========================
# Exclude cancelled flights
# =========================
rows_before_cancel = len(df)
df = df[df["CANCELLED"] != 1].copy()
rows_after_cancel = len(df)
print(f"\nRemoved {rows_before_cancel - rows_after_cancel:,} cancelled flights.")


# =========================
# Exclude diverted flights
# =========================
if "DIVERTED" in df.columns:
    rows_before_diverted = len(df)
    df = df[df["DIVERTED"] != 1].copy()
    rows_after_diverted = len(df)
    print(f"Removed {rows_before_diverted - rows_after_diverted:,} diverted flights.")
else:
    print("No DIVERTED column found. No diverted-flight filter applied.")


# =========================
# Drop rows where target cannot be determined
# Target source: ARR_DEL15 (recommended)
# =========================
rows_before_target = len(df)
df = df[df["ARR_DEL15"].notna()].copy()
rows_after_target = len(df)
print(f"Removed {rows_before_target - rows_after_target:,} rows with missing ARR_DEL15 target.")


# =========================
# Keep only valid binary target values
# =========================
df = df[df["ARR_DEL15"].isin([0, 1])].copy()
df["TARGET_DELAYED"] = df["ARR_DEL15"].astype(int)


# =========================
# Engineer time/date features
# =========================
df["MONTH"] = df["FL_DATE"].dt.month
# Keep BTS DAY_OF_WEEK if present; otherwise derive it
if "DAY_OF_WEEK" not in df.columns:
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek

df["DAY"] = df["FL_DATE"].dt.day
if "DAY_OF_MONTH" not in df.columns:
    df["DAY_OF_MONTH"] = df["FL_DATE"].dt.day

df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([5, 6]).astype(int)

df[["CRS_DEP_HOUR", "CRS_DEP_MINUTE"]] = df["CRS_DEP_TIME"].apply(parse_hhmm)
df[["CRS_ARR_HOUR", "CRS_ARR_MINUTE"]] = df["CRS_ARR_TIME"].apply(parse_hhmm)

df["DEP_HOUR"] = df["CRS_DEP_HOUR"]
df["ROUTE"] = df["ORIGIN"].astype(str) + "_" + df["DEST"].astype(str)


# =========================
# Additional validation checks
# =========================
print("\nPost-cleaning validation summary:")
print(f"Current shape: {df.shape}")
print(f"Delayed flights (1): {df['TARGET_DELAYED'].sum():,}")
print(f"On-time / not-delayed flights (0): {(df['TARGET_DELAYED'] == 0).sum():,}")
print(f"Target delay rate: {df['TARGET_DELAYED'].mean():.2%}")
print(f"Missing CRS_DEP_HOUR after parsing: {df['CRS_DEP_HOUR'].isna().sum():,}")
print(f"Missing CRS_ARR_HOUR after parsing: {df['CRS_ARR_HOUR'].isna().sum():,}")
print(f"Missing ROUTE: {df['ROUTE'].isna().sum():,}")

print("\nMissing values in key model fields:")
key_fields = [
    "FL_DATE", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME",
    "CRS_DEP_HOUR", "CRS_DEP_MINUTE", "DEP_HOUR", "MONTH", "DAY_OF_WEEK", "DAY",
    "DAY_OF_MONTH", "IS_WEEKEND", "ROUTE", "CRS_ELAPSED_TIME", "TARGET_DELAYED"
]
existing_key_fields = [col for col in key_fields if col in df.columns]
print(df[existing_key_fields].isna().sum())


# =========================
# Save clean full dataset
# =========================
CLEAN_FULL_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_FULL_PATH, index=False)
print(f"\nSaved clean full dataset to: {CLEAN_FULL_PATH}")


# =========================
# Create model-ready Version 1 dataset
# Updated to reflect the strongest Version 1 features from the notebook
# while keeping only pre-departure / scheduled information
# =========================
model_columns = [
    "FL_DATE",
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
    "ROUTE",
    "TARGET_DELAYED"
]

existing_model_columns = [col for col in model_columns if col in df.columns]
model_df = df[existing_model_columns].copy()
model_df.to_csv(MODEL_READY_PATH, index=False)
print(f"Saved model-ready dataset to: {MODEL_READY_PATH}")


# =========================
# Preview output
# =========================
print("\nModel-ready dataset preview:")
print(model_df.head())

print("\nData preparation complete.")

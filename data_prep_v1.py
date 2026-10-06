#used BearcatGPT ai to add/clean up comments and add some additional validation checks.

import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from datetime import datetime

# File paths
RAW_BASE = Path(r"C:\Users\audre\OneDrive\Desktop\ML Model Code\Data\Raw")
PROCESSED_DIR = Path(r"C:\Users\audre\OneDrive\Desktop\ML Model Code\Data\Processed")
CLEAN_FULL_PATH = PROCESSED_DIR / "flights_clean_full.csv"
MODEL_READY_PATH = PROCESSED_DIR / "flights_model_ready_v1.csv"
VALIDATION_LOG_PATH = PROCESSED_DIR / "validation_log.json"

# Monthly file mapping — Jan-Jul use abbreviation, Aug-Dec use full month + _2025
MONTHLY_FILES = {
    "JAN": RAW_BASE / "T_ONTIME_REPORTING_JAN.csv",
    "FEB": RAW_BASE / "T_ONTIME_REPORTING_FEB.csv",
    "MAR": RAW_BASE / "T_ONTIME_REPORTING_MAR.csv",
    "APR": RAW_BASE / "T_ONTIME_REPORTING_APR.csv",
    "MAY": RAW_BASE / "T_ONTIME_REPORTING_MAY.csv",
    "JUN": RAW_BASE / "T_ONTIME_REPORTING_JUN.csv",
    "JUL": RAW_BASE / "T_ONTIME_REPORTING_JUL.csv",
    "AUG": RAW_BASE / "T_ONTIME_REPORTING_AUG_2025.csv",
    "SEP": RAW_BASE / "T_ONTIME_REPORTING_SEP_2025.csv",
    "OCT": RAW_BASE / "T_ONTIME_REPORTING_OCT_2025.csv",
    "NOV": RAW_BASE / "T_ONTIME_REPORTING_NOV_2025.csv",
    "DEC": RAW_BASE / "T_ONTIME_REPORTING_DEC_2025.csv",
}

# Columns that must be present for a row to be usable
REQUIRED_COLS = ["FL_DATE", "OP_CARRIER_FL_NUM", "CANCELLED", "ARR_DEL15"]

# String columns that must be preserved as text and never coerced to numeric
# OP_UNIQUE_CARRIER contains airline codes (AA, DL, UA) — keeping it as str
# ensures it is treated as categorical by the model pipeline, not as a number
STRING_COLS = ["OP_UNIQUE_CARRIER", "TAIL_NUM"]

# Numeric columns that should be taken during cleaning
# OP_UNIQUE_CARRIER and TAIL_NUM are intentionally excluded from this list
# so they are never overwritten by pd.to_numeric()
NUMERIC_COLS = [
    "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME",
    "DEP_TIME", "DEP_DELAY", "ARR_DELAY", "ARR_DEL15",
    "DEP_DEL15", "CANCELLED", "DIVERTED",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]

# Model-ready column list for Version 1
# V1: BTS flight history only — scheduled timing, route, carrier, time-based features
# No weather, no previous-flight information
MODEL_COLUMNS_V1 = [
    "FL_DATE",
    "OP_UNIQUE_CARRIER",        # Airline code (AA, DL, UA) — categorical, kept as string
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
    "TARGET_DELAYED",
]


def parse_hhmm(value):
    """Convert BTS HHMM integer format to (hour, minute). Returns (NaN, NaN) on failure."""
    if pd.isna(value):
        return pd.Series([np.nan, np.nan])
    try:
        value = int(float(value))
        hhmm = f"{value:04d}"
        hour, minute = int(hhmm[:2]), int(hhmm[2:])
        if hour == 24 and minute == 0:
            hour = 0
        if hour > 23 or minute > 59:
            return pd.Series([np.nan, np.nan])
        return pd.Series([hour, minute])
    except Exception:
        return pd.Series([np.nan, np.nan])


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 checksum of a file for data versioning."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def ingest_monthly_files(file_map: dict) -> tuple[pd.DataFrame, list]:
    """Load, tag, and concatenate all available monthly CSVs."""
    frames = []
    ingestion_log = []

    for month_tag, path in file_map.items():
        if not path.exists():
            print(f"  [SKIP] {month_tag}: file not found -> {path}")
            ingestion_log.append({"month": month_tag, "status": "missing", "path": str(path)})
            continue

        try:
            # Force string dtype for carrier and tail number from the first read
            # Without this, pandas infers the type from the first N rows and may
            # silently cast OP_UNIQUE_CARRIER to float if any early rows are blank,
            # turning "AA" into NaN before we ever see it
            dtype_overrides = {col: str for col in STRING_COLS}
            chunk = pd.read_csv(path, low_memory=False, dtype=dtype_overrides)
            chunk.columns = [c.strip().upper() for c in chunk.columns]
            chunk["SOURCE_MONTH"] = month_tag

            # Strip whitespace from string columns so "AA " and "AA" are the same
            # Replace empty strings with NaN so the imputer handles them correctly
            for col in STRING_COLS:
                if col in chunk.columns:
                    chunk[col] = chunk[col].str.strip()
                    chunk[col] = chunk[col].replace("", np.nan)

            checksum = file_md5(path)
            rows = len(chunk)
            frames.append(chunk)
            print(f"  [OK]   {month_tag}: {rows:,} rows  |  MD5: {checksum}")

            # Print a sample of carrier values immediately after load to confirm
            # they are reading as letter codes, not numbers or blanks
            if "OP_UNIQUE_CARRIER" in chunk.columns:
                sample = chunk["OP_UNIQUE_CARRIER"].dropna().unique()[:8].tolist()
                print(f"           OP_UNIQUE_CARRIER sample: {sample}")

            ingestion_log.append({
                "month": month_tag,
                "status": "loaded",
                "path": str(path),
                "rows": rows,
                "md5": checksum,
                "ingested_at": datetime.now().isoformat(),
            })
        except Exception as e:
            print(f"  [ERROR] {month_tag}: {e}")
            ingestion_log.append({"month": month_tag, "status": "error", "error": str(e)})

    if not frames:
        raise RuntimeError("No monthly files could be loaded. Check your RAW_BASE path.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined shape after ingestion: {combined.shape}")
    return combined, ingestion_log


def validate(df: pd.DataFrame) -> dict:
    """
    Run all validation checks from the proposal and return a summary dict.
    Checks:
      1. Missing required fields
      2. Duplicate rows
      3. Invalid CANCELLED / DIVERTED values
      4. Negative CRS_ELAPSED_TIME
      5. Scheduled arrival before departure (same-day, ignoring overnight)
      6. ARR_DEL15 outside {0, 1}
      7. Text in numeric fields
      8. Blank OP_UNIQUE_CARRIER
    """
    report = {}

    # 1. Missing required fields
    for col in REQUIRED_COLS:
        if col in df.columns:
            report[f"missing_{col}"] = int(df[col].isna().sum())

    # 2. Duplicate rows
    report["duplicate_rows"] = int(df.duplicated().sum())

    # 3. Invalid binary flags
    for flag in ["CANCELLED", "DIVERTED", "ARR_DEL15", "DEP_DEL15"]:
        if flag in df.columns:
            bad = (~df[flag].isin([0, 1]) & df[flag].notna()).sum()
            report[f"invalid_{flag}"] = int(bad)

    # 4. Negative elapsed time
    if "CRS_ELAPSED_TIME" in df.columns:
        report["negative_elapsed_time"] = int((df["CRS_ELAPSED_TIME"] < 0).sum())

    # 5. Arrival before departure (simple numeric check — ignores overnight flights)
    if "CRS_DEP_TIME" in df.columns and "CRS_ARR_TIME" in df.columns:
        report["arr_before_dep_same_day"] = int(
            (df["CRS_ARR_TIME"] < df["CRS_DEP_TIME"]).sum()
        )

    # 6. ARR_DEL15 outside {0, 1}
    if "ARR_DEL15" in df.columns:
        report["arr_del15_out_of_range"] = int(
            (~df["ARR_DEL15"].isin([0, 1]) & df["ARR_DEL15"].notna()).sum()
        )

    # 7. Text in key numeric fields
    for col in ["DEP_DELAY", "ARR_DELAY", "CRS_ELAPSED_TIME", "CRS_DEP_TIME", "CRS_ARR_TIME"]:
        if col in df.columns:
            non_numeric = (
                pd.to_numeric(df[col], errors="coerce").isna().sum() - df[col].isna().sum()
            )
            if non_numeric > 0:
                report[f"non_numeric_{col}"] = int(non_numeric)

    # 8. Blank OP_UNIQUE_CARRIER
    if "OP_UNIQUE_CARRIER" in df.columns:
        report["missing_op_unique_carrier"] = int(df["OP_UNIQUE_CARRIER"].isna().sum())
        report["unique_carriers_found"] = int(df["OP_UNIQUE_CARRIER"].dropna().nunique())
        report["carrier_sample"] = df["OP_UNIQUE_CARRIER"].dropna().unique()[:8].tolist()

    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps in order.
    String columns (OP_UNIQUE_CARRIER, TAIL_NUM) are explicitly protected
    from numeric coercion at every step.
    """
    # Snapshot string columns before any cleaning so they can be restored
    # if a downstream step accidentally overwrites them
    protected = {}
    for col in STRING_COLS:
        if col in df.columns:
            protected[col] = df[col].copy()

    # Parse FL_DATE
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df[df["FL_DATE"].notna()].copy()

    # Change only the explicitly listed numeric columns — never touch STRING_COLS
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Restore protected string columns in case any step above touched them
    for col, series in protected.items():
        df[col] = series.loc[df.index]

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {before - len(df):,}")

    # Remove cancelled flights — no arrival outcome to predict
    before = len(df)
    df = df[df["CANCELLED"] != 1].copy()
    print(f"  Cancelled flights removed: {before - len(df):,}")

    # Remove diverted flights
    if "DIVERTED" in df.columns:
        before = len(df)
        df = df[df["DIVERTED"] != 1].copy()
        print(f"  Diverted flights removed: {before - len(df):,}")

    # Drop rows where the target cannot be determined
    before = len(df)
    df = df[df["ARR_DEL15"].notna()].copy()
    df = df[df["ARR_DEL15"].isin([0, 1])].copy()
    print(f"  Rows dropped (missing/invalid ARR_DEL15): {before - len(df):,}")

    # Drop rows missing critical scheduling fields
    before = len(df)
    df = df[df["CRS_DEP_TIME"].notna() & df["CRS_ARR_TIME"].notna()].copy()
    print(f"  Rows dropped (missing CRS times): {before - len(df):,}")

    # Drop rows with non-positive elapsed time
    if "CRS_ELAPSED_TIME" in df.columns:
        before = len(df)
        df = df[df["CRS_ELAPSED_TIME"] > 0].copy()
        print(f"  Rows dropped (non-positive elapsed time): {before - len(df):,}")

    # Confirm string columns are still intact after all cleaning steps
    for col in STRING_COLS:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            sample = df[col].dropna().unique()[:5].tolist()
            print(f"  {col} after cleaning — missing: {n_missing:,} | sample: {sample}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all time-based and route features."""

    # Date parts
    df["MONTH"] = df["FL_DATE"].dt.month
    if "DAY_OF_WEEK" not in df.columns:
        df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek
    df["DAY"] = df["FL_DATE"].dt.day
    if "DAY_OF_MONTH" not in df.columns:
        df["DAY_OF_MONTH"] = df["FL_DATE"].dt.day
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([5, 6]).astype(int)

    # Season — meteorological definition
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall",   10: "Fall",  11: "Fall"}
    df["SEASON"] = df["MONTH"].map(season_map)

    # Scheduled time parsing
    df[["CRS_DEP_HOUR", "CRS_DEP_MINUTE"]] = df["CRS_DEP_TIME"].apply(parse_hhmm)
    df[["CRS_ARR_HOUR", "CRS_ARR_MINUTE"]] = df["CRS_ARR_TIME"].apply(parse_hhmm)
    df["DEP_HOUR"] = df["CRS_DEP_HOUR"]

    # Time-of-day bucket
    def dep_bucket(hour):
        if pd.isna(hour):
            return np.nan
        hour = int(hour)
        if hour < 6:
            return "Early Morning"
        elif hour < 12:
            return "Morning"
        elif hour < 17:
            return "Afternoon"
        elif hour < 21:
            return "Evening"
        else:
            return "Night"

    df["DEP_TIME_BUCKET"] = df["CRS_DEP_HOUR"].apply(dep_bucket)

    # Route — state level to match available columns
    df["ROUTE"] = df["ORIGIN_STATE_NM"].astype(str) + "_" + df["DEST_STATE_NM"].astype(str)

    # Binary target
    df["TARGET_DELAYED"] = df["ARR_DEL15"].astype(int)

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    validation_output = {}

    # Ingest
    print("Ingesting monthly files...")
    df, ingestion_log = ingest_monthly_files(MONTHLY_FILES)
    validation_output["ingestion"] = ingestion_log

    # Validate raw
    print("\nValidating raw data...")
    raw_validation = validate(df)
    validation_output["raw_validation"] = raw_validation
    print(json.dumps(raw_validation, indent=2))

    # Clean
    print("\nCleaning data...")
    df = clean(df)
    print(f"Shape after cleaning: {df.shape}")

    # Engineer features
    print("\nEngineering features...")
    df = engineer_features(df)

    # Validate post-clean
    print("\nValidating cleaned data...")
    clean_validation = validate(df)
    validation_output["clean_validation"] = clean_validation
    print(json.dumps(clean_validation, indent=2))

    # Confirm OP_UNIQUE_CARRIER is populated before saving
    if "OP_UNIQUE_CARRIER" in df.columns:
        carrier_sample = df["OP_UNIQUE_CARRIER"].dropna().unique()[:10].tolist()
        carrier_missing = df["OP_UNIQUE_CARRIER"].isna().sum()
        print(f"\nOP_UNIQUE_CARRIER pre-save check:")
        print(f"  Missing: {carrier_missing:,}")
        print(f"  Sample values: {carrier_sample}")
        if carrier_missing == len(df):
            raise RuntimeError(
                "OP_UNIQUE_CARRIER is entirely blank. "
                "Check that the raw CSVs contain this column and re-run ingestion."
            )
    else:
        raise RuntimeError("OP_UNIQUE_CARRIER column not found after cleaning.")

    # Save clean full dataset
    df.to_csv(CLEAN_FULL_PATH, index=False)
    clean_md5 = file_md5(CLEAN_FULL_PATH)
    print(f"\nSaved clean full dataset: {CLEAN_FULL_PATH}")
    print(f"  MD5: {clean_md5}")

    # Build model-ready dataset
    existing_cols = [col for col in MODEL_COLUMNS_V1 if col in df.columns]
    missing_cols = [col for col in MODEL_COLUMNS_V1 if col not in df.columns]
    if missing_cols:
        print(f"\nWarning: these model columns were not found and will be skipped: {missing_cols}")

    model_df = df[existing_cols].copy()

    # Final confirmation that OP_UNIQUE_CARRIER made it into the model-ready file
    if "OP_UNIQUE_CARRIER" in model_df.columns:
        print(f"\nOP_UNIQUE_CARRIER in model-ready file:")
        print(f"  Missing: {model_df['OP_UNIQUE_CARRIER'].isna().sum():,}")
        print(f"  Unique carriers: {model_df['OP_UNIQUE_CARRIER'].nunique()}")
        print(f"  Sample: {model_df['OP_UNIQUE_CARRIER'].dropna().unique()[:10].tolist()}")

    model_df.to_csv(MODEL_READY_PATH, index=False)
    model_md5 = file_md5(MODEL_READY_PATH)
    print(f"\nSaved model-ready dataset: {MODEL_READY_PATH}")
    print(f"  Shape: {model_df.shape}")
    print(f"  MD5: {model_md5}")

    # Readback verification — re-read the CSV and confirm OP_UNIQUE_CARRIER survived
    readback = pd.read_csv(MODEL_READY_PATH, dtype={"OP_UNIQUE_CARRIER": str}, nrows=5)
    print(f"\nReadback verification (first 5 rows of OP_UNIQUE_CARRIER):")
    print(readback["OP_UNIQUE_CARRIER"].tolist())

    # Save validation log
    validation_output["output_files"] = {
        "clean_full": {"path": str(CLEAN_FULL_PATH), "md5": clean_md5},
        "model_ready": {
            "path": str(MODEL_READY_PATH),
            "md5": model_md5,
            "shape": list(model_df.shape),
            "columns": existing_cols,
        },
    }
    validation_output["run_timestamp"] = datetime.now().isoformat()

    with open(VALIDATION_LOG_PATH, "w") as f:
        json.dump(validation_output, f, indent=2)
    print(f"\nValidation log saved: {VALIDATION_LOG_PATH}")

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()

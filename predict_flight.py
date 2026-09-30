#used BearcatGPT ai to add/clean up comments and add some additional validation checks.


import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib

# =========================
# File paths
# =========================
MODEL_PATH = Path(r"C:\Users\audre\OneDrive\Desktop\ML Model Code\ML-Airport-Delay-Prediction\models\v1_random_forest.joblib")
DATA_PATH = Path(r"C:\Users\audre\Documents\ML-Airport-Delay-Prediction\data\processed\flights_model_ready_v1.csv")


# =========================
# Helper functions
# =========================
def parse_hhmm(value):
    """Convert HHMM input into hour and minute."""
    value = int(str(value).zfill(4))
    hour = value // 100
    minute = value % 100

    if hour == 24 and minute == 0:
        hour = 0

    if hour > 23 or minute > 59:
        raise ValueError("Time must be a valid HHMM value, such as 730, 0930, or 1745.")

    return hour, minute


def get_weekend_flag(day_of_week):
    return 1 if day_of_week in [5, 6] else 0


def get_state_name(airport_code, training_df, airport_column, state_column):
    """Look up state name from the training dataset using airport code."""
    if airport_column in training_df.columns and state_column in training_df.columns:
        matches = training_df.loc[
            training_df[airport_column].astype(str).str.upper() == airport_code.upper(),
            state_column
        ].dropna()

        if not matches.empty:
            return matches.iloc[0]

    return "Unknown"


# =========================
# Load model and training columns
# =========================
print("Loading trained model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

print("Loading model-ready dataset for feature alignment...")
training_df = pd.read_csv(DATA_PATH)
print("Reference dataset loaded successfully.\n")


# =========================
# Collect user input
# =========================
print("Enter flight information for prediction.")
flight_date_input = input("Flight date (YYYY-MM-DD): ").strip()
carrier_input = input("Operating carrier code (example: AA, DL, UA): ").strip().upper()
flight_num_input = input("Flight number (example: 1274): ").strip()
origin_input = input("Origin airport code (example: JFK): ").strip().upper()
dest_input = input("Destination airport code (example: LAX): ").strip().upper()
crs_dep_time_input = input("Scheduled departure time in HHMM format (example: 730 or 1735): ").strip()
crs_arr_time_input = input("Scheduled arrival time in HHMM format (example: 945 or 2010): ").strip()
crs_elapsed_time_input = input("Scheduled elapsed time in minutes (example: 145): ").strip()


# =========================
# Build input row
# =========================
flight_date = pd.to_datetime(flight_date_input)
day_of_week = flight_date.dayofweek
# Convert pandas Monday=0..Sunday=6 to BTS-like Monday=1..Sunday=7
bts_day_of_week = day_of_week + 1

crs_dep_hour, crs_dep_minute = parse_hhmm(crs_dep_time_input)
crs_arr_hour, crs_arr_minute = parse_hhmm(crs_arr_time_input)

origin_state = get_state_name(origin_input, training_df, "ORIGIN", "ORIGIN_STATE_NM")
dest_state = get_state_name(dest_input, training_df, "DEST", "DEST_STATE_NM")

input_df = pd.DataFrame([
    {
        "FL_DATE": flight_date,
        "OP_UNIQUE_CARRIER": carrier_input,
        "OP_CARRIER_FL_NUM": int(flight_num_input),
        "ORIGIN": origin_input,
        "ORIGIN_STATE_NM": origin_state,
        "DEST": dest_input,
        "DEST_STATE_NM": dest_state,
        "CRS_DEP_TIME": int(crs_dep_time_input),
        "CRS_ARR_TIME": int(crs_arr_time_input),
        "CRS_ELAPSED_TIME": float(crs_elapsed_time_input),
        "MONTH": flight_date.month,
        "DAY_OF_WEEK": bts_day_of_week,
        "DAY": flight_date.day,
        "DAY_OF_MONTH": flight_date.day,
        "IS_WEEKEND": get_weekend_flag(day_of_week),
        "CRS_DEP_HOUR": crs_dep_hour,
        "CRS_DEP_MINUTE": crs_dep_minute,
        "CRS_ARR_HOUR": crs_arr_hour,
        "CRS_ARR_MINUTE": crs_arr_minute,
        "DEP_HOUR": crs_dep_hour,
        "ROUTE": f"{origin_input}_{dest_input}"
    }
])


# =========================
# Align input columns to model expectation
# =========================
if "TARGET_DELAYED" in training_df.columns:
    training_df = training_df.drop(columns=["TARGET_DELAYED"])

expected_columns = training_df.columns.tolist()
input_df = input_df.reindex(columns=expected_columns)


# =========================
# Make prediction
# =========================
prediction = model.predict(input_df)[0]
probability_delay = model.predict_proba(input_df)[0, 1]

if probability_delay >= 0.70:
    risk_level = "High"
elif probability_delay >= 0.40:
    risk_level = "Moderate"
else:
    risk_level = "Low"

print("\n================ FLIGHT DELAY PREDICTION ================")
print(f"Predicted probability of delay: {probability_delay:.2%}")
print(f"Risk level: {risk_level}")
print(f"Predicted class: {'Delayed' if prediction == 1 else 'Not Delayed'}")
print("=========================================================")

print("\nInterpretation guidance:")
print("- This probability is based on the historical patterns learned by the trained model.")
print("- It should be treated as a decision-support estimate, not a guarantee.")
print("- If the script cannot find the model file, make sure you saved the trained Random Forest model as v1_random_forest.joblib in the models folder.")

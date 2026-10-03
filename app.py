from pathlib import Path
from datetime import date, time

import airportsdata
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# Page setup
# =========================================================
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ Flight Delay Prediction")

st.write(
    "Enter the scheduled flight information below to estimate "
    "whether the flight is likely to be delayed."
)


# =========================================================
# Load trained Random Forest model
# =========================================================
MODEL_PATH = Path(__file__).parent / "v1_random_forest.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as exc:
    st.error(f"Unable to load the prediction model: {exc}")
    st.stop()


# =========================================================
# Get categories learned by the fitted model
# =========================================================
preprocessor = model.named_steps["preprocessor"]

categorical_features = list(
    preprocessor.transformers_[1][2]
)

encoder = (
    preprocessor
    .named_transformers_["cat"]
    .named_steps["onehot"]
)

category_lookup = dict(
    zip(categorical_features, encoder.categories_)
)

valid_origins = set(
    category_lookup["ORIGIN"]
)

valid_destinations = set(
    category_lookup["DEST"]
)

valid_origin_states = set(
    category_lookup["ORIGIN_STATE_NM"]
)

valid_destination_states = set(
    category_lookup["DEST_STATE_NM"]
)

valid_routes = set(
    category_lookup["ROUTE"]
)


# =========================================================
# Airport metadata
# =========================================================
airport_data = airportsdata.load("IATA")


def build_airport_options(valid_airports, valid_states):
    """
    Return airports recognized by both the trained model
    and the airportsdata package.
    """

    options = {}

    for code in sorted(valid_airports):

        airport = airport_data.get(code)

        if not airport:
            continue

        state = airport.get("subd")

        if not state or state not in valid_states:
            continue

        name = airport.get("name", "")
        city = airport.get("city", "")

        label = f"{code} — {name}"

        if city:
            label += f" ({city})"

        options[label] = code

    return options


origin_options = build_airport_options(
    valid_origins,
    valid_origin_states,
)

destination_options = build_airport_options(
    valid_destinations,
    valid_destination_states,
)


# =========================================================
# Helper functions
# =========================================================
def time_to_hhmm(selected_time):
    """
    Convert a Python time object to BTS HHMM format.
    """

    return (
        selected_time.hour * 100
        + selected_time.minute
    )


def create_prediction_row(
    flight_number,
    origin,
    destination,
    flight_date,
    departure_time,
):
    """
    Create the exact fields expected by the trained model.
    """

    origin_state = airport_data[origin]["subd"]

    destination_state = (
        airport_data[destination]["subd"]
    )

    route = f"{origin}_{destination}"

    # BTS DAY_OF_WEEK:
    # Monday = 1
    # Sunday = 7
    day_of_week = flight_date.isoweekday()

    # Match the existing training preprocessing
    is_weekend = int(
        day_of_week in [5, 6]
    )

    crs_dep_time = time_to_hhmm(
        departure_time
    )

    prediction_data = {

        "OP_CARRIER_FL_NUM":
            int(flight_number),

        "ORIGIN":
            origin,

        "ORIGIN_STATE_NM":
            origin_state,

        "DEST":
            destination,

        "DEST_STATE_NM":
            destination_state,

        "CRS_DEP_TIME":
            crs_dep_time,

        "MONTH":
            flight_date.month,

        "DAY_OF_WEEK":
            day_of_week,

        "DAY":
            flight_date.day,

        "IS_WEEKEND":
            is_weekend,

        "CRS_DEP_HOUR":
            departure_time.hour,

        "CRS_DEP_MINUTE":
            departure_time.minute,

        "ROUTE":
            route,
    }

    return pd.DataFrame(
        [prediction_data]
    )


# =========================================================
# User Interface
# =========================================================
st.subheader("Flight Information")


flight_number = st.number_input(
    "Flight Number",
    min_value=1,
    max_value=9999,
    value=1000,
    step=1,
)


origin_label = st.selectbox(
    "Origin Airport",
    options=list(
        origin_options.keys()
    ),
)


destination_label = st.selectbox(
    "Destination Airport",
    options=list(
        destination_options.keys()
    ),
)


flight_date = st.date_input(
    "Flight Date",
    value=date.today(),
)


departure_time = st.time_input(
    "Scheduled Departure Time",
    value=time(12, 0),
)


# =========================================================
# Prediction
# =========================================================
if st.button(
    "Predict Flight Delay",
    type="primary",
    use_container_width=True,
):

    origin = origin_options[
        origin_label
    ]

    destination = destination_options[
        destination_label
    ]

    # Prevent selecting the same airport
    if origin == destination:

        st.warning(
            "Origin and destination airports "
            "must be different."
        )

        st.stop()


    route = f"{origin}_{destination}"


    # Check whether model knows this route
    if route not in valid_routes:

        st.warning(
            "This origin/destination combination "
            "was not present in the model's "
            "training data. Please choose another route."
        )

        st.stop()


    input_df = create_prediction_row(
        flight_number=flight_number,
        origin=origin,
        destination=destination,
        flight_date=flight_date,
        departure_time=departure_time,
    )


    try:

        prediction = int(
            model.predict(input_df)[0]
        )

        probabilities = (
            model.predict_proba(input_df)[0]
        )

        delay_probability = float(
            probabilities[1]
        )

        on_time_probability = float(
            probabilities[0]
        )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )

        st.stop()


    # =====================================================
    # Display prediction
    # =====================================================
    st.divider()

    st.subheader(
        "Prediction Result"
    )


    if prediction == 1:

        st.error(
            "⚠️ Flight Delay Predicted"
        )

    else:

        st.success(
            "✅ Flight Predicted On Time"
        )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Delay Probability",
            f"{delay_probability:.1%}",
        )


    with col2:

        st.metric(
            "On-Time Probability",
            f"{on_time_probability:.1%}",
        )


    # Optional view of fields sent to model
    with st.expander(
        "Model Inputs"
    ):

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# Interface enhancement description
# =========================================================
st.divider()

st.caption(
    "The interface automatically derives state, route, "
    "calendar, and departure-time features from the "
    "information entered above."
)
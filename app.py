
import streamlit as st
import pandas as pd
import joblib
from datetime import date

# Load trained model and preprocessing files
model = joblib.load("best_xgb_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Bike Rental Prediction",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 Bike Rental Demand Prediction")
st.write("Machine Learning Based Bike Rental Prediction System")

st.divider()

st.header("Enter Bike Rental Details")

col1, col2, col3 = st.columns(3)

with col1:
    selected_date = st.date_input(
        "Select Date",
        value=date(2011, 1, 1),
        min_value=date(2011, 1, 1),
        max_value=date(2012, 12, 31)
    )

    hour = st.selectbox("Hour", list(range(24)), index=8)

    holiday = st.selectbox(
        "Holiday",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    season = st.selectbox(
        "Season",
        ["spring", "summer", "fall", "winter"]
    )

    workingday = st.selectbox(
        "Working Day",
        [0, 1],
        format_func=lambda x: "Working Day" if x == 1 else "No work"
    )

    weather = st.selectbox(
        "Weather Situation",
        ["Clear", "Mist", "Light Snow", "Heavy Rain"]
    )

with col3:
    temp = st.number_input(
        "Temperature (0–1)",
        min_value=0.0, max_value=1.0,
        value=0.50, step=0.01
    )

    atemp = st.number_input(
        "Feels-like Temperature (0–1)",
        min_value=0.0, max_value=1.0,
        value=0.50, step=0.01
    )

    hum = st.number_input(
        "Humidity (0–1)",
        min_value=0.0, max_value=1.0,
        value=0.50, step=0.01
    )

windspeed = st.number_input(
    "Windspeed (0–1)",
    min_value=0.0, max_value=1.0,
    value=0.20, step=0.01
)

if st.button("Predict Bike Rentals", type="primary"):

    # Date-based features
    yr = selected_date.year
    mnth = selected_date.month
    day = selected_date.day
    dayofyear = selected_date.timetuple().tm_yday

    # Dataset weekday: Sunday=0, Monday=1, ..., Saturday=6
    weekday = (selected_date.weekday() + 1) % 7

    # Time period based on original notebook bins
    if hour <= 5:
        day_period = "Night"
    elif hour <= 11:
        day_period = "Morning"
    elif hour <= 16:
        day_period = "Afternoon"
    elif hour <= 20:
        day_period = "Evening"
    else:
        day_period = "Late Night"

    # Build original numerical and binary features
    input_data = {
        "yr": yr,
        "mnth": mnth,
        "hr": hour,
        "holiday": holiday,
        "workingday": workingday,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "day": day,
        "dayofyear": dayofyear
    }

    # One-hot encoding: season (fall is baseline)
    for value in ["spring", "summer", "winter"]:
        input_data[f"season_{value}"] = int(season == value)

    # One-hot encoding: day period (Night is baseline)
    for value in ["Morning", "Afternoon", "Evening", "Late Night"]:
        input_data[f"day_period_{value}"] = int(day_period == value)

    # One-hot encoding: weather (Clear is baseline)
    for value in ["Heavy Rain", "Light Snow", "Mist"]:
        input_data[f"weathersit_{value}"] = int(weather == value)

    # One-hot encoding: weekday (0 is baseline)
    for value in range(1, 7):
        input_data[f"weekday_{value}"] = int(weekday == value)

    # Arrange features in exact training order
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Apply the saved scaler to the same four columns
    scaled_columns = ["temp", "atemp", "hum", "windspeed"]
    input_df[scaled_columns] = scaler.transform(
        input_df[scaled_columns]
    )

    # Predict
    prediction = model.predict(input_df)[0]

    st.divider()
    st.subheader("Prediction Result")
    st.success(
        f"Predicted Bike Rentals: {max(0, round(float(prediction)))}"
    )


import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bike Rental | P716",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_files():
    model = joblib.load("best_xgb_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, feature_columns, scaler

model, feature_columns, scaler = load_files()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.hero {
    background: linear-gradient(120deg, #172554, #123047, #064e3b);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #31516a;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

.metric-card {
    background: #151f32;
    border: 1px solid #293951;
    padding: 22px;
    border-radius: 15px;
    min-height: 125px;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.result-card {
    background: linear-gradient(120deg, #064e3b, #14532d);
    border: 1px solid #22c55e;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
}

.result-number {
    color: white;
    font-size: 48px;
    font-weight: 800;
}

.result-label {
    color: #bbf7d0;
    font-size: 16px;
}

.section-title {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

div.stButton > button {
    background: #0d9488;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #0f766e;
    color: white;
    border: none;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🚲 BIKE RENTAL")
    st.caption("P716 | Data Science Project")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Prediction",
            "Model Performance",
            "About Project"
        ]
    )

    st.divider()
    st.caption("Developed for ExcelR P716")
    st.caption("Machine Learning Application")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🚲 Bike Rental Demand Prediction</h1>
    <p>Smart Machine Learning Dashboard | P716 Data Science Project</p>
</div>
""", unsafe_allow_html=True)


# ---------------- OVERVIEW ----------------
if page == "Overview":

    st.markdown("## Dashboard Overview")
    st.write(
        "Predict bike rental demand using weather, date and time-based "
        "features with a trained XGBoost Machine Learning model."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Selected ML Model</div>
            <div class="metric-value">XGBoost</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model R² Score</div>
            <div class="metric-value">96.05%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Prediction Type</div>
            <div class="metric-value">Regression</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Project Highlights</div>',
                unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        st.info("📊 Exploratory Data Analysis and Data Visualization")
        st.info("⚙️ Feature Engineering and Data Preprocessing")
        st.info("🤖 Machine Learning Model Comparison")

    with b:
        st.success("🌳 Tuned XGBoost Regression Model")
        st.success("📈 Model Evaluation and Performance")
        st.success("🚀 Streamlit Web Application Deployment")

    st.markdown("---")
    st.markdown("### Start Prediction")
    st.write("Go to the Prediction section to estimate bike rental demand.")

    if st.button("Open Prediction Page"):
        st.session_state["navigation_prediction"] = True
        st.info("Select Prediction from the sidebar to continue.")


# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.markdown("## 🔮 Bike Rental Prediction")
    st.write("Enter the required details to predict bike rental demand.")

    with st.form("prediction_form"):

        st.markdown("### 📅 Date and Time")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_date = st.date_input(
                "Select Date",
                value=date(2012, 1, 1),
                min_value=date(2011, 1, 1),
                max_value=date(2012, 12, 31)
            )

        with col2:
            hour = st.selectbox(
                "Hour",
                list(range(24)),
                index=8
            )

        with col3:
            season = st.selectbox(
                "Season",
                ["spring", "summer", "fall", "winter"]
            )

        st.markdown("### 🌤️ Weather Information")

        col4, col5, col6 = st.columns(3)

        with col4:
            temp = st.number_input(
                "Temperature (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01
            )

        with col5:
            atemp = st.number_input(
                "Feels-like Temperature (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01
            )

        with col6:
            hum = st.number_input(
                "Humidity (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01
            )

        col7, col8, col9 = st.columns(3)

        with col7:
            windspeed = st.number_input(
                "Windspeed (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=0.20,
                step=0.01
            )

        with col8:
            weather = st.selectbox(
                "Weather Situation",
                ["Clear", "Mist", "Light Snow", "Heavy Rain"]
            )

        with col9:
            holiday = st.selectbox(
                "Holiday",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        workingday = st.selectbox(
            "Working Day",
            [0, 1],
            format_func=lambda x: "Working Day" if x == 1 else "No Work"
        )

        st.markdown("---")

        submitted = st.form_submit_button(
            "🚀 Predict Bike Rentals",
            use_container_width=True
        )

    if submitted:

        # Date-based features
        yr = selected_date.year - 2011
        mnth = selected_date.month
        day = selected_date.day
        dayofyear = selected_date.timetuple().tm_yday

        # Dataset weekday: Sunday=0, Monday=1, ..., Saturday=6
        weekday = (selected_date.weekday() + 1) % 7

        # Time period
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

        # Original features
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

        # Season encoding
        for value in ["spring", "summer", "winter"]:
            input_data[f"season_{value}"] = int(season == value)

        # Day period encoding
        for value in ["Morning", "Afternoon", "Evening", "Late Night"]:
            input_data[f"day_period_{value}"] = int(day_period == value)

        # Weather encoding
        for value in ["Heavy Rain", "Light Snow", "Mist"]:
            input_data[f"weathersit_{value}"] = int(weather == value)

        # Weekday encoding
        for value in range(1, 7):
            input_data[f"weekday_{value}"] = int(weekday == value)

        # Arrange columns in training order
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Scale numerical columns
        scaled_columns = ["temp", "atemp", "hum", "windspeed"]

        input_df[scaled_columns] = scaler.transform(
            input_df[scaled_columns]
        )

        # Prediction
        prediction = model.predict(input_df)[0]
        prediction = max(0, round(float(prediction)))

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Bike Rentals</div>
            <div class="result-number">{prediction:,}</div>
            <div class="result-label">Predicted rental demand</div>
        </div>
        """, unsafe_allow_html=True)

        st.success("Prediction completed successfully!")

        st.caption(
            "Note: This is a Machine Learning estimate, not a guaranteed "
            "actual rental count."
        )


# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":

    st.markdown("## 📈 Model Performance")
    st.write("Performance of the tuned XGBoost regression model.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("R² Score", "0.9605")

    with c2:
        st.metric("MAE", "21.89")

    with c3:
        st.metric("RMSE", "36.32")

    st.markdown("---")
    st.markdown("### Performance Interpretation")

    st.write("""
    - **R² Score:** 0.9605 — the model explained approximately 96.05%
      of the target variance on the evaluated test split.
    - **MAE:** 21.89 — the average absolute prediction error was
      approximately 21.89 rentals.
    - **RMSE:** 36.32 — the root mean squared prediction error was
      approximately 36.32 rentals.
    """)

    st.info(
        "These metrics are from the project's evaluated test split. "
        "Performance on future dates may differ."
    )


# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":

    st.markdown("## ℹ️ About This Project")

    st.write("""
    **Bike Rental Demand Prediction** is a Data Science and Machine
    Learning project developed as part of ExcelR Project P716.

    The application predicts the expected number of bike rentals using
    date, time, season and weather-related input features.
    """)

    st.markdown("### Technology Stack")

    tech1, tech2 = st.columns(2)

    with tech1:
        st.info("🐍 Python")
        st.info("📊 Pandas")
        st.info("🔢 NumPy")

    with tech2:
        st.info("🤖 XGBoost")
        st.info("⚙️ Scikit-learn")
        st.info("🌐 Streamlit")

    st.markdown("---")
    st.markdown("### Project Information")
    st.write("**Project:** Bike Rental Demand Prediction")
    st.write("**Project ID:** P716")
    st.write("**Model:** Tuned XGBoost Regressor")
    st.write("**Application:** Streamlit Web App")


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center style='color:#94a3b8;'>Bike Rental Demand Prediction "
    "| ExcelR P716 | Machine Learning Project</center>",
    unsafe_allow_html=True
)

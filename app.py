
import streamlit as st
import pandas as pd
import joblib
from datetime import date

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Bike Rental Prediction | P716",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODEL FILES
# =====================================================

@st.cache_resource
def load_files():
    model = joblib.load("best_xgb_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, feature_columns, scaler


try:
    model, feature_columns, scaler = load_files()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()


# =====================================================
# CUSTOM CSS - PREMIUM DARK THEME
# =====================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

.main {
    background-color: #0b1120;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #e5e7eb !important;
}

/* HEADINGS */

h1, h2, h3, h4 {
    color: #f8fafc !important;
}

.stApp p {
    color: #cbd5e1;
}

/* HERO SECTION */

.hero-card {
    background: linear-gradient(
        120deg,
        #172554,
        #123047,
        #064e3b
    );
    border: 1px solid #31516a;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero-card h1 {
    color: #ffffff !important;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero-card p {
    color: #cbd5e1 !important;
    font-size: 16px;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background-color: #151f32 !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* INFO AND SUCCESS ALERTS */

div[data-testid="stAlert"] {
    background-color: #172554 !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

div[data-testid="stAlert"] * {
    color: #f1f5f9 !important;
}

/* FORM */

div[data-testid="stForm"] {
    background-color: #151f32 !important;
    border: 1px solid #334155 !important;
    border-radius: 15px !important;
    padding: 22px !important;
}

/* INPUT LABELS */

.stApp label {
    color: #e2e8f0 !important;
}

/* INPUT BOXES */

.stApp input,
.stApp textarea {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border-color: #475569 !important;
}

/* SELECT BOX */

.stApp [data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border-color: #475569 !important;
}

.stApp [data-baseweb="select"] * {
    color: #ffffff !important;
}

/* BUTTONS */

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background-color: #0d9488 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 45px;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #0f766e !important;
    color: #ffffff !important;
}

/* PREDICTION RESULT */

.result-card {
    background: linear-gradient(
        120deg,
        #064e3b,
        #14532d
    );
    border: 1px solid #22c55e;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}

.result-card h2 {
    color: #bbf7d0 !important;
    font-size: 20px;
}

.result-card h1 {
    color: #ffffff !important;
    font-size: 52px;
    margin: 15px 0;
}

.result-card p {
    color: #bbf7d0 !important;
}

/* DIVIDER */

hr {
    border-color: #293951 !important;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #94a3b8 !important;
    padding: 20px;
    font-size: 13px;
}

.footer p {
    color: #94a3b8 !important;
}
.footer {
    text-align: center;
    color: #94a3b8 !important;
    padding: 20px;
    font-size: 13px;
}

/* NEW COLOR CODE */
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {
    color: #102a43 !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

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


# =====================================================
# MAIN HEADER
# =====================================================

st.markdown("""
<div class="hero-card">
    <h1>🚲 Bike Rental Demand Prediction</h1>
    <p>
        Smart Machine Learning Dashboard |
        P716 Data Science Project
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# OVERVIEW PAGE
# =====================================================

if page == "Overview":

    st.header("Dashboard Overview")

    st.write(
        "Predict bike rental demand using date, time, "
        "season and weather information with a trained "
        "XGBoost Machine Learning model."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Machine Learning Model",
            "XGBoost"
        )

    with col2:
        st.metric(
            "R² Score",
            "96.05%"
        )

    with col3:
        st.metric(
            "Prediction Type",
            "Regression"
        )

    st.divider()

    st.subheader("Project Highlights")

    left, right = st.columns(2)

    with left:
        st.info("📊 Exploratory Data Analysis")
        st.info("⚙️ Feature Engineering")
        st.info("🧹 Data Preprocessing")

    with right:
        st.info("🌳 Tuned XGBoost Model")
        st.info("📈 Model Evaluation")
        st.info("🚀 Streamlit Deployment")

    st.divider()

    st.subheader("How It Works")

    st.write("""
    1. Enter the date and time.
    2. Select season and weather conditions.
    3. Enter temperature, humidity and windspeed.
    4. Click Predict Bike Rentals.
    5. View the estimated bike rental demand.
    """)


# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.header("🔮 Bike Rental Prediction")

    st.write(
        "Enter the required details to estimate bike rental demand."
    )

    with st.form("prediction_form"):

        st.subheader("📅 Date and Time")

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

        st.divider()

        st.subheader("🌤️ Weather Information")

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
                [
                    "Clear",
                    "Mist",
                    "Light Snow",
                    "Heavy Rain"
                ]
            )

        with col9:
            holiday = st.selectbox(
                "Holiday",
                [0, 1],
                format_func=lambda x:
                "Yes" if x == 1 else "No"
            )

        workingday = st.selectbox(
            "Working Day",
            [0, 1],
            format_func=lambda x:
            "Working Day" if x == 1 else "No Work"
        )

        st.divider()

        submitted = st.form_submit_button(
            "🚀 Predict Bike Rentals",
            use_container_width=True
        )

    # =================================================
    # PREDICTION LOGIC
    # =================================================

    if submitted:

        # Date features
        yr = selected_date.year - 2011
        mnth = selected_date.month
        day = selected_date.day
        dayofyear = selected_date.timetuple().tm_yday

        # Dataset weekday:
        # Sunday=0, Monday=1 ... Saturday=6
        weekday = (selected_date.weekday() + 1) % 7

        # Day period
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

            input_data[f"season_{value}"] = int(
                season == value
            )

        # Day period encoding
        for value in [
            "Morning",
            "Afternoon",
            "Evening",
            "Late Night"
        ]:

            input_data[f"day_period_{value}"] = int(
                day_period == value
            )

        # Weather encoding
        for value in [
            "Heavy Rain",
            "Light Snow",
            "Mist"
        ]:

            input_data[f"weathersit_{value}"] = int(
                weather == value
            )

        # Weekday encoding
        for value in range(1, 7):

            input_data[f"weekday_{value}"] = int(
                weekday == value
            )

        # DataFrame in training feature order
        input_df = pd.DataFrame([input_data])

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Apply saved scaler
        scaled_columns = [
            "temp",
            "atemp",
            "hum",
            "windspeed"
        ]

        input_df[scaled_columns] = scaler.transform(
            input_df[scaled_columns]
        )

        # Model prediction
        prediction = model.predict(input_df)[0]

        prediction = max(
            0,
            round(float(prediction))
        )

        # Display result
        st.divider()

        st.subheader("🎯 Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">
                <h2>Estimated Bike Rentals</h2>
                <h1>{prediction:,}</h1>
                <p>Predicted rental demand</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "Prediction completed successfully!"
        )

        st.caption(
            "Note: This is a Machine Learning estimate, "
            "not a guaranteed actual rental count."
        )


# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")

    st.write(
        "Performance metrics of the tuned XGBoost regression model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "R² Score",
            "0.9605"
        )

    with col2:
        st.metric(
            "MAE",
            "21.89"
        )

    with col3:
        st.metric(
            "RMSE",
            "36.32"
        )

    st.divider()

    st.subheader("Performance Interpretation")

    st.markdown("""
    **R² Score: 0.9605**

    The model explained approximately 96.05% of the target
    variance on the evaluated test split.

    **MAE: 21.89**

    The average absolute prediction error was approximately
    21.89 rentals.

    **RMSE: 36.32**

    The root mean squared prediction error was approximately
    36.32 rentals.
    """)

    st.info(
        "These metrics are from the project's evaluated test split. "
        "Performance on future dates may differ."
    )


# =====================================================
# ABOUT PROJECT PAGE
# =====================================================

elif page == "About Project":

    st.header("ℹ️ About Project")

    st.write("""
    Bike Rental Demand Prediction is a Data Science and
    Machine Learning project developed as part of ExcelR
    Project P716.

    This application estimates the expected number of bike
    rentals using date, time, season and weather-related features.
    """)

    st.divider()

    st.subheader("Technology Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.info("🐍 Python")
        st.info("📊 Pandas")
        st.info("🔢 NumPy")

    with col2:
        st.info("🤖 XGBoost")
        st.info("⚙️ Scikit-learn")
        st.info("🌐 Streamlit")

    st.divider()

    st.subheader("Project Information")

    st.write("**Project Name:** Bike Rental Demand Prediction")
    st.write("**Project ID:** P716")
    st.write("**Model:** Tuned XGBoost Regressor")
    st.write("**Application:** Streamlit Web Application")


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">
    <p>Bike Rental Demand Prediction | ExcelR P716</p>
    <p>Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------- Helper to get next week's date -----------
def get_next_week_date():
    today = datetime.today()
    next_week = today + timedelta(days=(6 - today.weekday()))
    return next_week

# -------- Feature definitions -----------
month_names = [
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]

rf_features = [
    'Holiday_Flag', 'Temperature', 'Fuel_Price',
    'CPI', 'Unemployment', 'Year',
    'Month', 'Week', 'Day', 'DayOfWeek',
    'Is_Weekend'
] + [f'MonthName_{month}' for month in month_names] \
  + ['Weekly_Sales_lag1', 'Weekly_Sales_lag2', 'Weekly_Sales_lag3']

prophet_regressors = [
    'Holiday_Flag', 'Temperature', 'Fuel_Price',
    'CPI', 'Unemployment', 'Year',
    'Month', 'Week', 'Day', 'DayOfWeek',
    'Is_Weekend'
] + [f'MonthName_{month}' for month in month_names]

# -------- Streamlit UI -----------
st.title("🧮 Weekly Sales Forecast App — with Confidence Interval")

model_choice = st.radio("Choose a model for prediction:", ["Random Forest", "Prophet"])

st.header("Input Feature Values")

# Collect user input for all features
input_dict = {}

input_dict['Holiday_Flag'] = st.selectbox("Holiday Flag", [0, 1])
input_dict['Temperature'] = st.number_input("Temperature", value=70.0)
input_dict['Fuel_Price'] = st.number_input("Fuel Price", value=2.5)
input_dict['CPI'] = st.number_input("CPI", value=220.0)
input_dict['Unemployment'] = st.number_input("Unemployment", value=7.5)
input_dict['Year'] = st.number_input("Year (e.g., 2024)", value=datetime.now().year, step=1)
input_dict['Month'] = st.slider("Month (1-12)", 1, 12, datetime.now().month)
input_dict['Week'] = st.slider("Week Number (1-52)", 1, 52, datetime.now().isocalendar().week)
input_dict['Day'] = st.slider("Day (1-31)", 1, 31, datetime.now().day)
input_dict['DayOfWeek'] = st.slider("Day of week (0=Mon, 6=Sun)", 0, 6, datetime.now().weekday())
input_dict['Is_Weekend'] = st.selectbox("Is Weekend?", [0, 1])

# MonthName one-hot
chosen_month = month_names[input_dict['Month'] - 1]
for month in month_names:
    colname = f'MonthName_{month}'
    input_dict[colname] = 1 if month == chosen_month else 0

# Supply constant/placeholder for lags, if model expects them
input_dict['Weekly_Sales_lag1'] = 0     # or better, np.nan or a 'typical' value from your data
input_dict['Weekly_Sales_lag2'] = 0
input_dict['Weekly_Sales_lag3'] = 0

# ----------- Predict button -----------
if st.button("Predict Next Week's Sales"):
    if model_choice == "Random Forest":
        try:
            model = joblib.load("random_forest_model.pkl")
            input_df = pd.DataFrame([input_dict], columns=rf_features)
            tree_preds = np.array([tree.predict(input_df)[0] for tree in model.estimators_])
            lower = np.percentile(tree_preds, 2.5)
            upper = np.percentile(tree_preds, 97.5)
            pred = np.mean(tree_preds)
            st.markdown(f"""
            ### 🟩 Weekly Sales Prediction (Random Forest)
            - **Prediction:** ${pred:,.2f}
            - **95% Confidence Interval:** ${lower:,.2f} ... ${upper:,.2f}
            """)
            st.caption(
                "This interval reflects model uncertainty from tree variation (not full predictive uncertainty), so treat with care."
            )
        except Exception as e:
            st.error(f"Error loading Random Forest model or predicting: {e}")
    else:
        try:
            model = joblib.load("prophet_model.pkl")
            future_date = get_next_week_date()
            prophet_input = {'ds': [future_date]}
            for feature in prophet_regressors:
                prophet_input[feature] = [input_dict[feature]]
            future_df = pd.DataFrame(prophet_input)
            forecast = model.predict(future_df)
            pred = forecast['yhat'].iloc[0]
            lower = forecast['yhat_lower'].iloc[0]
            upper = forecast['yhat_upper'].iloc[0]
            st.markdown(f"""
            ### 📈 Weekly Sales Prediction (Prophet)
            - **Prediction:** ${pred:,.2f}
            - **Confidence Interval:** ${lower:,.2f} ... ${upper:,.2f}
            """)
            st.caption(
                "Prophet's confidence interval is an approximate coverage interval for model and noise uncertainty."
            )
        except Exception as e:
            st.error(f"Error loading Prophet model or predicting: {e}")

st.caption("Models must be pre-trained and saved as .pkl files. Input values should reflect your data's typical range. Confidence intervals are estimates.")
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(page_title="Bankruptcy Prediction App", layout="centered")

# Title
st.title("🏦 Bankruptcy Prediction App")

# Load trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("xgb_tuned_bst.joblib")

model = load_model()

# Load sample test data (or let user upload)
@st.cache
def load_data():
    return pd.read_csv("X_test.csv")

X_test = load_data()

# Display first few rows
st.subheader("Sample Test Data")
st.dataframe(X_test.head())

# Select a row for SHAP explanation
index = st.slider("Select data index to explain:", 0, len(X_test)-1, 0)

input_data = X_test.iloc[[index]]
st.write("Selected Input:")
st.write(input_data)

import streamlit as st
import numpy as np
import pandas as pd

st.title("Bankruptcy Prediction App")
st.subheader("Enter Company Financial Metrics")

user_input = {}

st.markdown("### Key Financial Ratios (User Input)")
user_input['Attr_1'] = st.number_input("Attr_1: Net profit / total assets", value=0.05)
user_input['Attr_2'] = st.number_input("Attr_2: Total liabilities / total assets", value=0.5)
user_input['Attr_3'] = st.number_input("Attr_3: Working capital / total assets", value=0.1)
user_input['Attr_4'] = st.number_input("Attr_4: Current assets / short-term liabilities", value=1.5)
user_input['Attr_5'] = st.number_input("Attr_5", value=10.0)
user_input['Attr_6'] = st.number_input("Attr_6: Retained earnings / total assets", value=0.2)
user_input['Attr_7'] = st.number_input("Attr_7: EBIT / total assets", value=0.15)
user_input['Attr_8'] = st.number_input("Attr_8: Book value of equity / total liabilities", value=0.3)
user_input['Attr_9'] = st.number_input("Attr_9: Sales / total assets", value=1.2)
user_input['Attr_10'] = st.number_input("Attr_10: Equity / total assets", value=0.4)
user_input['Attr_11'] = st.number_input("Attr_11: (Gross profit + extraordinary items + financial expenses) / total assets", value=0.1)
user_input['Attr_12'] = st.number_input("Attr_12: Gross profit / short-term liabilities", value=0.2)
user_input['Attr_13'] = st.number_input("Attr_13: (Gross profit + depreciation) / sales", value=0.15)
user_input['Attr_14'] = st.number_input("Attr_14: (Gross profit + interest) / total assets", value=0.1)
user_input['Attr_15'] = st.number_input("Attr_15: (Total liabilities × 365) / (Gross profit + depreciation)", value=200.0)
user_input['Attr_16'] = st.number_input("Attr_16: (Gross profit + depreciation) / total liabilities", value=0.2)
user_input['Attr_17'] = st.number_input("Attr_17: Total assets / total liabilities", value=1.5)
user_input['Attr_18'] = st.number_input("Attr_18: Gross profit / total assets", value=0.1)
user_input['Attr_19'] = st.number_input("Attr_19: Gross profit / sales", value=0.3)
user_input['Attr_20'] = st.number_input("Attr_20: (Inventory × 365) / sales", value=60.0)
user_input['Attr_21'] = st.number_input("Attr_21: Sales (n) / Sales (n−1)", value=1.1)
user_input['Attr_22'] = st.number_input("Attr_22: Profit on operating activities / total assets", value=0.12)
user_input['Attr_23'] = st.number_input("Attr_23: Net profit / sales", value=0.1)
user_input['Attr_24'] = st.number_input("Attr_24: Gross profit (in 3 years) / total assets", value=0.25)
user_input['Attr_25'] = st.number_input("Attr_25: (Equity − share capital) / total assets", value=0.15)
user_input['Attr_26'] = st.number_input("Attr_26: (Net profit + depreciation) / total liabilities", value=0.2)
user_input['Attr_27'] = st.number_input("Attr_27: Profit on operating activities / financial expenses", value=3.0)
user_input['Attr_28'] = st.number_input("Attr_28: Working capital / fixed assets", value=0.4)
user_input['Attr_29'] = st.number_input("Attr_29: Logarithm of total assets", value=8.5)
user_input['Attr_30'] = st.number_input("Attr_30: (Total liabilities − cash) / sales", value=1.0)
user_input['Attr_31'] = st.number_input("Attr_31: (Gross profit + interest) / sales", value=0.2)
user_input['Attr_32'] = st.number_input("Attr_32: (Current liabilities × 365) / cost of products sold", value=90.0)
user_input['Attr_33'] = st.number_input("Attr_33: Operating expenses / short-term liabilities", value=0.6)
user_input['Attr_34'] = st.number_input("Attr_34: Operating expenses / total liabilities", value=0.3)
user_input['Attr_35'] = st.number_input("Attr_35: Profit on sales / total assets", value=0.12)
user_input['Attr_36'] = st.number_input("Attr_36: Total sales / total assets", value=1.2)
user_input['Attr_37'] = st.number_input("Attr_37: (Current assets − inventories) / long-term liabilities", value=1.5)
user_input['Attr_38'] = st.number_input("Attr_38: Constant capital / total assets", value=0.4)
user_input['Attr_39'] = st.number_input("Attr_39: Profit on sales / sales", value=0.1)
user_input['Attr_40'] = st.number_input("Attr_40: (Current assets − inventory − receivables) / short-term liabilities", value=0.3)
user_input['Attr_41'] = st.number_input("Attr_41: Total liabilities / [missing info]", value=0.8)
user_input['Attr_42'] = st.number_input("Attr_42: Profit on operating activities / sales", value=0.1)
user_input['Attr_43'] = st.number_input("Attr_43: Rotation receivables + inventory turnover in days", value=70.0)
user_input['Attr_44'] = st.number_input("Attr_44: (Receivables × 365) / sales", value=50.0)
user_input['Attr_45'] = st.number_input("Attr_45: Net profit / inventory", value=1.5)
user_input['Attr_46'] = st.number_input("Attr_46: (Current assets − inventory) / short-term liabilities", value=1.0)
user_input['Attr_47'] = st.number_input("Attr_47: (Inventory × 365) / cost of products sold", value=60.0)
user_input['Attr_48'] = st.number_input("Attr_48: EBITDA / total assets", value=0.2)
user_input['Attr_49'] = st.number_input("Attr_49: EBITDA / sales", value=0.15)
user_input['Attr_50'] = st.number_input("Attr_50: Current assets / total liabilities", value=1.0)
user_input['Attr_51'] = st.number_input("Attr_51: Short-term liabilities / total assets", value=0.4)
user_input['Attr_52'] = st.number_input("Attr_52: (Short-term liabilities × 365) / cost of products sold", value=90.0)
user_input['Attr_53'] = st.number_input("Attr_53: Equity / fixed assets", value=0.6)
user_input['Attr_54'] = st.number_input("Attr_54: Constant capital / fixed assets", value=0.4)
user_input['Attr_55'] = st.number_input("Attr_55: Working capital", value=10.0)
user_input['Attr_56'] = st.number_input("Attr_56: (Sales − cost of products sold) / sales", value=0.3)
user_input['Attr_57'] = st.number_input("Attr_57: (Current assets − inventory − short-term liabilities) / (Sales − gross profit − depreciation)", value=0.4)
user_input['Attr_58'] = st.number_input("Attr_58: Total costs / total sales", value=0.7)
user_input['Attr_59'] = st.number_input("Attr_59: Long-term liabilities / equity", value=0.8)
user_input['Attr_60'] = st.number_input("Attr_60: Sales / inventory", value=5.0)
user_input['Attr_61'] = st.number_input("Attr_61: Sales / receivables", value=6.0)
user_input['Attr_62'] = st.number_input("Attr_62: (Short-term liabilities × 365) / sales", value=90.0)
user_input['Attr_63'] = st.number_input("Attr_63: Sales / short-term liabilities", value=2.0)
user_input['Attr_64'] = st.number_input("Attr_64: Sales / fixed assets", value=4.0)
user_input['forecast_window'] = st.slider("Forecast Window (years ahead)", min_value=1, max_value=5, value=3)


user_df = pd.DataFrame([user_input])

import joblib

# Load your trained XGBoost model (Pipeline)
model = joblib.load("xgb_tuned_bst.joblib")  # path should be adjusted as per your app location

# Make prediction
if st.button("Predict Bankruptcy"):
    prediction = model.predict(user_df)[0]
    probability = model.predict_proba(user_df)[0][1]

    if prediction == 1:
        st.error(f"🚨 High Risk of Bankruptcy (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Bankruptcy (Probability: {probability:.2f})")


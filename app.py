import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.special import expit

# --- Page Configuration ---
st.set_page_config(page_title="Bankruptcy Prediction App", layout="wide")
st.title("🏦 Bankruptcy Predictor")
st.markdown("This app uses a machine learning model to predict bankruptcy risk. The most influential factors are listed at the top.")

# --- Model Loading ---
# Cache the model to prevent reloading on each interaction.
@st.cache_resource
def load_model():
    try:
        # Ensure 'xgb_tuned_bst.joblib' is in the same directory as your script.
        model = joblib.load("xgb_tuned_bst.joblib")
        return model
    except FileNotFoundError:
        st.error("Model file 'xgb_tuned_bst.joblib' not found. Please ensure it is in the correct directory.")
        return None

model = load_model()

# --- User Inputs ---
# This dictionary will hold all inputs from the user.
user_input = {}


# --- Forecast Window ---
st.markdown("###  Forecast Settings")
forecast_window = st.slider(
    "Forecast Window (years ahead)", 
    min_value=1, 
    max_value=5, 
    value=1, 
    help="Select the prediction horizon. This is a key factor for the model."
)
user_input.update({f"fw_{i}": 1 if forecast_window == i else 0 for i in range(1, 6)})


st.markdown("### Financial Ratio Inputs")
st.info("Adjust the sliders below. The most predictive factors are at the top. Expand other categories as needed.")

# --- Feature Definitions with New Labels and Median Defaults ---
# Based on the feature importance, metadata, and descriptive statistics provided.
# The default value for each slider is now the median (50th percentile) from your dataset.

feature_metadata = {
    "Attr1": "Net profit / total assets",
    "Attr2": "Total liabilities / total assets",
    "Attr3": "Working capital / total assets",
    "Attr4": "Current assets / short-term liabilities",
    "Attr5": "(Cash + ST securities + receivables − ST liabilities) / (Operating expenses − depreciation) × 365",
    "Attr6": "Retained earnings / total assets",
    "Attr7": "EBIT / total assets",
    "Attr8": "Book value of equity / total liabilities",
    "Attr9": "Sales / total assets",
    "Attr10": "Equity / total assets",
    "Attr11": "(Gross profit + extraordinary items + financial expenses) / total assets",
    "Attr12": "Gross profit / short-term liabilities",
    "Attr13": "(Gross profit + depreciation) / sales",
    "Attr14": "(Gross profit + interest) / total assets",
    "Attr15": "(Total liabilities × 365) / (Gross profit + depreciation)",
    "Attr16": "(Gross profit + depreciation) / total liabilities",
    "Attr17": "Total assets / total liabilities",
    "Attr18": "Gross profit / total assets",
    "Attr19": "Gross profit / sales",
    "Attr20": "(Inventory × 365) / sales",
    "Attr21": "Sales (n) / Sales (n−1)",
    "Attr22": "Profit on operating activities / total assets",
    "Attr23": "Net profit / sales",
    "Attr24": "Gross profit (in 3 years) / total assets",
    "Attr25": "(Equity − share capital) / total assets",
    "Attr26": "(Net profit + depreciation) / total liabilities",
    "Attr27": "Profit on operating activities / financial expenses",
    "Attr28": "Working capital / fixed assets",
    "Attr29": "Logarithm of total assets",
    "Attr30": "(Total liabilities − cash) / sales",
    "Attr31": "(Gross profit + interest) / sales",
    "Attr32": "(Current liabilities × 365) / cost of products sold",
    "Attr33": "Operating expenses / short-term liabilities",
    "Attr34": "Operating expenses / total liabilities",
    "Attr35": "Profit on sales / total assets",
    "Attr36": "Total sales / total assets",
    "Attr37": "(Current assets − inventories) / long-term liabilities",
    "Attr38": "Constant capital / total assets",
    "Attr39": "Profit on sales / sales",
    "Attr40": "(Current assets − inventory − receivables) / short-term liabilities",
    "Attr41": "Total liabilities / ((Op. profit + depreciation) × (12 / 365))",
    "Attr42": "Profit on operating activities / sales",
    "Attr43": "Rotation receivables + inventory turnover in days",
    "Attr44": "(Receivables × 365) / sales",
    "Attr45": "Net profit / inventory",
    "Attr46": "(Current assets − inventory) / short-term liabilities",
    "Attr47": "(Inventory × 365) / cost of products sold",
    "Attr48": "EBITDA / total assets",
    "Attr49": "EBITDA / sales",
    "Attr50": "Current assets / total liabilities",
    "Attr51": "Short-term liabilities / total assets",
    "Attr52": "(ST liabilities × 365) / cost of products sold",
    "Attr53": "Equity / fixed assets",
    "Attr54": "Constant capital / fixed assets",
    "Attr55": "Working capital",
    "Attr56": "(Sales − cost of products sold) / sales",
    "Attr57": "(CA − inventory − ST liabilities) / (Sales − GP − depreciation)",
    "Attr58": "Total costs / total sales",
    "Attr59": "Long-term liabilities / equity",
    "Attr60": "Sales / inventory",
    "Attr61": "Sales / receivables",
    "Attr62": "(ST liabilities × 365) / sales",
    "Attr63": "Sales / short-term liabilities",
    "Attr64": "Sales / fixed assets"
}


with st.sidebar.expander("📘 SHAP Feature Guide", expanded=True):
    st.markdown("""
    SHAP (SHapley Additive exPlanations) visualizes how features influence the bankruptcy risk prediction.
    
    - 🔴 Red bars increase bankruptcy risk  
    - 🔵 Blue bars reduce it  
    - The SHAP value reflects each feature’s impact on the model’s decision
    """)
    
    st.markdown("---")
    st.markdown("#### 💡 Feature Glossary")

    for key in sorted(feature_metadata.keys(), key=lambda x: int(x.replace("Attr", ""))):
        label = feature_metadata[key]
        st.markdown(f"- **{key}**: {label}")



feature_definitions = {
    # Top 12 Most Important Features
    "⭐ Top Predictive Factors": [
        ("Attr26", "(Net profit + depreciation) / total liabilities", -5.0, 5.0, 0.15510, 0.01),
        ("Attr24", "Gross profit (in 3 years) / total assets", -5.0, 5.0, 0.029916, 0.01),
        ("Attr27", "Profit on operating activities / financial expenses", -10.0, 10.0, 1.0841, 0.1),
        ("Attr34", "Operating expenses / total liabilities", 0.0, 5.0, 0.46539, 0.01),
        ("Attr16", "(Gross profit + depreciation) / total liabilities", -1.0, 5.0, 0.245845, 0.01),
        ("Attr38", "Constant capital / total assets", 0.0, 2.0, 0.61215, 0.01),
        ("Attr6", "Retained earnings / total assets", -1.0, 1.0, 0.0, 0.01),
        ("Attr10", "Equity / total assets", -1.0, 2.0, 0.50597, 0.01),
        ("Attr5", "Liquidity buffer indicator (Attr5)", -500.0, 500.0, -1.0345, 1.0),
        ("Attr35", "Profit on sales / total assets", -1.0, 1.0, 0.060655, 0.01),
        ("Attr13", "(Gross profit + depreciation) / sales", -1.0, 5.0, 0.075306, 0.01),
        ("Attr46", "(Current assets − inventory) / short-term liabilities", -1.0, 5.0, 1.02665, 0.1),
    ],
    "Profitability": [
        ("Attr7", "EBIT / total assets", -1.0, 1.0, 0.059653, 0.01),
        ("Attr39", "Profit on sales / sales", -1.0, 1.0, 0.036874, 0.01),
        ("Attr22", "Profit on operating activities / total assets", -1.0, 1.0, 0.062262, 0.01),
        ("Attr18", "Gross profit / total assets", -1.0, 1.0, 0.059653, 0.01),
        ("Attr19", "Gross profit / sales", 0.0, 1.0, 0.035874, 0.01),
        ("Attr11", "(Gross profit + extraordinary items + financial expenses) / total assets", -1.0, 1.0, 0.059653, 0.01),
    ],
    "Liquidity & Solvency": [
        ("Attr48", "EBITDA / total assets", -1.0, 1.0, 0.038015, 0.01),
        ("Attr44", "(Receivables × 365) / sales", 0.0, 500.0, 54.7675, 10.0),
        ("Attr33", "Operating expenses / short-term liabilities", 0.0, 5.0, 4.6255, 0.01),
        ("Attr12", "Gross profit / short-term liabilities", -1.0, 1.0, 0.17672, 0.01),
    ],
    "Efficiency & Activity": [
        ("Attr49", "EBITDA / sales", -1.0, 1.0, 0.01097, 0.01),
        ("Attr61", "Sales / receivables", 0.0, 150.0, 5.2943, 0.1),
        ("Attr45", "Net profit / inventory", -10.0, 10.0, 0.006366, 0.1),
        ("Attr47", "(Inventory × 365) / cost of products sold", 0.0, 500.0, 9.7917, 10.0),
        ("Attr9", "Sales / total assets", 0.0, 10.0, 1.19535, 0.1),
        ("Attr21", "Sales (n) / Sales (n−1)", 0.0, 5.0, 1.0452, 0.1),
        ("Attr30", "(Total liabilities − cash) / sales", -5.0, 5.0, 0.221705, 0.1),
        ("Attr62", "(Short-term liabilities × 365) / sales", 0.0, 500.0, 71.326, 10.0),
        ("Attr20", "(Inventory × 365) / sales", 0.0, 500.0, 35.1495, 10.0),
        ("Attr60", "Sales / inventory", 0.0, 500.0, 1088.35, 10.0),
    ],
    "Leverage & Other": [
        ("Attr63", "Sales / short-term liabilities", 0.0, 10.0, 5.0876, 0.1),
        ("Attr42", "Profit on operating activities / sales", -1.0, 1.0, 0.085515, 0.01),
    ]
}

# Dynamically create sliders from the definitions
for group_name, group_features in feature_definitions.items():
    # The top factors expander is open by default
    is_expanded = "Top Predictive Factors" in group_name
    with st.expander(group_name, expanded=is_expanded):
        # Use a 3-column layout for a clean look
        cols = st.columns(3)
        for i, (attr, label, min_val, max_val, default, step) in enumerate(group_features):
            user_input[attr] = cols[i % 3].slider(label, min_val, max_val, float(default), float(step), key=attr)


# --- Prediction Button and Logic ---
if st.button("Predict Bankruptcy Risk", type="primary", use_container_width=True):
    if model is not None:
        try:
            # --- Default values for features NOT in the UI (using median) ---
            # These are necessary for the model but less critical for user input.
            default_values = {
                'Attr1': 0.04966, 'Attr2': 0.4719, 'Attr3': 0.19661, 'Attr4': 1.5698,
                'Attr8': 1.0704, 'Attr14': 0.05965, 'Attr15': 846.26, 'Attr17': 2.1164,
                'Attr23': 0.01843, 'Attr25': 0.01097, 'Attr28': 0.46539, 'Attr29': 4.014,
                'Attr31': 0.03587, 'Attr32': 78.325, 'Attr36': 1.6434, 'Attr37': 3.0963,
                'Attr40': 0.09176, 'Attr41': 0.0994, 'Attr43': 99.4015, 'Attr50': 1.2222,
                'Attr51': 0.34101, 'Attr52': 95.096, 'Attr53': 1.3767, 'Attr54': 1.2053,
                'Attr55': 4993.3, 'Attr56': 0.12909, 'Attr57': 0.11967, 'Attr58': 0.95096,
                'Attr59': 0.23605, 'Attr64': 8.59885
            }
            # Ensure all required features are present in the final input dictionary
            for feat in model.feature_names_in_:
                if feat not in user_input:
                    user_input[feat] = default_values.get(feat, 0.0)

            # Create a DataFrame with columns in the exact order the model expects
            input_df = pd.DataFrame([user_input])
            input_df = input_df[model.feature_names_in_]

            # --- Prediction ---
            prob = model.predict_proba(input_df)[0][1]

            st.markdown("---")
            st.markdown("### 📈 Prediction Result")
            
            # Display the probability with a progress bar and clear risk assessment
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="Bankruptcy Probability", value=f"{prob:.2%}")
                if prob >= 0.5:
                    st.error("🚨 High Risk")
                elif prob >= 0.2:
                    st.warning("⚠️ Medium Risk")
                else:
                    st.success("✅ Low Risk")
            with col2:
                # FIX: Cast the probability to a standard Python float to avoid TypeError
                st.progress(float(prob))
                st.caption("This bar shows the model's predicted probability of bankruptcy within the selected forecast window.")
                


            # --- SHAP Waterfall Plot for Explainability ---
            st.markdown("###  Feature Contribution Analysis (SHAP Plot)")
            st.caption("Note: f(x) in the plot is in log-odds scale.")

            xgb_model = model.named_steps['xgb']
            explainer = shap.Explainer(xgb_model)
            shap_values = explainer(input_df)

            # Create the waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=14, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf() # Clear the figure from memory after displaying


        except Exception as e:
            st.exception(f"❌ An error occurred during prediction: {e}")


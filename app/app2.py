import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import shap
import matplotlib.pyplot as plt

# -----------------------------
# PATH SETUP
# -----------------------------
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from src.utils import load_object
from src.feature_engineering import create_features

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction System",
    layout="wide"
)

st.title("📉 Customer Churn Prediction System")
st.markdown("End-to-end ML system with prediction + SHAP explainability")

# -----------------------------
# LOAD MODEL
# -----------------------------
model_path = os.path.join(APP_ROOT, "artifacts", "best_model.pkl")
model = load_object(model_path)

# Extract pipeline parts
if hasattr(model, "named_steps"):
    preprocessor = model.named_steps["preprocessor"]
    ml_model = model.named_steps["model"]
elif hasattr(model, "best_estimator_"):
    preprocessor = model.best_estimator_.named_steps["preprocessor"]
    ml_model = model.best_estimator_.named_steps["model"]
else:
    raise AttributeError("Loaded model has no pipeline named_steps or best_estimator_.")

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "SHAP Explainability", "Business Insights"]
)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
if page == "Prediction":

    st.header("🔮 Predict Customer Churn")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 0, 150, 70)
    total_charges = st.number_input("Total Charges", value=1000.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    input_df = pd.DataFrame({
        "customerID": ["0000-000000"],
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": ["No"],
        "tenure": [tenure],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["DSL"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": [contract],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    input_df = create_features(input_df)

    if st.button("Predict Churn"):

        raw_pred = model.predict(input_df)[0]
        if hasattr(model, "label_encoder"):
            prediction = model.label_encoder.inverse_transform([raw_pred])[0]
        else:
            prediction = raw_pred

        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Result")

        if str(prediction).lower() in ["yes", "1", "true"]:
            st.error(f"⚠ Customer likely to churn ({probability:.2%})")
        else:
            st.success(f"✅ Customer likely to stay ({1 - probability:.2%})")

        st.progress(float(probability))

# -----------------------------
# SHAP EXPLAINABILITY PAGE
# -----------------------------
elif page == "SHAP Explainability":

    st.header("📊 SHAP Model Explainability")

    st.markdown("Understanding what drives churn predictions")

    # Load dataset sample for explanation
    sample_path = os.path.join(APP_ROOT, "data", "processed", "cleaned.csv")
    df_sample = pd.read_csv(sample_path).sample(200, random_state=42)

    df_sample = create_features(df_sample)
    X_sample = df_sample.drop("Churn", axis=1)

    # Transform data using pipeline preprocessor
    X_transformed = preprocessor.transform(X_sample)

    # SHAP explainer (tree-based models)
    explainer = shap.Explainer(ml_model, X_transformed)
    shap_values = explainer(X_transformed)

    st.subheader("📌 Global Feature Importance")

    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, X_transformed, show=False)
    st.pyplot(fig1)

    st.subheader("📍 Local Explanation (Single Prediction)")

    index = st.slider("Select Customer Index", 0, len(X_sample)-1, 0)

    single_sample = X_transformed[index:index+1]
    single_shap = explainer(single_sample)

    fig2, ax2 = plt.subplots()
    shap.waterfall_plot(single_shap[0], show=False)
    st.pyplot(fig2)

# -----------------------------
# BUSINESS INSIGHTS PAGE
# -----------------------------
elif page == "Business Insights":

    st.header("📊 Business Insights")

    st.markdown("""
### 🔥 Key Churn Drivers
- High Monthly Charges
- Month-to-Month Contracts
- Low Tenure Customers
- Electronic Payment Methods

### 💡 Business Actions
- Promote long-term contracts
- Offer discounts for high-risk users
- Improve onboarding for new customers
- Loyalty programs for early-stage users
""")
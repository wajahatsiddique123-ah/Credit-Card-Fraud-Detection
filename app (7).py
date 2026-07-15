import streamlit as st
import pandas as pd
import pickle

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------
# Load Model
# --------------------------
model = pickle.load(open("fraud_model.pkl", "rb"))

# --------------------------
# Title
# --------------------------
st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a credit card transaction is **Fraudulent** or **Genuine** using Logistic Regression.")

st.sidebar.header("Project Information")
st.sidebar.info("""
Model : Logistic Regression

Dataset : Kaggle Credit Card Fraud Detection

Target:
0 → Genuine
1 → Fraud
""")

st.markdown("---")

# --------------------------
# Feature Names
# --------------------------
features = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26","V27",
    "V28","Amount"
]

values = []

col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    if i % 2 == 0:
        value = col1.number_input(feature, value=0.0, format="%.6f")
    else:
        value = col2.number_input(feature, value=0.0, format="%.6f")
    values.append(value)

st.markdown("---")

# --------------------------
# Prediction
# --------------------------
if st.button("Predict Transaction"):

    input_df = pd.DataFrame([values], columns=features)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")

    st.subheader("Prediction Probability")

    st.write(f"**Genuine Probability:** {probability[0]*100:.2f}%")
    st.write(f"**Fraud Probability:** {probability[1]*100:.2f}%")

    st.progress(float(probability[1]))

st.markdown("---")
st.caption("Developed using Python, Streamlit and Logistic Regression")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Paths
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/random_forest_model.pkl"
# Add selector or PCA paths if used

# Load model and scaler safely
def load_joblib(path, name):
    if not os.path.exists(path):
        st.error(f"{name} file not found: {path}")
        return None
    try:
        return joblib.load(path)
    except EOFError:
        st.error(f"{name} file is corrupted or incomplete!")
        return None

scaler = load_joblib(SCALER_PATH, "Scaler")
model = load_joblib(MODEL_PATH, "Model")

if not scaler or not model:
    st.stop()  # Stop app if essential files are missing

st.title("Heart Disease Risk Prediction")
st.markdown("Enter patient information below to get a **real-time heart disease risk score**.")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
bp = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)

# Create input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cholesterol': [cholesterol],
    'bp': [bp]
})

# Preprocessing
input_df = pd.get_dummies(input_df, drop_first=True)

# Ensure scaler matches columns from training
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Error scaling input: {e}")
    st.stop()

# Prediction
if st.button("Predict Heart Disease Risk"):
    try:
        risk_prob = model.predict_proba(input_scaled)[0][1]
        st.success(f"Predicted Heart Disease Risk: {risk_prob*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

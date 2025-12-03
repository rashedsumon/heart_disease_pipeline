import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_loader import load_data

# Load models and transformers
scaler = joblib.load("models/RandomForest.pkl")  # Example, replace with actual if needed
model = joblib.load("models/RandomForest.pkl")
selector = joblib.load("models/RandomForest.pkl")  # Placeholder
pca = joblib.load("models/RandomForest.pkl")  # Placeholder

st.title("Heart Disease Risk Prediction")

st.markdown("""
Enter patient information below to get a **real-time heart disease risk score**.
""")

# Example inputs (expand based on dataset features)
age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
bp = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)

# Collect input into DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cholesterol': [cholesterol],
    'bp': [bp]
})

# Preprocessing (dummy example, extend for full features)
input_df = pd.get_dummies(input_df, drop_first=True)
input_scaled = scaler.transform(input_df)  # Example
# input_selected = selector.transform(input_scaled)  # Uncomment if selector is used
# input_pca = pca.transform(input_selected)        # Uncomment if PCA is used

# Prediction
if st.button("Predict Heart Disease Risk"):
    risk_prob = model.predict_proba(input_scaled)[0][1]  # Probability of heart disease
    st.success(f"Predicted Heart Disease Risk: {risk_prob*100:.2f}%")

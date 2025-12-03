import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from data_loader import load_data
from utils import preprocess_data

# -------------------------------
# Streamlit page configuration
# -------------------------------
st.set_page_config(page_title="Heart Disease Risk Prediction", layout="wide")
st.title("💓 Heart Disease Risk Prediction")

# -------------------------------
# Load dataset
# -------------------------------
try:
    df = load_data()
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
except FileNotFoundError as e:
    st.error(f"Dataset not found: {e}")
    st.stop()

# -------------------------------
# Sidebar: User input
# -------------------------------
st.sidebar.header("Enter Patient Details")

def user_input_features():
    features = {
        'age': st.sidebar.number_input("Age", 1, 120, 50),
        'sex': st.sidebar.selectbox("Sex (1=Male, 0=Female)", [0, 1]),
        'cp': st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1),
        'trestbps': st.sidebar.number_input("Resting BP", 80, 200, 120),
        'chol': st.sidebar.number_input("Cholesterol", 100, 600, 200),
        'fbs': st.sidebar.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1]),
        'restecg': st.sidebar.slider("Resting ECG (0-2)", 0, 2, 1),
        'thalach': st.sidebar.number_input("Max Heart Rate", 50, 250, 150),
        'exang': st.sidebar.selectbox("Exercise Induced Angina", [0, 1]),
        'oldpeak': st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0),
        'slope': st.sidebar.slider("Slope of ST Segment (0-2)", 0, 2, 1),
        'ca': st.sidebar.slider("Major Vessels (0-3)", 0, 3, 0),
        'thal': st.sidebar.slider("Thalassemia (0-3)", 0, 3, 2)
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# -------------------------------
# Sidebar: Model selection
# -------------------------------
st.sidebar.header("Select Model")
model_options = {
    "Random Forest": "models/model_rf.pkl",
    "Logistic Regression": "models/model_lr.pkl",
    "SVC": "models/model_svc.pkl"
}
selected_model_name = st.sidebar.selectbox("Choose a model", list(model_options.keys()))
model_path = model_options[selected_model_name]
scaler_path = "models/scaler.pkl"

# -------------------------------
# Load model and scaler safely
# -------------------------------
if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    st.warning(f"Model or scaler file not found. Please train your models first.")
    st.stop()
else:
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except EOFError:
        st.error("Model or scaler file is corrupted. Please re-train and re-upload the files.")
        st.stop()

    # Preprocess input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # -------------------------------
    # Display results
    # -------------------------------
    st.subheader(f"Prediction using {selected_model_name}")
    st.write("Heart Disease Risk:", "Yes" if prediction[0] == 1 else "No")

    st.subheader("Prediction Probability")
    st.dataframe(pd.DataFrame(prediction_proba, columns=model.classes_))

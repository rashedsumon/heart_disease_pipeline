import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_loader import load_data
from utils import preprocess_data

st.set_page_config(page_title="Heart Disease Risk Prediction", layout="wide")

st.title("💓 Heart Disease Risk Prediction")

# Load dataset
df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# User Input for new patient
st.sidebar.header("Enter Patient Details")
def user_input_features():
    age = st.sidebar.number_input("Age", 1, 120, 50)
    sex = st.sidebar.selectbox("Sex (1=Male,0=Female)", [0, 1])
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.sidebar.number_input("Resting BP", 80, 200, 120)
    chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])
    restecg = st.sidebar.slider("Resting ECG (0-2)", 0, 2, 1)
    thalach = st.sidebar.number_input("Max Heart Rate", 50, 250, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.sidebar.slider("Slope of ST Segment (0-2)", 0, 2, 1)
    ca = st.sidebar.slider("Major Vessels (0-3)", 0, 3, 0)
    thal = st.sidebar.slider("Thalassemia (0-3)", 0, 3, 2)

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Load trained model
model_path = "models/model_rf.pkl"
scaler_path = "models/scaler.pkl"

if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    st.warning("Models not found. Please train models first.")
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader("Prediction")
    st.write("Heart Disease Risk:", "Yes" if prediction[0]==1 else "No")
    st.subheader("Prediction Probability")
    st.write(prediction_proba)

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from data_loader import load_data
from utils import preprocess_data

# -------------------------------
# Load and preprocess data
# -------------------------------
print("Loading dataset...")
df = load_data()
X, y = preprocess_data(df)

# -------------------------------
# Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Scale features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# -------------------------------
# Train Random Forest
# -------------------------------
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
joblib.dump(rf_model, "models/model_rf.pkl")
print("Random Forest saved as 'models/model_rf.pkl'.")

# -------------------------------
# Train Logistic Regression
# -------------------------------
print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
joblib.dump(lr_model, "models/model_lr.pkl")
print("Logistic Regression saved as 'models/model_lr.pkl'.")

# -------------------------------
# Train SVC
# -------------------------------
print("Training Support Vector Classifier (SVC)...")
svc_model = SVC(probability=True, random_state=42)
svc_model.fit(X_train_scaled, y_train)
joblib.dump(svc_model, "models/model_svc.pkl")
print("SVC saved as 'models/model_svc.pkl'.")

# -------------------------------
# Save scaler
# -------------------------------
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved as 'models/scaler.pkl'.")

print("\nAll models and scaler are ready for use in the Streamlit app!")

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("data/heart_disease.csv")  # Replace with your CSV

# Example features and target
X = df[['age', 'cholesterol', 'bp']]       # Use all relevant features
y = df['target']                           # Replace 'target' with your label column

# One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save scaler and model
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(model, "models/random_forest_model.pkl")

print("Scaler and model saved successfully!")

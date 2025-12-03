import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/heart_disease.csv")  # replace with your CSV

# Features and target
X = df[['age', 'cholesterol', 'bp']]       # adjust to match your dataset
y = df['target']                           # adjust column name

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Save scaler and model
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(model, "models/random_forest_model.pkl")

print("Scaler and model saved successfully!")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_col='target', apply_pca=True, n_components=5):
    """Preprocess dataset"""
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values
    df = df.fillna(df.median())

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA dimensionality reduction
    if apply_pca:
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
        return X_scaled, y, scaler, pca

    return X_scaled, y, scaler, None

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

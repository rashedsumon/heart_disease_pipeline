import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
import joblib

def preprocess_and_train(df, target_col='target'):
    """
    Preprocesses data, trains multiple ML models, optionally performs PCA and K-Means,
    performs hyperparameter tuning, and exports trained models.
    """

    # 1. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Handle categorical features
    X = pd.get_dummies(X, drop_first=True)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Dimensionality reduction (PCA)
    pca = PCA(n_components=min(10, X_train_scaled.shape[1]))
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # 6. Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(10, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)

    # 7. Define models with hyperparameter grids
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            'C': [0.1, 1, 10]
        }),
        "RandomForest": (RandomForestClassifier(), {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10]
        }),
        "SVM": (SVC(probability=True), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        })
    }

    best_models = {}
    for name, (model, params) in models.items():
        print(f"Training {name}...")
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid.fit(X_train_sel, y_train)
        y_pred = grid.predict(X_test_sel)
        print(f"{name} Best Params: {grid.best_params_}")
        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
        best_models[name] = grid.best_estimator_
        # Save model
        joblib.dump(grid.best_estimator_, f"models/{name}.pkl")

    # Optional: Unsupervised patient segmentation
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_train_scaled)
    df['Cluster'] = kmeans.predict(X)
    joblib.dump(kmeans, "models/kmeans.pkl")
    print("K-Means clustering saved.")

    return best_models, scaler, selector, pca

if __name__ == "__main__":
    from data_loader import load_data
    import os
    os.makedirs('models', exist_ok=True)
    df = load_data()
    models, scaler, selector, pca = preprocess_and_train(df)

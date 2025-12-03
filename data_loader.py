

## **4. data_loader.py**


import pandas as pd
import kagglehub
import os

DATA_PATH = "data/heart_disease.csv"

def download_dataset():
    """Download dataset using kagglehub"""
    path = kagglehub.dataset_download("mahatiratusher/heart-disease-risk-prediction-dataset")
    print("Dataset downloaded to:", path)
    return path

def load_data():
    """Load CSV dataset"""
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        download_dataset()
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded. Shape:", df.shape)
    return df

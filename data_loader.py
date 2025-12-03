import pandas as pd
import kagglehub
import os

DATA_PATH = "data/heart_disease.csv"

def download_dataset():
    """Download dataset using kagglehub to DATA_PATH"""
    if not os.path.exists("data"):
        os.makedirs("data")  # ensure folder exists
    path = kagglehub.dataset_download(
        "mahatiratusher/heart-disease-risk-prediction-dataset",
        download_path="data",  # save into data folder
        unzip=True  # if it's a zip, extract it
    )
    print("Dataset downloaded to:", path)
    return path

def load_data():
    """Load CSV dataset"""
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        download_dataset()
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH} after download!")
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded. Shape:", df.shape)
    return df

import pandas as pd
import kagglehub
import os
import shutil

DATA_PATH = "data/heart_disease.csv"

def download_dataset():
    """Download dataset using kagglehub and move to DATA_PATH"""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    tmp_path = kagglehub.dataset_download("mahatiratusher/heart-disease-risk-prediction-dataset")
    print("Dataset downloaded to temporary path:", tmp_path)
    
    # If tmp_path is a directory, find the CSV inside
    if os.path.isdir(tmp_path):
        files = [f for f in os.listdir(tmp_path) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError("No CSV found in downloaded dataset!")
        csv_file = os.path.join(tmp_path, files[0])
    else:
        csv_file = tmp_path
    
    # Move CSV to DATA_PATH
    shutil.move(csv_file, DATA_PATH)
    print("Dataset moved to:", DATA_PATH)
    return DATA_PATH

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

import os
import pandas as pd
import kagglehub
import shutil

DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "heart_disease.csv")

def download_dataset():
    """
    Download the Heart Disease dataset from Kaggle using kagglehub
    and move the CSV to DATA_PATH.
    """
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download dataset (kagglehub returns temp path)
    tmp_path = kagglehub.dataset_download("mahatiratusher/heart-disease-risk-prediction-dataset")
    print(f"Dataset downloaded to temporary path: {tmp_path}")
    
    # If tmp_path is a directory, locate the CSV
    if os.path.isdir(tmp_path):
        csv_files = [f for f in os.listdir(tmp_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the downloaded dataset!")
        csv_file = os.path.join(tmp_path, csv_files[0])
    else:
        csv_file = tmp_path
    
    # Move CSV to DATA_PATH
    shutil.move(csv_file, DATA_PATH)
    print(f"Dataset moved to: {DATA_PATH}")
    return DATA_PATH

def load_data():
    """
    Load the Heart Disease CSV dataset.
    Downloads it automatically if missing.
    Returns a pandas DataFrame.
    """
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Downloading...")
        download_dataset()
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH} after download!")
    
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

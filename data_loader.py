


## **4️⃣ data_loader.py**


import kagglehub
import pandas as pd

def load_data():
    """
    Automatically downloads the heart disease dataset from KaggleHub
    and loads it as a pandas DataFrame.
    """
    path = kagglehub.dataset_download("mahatiratusher/heart-disease-risk-prediction-dataset")
    print("Dataset downloaded to:", path)

    # Assuming the CSV file is inside a folder named after the dataset
    csv_path = f"{path}/heart.csv"  # Adjust filename if different
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())

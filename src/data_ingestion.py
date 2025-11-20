from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def ingest_data(out_path="data/raw.csv"):
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved raw data to {out_path}")

if __name__ == "__main__":
    ingest_data()

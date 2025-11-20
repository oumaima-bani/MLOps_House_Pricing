import pandas as pd
import os

# Feature-engine transformers
from feature_engine.selection import DropCorrelatedFeatures


def process(in_path="data/raw.csv", out_path="data/processed.csv"):
    df = pd.read_csv(in_path)

    # -------------------------
    # 1. Drop rows with missing values (simple version)
    # -------------------------
    df = df.dropna().reset_index(drop=True)

    # Separate target / features
    X = df.drop(columns=["target"])
    y = df["target"]

    # -------------------------
    # 2. Drop highly correlated features
    # -------------------------
    drop_corr = DropCorrelatedFeatures(
        variables=None,
        method="pearson",
        threshold=0.9
    )

    X_reduced = drop_corr.fit_transform(X)

    # -------------------------
    # 3. Reassemble final DataFrame
    # -------------------------
    df_out = pd.concat([X_reduced, y], axis=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print(f"Saved processed data to {out_path}")


if __name__ == "__main__":
    process()

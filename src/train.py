import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import joblib

def train_and_save(processed_path="data/processed.csv", model_dir="api/models"):
    df = pd.read_csv(processed_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("house_price_experiment")
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model with MLflow
        mlflow.sklearn.log_model(model, name="sklearn_model")
        mlflow.log_param("n_estimators", 100)

        # Evaluate
        score = model.score(X_test, y_test)
        mlflow.log_metric("r2_score", float(score))
        print("R2_Score:", score)

        # Save sklearn model locally (optional)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "sk_model.joblib"))

        # Convert to ONNX
        initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(os.path.join(model_dir, "model.onnx"), "wb") as f:
            f.write(onnx_model.SerializeToString())

        # Add ONNX artifact to mlflow
        mlflow.log_artifact(os.path.join(model_dir, "model.onnx"))
    print("Training complete. Models saved.")

if __name__ == "__main__":
    train_and_save()

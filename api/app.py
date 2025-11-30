import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np
# No longer need to import hf_hub_download
# from huggingface_hub import hf_hub_download 

app = FastAPI(title="House Price Inference API")

# Create a Pydantic model for input features
class Features(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float

# The model is now expected to be in the same directory as app.py
model_path = "model.onnx" 

# Check if the model file exists
if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found at {model_path}")

# Load ONNX model once from the local file
session = rt.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
expected_num_features = session.get_inputs()[0].shape[1]

@app.post("/predict")
def predict(features: Features):
    # Convert Pydantic object to numpy array
    x_input = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                         features.AveBedrms, features.Population, features.AveOccup,
                         features.Latitude]], dtype=np.float32)

    # Check input length
    if x_input.shape[1] != expected_num_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_num_features} features, got {x_input.shape[1]}"
        )

    # Run prediction
    preds = session.run(None, {input_name: x_input})[0]

    # Convert prediction to actual dollars
    predicted_price = float(preds[0][0]) * 100_000

    return {"prediction_usd": predicted_price}

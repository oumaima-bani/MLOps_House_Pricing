from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np

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

# Load ONNX model once
session = rt.InferenceSession("models/model.onnx")
input_name = session.get_inputs()[0].name
expected_num_features = session.get_inputs()[0].shape[1]  # This will be 7

@app.post("/predict")
def predict(features: Features):
    # Convert Pydantic object to numpy array in the correct order
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

    # Convert prediction from 100,000s of dollars to actual dollars
    predicted_price = float(preds[0][0]) * 100_000

    return {"prediction_usd": predicted_price}





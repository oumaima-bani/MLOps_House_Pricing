import pandas as pd
import onnxruntime as rt
import numpy as np

def evaluate(model_path="models/model.onnx", data_path="data/processed.csv"):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"]).values.astype('float32')
    y = df["target"].values

    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    preds = sess.run(None, {input_name: X})[0].ravel()
    # simple score
    from sklearn.metrics import r2_score
    print("R2 Score:", r2_score(y, preds))

if __name__ == "__main__":
    evaluate()

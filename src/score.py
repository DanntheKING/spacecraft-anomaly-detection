import os
import json
import joblib
import numpy as np


def init():
    global model
    global scaler

    model_dir = os.environ["AZUREML_MODEL_DIR"]

    model = joblib.load(os.path.join(model_dir, "isolation_forest.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))


def run(raw_data):
    data = json.loads(raw_data)

    X = np.array(data["data"])
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    preds = (preds == -1).astype(int)

    return {
        "predictions": preds.tolist(),
        "meaning": {
            "0": "normal",
            "1": "anomaly"
        }
    }

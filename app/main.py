from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model_v1.pkl")

model = joblib.load(model_path)

@app.get("/")
def home():
    return {"message": "Student Performance Model API is running"}

@app.post("/predict")
def predict(features: list):
    features_array = np.array([features])
    prediction = model.predict(features_array)
    return {"prediction": prediction.tolist()}

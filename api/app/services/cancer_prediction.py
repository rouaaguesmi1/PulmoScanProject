# app/api/lcp_knn_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, conint
import joblib
import numpy as np
import os

router = APIRouter()

# Load the model at startup
model_path = "app\models\LungCancerPrediction\lcp_model.joblib"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
    
knn_model = joblib.load(model_path)

# Class mapping
class_names = ['Low', 'Medium', 'High']

# Define the request schema
class PatientFeatures(BaseModel):
    age: conint(ge=0, le=120)
    gender: conint(ge=0, le=1)
    air_pollution: int
    alcohol_use: int
    dust_allergy: int
    occupational_hazards: int
    genetic_risk: int
    chronic_lung_disease: int
    balanced_diet: int
    obesity: int
    smoking: int
    passive_smoker: int
    chest_pain: int
    coughing_of_blood: int
    fatigue: int
    weight_loss: int
    shortness_of_breath: int
    wheezing: int
    swallowing_difficulty: int
    clubbing_of_finger_nails: int
    frequent_cold: int
    dry_cough: int
    snoring: int

class PredictionResponse(BaseModel):
    predicted_class: str


@router.post("/predict_cancer_level", response_model=PredictionResponse)
def predict_cancer_level(features: PatientFeatures):
    try:
        # Convert to numpy array
        input_array = np.array([list(features.dict().values())])
        pred = knn_model.predict(input_array)
        result = class_names[pred[0]]
        return PredictionResponse(predicted_class=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

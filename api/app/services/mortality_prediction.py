from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import os # Recommended for robust path handling
from pathlib import Path # Recommended for robust path handling

router = APIRouter()

# Load model
# It's good practice to use os.path.join or pathlib for path construction
# Assuming this script is in a subdirectory like 'app/routers' and 'models' is in 'app/models'
# If the current working directory is the project root containing 'app':
MODEL_FILE_PATH = os.path.join("app", "models", "Lung CancerMortalityPrediction", "survival_model.joblib")
# Or, for more robustness relative to this file's location (e.g. if script is in app/routers/your_router.py):
# SCRIPT_DIR = Path(__file__).resolve().parent
# MODEL_FILE_PATH = SCRIPT_DIR.parent / "models" / "Lung CancerMortalityPrediction" / "survival_model.joblib"
# For this fix, I'll assume the original path works in your environment, but keep this in mind.
try:
    model = joblib.load(MODEL_FILE_PATH) # Changed to use forward slashes for better cross-platform compatibility
except FileNotFoundError:
    alt_model_path = os.path.join(os.getcwd(), "app", "models", "Lung CancerMortalityPrediction", "survival_model.joblib")
    try:
        model = joblib.load(alt_model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model. Ensure the path is correct and model file exists. Error: {e}")

import pandas as pd
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException # Assuming FastAPI setup
from typing import Optional, List, Any, Dict

# Assume 'model' is your loaded scikit-learn model
# e.g., import joblib
# model = joblib.load('your_model.pkl')


# This should be EXACTLY as your model was trained, including order
MODEL_FEATURE_ORDER = [
    "age",
    "gender",
    "diagnosis_date",
    "cancer_stage",
    "beginning_of_treatment_date",
    "family_history",
    "bmi",
    "cholesterol_level",
    "hypertension",
    "asthma",
    "cirrhosis",
    "other_cancer",
    "end_treatment_date",
    "treatment_duration", # This was calculated
    "smoke_current_smoker", # Corrected case
    "smoke_former_smoker",  # Corrected case
    "smoke_never_smoked",   # Corrected case
    "smoke_passive_smoker", # Corrected case
    "treatment_chemotherapy", # Corrected case
    "treatment_combined",     # Corrected case
    "treatment_radiation",    # Corrected case
    "treatment_surgery"       # Corrected case
]

class MortalityInput(BaseModel):
    age: float
    gender: int
    diagnosis_date: str = Field(example="2023-01-01")
    cancer_stage: int
    beginning_of_treatment_date: str = Field(example="2023-01-15")
    family_history: int
    bmi: float
    cholesterol_level: int
    hypertension: int
    asthma: int
    cirrhosis: int
    other_cancer: int
    end_treatment_date: str = Field(example="2023-06-01")
    # These need to match the keys in the input JSON if you don't want to rename them manually later
    # Or, you can use Field(alias="...") if you want different Python attribute names
    smoke_Current_Smoker: int # Will be renamed
    smoke_Former_Smoker: int  # Will be renamed
    smoke_Never_Smoked: int   # Will be renamed
    smoke_Passive_Smoker: int # Will be renamed
    treatment_Chemotherapy: int # Will be renamed
    treatment_Combined: int     # Will be renamed
    treatment_Radiation: int    # Will be renamed
    treatment_Surgery: int      # Will be renamed

# Your utility function (seems fine, but we'll use direct conversion for simplicity later)
def convert_to_timestamp(date_str: str, field_name: str) -> Optional[int]:
    try:
        return int(pd.to_datetime(date_str).timestamp())
    except Exception:
        # It's better to raise HTTPException here if this is directly in an endpoint path operation
        # For now, keeping ValueError as per original
        raise ValueError(f"Invalid date format for '{field_name}': '{date_str}'. Expected format like 'YYYY-MM-DD'.")

@router.post("/predict")
def predict_survival(data: MortalityInput):
    input_dict = data.dict()
    processed_dict: Dict[str, Any] = {}

    print("\n📥 Raw input:", input_dict)

    # --- 1. Rename keys to match training data ---
    key_mapping = {
        "smoke_Current_Smoker": "smoke_current_smoker",
        "smoke_Former_Smoker": "smoke_former_smoker",
        "smoke_Never_Smoked": "smoke_never_smoked",
        "smoke_Passive_Smoker": "smoke_passive_smoker",
        "treatment_Chemotherapy": "treatment_chemotherapy",
        "treatment_Combined": "treatment_combined",
        "treatment_Radiation": "treatment_radiation",
        "treatment_Surgery": "treatment_surgery"
    }
    for old_key, new_key in key_mapping.items():
        if old_key in input_dict:
            processed_dict[new_key] = input_dict.pop(old_key)

    # Copy remaining keys
    for key, value in input_dict.items():
        if key not in processed_dict: # Avoid overwriting already processed keys (though pop prevents this)
             processed_dict[key] = value

    # --- 2. Handle Date Conversions and Feature Engineering ---
    date_columns_to_convert_to_timestamp = [
        "diagnosis_date",
        "beginning_of_treatment_date",
        "end_treatment_date"
    ]

    try:
        # Convert date strings to datetime objects first for calculations
        dt_diagnosis_date = pd.to_datetime(processed_dict["diagnosis_date"])
        dt_beginning_of_treatment_date = pd.to_datetime(processed_dict["beginning_of_treatment_date"])
        dt_end_treatment_date = pd.to_datetime(processed_dict["end_treatment_date"])

        # Calculate treatment duration in days
        treatment_duration_days = (dt_end_treatment_date - dt_beginning_of_treatment_date).days
        processed_dict["treatment_duration"] = treatment_duration_days

        # Now convert date columns to Unix timestamps (int64 as per training)
        processed_dict["diagnosis_date"] = int(dt_diagnosis_date.timestamp())
        processed_dict["beginning_of_treatment_date"] = int(dt_beginning_of_treatment_date.timestamp())
        processed_dict["end_treatment_date"] = int(dt_end_treatment_date.timestamp())

    except Exception as e:
        # More specific error handling for date parsing is good
        # The convert_to_timestamp function already does this, but here we handle general errors
        raise HTTPException(status_code=400, detail=f"Error processing date fields or calculating duration: {str(e)}")


    # --- 3. Ensure Feature Order and Prepare for Model ---
    features_for_model: List[Any] = []
    missing_features = []
    for feature_name in MODEL_FEATURE_ORDER:
        if feature_name in processed_dict:
            features_for_model.append(processed_dict[feature_name])
        else:
            # This should not happen if MODEL_FEATURE_ORDER and processing are correct
            missing_features.append(feature_name)

    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: The following required features were not prepared: {', '.join(missing_features)}"
        )

    print("\nProcessed features for model (in order):", features_for_model)
    print("Feature names for model (in order):", MODEL_FEATURE_ORDER)


    # --- 4. Prediction ---
    # The model expects a 2D array-like structure (e.g., a list of lists, or a 2D NumPy array)
    # Since we are predicting for a single instance, it should be [[val1, val2, ...]]
    try:
        # Create a DataFrame with the correct column order and dtypes (optional but good practice)
        # This helps ensure dtypes are also somewhat aligned if your model is sensitive.
        # Pandas will infer dtypes; for critical cases, specify them.
        input_df = pd.DataFrame([features_for_model], columns=MODEL_FEATURE_ORDER)

        # Ensure dtypes match the training data as closely as possible if needed
        # For example, if 'age' was float32 during training, you might do:
        # input_df['age'] = input_df['age'].astype('float32')
        # However, for most sklearn models, float64 input for float32 trained feature is fine.
        # int64 input for int32 trained feature is also generally fine.
        # The crucial part is numerical vs. categorical and the values themselves.

        # Example: Ensure specific Dtypes from your training data info
        # This is a more robust way if dtypes are critical
        # For simplicity, we'll assume pandas infers reasonably well for now.
        # dtypes_from_training = {
        #     "age": "float64", "gender": "int64", "diagnosis_date": "int64",
        #     "cancer_stage": "int64", "beginning_of_treatment_date": "int64",
        #     "family_history": "int64", "bmi": "float64", "cholesterol_level": "int64",
        #     "hypertension": "int64", "asthma": "int64", "cirrhosis": "int64",
        #     "other_cancer": "int64", "end_treatment_date": "int64",
        #     "treatment_duration": "int64", "smoke_current_smoker": "int32",
        #     "smoke_former_smoker": "int32", "smoke_never_smoked": "int32",
        #     "smoke_passive_smoker": "int32", "treatment_chemotherapy": "int32",
        #     "treatment_combined": "int32", "treatment_radiation": "int32",
        #     "treatment_surgery": "int32"
        # }
        # for col, dtype in dtypes_from_training.items():
        #    input_df[col] = input_df[col].astype(dtype)


        print("\nDataFrame to model:", input_df.info())
        print(input_df.head())

        # PREDICTION STEP (replace with your actual model prediction)
        # prediction = model.predict(input_df)
        # prediction_proba = model.predict_proba(input_df) # If applicable

        # Dummy prediction for now
        prediction = [0] # e.g., 0 for non-survivor, 1 for survivor
        prediction_proba = [[0.7, 0.3]] # e.g. [prob_class_0, prob_class_1]
        survived_status = "Will Survive" if prediction[0] == 1 else "Won't Survive"
    except Exception as e:
        # Catch issues during DataFrame creation or model prediction
        raise HTTPException(status_code=500, detail=f"Error during model prediction preparation: {str(e)}")


    return {
        "survived": survived_status,
        "prediction": prediction[0],
        "prediction_probability_class_0": prediction_proba[0][0],
        "prediction_probability_class_1": prediction_proba[0][1],
        "processed_input_features_ordered": dict(zip(MODEL_FEATURE_ORDER, features_for_model)) # for debugging
    }


# survived: Indicates whether the patient survived (e.g., yes, no). patient survived (e.g., yes, no).
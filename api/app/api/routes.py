from fastapi import APIRouter

# Import routers from each model-specific service
from app.services import (
    cancer_classification as cancer_subtype,
    # Uncomment these as you implement them
    nodule_detection,
    nodule_classification ,
    # fibrosis_progression,
    cancer_prediction,
    mortality_prediction,
    image_enhancement,
    patient_cancer_classification,
)

# Create a root router
router = APIRouter()
 
# Include each service's router with its own prefix
router.include_router(cancer_subtype.router, prefix="/cancer-subtype", tags=["Cancer Subtype Classification"])
router.include_router(patient_cancer_classification.router, prefix="/patient-cancer-classification", tags=["Patient Cancer Classification"])
router.include_router(nodule_detection.router, prefix="/nodule-detection", tags=["Nodule Candidate Detection"])
router.include_router(nodule_classification.router, prefix="/nodule-classification", tags=["Nodule Classification"])
# router.include_router(fibrosis_progression.router, prefix="/fibrosis-progression", tags=["Pulmonary Fibrosis Progression"])
router.include_router(cancer_prediction.router, prefix="/cancer-prediction", tags=["Lung Cancer Prediction"])
router.include_router(mortality_prediction.router, prefix="/mortality-prediction", tags=["Mortality Prediction"])
router.include_router(image_enhancement.router, prefix="/image-enhancement", tags=["Medical Image Enhancement"])


#uvicorn app.main:app --reload
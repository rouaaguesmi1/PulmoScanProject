from fastapi import APIRouter, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io
from app.models.CancerClassification.model_arch import CustomSotaCNN
import numpy as np
from pydantic import BaseModel

router = APIRouter()

class_names = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']


model = CustomSotaCNN(num_classes=4)
try:
    # Make sure the model weights are for a model that expects 3 input channels.
    model.load_state_dict(torch.load("app/models/CancerClassification/best_custom_sota_model.pth", map_location="cpu"))
except RuntimeError as e:
    print(f"Error loading state_dict. This might indicate a mismatch between your model architecture and the saved weights: {e}")
    print("Ensure the CustomSotaCNN definition in app.models.custom_model.py expects 3 input channels for the stem.")
    raise e
model.eval()

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

# ImageNet stats (ensure these match what your model was trained with)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Model input dimensions (used by transform)
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Kept your torchvision transform as it's standard for PyTorch
image_transform = transforms.Compose([ # Renamed for clarity
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def run_inference(uploaded_file: UploadFile): # Explicitly type hint UploadFile
    try:
        image_bytes = uploaded_file.file.read()
        # Attempt to open the image
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        # Handle cases where the file is not a valid image
        return PredictionResponse(predicted_class=f"Error: Could not open image file. {e}", confidence=0.0)

    # Crucial: Convert to RGB. This handles grayscale, RGBA, etc.
    # For grayscale, it duplicates the channel. For RGBA, it discards alpha.
    pil_image_rgb = pil_image.convert('RGB')

    # For debugging, you can check the mode:
    # print(f"PIL Image mode after convert('RGB'): {pil_image_rgb.mode}") # Should be 'RGB'

    input_tensor = image_transform(pil_image_rgb).unsqueeze(0) # Add batch dimension

    # For debugging, check tensor shape:
    # print(f"Input tensor shape: {input_tensor.shape}") # Should be [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    return PredictionResponse(
        predicted_class=class_names[predicted_idx.item()],
        confidence=round(confidence.item(), 4)
    )


@router.post("/", response_model=PredictionResponse) # Add response_model for Swagger/OpenAPI docs
async def predict_image(file: UploadFile = File(...)):
    return run_inference(file)

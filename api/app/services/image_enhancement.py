from fastapi import APIRouter, UploadFile, File
from PIL import Image, ImageEnhance
import io
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

class ContrastAdjustmentResponse(BaseModel):
    message: str

# The default contrast factor
DEFAULT_CONTRAST_FACTOR = 2.0  # 1.0 is the original image, higher values increase contrast

def adjust_contrast(uploaded_file: UploadFile, contrast_factor: float = DEFAULT_CONTRAST_FACTOR):
    try:
        # Read the image file
        image_bytes = uploaded_file.file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"message": f"Error: Could not open image file. {e}"}

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image_enhanced = enhancer.enhance(contrast_factor)

    # Save the enhanced image to a BytesIO object to send as response
    img_io = io.BytesIO()
    pil_image_enhanced.save(img_io, format="PNG")
    img_io.seek(0)  # Move to the beginning of the BytesIO object

    return img_io

@router.post("/adjust_contrast/", response_model=ContrastAdjustmentResponse)
async def adjust_contrast_endpoint(file: UploadFile = File(...), contrast_factor: float = DEFAULT_CONTRAST_FACTOR):
    """
    Endpoint to adjust the contrast of the uploaded image.
    - `contrast_factor` (float) is used to increase or decrease contrast. Default is 2.0 (higher = more contrast).
    """
    enhanced_image = adjust_contrast(file, contrast_factor)
    
    if isinstance(enhanced_image, dict):  # Error case
        return enhanced_image
    
    return StreamingResponse(enhanced_image, media_type="image/png")

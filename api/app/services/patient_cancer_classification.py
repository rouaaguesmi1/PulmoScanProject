import os
import shutil
import tempfile
import zipfile
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import SimpleITK as sitk
import numpy as np
import scipy.ndimage
from skimage.measure import label as skimage_label, regionprops
from skimage.morphology import disk, binary_closing
from skimage.segmentation import clear_border
import scipy.ndimage as ndi

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
router = APIRouter()
# Assuming models_arch and schemas are in app.models.PatientCancerClassification
try:
    from ..models.PatientCancerClassification.patient_classifier_arch import Simp3DNet, MODEL_INPUT_SHAPE_CFG
    from ..models.PatientCancerClassification.patient_schemas import PatientCancerPredictionResponse
except ImportError:
    raise ImportError("Could not import Simp3DNet or Schemas. Check path in patient_classification_service.py")

# --- Configuration & Parameters ---
# MODEL_FILENAME should match the name of your trained .pth file for Simp3DNet (64x64x64)
MODEL_FILENAME = "simp3dnet_model_50_each_64cube_best.pth" # Example filename
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "ml_models"
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "app" / "models" / "PatientCancerClassification" / MODEL_FILENAME

UPLOAD_DIR_BASE = Path(__file__).resolve().parent.parent.parent / "uploads_patient_scans"
UPLOAD_DIR_BASE.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Patient Classification Service: Global DEVICE set to: {DEVICE}")

# Preprocessing Parameters from DSB17 Training Script
TARGET_SPACING_DSB = np.array([1.5, 1.5, 1.5]) # z, y, x
FINAL_SCAN_SIZE_DSB = MODEL_INPUT_SHAPE_CFG # (D, H, W) e.g., (64, 64, 64)
CLIP_BOUND_HU_DSB = np.array([-1000.0, 400.0])
PIXEL_MEAN_DSB = 0.25
LUNG_THRESHOLD_HU_DSB = -320 # Used in get_segmented_lungs

# Model & Inference Parameters
CLASSIFICATION_THRESHOLD = 0.5 # Example, adjust based on validation

# Global variable for the model
patient_classification_model: Optional[Simp3DNet] = None

warnings.filterwarnings("ignore", category=UserWarning, module='SimpleITK')
warnings.filterwarnings("ignore", category=UserWarning, module='torch')


# --- Preprocessing Functions (Adapted from DSB17 Training Script) ---

def load_scan_series(dicom_folder_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Loads a DICOM series, returns image_array (z,y,x), origin (z,y,x), spacing (z,y,x), and series_uid."""
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_folder_path)
        if not series_ids:
            print(f"No DICOM series found in {dicom_folder_path}")
            return None, None, None, None
        
        # Use the first series found
        series_uid = series_ids[0]
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder_path, series_uid)
        
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        # Optional: Configure metadata reading if needed, e.g. series_reader.MetaDataDictionaryArrayUpdateOn()
        
        itkimage = series_reader.Execute()
        
        image_array_zyx = sitk.GetArrayFromImage(itkimage) # z, y, x
        origin_zyx = np.array(list(reversed(itkimage.GetOrigin())))
        spacing_zyx = np.array(list(reversed(itkimage.GetSpacing())))
        
        return image_array_zyx, origin_zyx, spacing_zyx, series_uid
    except Exception as e:
        print(f"Error reading DICOM series from {os.path.basename(dicom_folder_path)}: {e}")
        return None, None, None, None

def resample_volume_scipy_dsb(image_np_zyx: np.ndarray, original_spacing_zyx: np.ndarray,
                              target_spacing_zyx: np.ndarray = TARGET_SPACING_DSB) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Resamples a 3D image to a new spacing using scipy.ndimage.zoom (order=1)."""
    resize_factor = original_spacing_zyx / target_spacing_zyx
    new_real_shape = image_np_zyx.shape * resize_factor
    new_shape = np.round(new_real_shape).astype(int)
    new_shape[new_shape == 0] = 1 # Prevent zero dimensions

    real_resize_factor = new_shape / np.array(image_np_zyx.shape)
    # actual_new_spacing_zyx = original_spacing_zyx / real_resize_factor # Not strictly needed for inference path

    try:
        # order=1 (bilinear interpolation) as used in many pipelines
        resampled_image_np = scipy.ndimage.zoom(image_np_zyx, real_resize_factor, mode='nearest', order=1)
        return resampled_image_np, target_spacing_zyx # Return target as new spacing
    except Exception as e:
        print(f"Error during scipy resampling (DSB): {e}")
        return None, None

def get_segmented_lungs_slice_dsb(im_slice: np.ndarray, hu_threshold: int = LUNG_THRESHOLD_HU_DSB) -> np.ndarray:
    """Segments lungs from a single 2D slice (DSB version). Returns a masked slice."""
    if im_slice.ndim != 2:
        print("  get_segmented_lungs_slice_dsb: input not 2D, returning original.")
        return im_slice # Should not happen if called correctly

    binary = im_slice < hu_threshold
    cleared = clear_border(binary)
    label_image = skimage_label(cleared)
    
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    
    # Determine area threshold: if 2+ regions, use area of 2nd largest. If 1, use largest. Else 0.
    area_threshold = 0
    if len(areas) >= 2:
        area_threshold = areas[-2]
    elif len(areas) == 1:
        area_threshold = areas[-1]

    final_binary_mask = np.zeros_like(label_image, dtype=bool)
    if area_threshold > 0:
        for region in regionprops(label_image):
            if region.area >= area_threshold: # Keep regions with area >= threshold
                for coordinates in region.coords:
                    final_binary_mask[coordinates[0], coordinates[1]] = True
    
    # Morphological operations from DSB script
    selem_closing = disk(2) # disk(2) in notebook
    closed_mask = binary_closing(final_binary_mask, selem_closing)
    
    selem_dilate = disk(5) # disk(5) in notebook
    dilated_mask = ndi.binary_dilation(closed_mask, structure=selem_dilate)

    # Apply mask: set non-lung regions to a value outside typical HU range for air/tissue
    # The training script used CLIP_BOUND_HU[0] - 1
    background_val = CLIP_BOUND_HU_DSB[0] - 1 
    segmented_slice = np.copy(im_slice)
    segmented_slice[~dilated_mask] = background_val # Apply mask by setting non-lung to background
    
    return segmented_slice

def apply_lung_segmentation_to_volume_dsb(resampled_image_np: np.ndarray) -> np.ndarray:
    """Applies lung segmentation slice by slice to the 3D volume."""
    segmented_lungs_np = np.copy(resampled_image_np) # Output array
    print("  Segmenting lungs (slice by slice, DSB method)...")
    for i in range(resampled_image_np.shape[0]): # Iterate over Z slices
        segmented_lungs_np[i] = get_segmented_lungs_slice_dsb(resampled_image_np[i])
    return segmented_lungs_np

def normalize_hu_dsb(image_np: np.ndarray, clip_bounds: np.ndarray = CLIP_BOUND_HU_DSB) -> np.ndarray:
    """Normalizes Hounsfield Units (HU) to the range [0, 1] after clipping."""
    min_b, max_b = clip_bounds
    image_np = np.clip(image_np, min_b, max_b)
    image_np = (image_np - min_b) / (max_b - min_b)
    return image_np.astype(np.float32)

def zero_center_dsb(image_np: np.ndarray, pixel_mean: float = PIXEL_MEAN_DSB) -> np.ndarray:
    """Zero-centers the normalized image using PIXEL_MEAN."""
    return (image_np - pixel_mean).astype(np.float32)

def resize_scan_to_target_dsb(image_np_zyx: np.ndarray, target_shape_zyx: Tuple[int, int, int] = FINAL_SCAN_SIZE_DSB) -> Optional[np.ndarray]:
    """Resizes a 3D image to a target shape (D, H, W) using zoom and padding/cropping."""
    if image_np_zyx.shape == target_shape_zyx:
        return image_np_zyx.astype(np.float32)

    resize_factor = np.array(target_shape_zyx) / np.array(image_np_zyx.shape)
    
    try:
        # order=1 (bilinear) for resizing, mode='nearest' for padding edges if needed
        resized_image = scipy.ndimage.zoom(image_np_zyx, resize_factor, order=1, mode='nearest')
        
        # Correct shape if zoom result is slightly off due to rounding
        if resized_image.shape != target_shape_zyx:
            current_shape_arr = np.array(resized_image.shape)
            target_shape_arr = np.array(target_shape_zyx)
            diff = target_shape_arr - current_shape_arr
            
            pad_amounts = np.maximum(diff, 0)
            crop_amounts = np.maximum(-diff, 0)
            
            pad_width = []
            for p_val in pad_amounts:
                pad_width.append((p_val // 2, p_val - (p_val // 2))) # (before, after)
            
            if np.any(pad_amounts > 0):
                 # Pad with edge value, consistent with some segmentation approaches
                resized_image = np.pad(resized_image, tuple(pad_width), mode='edge')

            crop_slices = []
            for c_val, current_dim_size in zip(crop_amounts, resized_image.shape):
                start = c_val // 2
                end = current_dim_size - (c_val - (c_val // 2))
                crop_slices.append(slice(start, end))

            if np.any(crop_amounts > 0):
                resized_image = resized_image[tuple(crop_slices)]
        
        if resized_image.shape != target_shape_zyx:
            print(f"ERROR: Resize failed. Target: {target_shape_zyx}, Got: {resized_image.shape}. Attempting simple crop/pad to fix.")
            # Force crop or pad to exact size (less ideal but a fallback)
            final_image = np.full(target_shape_zyx, resized_image.min(), dtype=resized_image.dtype) # Pad with min
            slices_original = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(resized_image.shape, target_shape_zyx))
            slices_final = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(resized_image.shape, target_shape_zyx))
            final_image[slices_final] = resized_image[slices_original]
            resized_image = final_image

        if resized_image.shape != target_shape_zyx:
            print(f"CRITICAL ERROR: Resize to target failed. Final shape {resized_image.shape} != {target_shape_zyx}")
            return None
            
        return resized_image.astype(np.float32)
    except Exception as e:
        print(f"Error resizing image of shape {image_np_zyx.shape} to target {target_shape_zyx}: {e}")
        return None

def preprocess_dicom_series_for_patient_classification(
    dicom_folder_path: str
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads and preprocesses a DICOM series for patient-level cancer classification.
    Returns: (processed_image_np_zyx, metadata, series_uid_or_folder_name)
    """
    patient_id_or_folder = Path(dicom_folder_path).name
    print(f"Processing scan series from: {dicom_folder_path}")

    image_np_zyx, origin_zyx, spacing_zyx, series_uid = load_scan_series(dicom_folder_path)
    if image_np_zyx is None:
        return None, None, patient_id_or_folder
    
    effective_id = series_uid if series_uid else patient_id_or_folder
    print(f"  Scan ID: {effective_id}, Original shape: {image_np_zyx.shape}, spacing: {np.round(spacing_zyx,3)}")

    resampled_image_np, new_spacing_zyx = resample_volume_scipy_dsb(
        image_np_zyx, spacing_zyx, TARGET_SPACING_DSB
    )
    if resampled_image_np is None:
        print(f"  Failed to resample {effective_id}. Skipping.")
        return None, None, effective_id
    print(f"  Resampled shape: {resampled_image_np.shape}, new spacing: {np.round(new_spacing_zyx,3)}")

    segmented_lungs_volume = apply_lung_segmentation_to_volume_dsb(resampled_image_np)
    normalized_image = normalize_hu_dsb(segmented_lungs_volume, clip_bounds=CLIP_BOUND_HU_DSB)
    centered_image = zero_center_dsb(normalized_image, pixel_mean=PIXEL_MEAN_DSB)
    
    final_image_np = resize_scan_to_target_dsb(centered_image, target_shape_zyx=FINAL_SCAN_SIZE_DSB)
    if final_image_np is None:
        print(f"  Failed to resize to target for {effective_id}. Skipping.")
        return None, None, effective_id
        
    print(f"  Preprocessing complete for {effective_id}. Final shape: {final_image_np.shape}")

    metadata = {
        'original_dicom_folder': dicom_folder_path,
        'series_uid': series_uid,
        'original_shape_zyx': image_np_zyx.shape,
        'original_spacing_zyx': spacing_zyx,
        'resampled_spacing_zyx': new_spacing_zyx,
        'final_processed_shape_zyx': final_image_np.shape
    }
    return final_image_np, metadata, effective_id

# --- Model Loading ---
def load_patient_classification_model():
    global patient_classification_model
    if not MODEL_PATH.exists():
        print(f"Error: Patient classification model file not found at {MODEL_PATH}")
        patient_classification_model = None
        return

    try:
        model_instance = Simp3DNet(input_channels=1, num_classes=1) # Matches Simp3DNet definition
        print(f"Attempting to load patient classification model state dict from: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        model_instance.load_state_dict(state_dict)
        model_instance.to(DEVICE)
        model_instance.eval()
        patient_classification_model = model_instance
        print(f"Patient Classification Model loaded successfully from {MODEL_PATH} to {DEVICE}.")
    except Exception as e:
        patient_classification_model = None
        print(f"Error loading Patient Classification model: {e}")
        import traceback
        traceback.print_exc()

# --- FastAPI Router Definition ---
router_patient_clf = APIRouter() # Use a different name if integrating into a larger app
load_patient_classification_model() # Load model at startup

def find_dicom_series_folder(extracted_path: Path) -> Optional[Path]:
    """Tries to find the main DICOM series folder within an extracted archive."""
    # 1. Check if DICOM files are directly in extracted_path
    if any(f.name.lower().endswith(".dcm") for f in extracted_path.iterdir() if f.is_file()):
        if sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(extracted_path)):
            return extracted_path

    # 2. Check subdirectories
    for item in extracted_path.iterdir():
        if item.is_dir():
            # Heuristic: check if this subdir contains DICOMs or has a series ID
            if any(f.name.lower().endswith(".dcm") for f in item.iterdir() if f.is_file()):
                 if sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(item)):
                    return item
            # Deeper check (1 level)
            for sub_item in item.iterdir():
                if sub_item.is_dir():
                    if any(f.name.lower().endswith(".dcm") for f in sub_item.iterdir() if f.is_file()):
                        if sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(sub_item)):
                            return sub_item
    return None


@router.post("/predict_patient_cancer_status", response_model=PatientCancerPredictionResponse)
async def predict_patient_cancer_status_endpoint(
    scan_zip: UploadFile = File(..., description="ZIP file containing DICOM series for one patient."),
    patient_id_form: Optional[str] = Form(None, description="Optional patient identifier.")
):
    global patient_classification_model
    if patient_classification_model is None:
        raise HTTPException(status_code=503, detail="Patient classification model not loaded. Server not ready.")

    if not scan_zip.filename or not scan_zip.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .zip archive.")

    # Use provided patient_id or derive from filename
    patient_id_from_file = scan_zip.filename[:-4] if scan_zip.filename else "unknown_scan"
    effective_patient_id = patient_id_form if patient_id_form else patient_id_from_file

    with tempfile.TemporaryDirectory(prefix="patient_clf_upload_", dir=UPLOAD_DIR_BASE) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        zip_save_path = temp_dir / scan_zip.filename
        extracted_scan_path = temp_dir / "extracted"
        extracted_scan_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save the zip file
            with open(zip_save_path, "wb") as buffer:
                shutil.copyfileobj(scan_zip.file, buffer)
            scan_zip.file.close()
            print(f"ZIP file saved to temp: {zip_save_path}")

            # Extract the zip file
            with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_scan_path)
            print(f"ZIP file extracted to: {extracted_scan_path}")
            
            # Try to find the actual DICOM series folder
            dicom_series_folder = find_dicom_series_folder(extracted_scan_path)
            if not dicom_series_folder:
                raise HTTPException(status_code=422, 
                                    detail=f"Could not automatically find a valid DICOM series folder in the uploaded ZIP for {effective_patient_id}.")
            print(f"Identified DICOM series folder: {dicom_series_folder}")

            # 1. Preprocess the scan
            processed_scan_np_zyx, scan_metadata, series_id_used = \
                preprocess_dicom_series_for_patient_classification(str(dicom_series_folder))

            if processed_scan_np_zyx is None or scan_metadata is None:
                error_msg = f"Failed to preprocess scan {series_id_used or effective_patient_id}."
                print(error_msg)
                return PatientCancerPredictionResponse(
                    patient_id=effective_patient_id,
                    message="Prediction failed during preprocessing.",
                    error_details=error_msg
                )
            
            # 2. Prepare tensor for model
            # Expected: (B, C, D, H, W) -> (1, 1, *FINAL_SCAN_SIZE_DSB)
            # processed_scan_np_zyx is (D, H, W)
            scan_tensor = torch.from_numpy(processed_scan_np_zyx).float().unsqueeze(0).unsqueeze(0)
            scan_tensor = scan_tensor.to(DEVICE)

            # 3. Get prediction
            patient_classification_model.eval() # Ensure model is in eval mode
            with torch.no_grad():
                output_logit = patient_classification_model(scan_tensor)
                probability = torch.sigmoid(output_logit).item()
                predicted_class = 1 if probability >= CLASSIFICATION_THRESHOLD else 0
            
            print(f"Prediction for {series_id_used or effective_patient_id}: Prob={probability:.4f}, Class={predicted_class}")
            
            return PatientCancerPredictionResponse(
                patient_id=series_id_used or effective_patient_id,
                message="Patient cancer status prediction successful.",
                probability_cancer=probability,
                predicted_class=predicted_class,
                model_threshold_used=CLASSIFICATION_THRESHOLD,
                processed_scan_shape_zyx=list(scan_metadata['final_processed_shape_zyx'])
            )

        except HTTPException: # Re-raise HTTPExceptions
            raise
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid or corrupted ZIP file.")
        except RuntimeError as e: # Catch SimpleITK or other runtime errors
            error_msg = f"Runtime error processing scan {effective_patient_id}: {str(e)}"
            print(error_msg)
            import traceback; traceback.print_exc()
            return PatientCancerPredictionResponse(
                patient_id=effective_patient_id,
                message="Prediction failed due to runtime error.",
                error_details=error_msg
            )
        except Exception as e:
            error_msg = f"An unexpected error occurred for {effective_patient_id}: {str(e)}"
            print(error_msg)
            import traceback; traceback.print_exc()
            # Use HTTPException for general server errors if not caught by specific handlers
            raise HTTPException(status_code=500, detail=error_msg)
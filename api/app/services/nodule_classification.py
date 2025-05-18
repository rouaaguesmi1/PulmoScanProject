# -*- coding: utf-8 -*-
# Fix this shit later I"m tired Right Now
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import os
import shutil
import tempfile
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import SimpleITK as sitk
import numpy as np
import scipy.ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_closing
from skimage.segmentation import clear_border
import scipy.ndimage as ndi

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from ..models.NoduleClassification.candidate_schemas import (
    CandidateCoordinateItem,
    CandidatePredictionItem,
    # CandidateClassificationRequest, # We'll get candidates as JSON string in Form
    CandidateClassificationResponse
)
# Assuming model_arch.py is in app/models_arch/
try:
    from ..models.NoduleClassification.candidate_classifier_arch import Simple3DCNN, PATCH_SIZE_CFG
except ImportError:
    raise ImportError("Could not import Simple3DCNN. Check path in candidate_classification_service.py")


# --- Configuration & Parameters ---
# MODEL_FILENAME should match the name of your trained .pth file
MODEL_FILENAME = "candidate_detection_model_best.pth" # From notebook
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "ml_models" # Assuming ml_models is at project root
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "app" / "models" / "NoduleClassification" / MODEL_FILENAME

UPLOAD_DIR_BASE = Path(__file__).resolve().parent.parent.parent / "uploads_candidates"
UPLOAD_DIR_BASE.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Candidate Classification Service: Global DEVICE set to: {DEVICE}")

# Preprocessing Parameters from Notebook
TARGET_SPACING_LUNA = np.array([1.0, 1.0, 1.0]) # z, y, x
LUNG_THRESHOLD_HU = -320 # For get_segmented_lungs
NORM_MIN_BOUND = -1000.0
NORM_MAX_BOUND = 400.0
NORM_PIXEL_MEAN = 0.25 # For zero_centering

# Model & Inference Parameters
PATCH_SIZE = PATCH_SIZE_CFG # (z, y, x) from model architecture file
# This threshold would be determined from your validation set (e.g., Youden's J)
CLASSIFICATION_THRESHOLD = 0.5 # Example, adjust based on notebook's "optimal_threshold"

# Global variable for the model
classification_model: Optional[Simple3DCNN] = None

warnings.filterwarnings("ignore", category=UserWarning, module='SimpleITK')
warnings.filterwarnings("ignore", category=UserWarning, module='torch')


# --- Preprocessing Functions (Adapted from LUNA16 Notebook) ---

def load_itk_image(filename: Path) -> Tuple[Optional[sitk.Image], Optional[str]]:
    """Loads a SimpleITK image, returns image and series UID."""
    try:
        itkimage = sitk.ReadImage(str(filename))
        seriesuid = os.path.basename(filename).replace('.mhd', '')
        return itkimage, seriesuid
    except Exception as e:
        print(f"Error reading ITK image {filename}: {e}")
        return None, None

def get_image_properties_luna(itkimage: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Returns origin (z,y,x) and spacing (z,y,x) from ITK image."""
    origin_zyx = np.array(list(reversed(itkimage.GetOrigin())))
    spacing_zyx = np.array(list(reversed(itkimage.GetSpacing())))
    return origin_zyx, spacing_zyx

def world_to_voxel_luna(world_coords_xyz: np.ndarray, origin_zyx: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray:
    """Converts world coordinates (x,y,z) to voxel coordinates (z,y,x)."""
    # world_coords_xyz is (X, Y, Z)
    # origin_zyx is (Z_orig, Y_orig, X_orig)
    # spacing_zyx is (Z_spac, Y_spac, X_spac)
    # We need to align: (X - X_orig)/X_spac, etc.
    origin_xyz = origin_zyx[::-1] # Convert origin to (X,Y,Z) order
    spacing_xyz = spacing_zyx[::-1] # Convert spacing to (X,Y,Z) order

    stretched_voxel_coords = np.absolute(world_coords_xyz - origin_xyz)
    voxel_coords_xyz = stretched_voxel_coords / spacing_xyz
    # Return as (z,y,x)
    return np.round(voxel_coords_xyz).astype(int)[::-1]


def resample_volume_scipy(image_np_zyx: np.ndarray, original_spacing_zyx: np.ndarray,
                          target_spacing_zyx: np.ndarray = TARGET_SPACING_LUNA) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Resamples a 3D image to a new spacing using scipy.ndimage.zoom."""
    resize_factor = original_spacing_zyx / target_spacing_zyx
    new_real_shape = image_np_zyx.shape * resize_factor
    new_shape = np.round(new_real_shape).astype(int) # Ensure integer shape
    
    # Handle potential zero dimensions if resize_factor is very small or shape is 1
    new_shape[new_shape == 0] = 1

    real_resize_factor = new_shape / np.array(image_np_zyx.shape)
    actual_new_spacing_zyx = original_spacing_zyx / real_resize_factor

    try:
        # order=0 (nearest neighbor) as used in notebook, order=1 (bilinear) might be better for intensities
        resampled_image_np = scipy.ndimage.zoom(image_np_zyx, real_resize_factor, mode='nearest', order=0)
        return resampled_image_np, actual_new_spacing_zyx
    except Exception as e:
        print(f"Error during scipy resampling: {e}")
        return None, None

def get_segmented_lungs_slice(im_slice: np.ndarray, threshold_hu: int = LUNG_THRESHOLD_HU) -> np.ndarray:
    """Segments lungs from a single 2D slice. Returns a binary mask."""
    binary = im_slice < threshold_hu
    cleared = clear_border(binary)
    label_image = label(cleared)
    
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    threshold_area = 0
    if len(areas) > 1: threshold_area = areas[-2]
    elif len(areas) == 1: threshold_area = areas[-1]

    final_binary_mask = np.zeros_like(label_image, dtype=bool)
    if threshold_area > 0: # Proceed only if regions were found
        for region in regionprops(label_image):
            if region.area >= threshold_area: # Keep regions with area >= threshold_area
                for coordinates in region.coords:
                    final_binary_mask[coordinates[0], coordinates[1]] = True
    
    # Morphological closing and dilation as in notebook
    selem_closing = disk(2)
    closed_mask = binary_closing(final_binary_mask, selem_closing)
    selem_dilate = disk(5)
    dilated_mask = ndi.binary_dilation(closed_mask, structure=selem_dilate)
    return dilated_mask # Return the mask

def apply_lung_segmentation_to_volume(resampled_image_np: np.ndarray) -> np.ndarray:
    """Applies lung segmentation slice by slice and masks the image."""
    segmented_lungs_np = np.copy(resampled_image_np)
    print("  Segmenting lungs (slice by slice)...")
    for i in range(resampled_image_np.shape[0]): # Iterate over Z slices
        lung_mask_slice = get_segmented_lungs_slice(resampled_image_np[i])
        segmented_lungs_np[i][~lung_mask_slice] = NORM_MIN_BOUND -100 # Set non-lung to very low HU (notebook used -2000)
    return segmented_lungs_np

def normalize_hu_luna(image_np: np.ndarray) -> np.ndarray:
    """Normalizes Hounsfield Units (HU) to the range [0, 1]."""
    image_np = (image_np - NORM_MIN_BOUND) / (NORM_MAX_BOUND - NORM_MIN_BOUND)
    image_np[image_np > 1] = 1.
    image_np[image_np < 0] = 0.
    return image_np

def zero_center_luna(image_np: np.ndarray) -> np.ndarray:
    """Zero-centers the normalized image using PIXEL_MEAN."""
    return image_np - NORM_PIXEL_MEAN


def preprocess_scan_for_classification(
    mhd_file_path: Path
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads, preprocesses (resample, segment, normalize, zero-center) a CT scan.
    Returns: (processed_image_np_zyx, metadata, seriesuid)
    Metadata includes: original_origin_zyx, resampled_spacing_zyx, resampled_shape_zyx
    """
    print(f"Processing scan: {mhd_file_path.name}")
    itkimage, seriesuid = load_itk_image(mhd_file_path)
    if not itkimage or not seriesuid:
        return None, None, None

    image_np_zyx = sitk.GetArrayFromImage(itkimage) # z, y, x
    original_origin_zyx, original_spacing_zyx = get_image_properties_luna(itkimage)
    print(f"  Original shape: {image_np_zyx.shape}, spacing: {original_spacing_zyx}")

    resampled_image_np, resampled_spacing_zyx = resample_volume_scipy(
        image_np_zyx, original_spacing_zyx, TARGET_SPACING_LUNA
    )
    if resampled_image_np is None:
        print(f"  Failed to resample {seriesuid}. Skipping.")
        return None, None, seriesuid
    print(f"  Resampled shape: {resampled_image_np.shape}, new spacing: {resampled_spacing_zyx}")

    segmented_lungs_volume = apply_lung_segmentation_to_volume(resampled_image_np)
    normalized_image = normalize_hu_luna(segmented_lungs_volume)
    final_image_np = zero_center_luna(normalized_image).astype(np.float32)
    print(f"  Preprocessing complete for {seriesuid}. Final shape: {final_image_np.shape}")

    metadata = {
        'original_mhd_path': str(mhd_file_path),
        'original_origin_zyx': original_origin_zyx,
        'original_spacing_zyx': original_spacing_zyx,
        'resampled_spacing_zyx': resampled_spacing_zyx,
        'resampled_shape_zyx': final_image_np.shape,
    }
    return final_image_np, metadata, seriesuid

# --- Patch Extraction (Adapted from Notebook) ---
def extract_patch_from_volume(image_np_zyx: np.ndarray, center_voxel_zyx: np.ndarray,
                              patch_size_zyx: Tuple[int, int, int] = PATCH_SIZE) -> Optional[np.ndarray]:
    patch_z, patch_y, patch_x = patch_size_zyx
    center_z, center_y, center_x = center_voxel_zyx

    start_z = center_z - patch_z // 2
    end_z = start_z + patch_z
    start_y = center_y - patch_y // 2
    end_y = start_y + patch_y
    start_x = center_x - patch_x // 2
    end_x = start_x + patch_x

    pad_z_before = max(0, -start_z)
    pad_z_after = max(0, end_z - image_np_zyx.shape[0])
    pad_y_before = max(0, -start_y)
    pad_y_after = max(0, end_y - image_np_zyx.shape[1])
    pad_x_before = max(0, -start_x)
    pad_x_after = max(0, end_x - image_np_zyx.shape[2])

    slice_start_z, slice_end_z = max(0, start_z), min(image_np_zyx.shape[0], end_z)
    slice_start_y, slice_end_y = max(0, start_y), min(image_np_zyx.shape[1], end_y)
    slice_start_x, slice_end_x = max(0, start_x), min(image_np_zyx.shape[2], end_x)

    patch = image_np_zyx[slice_start_z:slice_end_z, slice_start_y:slice_end_y, slice_start_x:slice_end_x]

    if any([pad_z_before, pad_z_after, pad_y_before, pad_y_after, pad_x_before, pad_x_after]):
        pad_value = image_np_zyx.min() # As per notebook's padding strategy
        pad_width = (
            (pad_z_before, pad_z_after),
            (pad_y_before, pad_y_after),
            (pad_x_before, pad_x_after)
        )
        try:
            patch = np.pad(patch, pad_width, mode='constant', constant_values=pad_value)
        except Exception as e:
            print(f"Error during padding: {e}. Patch shape: {patch.shape}, Pad width: {pad_width}")
            return None
    
    if patch.shape != patch_size_zyx:
        print(f"Warning: Extracted patch shape {patch.shape} != target {patch_size_zyx}. Center: {center_voxel_zyx}")
        return None # Critical to return None if shape is wrong for the CNN
    return patch


# --- Model Loading ---
def load_classification_model():
    global classification_model
    if not MODEL_PATH.exists():
        print(f"Error: Classification model file not found at {MODEL_PATH}")
        classification_model = None
        return

    try:
        model_instance = Simple3DCNN(input_channels=1, num_classes=1, patch_size_config=PATCH_SIZE)
        print(f"Attempting to load model state dict from: {MODEL_PATH}")
        # Load state dict to CPU first
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        model_instance.load_state_dict(state_dict)
        model_instance.to(DEVICE)
        model_instance.eval()
        classification_model = model_instance
        print(f"Candidate Classification Model loaded successfully from {MODEL_PATH} to {DEVICE}.")
    except Exception as e:
        classification_model = None
        print(f"Error loading Candidate Classification model: {e}")
        import traceback
        traceback.print_exc()

# --- FastAPI Router Definition ---
router = APIRouter()
load_classification_model() # Load model at startup


@router.post("/classify_candidates", response_model=CandidateClassificationResponse)
async def classify_candidates_endpoint(
    mhd_file: UploadFile = File(..., description="MHD header file for the CT scan."),
    raw_file: UploadFile = File(..., description="RAW data file for the CT scan (must match MHD)."),
    candidates_json_str: str = Form(..., description="JSON string of List[CandidateCoordinateItem]")
):
    global classification_model
    if classification_model is None:
        raise HTTPException(status_code=503, detail="Candidate classification model not loaded. Server not ready.")

    if not mhd_file.filename or not mhd_file.filename.endswith(".mhd"):
        raise HTTPException(status_code=400, detail="MHD file must have a .mhd extension.")

    try:
        candidates_data = json.loads(candidates_json_str)
        # Validate with Pydantic (optional but good practice)
        input_candidates = [CandidateCoordinateItem(**cand) for cand in candidates_data]
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for candidates_json_str.")
    except Exception as e: # Pydantic validation error
        raise HTTPException(status_code=400, detail=f"Invalid candidate data: {e}")

    base_filename = mhd_file.filename[:-4]
    expected_raw_filename = base_filename + ".raw" # Ensure raw file has same base name

    with tempfile.TemporaryDirectory(prefix="cand_cls_upload_", dir=UPLOAD_DIR_BASE) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        mhd_save_path = temp_dir / mhd_file.filename
        raw_save_path = temp_dir / expected_raw_filename # Use expected name

        try:
            with open(mhd_save_path, "wb") as buffer:
                shutil.copyfileobj(mhd_file.file, buffer)
            mhd_file.file.close()
            with open(raw_save_path, "wb") as buffer:
                shutil.copyfileobj(raw_file.file, buffer)
            raw_file.file.close()
            print(f"Files saved to temp: {mhd_save_path}, {raw_save_path}")

            # 1. Preprocess the entire scan
            processed_scan_np_zyx, scan_metadata, seriesuid = preprocess_scan_for_classification(mhd_save_path)

            if processed_scan_np_zyx is None or scan_metadata is None:
                raise HTTPException(status_code=422, detail=f"Failed to preprocess scan {seriesuid or mhd_file.filename}.")

            # 2. Classify each candidate
            predictions_list: List[CandidatePredictionItem] = []
            classification_model.eval() # Ensure model is in eval mode

            with torch.no_grad():
                for cand_item in input_candidates:
                    cand_pred_item_args = cand_item.model_dump() # Start with input data
                    try:
                        world_coords_xyz = np.array([cand_item.coordX, cand_item.coordY, cand_item.coordZ])
                        voxel_coords_zyx = world_to_voxel_luna(
                            world_coords_xyz,
                            scan_metadata['original_origin_zyx'], # Use original origin for conversion
                            scan_metadata['resampled_spacing_zyx'] # Use resampled spacing
                        )

                        # Sanity check voxel coordinates against resampled shape
                        res_shape = scan_metadata['resampled_shape_zyx']
                        if not (0 <= voxel_coords_zyx[0] < res_shape[0] and \
                                0 <= voxel_coords_zyx[1] < res_shape[1] and \
                                0 <= voxel_coords_zyx[2] < res_shape[2]):
                            print(f"Candidate {cand_item.id} voxel {voxel_coords_zyx} out of bounds {res_shape}")
                            cand_pred_item_args["probability_nodule"] = -1.0
                            cand_pred_item_args["predicted_class"] = -1
                            cand_pred_item_args["error_message"] = "Candidate voxel coords out of resampled scan bounds."
                            predictions_list.append(CandidatePredictionItem(**cand_pred_item_args))
                            continue

                        patch_np = extract_patch_from_volume(
                            processed_scan_np_zyx,
                            voxel_coords_zyx,
                            PATCH_SIZE
                        )

                        if patch_np is None:
                            print(f"Failed to extract patch for candidate {cand_item.id} at {voxel_coords_zyx}")
                            cand_pred_item_args["probability_nodule"] = -1.0
                            cand_pred_item_args["predicted_class"] = -1
                            cand_pred_item_args["error_message"] = "Patch extraction failed (e.g., boundary, shape mismatch)."
                            predictions_list.append(CandidatePredictionItem(**cand_pred_item_args))
                            continue

                        patch_tensor = torch.from_numpy(patch_np).float().unsqueeze(0).unsqueeze(0) # (B, C, D, H, W)
                        patch_tensor = patch_tensor.to(DEVICE)

                        output_logit = classification_model(patch_tensor)
                        probability = torch.sigmoid(output_logit).item()
                        predicted_class = 1 if probability >= CLASSIFICATION_THRESHOLD else 0
                        
                        cand_pred_item_args["probability_nodule"] = probability
                        cand_pred_item_args["predicted_class"] = predicted_class
                        predictions_list.append(CandidatePredictionItem(**cand_pred_item_args))

                    except Exception as e_cand:
                        print(f"Error processing candidate {cand_item.id}: {e_cand}")
                        cand_pred_item_args["probability_nodule"] = -1.0
                        cand_pred_item_args["predicted_class"] = -1
                        cand_pred_item_args["error_message"] = str(e_cand)
                        predictions_list.append(CandidatePredictionItem(**cand_pred_item_args))

        except HTTPException: # Re-raise HTTPExceptions
            raise
        except RuntimeError as e: # Catch SimpleITK or other runtime errors during file processing
            print(f"Runtime Error (likely SimpleITK): {e}")
            raise HTTPException(status_code=400, detail=f"Error processing image data: {str(e)}")
        except Exception as e:
            import traceback
            print(f"General Error: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

        return CandidateClassificationResponse(
            message="Candidate classification complete.",
            processed_scan_filename_base=seriesuid or base_filename,
            predictions=predictions_list,
            scan_resampled_shape_zyx=list(scan_metadata['resampled_shape_zyx']) if scan_metadata else None,
            model_threshold_used=CLASSIFICATION_THRESHOLD
        )
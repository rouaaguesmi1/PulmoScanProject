# app/services/nodule_detection_service.py
import os
import shutil
import tempfile
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
# from torch.cuda.amp import autocast # Deprecated
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, binary_dilation, generate_binary_structure, label as ndimage_label
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

# Assuming model_arch.py is in app/models/NoduleDetection/model_arch.py relative to project root
try:
    # Adjust the import path for model_arch.py to match your project structure
    # If nodule_detection_service.py is in app/services/
    # and model_arch.py is in app/models/NoduleDetection/
    # then the relative path from app/services/ to app/models/ is ../models/
    from ..models.NoduleDetection.model_arch import UNet3D_MSA
except ImportError as e:
    print(f"ImportError for model_arch: {e}. Falling back to direct import if possible (might fail).")
    # Fallback for different structures or direct execution (less likely in FastAPI context)
    # This fallback might need adjustment based on your exact execution context if the primary import fails.
    try:
        from model_arch import UNet3D_MSA # If model_arch is in the same directory or Python path
    except ImportError:
        raise ImportError("Could not import UNet3D_MSA. Check the import path in nodule_detection_service.py")


# --- Configuration & Parameters ---
# MODEL_FILENAME is not used if MODEL_PATH is absolute
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "NoduleDetection" / "best_model_msa_unetinf.pth"

UPLOAD_DIR_BASE = Path(__file__).resolve().parent.parent.parent / "uploads"
UPLOAD_DIR_BASE.mkdir(parents=True, exist_ok=True)

# This DEVICE is determined at startup.
# The device_to_use in run_inference_on_volume will take this value.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Nodule Detection Service (Startup): Global DEVICE set to: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"Nodule Detection Service (Startup): torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"Nodule Detection Service (Startup): torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            print(f"Nodule Detection Service (Startup): torch.cuda.current_device(): {torch.cuda.current_device()}")
            print(f"Nodule Detection Service (Startup): torch.cuda.get_device_name(torch.cuda.current_device()): {torch.cuda.get_device_name(torch.cuda.current_device())}")
        except Exception as e:
            print(f"Nodule Detection Service (Startup): Error getting CUDA device details: {e}")
else:
    print("Nodule Detection Service (Startup): Running on CPU.")


# Preprocessing Parameters
TARGET_SPACING = np.array([1.0, 1.0, 1.0])
CLIP_BOUNDS = (-1000, 400)
NORM_RANGE = (0.0, 1.0)
LUNG_THRESHOLD = -320

# Model & Inference Parameters
PATCH_SIZE = (96, 96, 96) # z, y, x
STRIDE = (48, 48, 48)     # z, y, x
INFERENCE_BATCH_SIZE = 1 # Start with 1 for stability, especially with 8GB RAM and RTX 2050
HEATMAP_SIGMA = 3.0
NMS_THRESHOLD_HEATMAP = 0.4 # For find_heatmap_peaks
NMS_FOOTPRINT_HEATMAP = 5
DETECTION_NMS_DISTANCE = 5.0 # For nms_3d_detections on final world coordinate detections

# Global variable for the model
nodule_model: torch.nn.Module | None = None

warnings.filterwarnings("ignore", category=UserWarning, module='SimpleITK')
warnings.filterwarnings("ignore", category=UserWarning, module='torch') # General torch user warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch') # Specifically for AMP autocast warning


# --- Pydantic Schemas ---
class NodulePredictionItem(BaseModel):
    coordX: float
    coordY: float
    coordZ: float
    voxelX_resampled: int
    voxelY_resampled: int
    voxelZ_resampled: int
    probability: float
    estimated_radius_mm: float

class NoduleDetectionResponse(BaseModel):
    message: str
    nodules: List[NodulePredictionItem]
    scan_shape_zyx_resampled: List[int] | None = None


# --- Preprocessing Functions ---
def load_ct_scan(mhd_path: Path) -> sitk.Image:
    return sitk.ReadImage(str(mhd_path))

def get_scan_properties(sitk_image: sitk.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.array(sitk_image.GetOrigin())
    spacing = np.array(sitk_image.GetSpacing())
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
    return origin, spacing, direction

def world_to_voxel(world_coords: np.ndarray, origin: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    stretched_voxel_coords = np.absolute(world_coords - origin)
    voxel_coords = stretched_voxel_coords / spacing
    return voxel_coords

def voxel_to_world(voxel_coords_zyx: np.ndarray, origin_xyz: np.ndarray, spacing_xyz: np.ndarray) -> np.ndarray:
    voxel_coords_xyz_order = np.array([voxel_coords_zyx[2], voxel_coords_zyx[1], voxel_coords_zyx[0]])
    world_coords = (voxel_coords_xyz_order * spacing_xyz) + origin_xyz
    return world_coords

def resample_volume(sitk_image: sitk.Image, target_spacing_zyx: np.ndarray = TARGET_SPACING) -> sitk.Image:
    original_spacing_xyz = np.array(sitk_image.GetSpacing())
    original_size_xyz = np.array(sitk_image.GetSize())
    target_spacing_sitk_xyz = target_spacing_zyx[::-1]
    new_size_xyz = original_size_xyz * (original_spacing_xyz / target_spacing_sitk_xyz)
    new_size_xyz = np.ceil(new_size_xyz).astype(np.int32)
    new_size_xyz = [int(s) for s in new_size_xyz]
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size_xyz)
    resample_filter.SetOutputSpacing(target_spacing_sitk_xyz)
    resample_filter.SetInterpolator(sitk.sitkLinear)
    resample_filter.SetOutputOrigin(sitk_image.GetOrigin())
    resample_filter.SetOutputDirection(sitk_image.GetDirection())
    resample_filter.SetDefaultPixelValue(float(CLIP_BOUNDS[0]))
    resampled_image = resample_filter.Execute(sitk_image)
    return resampled_image

def normalize_hu(volume_np: np.ndarray, clip_bounds: Tuple[int, int] = CLIP_BOUNDS, norm_range: Tuple[float, float] = NORM_RANGE) -> np.ndarray:
    volume_np = np.clip(volume_np, clip_bounds[0], clip_bounds[1])
    min_val, max_val = clip_bounds
    norm_min, norm_max = norm_range
    if (max_val - min_val) == 0:
        return np.full_like(volume_np, norm_min, dtype=np.float32)
    volume_np = (volume_np - min_val) / (max_val - min_val)
    volume_np = volume_np * (norm_max - norm_min) + norm_min
    return volume_np.astype(np.float32)

def get_lung_mask(volume_np: np.ndarray, threshold: int = LUNG_THRESHOLD) -> np.ndarray:
    binary_image = volume_np < threshold
    struct = generate_binary_structure(3, 2)
    closed_image = binary_dilation(binary_image, structure=struct, iterations=10)
    label_map, num_features = ndimage_label(closed_image)
    if num_features > 0:
        component_sizes = np.bincount(label_map.ravel())
        background_label = 0
        if len(component_sizes) > background_label:
             component_sizes[background_label] = 0
        else:
            print("Warning: Lung mask - bincount issue, returning closed image.")
            return closed_image.astype(bool)
        if num_features > 1:
            sorted_labels_indices = np.argsort(component_sizes)[::-1]
            lung_mask_np = np.zeros_like(label_map, dtype=bool)
            if sorted_labels_indices[0] != background_label and component_sizes[sorted_labels_indices[0]] > 0:
                lung_mask_np[label_map == sorted_labels_indices[0]] = True
            if len(sorted_labels_indices) > 1 and sorted_labels_indices[1] != background_label and component_sizes[sorted_labels_indices[1]] > 0:
                 lung_mask_np[label_map == sorted_labels_indices[1]] = True
            lung_mask_np = binary_dilation(lung_mask_np, structure=struct, iterations=5)
            return lung_mask_np
        else:
             print(f"Warning: Lung mask - only {num_features} component(s) found ({label_map.max()} labels). Returning closed image mask.") # Added more detail
             return closed_image.astype(bool)
    else:
        print("Warning: Lung mask - no components found after labeling. Returning empty mask.")
        return np.zeros_like(volume_np, dtype=bool)

def preprocess_scan_from_path(mhd_path: Path) -> Tuple[np.ndarray | None, Dict[str, Any] | None]:
    try:
        sitk_image = load_ct_scan(mhd_path)
        original_origin, original_spacing, original_direction = get_scan_properties(sitk_image)
        resampled_image = resample_volume(sitk_image, TARGET_SPACING)
        resampled_origin, resampled_spacing, _ = get_scan_properties(resampled_image)
        volume_np_for_mask = sitk.GetArrayFromImage(resampled_image)
        lung_mask_np = get_lung_mask(volume_np_for_mask, LUNG_THRESHOLD)
        if np.sum(lung_mask_np) == 0: # If lung mask is empty
            print("Warning: Preprocessing resulted in an empty lung mask. Aborting processing for this scan.")
            return None, None # Indicate failure
        volume_np_normalized = normalize_hu(volume_np_for_mask, CLIP_BOUNDS, NORM_RANGE)
        processed_volume_np = volume_np_normalized * lung_mask_np
        metadata = {
            'original_mhd_path': str(mhd_path),
            'original_origin': original_origin, 'original_spacing': original_spacing, 'original_direction': original_direction,
            'resampled_origin': resampled_origin, 'resampled_spacing': resampled_spacing,
            'resampled_shape_xyz': resampled_image.GetSize(), 'resampled_shape_zyx': processed_volume_np.shape,
            'clip_bounds': CLIP_BOUNDS, 'norm_range': NORM_RANGE
        }
        return processed_volume_np, metadata
    except Exception as e:
        print(f"Error during preprocessing scan {mhd_path}: {e}")
        return None, None


# --- Inference Helper Functions ---
def find_heatmap_peaks(heatmap: np.ndarray, threshold: float = NMS_THRESHOLD_HEATMAP, footprint_size: int = NMS_FOOTPRINT_HEATMAP) -> Tuple[np.ndarray, np.ndarray]:
    footprint = generate_binary_structure(heatmap.ndim, footprint_size)
    local_max = maximum_filter(heatmap, footprint=footprint, mode='constant', cval=0.0) == heatmap
    threshold_mask = heatmap > threshold
    peaks_mask = local_max & threshold_mask
    peak_coords_zyx = np.argwhere(peaks_mask)
    peak_values = heatmap[peaks_mask]
    if len(peak_coords_zyx) > 0:
        sorted_indices = np.argsort(peak_values)[::-1]
        peak_coords_zyx = peak_coords_zyx[sorted_indices]
        peak_values = peak_values[sorted_indices]
    return peak_coords_zyx, peak_values

def convert_resampled_peak_to_world(peak_voxel_zyx: np.ndarray, scan_metadata: Dict[str, Any]) -> np.ndarray:
    resampled_origin_xyz = scan_metadata['resampled_origin']
    resampled_spacing_xyz = scan_metadata['resampled_spacing']
    world_coord_xyz = voxel_to_world(peak_voxel_zyx, resampled_origin_xyz, resampled_spacing_xyz)
    return world_coord_xyz

def nms_3d_detections(candidates: List[Dict[str, Any]], distance_threshold: float = DETECTION_NMS_DISTANCE) -> List[Dict[str, Any]]:
    if not candidates: return []
    candidates = sorted(candidates, key=lambda x: x['probability'], reverse=True)
    retained_candidates, suppressed_indices = [], [False] * len(candidates)
    for i in range(len(candidates)):
        if suppressed_indices[i]: continue
        retained_candidates.append(candidates[i])
        for j in range(i + 1, len(candidates)):
            if suppressed_indices[j]: continue
            coord_i = np.array([candidates[i]['coordX'], candidates[i]['coordY'], candidates[i]['coordZ']])
            coord_j = np.array([candidates[j]['coordX'], candidates[j]['coordY'], candidates[j]['coordZ']])
            if np.linalg.norm(coord_i - coord_j) < distance_threshold:
                suppressed_indices[j] = True
    return retained_candidates

# --- Main Inference Function ---
# app/services/nodule_detection_service.py
# ... (other imports and code from the previous correct version) ...
import time # Ensure this is at the top

# ... (Preprocessing functions, Pydantic models, etc. - keep them as they were)

# --- Main Inference Function ---
def run_inference_on_volume(
    processed_volume_np: np.ndarray,
    scan_metadata: Dict[str, Any],
    model_instance: torch.nn.Module,
    device_to_use: torch.device,
    patch_sz_zyx: Tuple[int, int, int] = PATCH_SIZE,
    stride_sz_zyx: Tuple[int, int, int] = STRIDE,
    batch_sz: int = INFERENCE_BATCH_SIZE
) -> Tuple[List[Dict[str, Any]], np.ndarray | None]:

    # ***** DIAGNOSTIC PRINTS (keep these) *****
    print(f"Nodule Detection Inference (run_inference_on_volume): Received device_to_use: {device_to_use}")
    # Check if model has parameters before accessing them
    if list(model_instance.parameters()):
        print(f"Nodule Detection Inference (run_inference_on_volume): Model is currently on device: {next(model_instance.parameters()).device}")
        if next(model_instance.parameters()).device != device_to_use:
            print(f"Nodule Detection Inference (run_inference_on_volume): Model was on {next(model_instance.parameters()).device}, moving to {device_to_use}.")
            model_instance.to(device_to_use)
    else:
        print("Nodule Detection Inference (run_inference_on_volume): Model has no parameters or is empty. Cannot check device.")
        # Potentially move the empty model structure to device anyway if that's desired
        model_instance.to(device_to_use)


    if device_to_use.type == 'cuda':
        print(f"Nodule Detection Inference (run_inference_on_volume): torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"Nodule Detection Inference (run_inference_on_volume): torch.cuda.device_count(): {torch.cuda.device_count()}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                print(f"Nodule Detection Inference (run_inference_on_volume): torch.cuda.current_device(): {torch.cuda.current_device()}")
                print(f"Nodule Detection Inference (run_inference_on_volume): torch.cuda.get_device_name(torch.cuda.current_device()): {torch.cuda.get_device_name(torch.cuda.current_device())}")
            except Exception as e:
                 print(f"Nodule Detection Inference (run_inference_on_volume): Error getting CUDA device details: {e}")
    elif device_to_use.type == 'cpu':
        print("Nodule Detection Inference (run_inference_on_volume): Configured to run on CPU.")
    # ***** END OF DIAGNOSTIC PRINTS *****

    model_instance.eval()

    volume_tensor = torch.from_numpy(processed_volume_np).float().unsqueeze(0).unsqueeze(0)
    scan_shape_zyx = np.array(processed_volume_np.shape)
    patch_arr_zyx = np.array(patch_sz_zyx)
    stride_arr_zyx = np.array(stride_sz_zyx) # Make sure stride_arr_zyx is defined

    # vvvvvvvv REINSTATE AND REFINE PATCH GENERATION LOGIC vvvvvvvv
    patch_origins_list_tuples = set() # Use a set of tuples to store unique origins

    # Iterate with strides to generate initial patch origins
    for z_start_base in range(0, scan_shape_zyx[0], stride_arr_zyx[0]):
        for y_start_base in range(0, scan_shape_zyx[1], stride_arr_zyx[1]):
            for x_start_base in range(0, scan_shape_zyx[2], stride_arr_zyx[2]):
                # For strided patches, ensure the patch start doesn't make it go way past the end
                # such that a full patch cannot be formed even by shifting.
                # The actual starting point will be adjusted to fit the patch fully if possible.
                z = min(z_start_base, max(0, scan_shape_zyx[0] - patch_arr_zyx[0]))
                y = min(y_start_base, max(0, scan_shape_zyx[1] - patch_arr_zyx[1]))
                x = min(x_start_base, max(0, scan_shape_zyx[2] - patch_arr_zyx[2]))
                patch_origins_list_tuples.add( (max(0,z), max(0,y), max(0,x)) )


    # Add patches to cover the exact edges/corners if not already covered by stride
    # These are the coordinates for patches that end exactly at the volume boundary.
    # Max with 0 handles cases where scan_dim < patch_dim.
    edge_z = max(0, scan_shape_zyx[0] - patch_arr_zyx[0])
    edge_y = max(0, scan_shape_zyx[1] - patch_arr_zyx[1])
    edge_x = max(0, scan_shape_zyx[2] - patch_arr_zyx[2])

    # Add 8 corners (some will be duplicates if dimensions are small, set handles it)
    patch_origins_list_tuples.add((0, 0, 0))
    patch_origins_list_tuples.add((0, 0, edge_x))
    patch_origins_list_tuples.add((0, edge_y, 0))
    patch_origins_list_tuples.add((0, edge_y, edge_x))
    patch_origins_list_tuples.add((edge_z, 0, 0))
    patch_origins_list_tuples.add((edge_z, 0, edge_x))
    patch_origins_list_tuples.add((edge_z, edge_y, 0))
    patch_origins_list_tuples.add((edge_z, edge_y, edge_x))
    
    patch_origins_list = [np.array(o) for o in sorted(list(patch_origins_list_tuples))]
    # ^^^^^^^^^^ END OF PATCH GENERATION LOGIC ^^^^^^^^^^

    print(f"Nodule Detection: Processing scan with {len(patch_origins_list)} unique patches.")
    if not patch_origins_list:
        print("Nodule Detection: No valid patches generated. Returning empty results.")
        return [], None
    
    full_heatmap_aggregator = torch.zeros_like(volume_tensor, device='cpu', dtype=torch.float32)
    count_map_aggregator = torch.zeros_like(volume_tensor, device='cpu', dtype=torch.float32)
    patch_batch_tensors_cpu = [] 
    origin_batch_coords_processed_in_batch = [] # Stores origins for the current batch being built

    total_patches_to_process = len(patch_origins_list)
    num_batches = (total_patches_to_process + batch_sz - 1) // batch_sz

    with torch.no_grad():
        # We iterate up to total_patches_to_process, forming batches
        for i_patch_overall_idx in range(total_patches_to_process):
            origin_zyx = patch_origins_list[i_patch_overall_idx]
            current_batch_num_display = (i_patch_overall_idx // batch_sz) + 1 # For display
            
            # --- Granular Logging ---
            # print(f"--- Preparing Patch {i_patch_overall_idx+1}/{total_patches_to_process} for Batch {current_batch_num_display}/{num_batches} ---")
            # print(f"Patch {i_patch_overall_idx+1}: Origin ZYX: {origin_zyx.astype(int)}")

            z, y, x = origin_zyx.astype(int)
            
            # print(f"Patch {i_patch_overall_idx+1}: Slicing volume_tensor...")
            z_end = min(z + patch_arr_zyx[0], scan_shape_zyx[0])
            y_end = min(y + patch_arr_zyx[1], scan_shape_zyx[1])
            x_end = min(x + patch_arr_zyx[2], scan_shape_zyx[2])
            patch_data_cpu = volume_tensor[:, :, z:z_end, y:y_end, x:x_end]
            # print(f"Patch {i_patch_overall_idx+1}: Sliced shape: {patch_data_cpu.shape}")
            
            current_patch_shape = patch_data_cpu.shape
            padding_needed_zyx = [
                max(0, patch_arr_zyx[0] - current_patch_shape[2]),
                max(0, patch_arr_zyx[1] - current_patch_shape[3]),
                max(0, patch_arr_zyx[2] - current_patch_shape[4])
            ]
            
            if any(p > 0 for p in padding_needed_zyx):
                # print(f"Patch {i_patch_overall_idx+1}: Padding needed (ZYX): {padding_needed_zyx}")
                torch_pad_dims = (0, padding_needed_zyx[2], 0, padding_needed_zyx[1], 0, padding_needed_zyx[0])
                patch_data_cpu = torch.nn.functional.pad(patch_data_cpu, torch_pad_dims, mode='constant', value=0.0)
                # print(f"Patch {i_patch_overall_idx+1}: Padded shape: {patch_data_cpu.shape}")
            
            patch_batch_tensors_cpu.append(patch_data_cpu)
            origin_batch_coords_processed_in_batch.append(origin_zyx) # Store origin for this patch
            # print(f"Patch {i_patch_overall_idx+1}: Added to CPU batch list. Current CPU batch size: {len(patch_batch_tensors_cpu)}")


            # Check if batch is full or it's the last patch overall
            if len(patch_batch_tensors_cpu) == batch_sz or (i_patch_overall_idx == total_patches_to_process - 1):
                print(f"--- Processing Batch {current_batch_num_display}/{num_batches} (contains {len(patch_batch_tensors_cpu)} patches) ---")
                
                # print(f"Batch {current_batch_num_display}: Concatenating {len(patch_batch_tensors_cpu)} CPU tensors...")
                batch_tensor_input_cpu = torch.cat(patch_batch_tensors_cpu, dim=0)
                # print(f"Batch {current_batch_num_display}: CPU batch tensor shape: {batch_tensor_input_cpu.shape}. Moving to device: {device_to_use}...")
                
                batch_tensor_input_gpu = batch_tensor_input_cpu.to(device_to_use)
                # print(f"Batch {current_batch_num_display}: Batch tensor moved to GPU. Shape: {batch_tensor_input_gpu.shape}. Device: {batch_tensor_input_gpu.device}")
                
                # print(f"Batch {current_batch_num_display}: Starting model forward pass with autocast...")
                with torch.amp.autocast(device_type=device_to_use.type, enabled=(device_to_use.type == 'cuda')):
                    outputs_gpu = model_instance(batch_tensor_input_gpu)
                    # print(f"Batch {current_batch_num_display}: Model forward pass complete. Output shape on GPU: {outputs_gpu.shape}")
                    outputs_gpu = torch.sigmoid(outputs_gpu)
                    # print(f"Batch {current_batch_num_display}: Sigmoid applied. Output shape on GPU: {outputs_gpu.shape}")

                # print(f"Batch {current_batch_num_display}: Moving outputs to CPU...")
                outputs_cpu = outputs_gpu.cpu()
                # print(f"Batch {current_batch_num_display}: Outputs moved to CPU. Shape: {outputs_cpu.shape}")

                # print(f"Batch {current_batch_num_display}: Aggregating {outputs_cpu.shape[0]} heatmaps...")
                for i_output_in_batch, patch_output_heatmap in enumerate(outputs_cpu):
                    oz, oy, ox = origin_batch_coords_processed_in_batch[i_output_in_batch].astype(int)
                    heatmap_d, heatmap_h, heatmap_w = patch_output_heatmap.shape[1:]
                    agg_z_end, agg_y_end, agg_x_end = min(oz + heatmap_d, scan_shape_zyx[0]), min(oy + heatmap_h, scan_shape_zyx[1]), min(ox + heatmap_w, scan_shape_zyx[2])
                    patch_hm_d, patch_hm_h, patch_hm_w = agg_z_end - oz, agg_y_end - oy, agg_x_end - ox

                    if patch_hm_d > 0 and patch_hm_h > 0 and patch_hm_w > 0:
                        full_heatmap_aggregator[:, :, oz:agg_z_end, oy:agg_y_end, ox:agg_x_end] += \
                            patch_output_heatmap[:, 0:patch_hm_d, 0:patch_hm_h, 0:patch_hm_w]
                        count_map_aggregator[:, :, oz:agg_z_end, oy:agg_y_end, ox:agg_x_end] += 1
                # print(f"Batch {current_batch_num_display}: Heatmap aggregation complete.")
                
                patch_batch_tensors_cpu = [] # Clear for next batch
                origin_batch_coords_processed_in_batch = [] # Clear for next batch

                if current_batch_num_display % 10 == 0 or current_batch_num_display == num_batches :
                    print(f"Nodule Detection Inference: FINISHED Processing Batch {current_batch_num_display} / {num_batches}")
    
    print("Nodule Detection Inference: All batches processed. Averaging heatmap...")
    count_map_aggregator[count_map_aggregator == 0] = 1
    averaged_heatmap_np = (full_heatmap_aggregator / count_map_aggregator).squeeze().numpy()
    print("Nodule Detection Inference: Heatmap averaged. Finding peaks...")
    
    peak_voxels_zyx, peak_values = find_heatmap_peaks(averaged_heatmap_np, NMS_THRESHOLD_HEATMAP, NMS_FOOTPRINT_HEATMAP)
    print(f"Nodule Detection Inference: Found {len(peak_voxels_zyx)} peaks. Converting to world coordinates and NMS...")

    detected_nodules_world_coords = []
    # ... (rest of NMS and result formatting - keep as is)
    for voxel_zyx, confidence_val in zip(peak_voxels_zyx, peak_values):
        world_coord_xyz = convert_resampled_peak_to_world(voxel_zyx, scan_metadata)
        est_radius_mm = HEATMAP_SIGMA * float(np.mean(scan_metadata['resampled_spacing']))
        detected_nodules_world_coords.append({
            'coordX': float(world_coord_xyz[0]), 'coordY': float(world_coord_xyz[1]), 'coordZ': float(world_coord_xyz[2]),
            'voxelX_resampled': int(voxel_zyx[2]), 'voxelY_resampled': int(voxel_zyx[1]), 'voxelZ_resampled': int(voxel_zyx[0]),
            'probability': float(confidence_val), 'estimated_radius_mm': est_radius_mm
        })
    
    final_nodules_after_nms = nms_3d_detections(detected_nodules_world_coords, DETECTION_NMS_DISTANCE)

    print(f"Nodule Detection: Found {len(final_nodules_after_nms)} candidate nodules after NMS (final).")
    return final_nodules_after_nms, averaged_heatmap_np

# ... (Rest of the file: load_nodule_detection_model, predict_nodules_endpoint, etc. should remain the same as the last correct version)

# --- Model Loading Function ---
def load_nodule_detection_model():
    global nodule_model
    print(f"Nodule Detection Model Loading: Attempting to load model from: {MODEL_PATH}") # Added for clarity
    if not MODEL_PATH.exists():
        print(f"Error: Nodule model file not found at {MODEL_PATH}")
        return

    try:
        nodule_model_instance = UNet3D_MSA(
            in_channels=1, out_channels=1,
            features=[64, 128, 256, 512], # MUST MATCH YOUR TRAINED MODEL
            use_attention=True
        ) # Create on CPU first
        
        # Load state dict to CPU first to avoid potential CUDA OOM during load if model is large
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        nodule_model_instance.load_state_dict(state_dict)
        
        nodule_model_instance.to(DEVICE) # Now move the populated model to the target DEVICE
        nodule_model_instance.eval()
        nodule_model = nodule_model_instance # Assign to global variable
        print(f"Nodule Detection Model loaded successfully from {MODEL_PATH} and moved to {DEVICE}.")
        print(f"Nodule Detection Model (load_nodule_detection_model): Model is on device: {next(nodule_model.parameters()).device}")

    except Exception as e:
        nodule_model = None
        print(f"Error loading Nodule Detection model from {MODEL_PATH}: {e}")
        import traceback
        traceback.print_exc()


# --- FastAPI Router Definition ---
router = APIRouter()
load_nodule_detection_model() # Load model at startup


@router.post("/predict", response_model=NoduleDetectionResponse)
async def predict_nodules_endpoint(
    mhd_file: UploadFile = File(..., description="MHD header file for the CT scan."),
    raw_file: UploadFile = File(..., description="RAW data file for the CT scan (must match MHD).")
):
    if nodule_model is None:
        raise HTTPException(status_code=503, detail="Nodule detection model not loaded or failed to load. Server is not ready.")

    if not mhd_file.filename or not mhd_file.filename.endswith(".mhd"): # Added check for filename existence
        raise HTTPException(status_code=400, detail="MHD file must have a .mhd extension.")
    
    base_filename = mhd_file.filename[:-4]
    expected_raw_filename = base_filename + ".raw"
    
    with tempfile.TemporaryDirectory(prefix="nodule_upload_", dir=UPLOAD_DIR_BASE) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        mhd_save_path = temp_dir / mhd_file.filename
        raw_save_path = temp_dir / expected_raw_filename

        try:
            with open(mhd_save_path, "wb") as buffer:
                shutil.copyfileobj(mhd_file.file, buffer)
            mhd_file.file.close()
            with open(raw_save_path, "wb") as buffer:
                shutil.copyfileobj(raw_file.file, buffer)
            raw_file.file.close()
            print(f"Nodule Detection: Files saved to temp: {mhd_save_path}, {raw_save_path}")

            preprocess_start_time = torch.cuda.Event(enable_timing=True) if DEVICE.type == 'cuda' else None
            preprocess_end_time = torch.cuda.Event(enable_timing=True) if DEVICE.type == 'cuda' else None
            if preprocess_start_time: preprocess_start_time.record()
            cpu_preprocess_start = os.times().user if not preprocess_start_time else 0


            print("Nodule Detection: Preprocessing scan...")
            processed_volume_np, scan_metadata = preprocess_scan_from_path(mhd_save_path)

            if processed_volume_np is None or scan_metadata is None:
                 raise HTTPException(status_code=400, detail="Preprocessing failed, possibly due to empty lung mask or other error.")


            if preprocess_end_time: preprocess_end_time.record(); torch.cuda.synchronize(); print(f"Nodule Detection: Preprocessing took {preprocess_start_time.elapsed_time(preprocess_end_time) / 1000:.2f} seconds (GPU timer if applicable).")
            else: print(f"Nodule Detection: Preprocessing took {os.times().user - cpu_preprocess_start:.2f} seconds (CPU timer).")
            print(f"Nodule Detection: Preprocessing complete. Volume shape: {processed_volume_np.shape}")


            inference_start_time = torch.cuda.Event(enable_timing=True) if DEVICE.type == 'cuda' else None
            inference_end_time = torch.cuda.Event(enable_timing=True) if DEVICE.type == 'cuda' else None
            if inference_start_time: inference_start_time.record()
            cpu_inference_start = os.times().user if not inference_start_time else 0

            print("Nodule Detection: Running inference...")
            detected_nodules_list, _ = run_inference_on_volume(
                processed_volume_np, scan_metadata, nodule_model, DEVICE,
                PATCH_SIZE, STRIDE, INFERENCE_BATCH_SIZE
            )

            if inference_end_time: inference_end_time.record(); torch.cuda.synchronize(); print(f"Nodule Detection: Core inference took {inference_start_time.elapsed_time(inference_end_time) / 1000:.2f} seconds (GPU timer).")
            else: print(f"Nodule Detection: Core inference took {os.times().user - cpu_inference_start:.2f} seconds (CPU timer).")
            print(f"Nodule Detection: Inference complete.")

        except RuntimeError as e: # Catch RuntimeError for SimpleITK errors
            err_msg = str(e)
            # You can check if "SimpleITK" or specific ITK error messages are in err_msg
            # to make the error message more specific, but for now, this is safer.
            print(f"Nodule Detection Error (likely SimpleITK or other RuntimeError): {err_msg}")
            # Potentially add a check like: if "SimpleITK" in err_msg or "ITK" in err_msg:
            raise HTTPException(status_code=400, detail=f"Error processing image data (check MHD/RAW files and format): {err_msg}")
        except HTTPException: # Re-raise HTTPExceptions from preprocessing
            raise
        except Exception as e:
            import traceback
            print(f"Nodule Detection Error (General): {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An error occurred during nodule detection: {str(e)}")
            
        return NoduleDetectionResponse(
            message="Nodule detection successful.",
            nodules=[NodulePredictionItem(**nod) for nod in detected_nodules_list],
            scan_shape_zyx_resampled=list(processed_volume_np.shape) if processed_volume_np is not None else None
        )
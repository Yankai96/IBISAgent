import numpy as np
import nibabel as nib
from PIL import Image
from process_img import generate_trajectory

def get_slice(nii_data, axis, index):
    axis = axis.lower()
    if axis == 'x':
        slice_2d = nii_data[index, :, :]
    elif axis == 'y':
        slice_2d = nii_data[:, index, :]
    elif axis == 'z':
        slice_2d = nii_data[:, :, index]
    else:
        raise ValueError(f"Invalid axis: {axis}")
    return np.rot90(slice_2d)

def apply_windowing(image_slice, window_min, window_max):
    if window_min is None or window_max is None:
        window_min = np.percentile(image_slice, 1)
        window_max = np.percentile(image_slice, 99)
        
    slice_windowed = np.clip(image_slice, window_min, window_max)
    if (window_max - window_min) == 0:
        return np.zeros_like(slice_windowed, dtype=np.uint8)
    slice_normalized = ((slice_windowed - window_min) / (window_max - window_min)) * 255.0
    return slice_normalized.astype(np.uint8)

def process_nifti(nii_path, mask_path, predictor, slice_axis="z", slice_idx=0, mask_value=1, window_min=None, window_max=None, iou_threshold=0.8, max_steps=20):
    image_nii = nib.load(nii_path)
    image_data = image_nii.get_fdata()
    
    label_nii = nib.load(mask_path)
    label_data = label_nii.get_fdata()
    
    img_slice = get_slice(image_data, slice_axis, slice_idx)
    mask_slice = get_slice(label_data, slice_axis, slice_idx)
    
    ground_truth_mask = (mask_slice == mask_value).astype(np.uint8)
    
    img_slice_norm = apply_windowing(img_slice, window_min, window_max)
    img_pil = Image.fromarray(img_slice_norm).convert("RGB")
    
    trajectory = generate_trajectory(img_pil, ground_truth_mask, predictor, iou_threshold, max_steps)
    
    # ensure standard python types
    window = [float(window_min), float(window_max)] if window_min is not None and window_max is not None else None
    
    result = {
        "file": nii_path,
        "type": "nii",
        "mask_file": mask_path,
        "nii_data": {
            "size": list(image_data.shape),
            "slice_dir": slice_axis,
            "slice_idx": slice_idx,
            "slice_size": list(img_slice.shape),
            "mask_number": mask_value,
            "window": window
        },
        "img_data": None,
        "trajectory": trajectory
    }
    return result

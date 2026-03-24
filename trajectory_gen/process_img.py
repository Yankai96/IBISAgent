import cv2
import numpy as np
import zlib
import base64
import torch
from PIL import Image

def encode_mask_to_rle_zlib_b64(mask: np.ndarray) -> str:
    """Encode a binary mask to zlib-compressed base64 RLE."""
    pixels = mask.flatten(order='C')
    if len(pixels) == 0:
        return ""
    changes = np.concatenate([[0], np.where(pixels[:-1] != pixels[1:])[0] + 1, [len(pixels)]])
    runs = np.diff(changes)
    if pixels[0] == 1:
        runs = np.concatenate([[0], runs])
    runs_str = " ".join(map(str, runs))
    compressed = zlib.compress(runs_str.encode('utf-8'))
    return base64.b64encode(compressed).decode('utf-8')

def calculate_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0

def generate_trajectory(image_pil, ground_truth_mask, predictor, iou_threshold, max_steps):
    predictor.set_image(np.array(image_pil))
    
    predicted_mask = np.zeros_like(ground_truth_mask, dtype=np.uint8)
    click_history = []
    trajectory_data = []
    
    previous_low_res_mask = None
    previous_scores = None
    
    height, width = ground_truth_mask.shape
    device = getattr(predictor.model, "device", "cuda")
    
    for step in range(max_steps):
        current_iou = calculate_iou(predicted_mask, ground_truth_mask)
        if current_iou >= iou_threshold:
            break
            
        fn_mask = np.logical_and(ground_truth_mask, np.logical_not(predicted_mask))
        fp_mask = np.logical_and(np.logical_not(ground_truth_mask), predicted_mask)
        
        if not np.any(fn_mask) and not np.any(fp_mask):
            break
            
        fn_dist = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 5)
        fp_dist = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        if np.max(fn_dist) >= np.max(fp_dist):
            click_type, click_label = 'positive', 1
            max_coords = np.unravel_index(np.argmax(fn_dist), fn_dist.shape)
        else:
            click_type, click_label = 'negative', 0
            max_coords = np.unravel_index(np.argmax(fp_dist), fp_dist.shape)
            
        click_coords = [int(max_coords[1]), int(max_coords[0])] # x, y
        click_history.append((click_coords, click_label))
        
        all_coords = np.array([c[0] for c in click_history])
        all_labels = np.array([c[1] for c in click_history])
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(str(device) == "cuda")):
            if previous_low_res_mask is not None:
                masks, scores, low_res_masks = predictor.predict(
                    point_coords=all_coords,
                    point_labels=all_labels,
                    mask_input=previous_low_res_mask[np.argmax(previous_scores)][None, :, :],
                    multimask_output=True
                )
            else:
                masks, scores, low_res_masks = predictor.predict(
                    point_coords=all_coords,
                    point_labels=all_labels,
                    multimask_output=True
                )
                
        previous_low_res_mask = low_res_masks
        previous_scores = scores
        
        best_idx = np.argmax(scores)
        predicted_mask = masks[best_idx]
        
        coords_norm = [click_coords[0] / width, click_coords[1] / height]
        
        step_info = {
            "step": step,
            "coords_norm": coords_norm,
            "coords_original": click_coords,
            "mask_rle": encode_mask_to_rle_zlib_b64(predicted_mask > 0)
        }
        trajectory_data.append(step_info)
        
    return trajectory_data

def process_image(img_path, mask_path, predictor, iou_threshold=0.8, max_steps=20, reverse_mask=False):
    img_pil = Image.open(img_path).convert("RGB")
    width, height = img_pil.size
    
    mask_pil = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_pil)
    
    if reverse_mask:
        ground_truth_mask = (mask_np < 128).astype(np.uint8)
    else:
        ground_truth_mask = (mask_np > 128).astype(np.uint8)
        
    trajectory = generate_trajectory(img_pil, ground_truth_mask, predictor, iou_threshold, max_steps)
    
    format_str = img_path.split('.')[-1].lower()
    
    result = {
        "file": img_path,
        "type": "img",
        "mask_file": mask_path,
        "nii_data": None,
        "img_data": {
            "format": format_str,
            "size": [width, height],
            "reverse": reverse_mask
        },
        "trajectory": trajectory
    }
    return result

import argparse
import os
import json
from pathlib import Path

# SAM2 import
from sam2.sam2_image_predictor import SAM2ImagePredictor

from process_img import process_image
from process_nii import process_nifti

def main():
    parser = argparse.ArgumentParser(description="Generate trajectory for IBISAgent using custom data")
    
    # Common arguments
    parser.add_argument("--type", type=str, choices=["img", "nii"], required=True, help="Data type: 'img' for 2D images, 'nii' for 3D NIfTI")
    parser.add_argument("--image", type=str, required=True, help="Path to input original image or NIfTI file")
    parser.add_argument("--mask", type=str, required=True, help="Path to input mask file (binary image or NIfTI)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output JSON trajectory")
    
    parser.add_argument("--iou_threshold", type=float, default=0.8, help="Target IoU threshold to stop iteration (default: 0.8)")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum number of interaction steps (default: 20)")
    
    parser.add_argument("--model_path", type=str, default="facebook/sam2-hiera-large", help="SAM2 model path or HF identifier")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., 'cuda', 'cpu')")
    
    # 2D Image specific arguments
    parser.add_argument("--reverse_mask", action="store_true", help="For 'img': treat black as target and white as background (default is white=target)")
    
    # 3D NIfTI specific arguments
    parser.add_argument("--slice_axis", type=str, choices=["x", "y", "z"], default="z", help="For 'nii': slice axis (x, y, or z)")
    parser.add_argument("--slice_idx", type=int, default=0, help="For 'nii': index of the slice to extract")
    parser.add_argument("--mask_value", type=int, default=1, help="For 'nii': pixel value in the mask that represents the target")
    parser.add_argument("--window_min", type=float, default=None, help="For 'nii': minimum window value for visualization/normalization")
    parser.add_argument("--window_max", type=float, default=None, help="For 'nii': maximum window value for visualization/normalization")
    
    args = parser.parse_args()
    
    print(f"Loading SAM2 model from {args.model_path} on {args.device}...")
    try:
        predictor = SAM2ImagePredictor.from_pretrained(args.model_path, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print(f"Processing {args.type} data...")
    
    if args.type == "img":
        result = process_image(
            img_path=args.image,
            mask_path=args.mask,
            predictor=predictor,
            iou_threshold=args.iou_threshold,
            max_steps=args.max_steps,
            reverse_mask=args.reverse_mask
        )
    elif args.type == "nii":
        result = process_nifti(
            nii_path=args.image,
            mask_path=args.mask,
            predictor=predictor,
            slice_axis=args.slice_axis,
            slice_idx=args.slice_idx,
            mask_value=args.mask_value,
            window_min=args.window_min,
            window_max=args.window_max,
            iou_threshold=args.iou_threshold,
            max_steps=args.max_steps
        )
        
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        
    print(f"Trajectory successfully saved to {output_path}")

if __name__ == "__main__":
    main()

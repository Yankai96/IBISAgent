import argparse
import os
import numpy as np
from PIL import Image

from mllm_engine import MLLMEngine
from sam2_engine import SAM2Engine

SYSTEM_PROMPT = (
    "You are a precise and expert medical segmentation agent. Your mission is to accurately segment a target object in a medical image through a series of interactive point placements. You will be given an image and an instruction. You must carefully analyze the image state. If there is no mask, it is an initialization step. Your goal is to place a Positive Point on a clear, representative part of the target object. If a semi-transparent green mask is present, it is a refinement step. Your goal is to improve its accuracy. Place a Positive Point on a region of the target that the mask has missed, or a Negative Point on an area the mask has incorrectly included.\nYour response must strictly follow this structure: first, your detailed reasoning within <think> tags, and then your single, decisive move within <action> tags. The only valid actions are Positive Point (x, y), Negative Point (x, y), or Terminate. All coordinates (x, y) must be normalized to a 0.0-1.0 scale and formatted to four decimal places, for example: Positive Point (0.5000, 0.2500). Only use Terminate when the mask perfectly aligns with the target boundary. If you Terminate, you must also append a final, concise summary in an <answer> tag."
)

def overlay_mask(image_pil: Image.Image, mask: np.ndarray, color=(34, 139, 34), alpha=0.4):
    """
    将掩码以半透明颜色叠加到原始图像上。
    参数:
        image_pil: 原始图像 (PIL.Image)
        mask: 掩码数组，大于0表示前景
        color: 掩码颜色，默认森林绿 (34, 139, 34)
        alpha: 透明度，默认0.4
    返回:
        叠加后的新图像 (PIL.Image)
    """
    img_np = np.array(image_pil).astype(np.float32)
    mask_indices = mask > 0
    
    overlay = img_np.copy()
    overlay[mask_indices, 0] = color[0]
    overlay[mask_indices, 1] = color[1]
    overlay[mask_indices, 2] = color[2]
    
    blended = img_np * (1 - alpha) + overlay * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)

def main():
    parser = argparse.ArgumentParser(description="IBISAgent Multi-turn Interactive Segmentation")
    parser.add_argument("--image", type=str, required=True, help="Path to the input medical image")
    parser.add_argument("--prompt", type=str, required=True, help="User text prompt, e.g., 'Is there a colon tumor in this image?'")
    parser.add_argument("--mllm_path", type=str, default="infer/models/mllm", help="Path to the Qwen2.5-VL model")
    parser.add_argument("--sam2_cfg", type=str, default="infer/models/sam2/medsam2_cfg.yaml", help="Path to the MedSAM2 config file")
    parser.add_argument("--sam2_ckpt", type=str, default="infer/models/sam2/MedSAM2_latest.pt", help="Path to the MedSAM2 checkpoint")
    parser.add_argument("--max_turns", type=int, default=20, help="Maximum number of iterations")
    parser.add_argument("--use_history", type=int, default=0, help="Whether to use chat history (1 for True, 0 for False). Default is False.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the step-by-step results")
    
    args = parser.parse_args()
    use_history = bool(args.use_history)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print(">>> Initializing Models (This may take a while)...")
    print("=" * 60)
    
    mllm = MLLMEngine(args.mllm_path)
    sam2 = SAM2Engine(model_cfg=args.sam2_cfg, checkpoint_path=args.sam2_ckpt)
    
    original_image = Image.open(args.image).convert("RGB")
    sam2.set_image(original_image)
    
    mllm.init_session(SYSTEM_PROMPT)
    
    current_image_path = args.image
    current_image = original_image
    
    print("\n" + "=" * 60)
    print(f">>> Starting Multi-turn Inference (Max turns: {args.max_turns})")
    print(f"User Prompt: {args.prompt}")
    print("=" * 60)
    
    action_history = []
    
    for turn in range(1, args.max_turns + 1):
        print(f"\n[Turn {turn}]")
        
        is_first_turn = (turn == 1)
        current_prompt = f"<image>{args.prompt}" if is_first_turn else args.prompt
        
        print("Waiting for MLLM reasoning...")
        response = mllm.chat(user_prompt=current_prompt, image_path=current_image_path, is_first_turn=is_first_turn, use_history=use_history)
        
        print(f"-> Thinking:\n{response['think']}")
        print(f"-> Action: {response['action_raw']}")
        
        if response['action_type'] == 'Terminate':
            print(f"\n>>> Agent decided to Terminate at turn {turn}. Segmentation process finished.")
            break
            
        if response['action_type'] is None or response['coords'] is None:
            print(f"\n>>> Error: Failed to parse action '{response['action_raw']}'. Terminating process.")
            break
            
        point_label = 1 if response['action_type'] == 'Positive' else 0
        coords = response['coords']
        
        current_action = (response['action_type'], coords)
        action_history.append(current_action)
        
        if len(action_history) >= 3:
            last_three_actions = action_history[-3:]
            if all(act == current_action for act in last_three_actions):
                print(f"\n>>> Repeated action {current_action} detected 3 times consecutively. Terminating process.")
                break
        
        print("Invoking SAM2 with new point...")
        mask = sam2.predict(coords, point_label)
        
        overlaid_image = overlay_mask(original_image, mask, color=(34, 139, 34), alpha=0.4)
        
        current_image_path = os.path.join(args.output_dir, f"turn_{turn}_state.png")
        overlaid_image.save(current_image_path)
        current_image = overlaid_image
        
        print(f"-> New mask generated and saved to: {current_image_path}")

    print("\n" + "=" * 60)
    print(">>> Inference Complete.")
    final_output_path = os.path.join(args.output_dir, "final_segmentation.png")
    current_image.save(final_output_path)
    print(f"Final segmentation saved to: {final_output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

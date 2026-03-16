import torch
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra.utils import instantiate
from omegaconf import OmegaConf

class SAM2Engine:
    def __init__(self, model_cfg="infer/models/sam2/medsam2_cfg.yaml", checkpoint_path="infer/models/sam2/MedSAM2_latest.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SAM2Engine] Building MedSAM2 model from cfg: {model_cfg}...")
        
        cfg = OmegaConf.load(model_cfg)
        
        sam2_model = instantiate(cfg.model)
        
        print(f"[SAM2Engine] Loading checkpoint from: {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        sam2_model.load_state_dict(state_dict)
        sam2_model.to(device=self.device)
        sam2_model.eval() 
        
        self.predictor = SAM2ImagePredictor(sam2_model)
        
        self.reset()
        print("[SAM2Engine] Model and Checkpoint loaded successfully.")

    def reset(self):
        """重置状态，清空历史点击和掩码"""
        self.click_history = []  
        self.previous_low_res_mask = None
        self.previous_scores = None
        self.image_size = None

    def set_image(self, image_pil: Image.Image):
        """设置要处理的图像并计算特征"""
        self.reset()
        self.image_size = image_pil.size  
        print(f"[SAM2Engine] Computing image embeddings (size: {self.image_size})...")
        self.predictor.set_image(image_pil)
        print("[SAM2Engine] Image embeddings computed.")
        
    def predict(self, point_coords, point_label):
        """
        根据新的点击坐标和标签进行预测。
        参数:
            point_coords: (x, y) 相对坐标，范围在 [0.0, 1.0] 之间
            point_label: 1 代表正样本，0 代表负样本
        返回:
            最佳的二维掩码数组 (H, W)
        """
        abs_x = int(point_coords[0] * self.image_size[0])
        abs_y = int(point_coords[1] * self.image_size[1])
        
        self.click_history.append(([abs_x, abs_y], point_label))
        
        all_coords = np.array([c[0] for c in self.click_history], dtype=np.float32)
        all_labels = np.array([c[1] for c in self.click_history], dtype=np.int32)
        
        mask_input = None
        if self.previous_low_res_mask is not None and self.previous_scores is not None:
            best_idx = np.argmax(self.previous_scores)
            mask_input = self.previous_low_res_mask[best_idx][None, :, :]
            
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(self.device=="cuda")):
            masks, scores, low_res_masks = self.predictor.predict(
                point_coords=all_coords,
                point_labels=all_labels,
                mask_input=mask_input,
                multimask_output=True
            )
            
        self.previous_low_res_mask = low_res_masks
        self.previous_scores = scores
        
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        return best_mask

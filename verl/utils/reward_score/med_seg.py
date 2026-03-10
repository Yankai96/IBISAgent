import re
import math
import numpy as np
import base64
import zlib
import requests
from io import BytesIO
import traceback
from pycocotools import mask as coco_mask
from PIL import Image
import os
import numbers

def think_format_reward(predict_str: str) -> float:
    pattern = r"<think>\n(.*?)\n</think>\n(\s*)<action>\n(.*?)\n</action>"
    match = re.search(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0

def answer_format_reward(predict_str: str) -> float:
    if '<answer>' in predict_str:
        if '</answer>' not in predict_str:
            return 0

        if predict_str.index('<answer>') > predict_str.index('</answer>'):
            return 0

    return 1

def point_format_reward(predict_str: str) -> float:
    def is_valid_format(predict_str: str) -> bool:
        try:
            pattern = r'<action>\n(.*?)\n</action>'
            match = re.search(pattern, predict_str, re.DOTALL)
            if not match:
                return False
            content = match.group(1).strip()

            point_pattern = r'^(Positive|Negative) Point \(\s*(0(\.\d+)?|1(\.0+)?)\s*,\s*(0(\.\d+)?|1(\.0+)?)\s*\)$'
            if re.fullmatch(point_pattern, content):
                return True
            return False
        except Exception:
            return False

    return 1.0 if is_valid_format(predict_str) else 0.0

def iou_reward(predict_str: str, ground_truth: str, extra_info: dict) -> float:
    predict_match = re.search(r'<action>\n(Positive|Negative) Point \(\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*,\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*\)\n</action>', predict_str)
    if not predict_match:
        return 0.0

    predict_type = predict_match.group(1)
    predict_x = float(predict_match.group(2))
    predict_y = float(predict_match.group(3))
    label = 1 if predict_type == "Positive" else 0

    try:
        clicklist = list(extra_info.get("clicklist", []))
        labels = list(extra_info.get("labels", []))
        original_image_name = extra_info.get("original_image_name")
        gt_mask_name = extra_info.get("gt_mask_name")
        pre_iou = extra_info.get("pre_iou")

        image_path = original_image_name
        mask_path = gt_mask_name
        image_size = extra_info.get("image_size")

        int_x = int(predict_x * image_size[0])
        int_y = int(predict_y * image_size[1])
        clicklist.append([int_x, int_y])
        labels.append(label)

        predict_mask = sam2_request(
            image_path=image_path,
            clicklist=clicklist,
            labels=labels
        )

        mask_pil = Image.open(mask_path).convert('L')
        ground_truth_mask = np.array(mask_pil) > 127

        iou = calculate_iou(predict_mask, ground_truth_mask)

        iou_increase_score = iou_increase_reward(iou, pre_iou)
        iou_score = 1 if iou > 0.5 else 0.0

        return iou_increase_score + iou_score

    except Exception:
        print(f"Error process data, extra_info:{extra_info}")
        traceback.print_exc()

    return 0.0

def calculate_iou(predicted_mask, ground_truth_mask):
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou

def sam2_request(image_path: str, clicklist: list, labels: list):
    fixed_clicklist = []
    for c in clicklist:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            x, y = c[:2]
            x = int(round(x)) if isinstance(x, numbers.Number) else x
            y = int(round(y)) if isinstance(y, numbers.Number) else y
            fixed_clicklist.append([x, y])
        else:
            fixed_clicklist.append(c)
    clicklist = fixed_clicklist

    url = "http://127.0.0.1:6060/segment"
    payload = {
        "image_path": image_path,
        "clicklist": clicklist,
        "labels": labels
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
    except requests.exceptions.RequestException as e:
        print(f"[SAM2 Request Error] 网络异常: {e}")
        return np.array([], dtype=np.uint8)

    if response.status_code == 200:
        try:
            mask = np.load(BytesIO(response.content))
            mask = (mask > 0).astype(np.uint8)
            return mask
        except Exception as e:
            print(f"[SAM2 Decode Error] 无法解析掩码: {e}")
            return np.array([], dtype=np.uint8)
    else:
        print(f"[SAM2 Request Error] 状态码 {response.status_code}，内容: {response.text}")
        return np.array([], dtype=np.uint8)


def iou_increase_reward(predicted_iou: float, pre_iou: float) -> float:
    return 1 if predicted_iou > pre_iou else 0

def point_placement_reward(predict_str: str, ground_truth: str, extra_info: dict) -> float:
    predict_match = re.search(r'<action>\n(Positive|Negative) Point \(\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*,\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*\)\n</action>', predict_str)
    if not predict_match:
        return 0.0

    predict_type = predict_match.group(1)
    predict_x = float(predict_match.group(2))
    predict_y = float(predict_match.group(3))

    try:
        clicklist = list(extra_info.get("clicklist", []))
        labels = list(extra_info.get("labels", []))
        original_image_name = extra_info.get("original_image_name")
        gt_mask_name = extra_info.get("gt_mask_name")

        image_path = original_image_name
        mask_path = gt_mask_name
        image_size = extra_info.get("image_size")

        mask_pil = Image.open(mask_path).convert('L')
        ground_truth_mask = (np.array(mask_pil) > 127)

        current_pred_mask = np.array([], dtype=np.uint8)

        if clicklist:
            current_pred_mask = sam2_request(
                image_path=image_path,
                clicklist=clicklist,
                labels=labels
            )

        if current_pred_mask.size == 0:
            current_pred_mask = np.zeros(ground_truth_mask.shape, dtype=np.uint8)

        current_pred_mask = (current_pred_mask > 0)

        if current_pred_mask.shape != ground_truth_mask.shape:
            print(f"Placement Reward: Mask 形状不匹配。Pred: {current_pred_mask.shape}, GT: {ground_truth_mask.shape}")
            if not clicklist:
                current_pred_mask = np.zeros_like(ground_truth_mask, dtype=bool)
            else:
                return 0.0

        fn_mask = np.logical_and(ground_truth_mask, np.logical_not(current_pred_mask))

        fp_mask = np.logical_and(np.logical_not(ground_truth_mask), current_pred_mask)

        int_x = int(predict_x * image_size[0])
        int_y = int(predict_y * image_size[1])

        if not (0 <= int_y < image_size[1] and 0 <= int_x < image_size[0]):
            print("Placement Reward: 点击坐标超出图像边界。")
            return 0.0

        reward = 0.0
        if predict_type == "Positive":
            if fn_mask[int_y, int_x]:
                reward = 1.0
            else:
                reward = 0.0
        elif predict_type == "Negative":
            if fp_mask[int_y, int_x]:
                reward = 1.0
            else:
                reward = 0.0

        return reward

    except Exception:
        print("处理 point_placement_reward 过程出错, extra_info内容:", extra_info)
        traceback.print_exc()

    return 0.0

def compute_score(
    data_source,
    solution_str: str,
    ground_truth: str,
    extra_info: dict
) -> dict:

    def safe_float(x) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    think_format_score   = safe_float(think_format_reward(solution_str))
    point_format_score   = safe_float(point_format_reward(solution_str))
    answer_format_score  = safe_float(answer_format_reward(solution_str))
    iou_score            = safe_float(iou_reward(solution_str, ground_truth, extra_info))
    placement_score      = safe_float(point_placement_reward(solution_str, ground_truth, extra_info))

    total_score = (
        think_format_score
        + point_format_score
        + answer_format_score
        + iou_score
        + placement_score
    )

    return {
        "score": float(total_score),
        "think_format_score": float(think_format_score),
        "point_format_score": float(point_format_score),
        "answer_format_score": float(answer_format_score),
        "iou_score": float(iou_score),
        "placement_score": float(placement_score),
    }

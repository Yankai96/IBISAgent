import json
import numpy as np
import zlib
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
from pycocotools import mask as coco_mask
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MASK_COLOR = (0, 255, 0, 128)  # semi-transparent green
PIL_FORMAT = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}


@dataclass
class Config:
    dataset_name: str
    raw_data_path: Path
    image_dir: Path
    output_path: Path
    split: str = "train"


# -------------------------- RLE decoding --------------------------

def decode_rle_mask(rle_dict: dict, sample_name: str, click_idx: int, img_size: tuple) -> Image.Image:
    """Decode RLE (base64 -> zlib -> COCO RLE) and return a single-channel mask image.

    Args:
        img_size: (width, height) from PIL Image.size
    """
    try:
        counts_str = rle_dict.get("counts", "")
        rle_size = rle_dict.get("size", [0, 0])  # (height, width) in RLE convention

        if not isinstance(counts_str, str) or not counts_str:
            raise ValueError("counts must be a non-empty base64 string")

        counts_bytes = zlib.decompress(base64.b64decode(counts_str))

        # PIL size is (width, height); RLE expects (height, width)
        expected_size = (img_size[1], img_size[0])
        if rle_size != list(expected_size):
            swapped = [rle_size[1], rle_size[0]]
            if swapped != list(expected_size):
                raise ValueError(f"RLE size {rle_size} does not match image size {expected_size}")
            rle_size = list(expected_size)

        mask_array = coco_mask.decode({"counts": counts_bytes, "size": rle_size}).astype(np.uint8)
        return Image.fromarray(mask_array * 255, mode="L")

    except Exception as e:
        logger.error(f"Sample {sample_name}, click {click_idx}: RLE decode failed — {e}")
        return Image.new("L", img_size, 0)


# -------------------------- Image utilities --------------------------

def encode_image(img: Image.Image, fmt: str) -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def overlay_mask(origin_img: Image.Image, mask: Image.Image, color: tuple) -> bytes:
    """Overlay a mask onto the original image and return PNG bytes."""
    base = origin_img.convert("RGBA")
    overlay = Image.new("RGBA", origin_img.size, (0, 0, 0, 0))
    overlay.paste(color, mask=mask)
    result = Image.alpha_composite(base, overlay).convert("RGB")
    return encode_image(result, "PNG")


# -------------------------- Sample building --------------------------

def build_sample(
    data_idx: int,
    global_idx: int,
    clicks: list,
    all_image_bytes: list,
    user_content: str,
    img_name: str,
    mask_name: str,
    img_size: tuple,
    dataset_name: str,
    split: str,
) -> dict:
    click = clicks[data_idx]
    click_type = "Positive" if click["is_positive"] else "Negative"
    x, y = click["coor_norm"]
    answer = f"{click_type} Point ({x}, {y})"

    prev_clicks = clicks[:data_idx]
    return {
        "data_source": "med-seg",
        "prompt": [{"role": "user", "content": user_content}],
        "images": [{"bytes": all_image_bytes[data_idx]}],
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": global_idx,
            "answer": answer,
            "question": user_content,
            "dataset": dataset_name,
            "original_image_name": img_name,
            "gt_mask_name": mask_name,
            "clicklist": [c["coor"] for c in prev_clicks],
            "labels": [1 if c["is_positive"] else 0 for c in prev_clicks],
            "current_gt_mask": click.get("mask", {}).get("counts", ""),
            "image_size": img_size,
        },
    }


# -------------------------- Core conversion --------------------------

def find_image(image_dir: Path, name: str) -> tuple[Path, str] | None:
    for ext in PIL_FORMAT:
        path = image_dir / f"{name}{ext}"
        if path.exists():
            return path, PIL_FORMAT[ext]
    return None


def process_sample(sample: dict, image_dir: Path, dataset_name: str, split: str, global_idx: int) -> list[dict]:
    img_name = sample["image_name"]
    mask_name = sample["mask_name"]
    clicks = sample["clicks_list"]
    caption = sample["caption"] if isinstance(sample["caption"], str) else sample["caption"][0]
    user_content = f"<image>{caption}"

    result = find_image(image_dir, img_name)
    if result is None:
        logger.warning(f"Image not found for sample {img_name}, skipping")
        return []
    img_path, fmt = result

    origin_img = Image.open(img_path)
    origin_bytes = encode_image(origin_img, fmt)
    img_size = origin_img.size  # (width, height)

    overlay_bytes_list = [
        overlay_mask(origin_img, decode_rle_mask(click["mask"], mask_name, i, img_size), MASK_COLOR)
        for i, click in enumerate(clicks)
    ]
    all_image_bytes = [origin_bytes] + overlay_bytes_list

    samples = [
        build_sample(i, global_idx + i, clicks, all_image_bytes, user_content,
                     img_name, mask_name, img_size, dataset_name, split)
        for i in range(len(clicks))
    ]
    logger.info(f"Sample {mask_name}: generated {len(clicks)} entries")
    return samples


def convert(cfg: Config) -> None:
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cfg.raw_data_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    global_idx = 0

    for sample in tqdm(raw_data["data"], desc="Processing", unit="sample"):
        try:
            new_records = process_sample(sample, cfg.image_dir, cfg.dataset_name, cfg.split, global_idx)
            records.extend(new_records)
            global_idx += len(new_records)
        except Exception as e:
            name = sample.get("mask_name", sample.get("image_name", "unknown"))
            logger.error(f"Sample {name} failed: {e}, skipping")

    pd.DataFrame(records).to_parquet(cfg.output_path, engine="pyarrow", index=False)
    logger.info(f"Done. {len(records)} entries saved to {cfg.output_path}")


# -------------------------- Entry point --------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert med-seg data to RL format (Parquet)")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--data_root", required=True, help="Root directory containing the dataset (e.g. /path/to/med-seg-rl)")
    parser.add_argument("--output_root", required=True, help="Root directory for output Parquet files")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    cfg = Config(
        dataset_name=args.dataset,
        raw_data_path=data_root / args.dataset / f"{args.split}.json",
        image_dir=data_root / args.dataset / args.split,
        output_path=output_root / args.dataset / f"{args.split}_rl.parquet",
        split=args.split,
    )
    convert(cfg)


if __name__ == "__main__":
    main()

"""
dataset.py
----------
PyTorch Dataset for line-segment endpoint heatmap training.

Expected COCO-format annotations
---------------------------------
The JSON file follows the standard COCO structure, but each annotation stores
line-segment endpoints in one of two supported formats:

  Format A – "endpoints" key (preferred custom extension):
    {
      "id": 1, "image_id": 1, "category_id": 1,
      "endpoints": [x1, y1, x2, y2]
    }

  Format B – COCO "keypoints" field with exactly 2 keypoints:
    {
      "id": 1, "image_id": 1, "category_id": 1,
      "keypoints": [x1, y1, v1, x2, y2, v2],
      "num_keypoints": 2
    }

  Format C – "segmentation" field as a single polygon with 4 values:
    {
      "id": 1, "image_id": 1, "category_id": 1,
      "segmentation": [[x1, y1, x2, y2]]
    }

The dataset returns:
  image  : torch.Tensor (3, H, W), normalised to [0, 1]
  heatmap: torch.Tensor (2, H, W), values in [0, 1]
             channel 0 = first endpoint
             channel 1 = second endpoint
"""

import os
import json
from typing import Optional, Callable, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap import make_endpoint_heatmaps


# --------------------------------------------------------------------------- #
# Annotation parser                                                             #
# --------------------------------------------------------------------------- #

def _parse_endpoints(ann: dict) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract (x1, y1, x2, y2) from a single COCO annotation dict.
    Returns None if the annotation cannot be interpreted as a line segment.
    """
    # Format A: custom "endpoints" key
    if "endpoints" in ann:
        ep = ann["endpoints"]
        if len(ep) >= 4:
            return float(ep[0]), float(ep[1]), float(ep[2]), float(ep[3])

    # Format B: COCO keypoints (2 keypoints × 3 values = 6 values)
    if "keypoints" in ann:
        kp = ann["keypoints"]
        if len(kp) >= 6:
            return float(kp[0]), float(kp[1]), float(kp[3]), float(kp[4])

    # Format C: segmentation polygon with exactly 4 coordinates
    if "segmentation" in ann:
        seg = ann["segmentation"]
        if isinstance(seg, list) and len(seg) > 0:
            poly = seg[0]
            if isinstance(poly, list) and len(poly) >= 4:
                return float(poly[0]), float(poly[1]), float(poly[2]), float(poly[3])

    return None


def load_coco_annotations(json_path: str) -> dict:
    """
    Load a COCO JSON and return a mapping:
        image_id -> {
            "file_name": str,
            "width": int,
            "height": int,
            "endpoints": [(x1,y1,x2,y2), ...]
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build image_id -> image metadata lookup
    id_to_image = {img["id"]: img for img in coco.get("images", [])}

    # Aggregate annotations per image
    data: dict = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in id_to_image:
            continue

        ep = _parse_endpoints(ann)
        if ep is None:
            continue  # skip annotations that have no line-segment info

        if img_id not in data:
            img_meta = id_to_image[img_id]
            data[img_id] = {
                "file_name": img_meta["file_name"],
                "width": img_meta.get("width", 0),
                "height": img_meta.get("height", 0),
                "endpoints": [],
            }
        data[img_id]["endpoints"].append(ep)

    return data


# --------------------------------------------------------------------------- #
# PyTorch Dataset                                                               #
# --------------------------------------------------------------------------- #

class LineSegmentDataset(Dataset):
    """
    PyTorch Dataset for Gaussian endpoint heatmap training.

    Parameters
    ----------
    image_dir   : directory containing the JPEG/PNG images
    ann_file    : path to the COCO-format JSON annotation file
    input_size  : (height, width) to resize images and heatmaps to
    sigma       : Gaussian sigma for heatmap generation
    transform   : optional callable applied to the *image* tensor after
                  normalisation; useful for augmentation
    """

    def __init__(
        self,
        image_dir: str,
        ann_file: str,
        input_size: Tuple[int, int] = (512, 512),
        sigma: float = 16.0,
        transform: Optional[Callable] = None,
    ):
        self.image_dir = image_dir
        self.input_size = input_size   # (H, W)
        self.sigma = sigma
        self.transform = transform

        # Load and index annotations
        ann_data = load_coco_annotations(ann_file)
        # Keep only images that have at least one valid line segment
        self.samples = [v for v in ann_data.values() if len(v["endpoints"]) > 0]

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid line-segment annotations found in '{ann_file}'. "
                "Make sure annotations contain 'endpoints', 'keypoints', or "
                "'segmentation' fields with line-segment data."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        target_h, target_w = self.input_size

        # ------------------------------------------------------------------- #
        # Load image                                                            #
        # ------------------------------------------------------------------- #
        img_path = os.path.join(self.image_dir, sample["file_name"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (target_w, target_h),
                                 interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1] and convert to (C, H, W) float32 tensor
        image_tensor = torch.from_numpy(
            img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        )

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        # ------------------------------------------------------------------- #
        # Scale endpoints to resized image coordinates                         #
        # ------------------------------------------------------------------- #
        scale_x = target_w / orig_w if orig_w > 0 else 1.0
        scale_y = target_h / orig_h if orig_h > 0 else 1.0

        scaled_endpoints = [
            (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            for x1, y1, x2, y2 in sample["endpoints"]
        ]

        # ------------------------------------------------------------------- #
        # Generate 2-channel Gaussian heatmap                                  #
        # ------------------------------------------------------------------- #
        heatmap_np = make_endpoint_heatmaps(
            endpoints=scaled_endpoints,
            height=target_h,
            width=target_w,
            sigma=self.sigma,
        )
        heatmap_tensor = torch.from_numpy(heatmap_np)

        return image_tensor, heatmap_tensor

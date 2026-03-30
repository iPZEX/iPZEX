"""
val.py
------
Validation / inference script for Gaussian endpoint heatmap prediction.

Usage examples
--------------
# Evaluate on the validation split and print metrics:
    python val.py --checkpoint checkpoints/best.pt

# Run inference on a single image and save output:
    python val.py --checkpoint checkpoints/best.pt \\
                  --image path/to/image.jpg \\
                  --output_image result.png

# Evaluate on a custom annotation file:
    python val.py --checkpoint checkpoints/best.pt \\
                  --val-image-dir data/test2017 \\
                  --val-ann-file  data/annotations/instances_test2017.json
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import LineSegmentDataset
from src.model import build_model
from src.loss import WeightedMSELoss
from src.utils import (save_visualisation, endpoint_distance_error,
                        decode_endpoints, overlay_heatmap_on_image)


# =========================================================================== #
# CONFIG – defaults mirror train.py                                            #
# =========================================================================== #
DEFAULTS = dict(
    checkpoint="checkpoints/best.pt",
    val_image_dir="data/val2017",
    val_ann_file="data/annotations/instances_val2017.json",
    output_dir="val_output",
    input_size=512,
    sigma=16.0,
    base_channels=32,
    batch_size=8,
    num_workers=4,
    fg_weight=10.0,
    threshold=0.3,          # peak detection threshold
    vis_samples=8,          # max visualisations to save
    image=None,             # single-image inference mode
    output_image=None,      # where to save the single-image result
)
# =========================================================================== #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate / run inference with trained heatmap model"
    )
    for key, val in DEFAULTS.items():
        arg_type = type(val) if val is not None else str
        parser.add_argument(f"--{key.replace('_', '-')}", type=arg_type,
                            default=val)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Single-image inference                                                        #
# --------------------------------------------------------------------------- #

def infer_single_image(model, img_path: str, input_size: int,
                       sigma: float, threshold: float,
                       device: torch.device,
                       output_path: str = None):
    """Run the model on one image and (optionally) save a visualisation."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size, input_size),
                             interpolation=cv2.INTER_LINEAR)

    image_tensor = torch.from_numpy(
        img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    ).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(image_tensor)   # (1, 2, H, W)

    # Decode peaks
    endpoints = decode_endpoints(pred, threshold=threshold)[0]
    print(f"Detected endpoints (resized coords):")
    for ch, (x, y) in enumerate(endpoints):
        if x >= 0:
            # Scale back to original image coordinates
            ox = int(x * orig_w / input_size)
            oy = int(y * orig_h / input_size)
            print(f"  p{ch+1}: ({ox}, {oy})  [resized: ({x}, {y})]")
        else:
            print(f"  p{ch+1}: not detected")

    if output_path:
        save_visualisation(
            image_tensor=image_tensor[0].cpu(),
            heatmap_pred=pred[0].cpu(),
            heatmap_gt=None,
            save_path=output_path,
        )
        print(f"Visualisation saved → {output_path}")

    return endpoints


# --------------------------------------------------------------------------- #
# Dataset evaluation                                                            #
# --------------------------------------------------------------------------- #

def evaluate(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[val] Using device: {device}")

    # Load model
    model = build_model(base_channels=args.base_channels).to(device)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    # Accept both full checkpoint dicts and plain state dicts
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[val] Loaded checkpoint: {args.checkpoint}")

    # Dataset
    input_size = (args.input_size, args.input_size)
    val_ds = LineSegmentDataset(
        image_dir=args.val_image_dir,
        ann_file=args.val_ann_file,
        input_size=input_size,
        sigma=args.sigma,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    print(f"[val] Validation samples: {len(val_ds)}")

    criterion = WeightedMSELoss(fg_threshold=0.1, fg_weight=args.fg_weight)
    os.makedirs(args.output_dir, exist_ok=True)

    total_loss = 0.0
    total_dist = 0.0
    dist_steps = 0
    saved = 0

    with torch.no_grad():
        for batch_idx, (images, heatmaps) in enumerate(val_loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            preds = model(images)

            loss = criterion(preds, heatmaps)
            total_loss += loss.item()

            dist = endpoint_distance_error(preds, heatmaps, args.threshold)
            if not math.isnan(dist):
                total_dist += dist
                dist_steps += 1

            # Save a few visualisations
            for i in range(images.size(0)):
                if saved >= args.vis_samples:
                    break
                out_path = os.path.join(args.output_dir,
                                        f"sample_{saved:04d}.png")
                save_visualisation(
                    image_tensor=images[i].cpu(),
                    heatmap_pred=preds[i].cpu(),
                    heatmap_gt=heatmaps[i].cpu(),
                    save_path=out_path,
                )
                saved += 1

    avg_loss = total_loss / len(val_loader)
    avg_dist = total_dist / dist_steps if dist_steps > 0 else float("nan")
    print(f"\n[val] Results:")
    print(f"      Weighted MSE loss : {avg_loss:.6f}")
    print(f"      Mean endpoint dist: {avg_dist:.2f} px")
    print(f"      Visualisations    : {args.output_dir}/")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    if args.image:
        # Single-image inference mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(base_channels=args.base_channels).to(device)

        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)

        infer_single_image(
            model=model,
            img_path=args.image,
            input_size=args.input_size,
            sigma=args.sigma,
            threshold=args.threshold,
            device=device,
            output_path=args.output_image,
        )
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

"""
train.py
--------
Training script for Gaussian endpoint heatmap prediction.

Quick start
-----------
1. Activate your virtual environment and install requirements:
       pip install -r requirements.txt

2. Edit the CONFIG section below (or pass CLI arguments) to point at your
   COCO dataset:
       TRAIN_IMAGE_DIR  = "data/train2017"
       TRAIN_ANN_FILE   = "data/annotations/instances_train2017.json"
       VAL_IMAGE_DIR    = "data/val2017"
       VAL_ANN_FILE     = "data/annotations/instances_val2017.json"

3. Run:
       python train.py
   or, from VS Code, press F5 (see .vscode/launch.json).

Checkpoints are saved to OUTPUT_DIR after each epoch.
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Make sure the project root is on the path when running from VS Code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import LineSegmentDataset
from src.model import build_model
from src.loss import WeightedMSELoss
from src.utils import save_visualisation, endpoint_distance_error


# =========================================================================== #
# CONFIG – edit these defaults or override via CLI arguments                   #
# =========================================================================== #
DEFAULTS = dict(
    train_image_dir="data/train2017",
    train_ann_file="data/annotations/instances_train2017.json",
    val_image_dir="data/val2017",
    val_ann_file="data/annotations/instances_val2017.json",
    output_dir="checkpoints",
    vis_dir="vis",
    input_size=512,        # images are resized to (input_size × input_size)
    sigma=16.0,            # Gaussian sigma for heatmap generation (pixels)
    base_channels=32,      # UNet width; increase for better accuracy
    batch_size=8,
    num_workers=4,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    lr_step_size=20,       # reduce LR every N epochs
    lr_gamma=0.1,
    fg_weight=10.0,        # foreground weight in WeightedMSELoss
    save_every=5,          # save checkpoint every N epochs
    vis_every=5,           # save visualisations every N epochs
    vis_samples=4,         # how many samples to visualise
    seed=42,
)
# =========================================================================== #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Gaussian endpoint heatmap model on COCO line-segment data"
    )
    for key, val in DEFAULTS.items():
        arg_type = type(val) if val is not None else str
        parser.add_argument(f"--{key.replace('_', '-')}", type=arg_type,
                            default=val, help=f"(default: {val})")
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # Directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    input_size = (args.input_size, args.input_size)

    # ------------------------------------------------------------------- #
    # Datasets & DataLoaders                                                #
    # ------------------------------------------------------------------- #
    print("[train] Loading datasets …")
    train_ds = LineSegmentDataset(
        image_dir=args.train_image_dir,
        ann_file=args.train_ann_file,
        input_size=input_size,
        sigma=args.sigma,
    )
    val_ds = LineSegmentDataset(
        image_dir=args.val_image_dir,
        ann_file=args.val_ann_file,
        input_size=input_size,
        sigma=args.sigma,
    )
    print(f"[train] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------- #
    # Model, Loss, Optimiser                                                #
    # ------------------------------------------------------------------- #
    model = build_model(base_channels=args.base_channels).to(device)
    criterion = WeightedMSELoss(fg_threshold=0.1, fg_weight=args.fg_weight)
    optimiser = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimiser,
                                          step_size=args.lr_step_size,
                                          gamma=args.lr_gamma)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {num_params:,}")

    # ------------------------------------------------------------------- #
    # Training loop                                                         #
    # ------------------------------------------------------------------- #
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────── #
        model.train()
        train_loss = 0.0
        for images, heatmaps in train_loader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            preds = model(images)
            loss = criterion(preds, heatmaps)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────────── #
        model.eval()
        val_loss = 0.0
        val_dist = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch_idx, (images, heatmaps) in enumerate(val_loader):
                images = images.to(device)
                heatmaps = heatmaps.to(device)

                preds = model(images)
                loss = criterion(preds, heatmaps)
                val_loss += loss.item()

                dist = endpoint_distance_error(preds, heatmaps)
                if not math.isnan(dist):
                    val_dist += dist
                    val_steps += 1

                # Save a visualisation for the first mini-batch
                if batch_idx == 0 and epoch % args.vis_every == 0:
                    _save_batch_vis(images, preds, heatmaps,
                                   args.vis_dir, epoch, args.vis_samples)

        val_loss /= len(val_loader)
        val_dist = val_dist / val_steps if val_steps > 0 else float("nan")
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={train_loss:.5f}"
            f"  val_loss={val_loss:.5f}"
            f"  val_dist={val_dist:.2f}px"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
            f"  time={elapsed:.1f}s"
        )

        scheduler.step()

        # ── Checkpoint ─────────────────────────────────────────────────── #
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"[train] Checkpoint saved → {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"[train] Best model updated → {best_path}")

    print("[train] Training complete.")


# --------------------------------------------------------------------------- #
# Helper: save visualisations for a single batch                               #
# --------------------------------------------------------------------------- #

def _save_batch_vis(images, preds, heatmaps, vis_dir, epoch, n_samples):
    for i in range(min(n_samples, images.size(0))):
        save_path = os.path.join(vis_dir, f"epoch{epoch:03d}_sample{i:02d}.png")
        save_visualisation(
            image_tensor=images[i].cpu(),
            heatmap_pred=preds[i].cpu(),
            heatmap_gt=heatmaps[i].cpu(),
            save_path=save_path,
        )


if __name__ == "__main__":
    main()

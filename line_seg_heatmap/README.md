# Line-Segment Endpoint Heatmap Training

A complete, **VS Code-ready** PyTorch training codebase for predicting
line-segment endpoints via **Gaussian heatmaps**, using a **COCO-format**
dataset.

---

## Project structure

```
line_seg_heatmap/
├── .vscode/
│   └── launch.json          ← F5 run/debug configurations for VS Code
├── data/
│   ├── train2017/           ← training images (JPEG/PNG)
│   ├── val2017/             ← validation images
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   └── sample/
│       └── README.md
├── src/
│   ├── __init__.py
│   ├── heatmap.py           ← vectorised Gaussian heatmap generation
│   ├── dataset.py           ← COCO parser + PyTorch Dataset
│   ├── model.py             ← UNet encoder–decoder
│   ├── loss.py              ← MSE / weighted-MSE losses
│   └── utils.py             ← visualisation & evaluation helpers
├── train.py                 ← training entry point
├── val.py                   ← validation / inference entry point
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Dataset format

Put your data under `data/` in standard COCO layout:

```
data/
├── train2017/          (images)
├── val2017/            (images)
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Each annotation in the JSON must encode the two line-segment endpoints.
Three formats are supported automatically:

### Format A – custom `"endpoints"` key *(recommended)*

```json
{
  "id": 1, "image_id": 1, "category_id": 1,
  "endpoints": [x1, y1, x2, y2]
}
```

### Format B – COCO `"keypoints"` with 2 keypoints

```json
{
  "id": 1, "image_id": 1, "category_id": 1,
  "keypoints": [x1, y1, visibility1, x2, y2, visibility2],
  "num_keypoints": 2
}
```

### Format C – `"segmentation"` polygon with 4 coordinates

```json
{
  "id": 1, "image_id": 1, "category_id": 1,
  "segmentation": [[x1, y1, x2, y2]]
}
```

---

## Training

```bash
python train.py \
  --train-image-dir data/train2017 \
  --train-ann-file  data/annotations/instances_train2017.json \
  --val-image-dir   data/val2017 \
  --val-ann-file    data/annotations/instances_val2017.json
```

Or press **F5** in VS Code with the *"Train – heatmap model"* configuration.

Key arguments (all have sensible defaults):

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 8 | Mini-batch size |
| `--input-size` | 512 | Resize images to this square size |
| `--sigma` | 16.0 | Gaussian σ for heatmap generation (pixels) |
| `--base-channels` | 32 | UNet width; larger = more capacity |
| `--lr` | 1e-3 | Initial learning rate |
| `--fg-weight` | 10.0 | Foreground pixel weight in loss |

Checkpoints are saved to `checkpoints/` every `--save-every` epochs.
The best checkpoint (lowest validation loss) is always saved as `checkpoints/best.pt`.

---

## Validation

```bash
python val.py --checkpoint checkpoints/best.pt
```

Prints:
- Weighted MSE loss on the validation set
- Mean endpoint pixel distance

Side-by-side visualisations are saved to `val_output/`.

---

## Single-image inference

```bash
python val.py \
  --checkpoint   checkpoints/best.pt \
  --image        path/to/image.jpg \
  --output-image result.png
```

Prints the detected endpoint coordinates and (optionally) saves a
colour-overlay PNG showing the predicted heatmaps.

---

## How it works

### Gaussian heatmap (`src/heatmap.py`)

Each endpoint `(cx, cy)` creates a 2-D Gaussian blob on a float32 map:

```
G(x, y) = exp( -((x−cx)² + (y−cy)²) / (2σ²) )
```

The computation is vectorised with NumPy and only touches pixels within
`±√(2·th)·σ` of the centre (where `th = 4.6052` ≈ `ln(100)`), so values
below ~1 % of the peak are simply skipped.  When multiple segments overlap,
the **maximum** response is kept.

The model is trained to predict **two heatmap channels** simultaneously:
- Channel 0 → first endpoint (p1)
- Channel 1 → second endpoint (p2)

### Model (`src/model.py`)

A **UNet** with skip connections:

```
Input (3,H,W)
 → enc1 (DoubleConv, stride 1)  → s1
 → enc2 (MaxPool + DoubleConv)  → s2
 → enc3 (MaxPool + DoubleConv)  → s3
 → enc4 (MaxPool + DoubleConv)  → s4
 → bottleneck (MaxPool + DoubleConv)
 → dec4 (Upsample + cat s4 + DoubleConv)
 → dec3 (Upsample + cat s3 + DoubleConv)
 → dec2 (Upsample + cat s2 + DoubleConv)
 → dec1 (Upsample + cat s1 + DoubleConv)
 → head Conv2d(c, 2) + Sigmoid
Output (2,H,W)
```

### Loss (`src/loss.py`)

**WeightedMSELoss**: pixels with ground-truth value > 0.1 (i.e. near
the Gaussian peaks) are up-weighted by `fg_weight` (default 10×).  This
counteracts the class imbalance between the small foreground region and the
large background.

---

## Tips

- **Start small**: use `--base-channels 16 --input-size 256` for a quick
  sanity-check run.
- **Adjust σ**: larger σ makes the heatmap easier to learn but gives coarser
  localization.  Typical range: 8–32 pixels at the target resolution.
- **GPU memory**: `batch-size 8` + `input-size 512` requires ~4 GB VRAM with
  the default `base-channels 32`.  Reduce either if you run out of memory.
- **Overfitting check**: if train loss drops but val loss plateaus, try
  adding data augmentation via the `transform` argument in `LineSegmentDataset`.

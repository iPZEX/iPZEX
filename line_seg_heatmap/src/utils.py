"""
utils.py
--------
Visualisation and evaluation utilities for line-segment endpoint heatmaps.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Visualisation                                                                 #
# --------------------------------------------------------------------------- #

def heatmap_to_colormap(heatmap: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel float heatmap ([0,1]) to a BGR colour image
    using the JET colour map.

    Parameters
    ----------
    heatmap : np.ndarray (H, W), float32 in [0, 1]

    Returns
    -------
    np.ndarray (H, W, 3), uint8, BGR
    """
    hm_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)


def overlay_heatmap_on_image(image_rgb: np.ndarray,
                              heatmap: np.ndarray,
                              alpha: float = 0.5) -> np.ndarray:
    """
    Blend a Gaussian heatmap onto an RGB image for visualisation.

    Parameters
    ----------
    image_rgb : np.ndarray (H, W, 3), uint8
    heatmap   : np.ndarray (H, W), float32 in [0, 1]
    alpha     : weight of the heatmap overlay

    Returns
    -------
    np.ndarray (H, W, 3), uint8, BGR
    """
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hm_bgr = heatmap_to_colormap(heatmap)
    h, w = img_bgr.shape[:2]
    if hm_bgr.shape[:2] != (h, w):
        hm_bgr = cv2.resize(hm_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, hm_bgr, alpha, 0)


def save_visualisation(image_tensor: torch.Tensor,
                        heatmap_pred: torch.Tensor,
                        heatmap_gt: Optional[torch.Tensor],
                        save_path: str,
                        alpha: float = 0.5) -> None:
    """
    Save a side-by-side visualisation:
      [ image+pred_p1 | image+pred_p2 | image+gt_p1 | image+gt_p2 ]

    Parameters
    ----------
    image_tensor  : (3, H, W) float32 tensor in [0, 1]
    heatmap_pred  : (2, H, W) float32 tensor in [0, 1]
    heatmap_gt    : (2, H, W) float32 tensor in [0, 1], or None
    save_path     : output file path (PNG recommended)
    alpha         : heatmap overlay transparency
    """
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    panels = []
    labels = ["pred_p1", "pred_p2"]
    for ch, label in enumerate(labels):
        hm_np = heatmap_pred[ch].cpu().numpy()
        panel = overlay_heatmap_on_image(img_np, hm_np, alpha=alpha)
        _add_text(panel, label)
        panels.append(panel)

    if heatmap_gt is not None:
        for ch, label in enumerate(["gt_p1", "gt_p2"]):
            hm_np = heatmap_gt[ch].cpu().numpy()
            panel = overlay_heatmap_on_image(img_np, hm_np, alpha=alpha)
            _add_text(panel, label)
            panels.append(panel)

    combined = np.concatenate(panels, axis=1)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, combined)


def _add_text(img: np.ndarray, text: str) -> None:
    """Draw a small label in the top-left corner of *img* in-place."""
    cv2.putText(img, text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Endpoint extraction from heatmap                                              #
# --------------------------------------------------------------------------- #

def extract_peaks(heatmap: np.ndarray,
                  threshold: float = 0.3,
                  top_k: int = 1) -> List[Tuple[int, int]]:
    """
    Extract the top-k peak locations from a 2-D heatmap.

    Parameters
    ----------
    heatmap   : np.ndarray (H, W), float32 in [0, 1]
    threshold : minimum heatmap value to be considered a peak
    top_k     : maximum number of peaks to return

    Returns
    -------
    List of (x, y) pixel coordinates sorted by descending heatmap value.
    """
    # Suppress non-maximum values with a small local maximum filter
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(heatmap, kernel)
    local_max = (heatmap == dilated) & (heatmap >= threshold)
    ys, xs = np.where(local_max)

    if len(xs) == 0:
        return []

    # Sort by descending score
    scores = heatmap[ys, xs]
    order = np.argsort(-scores)
    ys, xs = ys[order], xs[order]

    return [(int(xs[i]), int(ys[i])) for i in range(min(top_k, len(xs)))]


def decode_endpoints(heatmap_batch: torch.Tensor,
                     threshold: float = 0.3) -> List[List[Tuple[int, int]]]:
    """
    Decode a batched 2-channel heatmap tensor into endpoint coordinates.

    Parameters
    ----------
    heatmap_batch : (B, 2, H, W) float32 tensor
    threshold     : minimum response to accept as a peak

    Returns
    -------
    List (length B) of lists [[p1, p2], ...] where p1, p2 are (x, y) tuples.
    Each inner list has two elements (one per channel).  If a channel has no
    peak above the threshold, its entry is (-1, -1).
    """
    results = []
    for b in range(heatmap_batch.size(0)):
        endpoints = []
        for ch in range(heatmap_batch.size(1)):
            hm_np = heatmap_batch[b, ch].cpu().numpy()
            peaks = extract_peaks(hm_np, threshold=threshold, top_k=1)
            endpoints.append(peaks[0] if peaks else (-1, -1))
        results.append(endpoints)
    return results


# --------------------------------------------------------------------------- #
# Metric                                                                        #
# --------------------------------------------------------------------------- #

def endpoint_distance_error(pred_hm: torch.Tensor,
                             gt_hm: torch.Tensor,
                             threshold: float = 0.3) -> float:
    """
    Mean L2 distance (in pixels) between predicted and ground-truth endpoint
    peaks across a batch.

    Endpoints that cannot be detected (no peak above threshold) are skipped.

    Parameters
    ----------
    pred_hm   : (B, 2, H, W)
    gt_hm     : (B, 2, H, W)
    threshold : peak detection threshold

    Returns
    -------
    Mean pixel distance as a Python float (or NaN if no valid pairs).
    """
    pred_pts = decode_endpoints(pred_hm, threshold)
    gt_pts = decode_endpoints(gt_hm, threshold)

    errors = []
    for b_pred, b_gt in zip(pred_pts, gt_pts):
        for (px, py), (gx, gy) in zip(b_pred, b_gt):
            if px < 0 or gx < 0:
                continue
            errors.append(np.hypot(px - gx, py - gy))

    return float(np.mean(errors)) if errors else float("nan")

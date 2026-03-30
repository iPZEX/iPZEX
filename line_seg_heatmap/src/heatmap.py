"""
heatmap.py
----------
Vectorised Gaussian heatmap generation for line-segment endpoints.

Each line segment has two endpoints (x1,y1) and (x2,y2).
We create two heatmap channels – one per endpoint – so the model can
predict both ends independently.

Usage
-----
>>> from src.heatmap import make_endpoint_heatmaps
>>> hm = make_endpoint_heatmaps(
...     endpoints=[(100, 200, 300, 400)],   # list of (x1,y1,x2,y2)
...     height=512, width=512,
...     sigma=16.0
... )
>>> hm.shape  # (2, 512, 512)  – channel 0 = p1, channel 1 = p2
"""

import math
import numpy as np


# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #
# Gaussian values below exp(-TH) ≈ 0.01 are clipped to zero.
# TH = -ln(0.01) ≈ 4.6052, meaning only pixels with Gaussian response
# above 1 % of the peak value are written.
_TH = 4.6052
# Pre-computed truncation multiplier: only compute within ±DELTA·sigma pixels.
_DELTA = math.sqrt(_TH * 2)  # ≈ 3.03


def put_heatmap(heatmap: np.ndarray,
                center_x: float,
                center_y: float,
                sigma: float = 16.0) -> None:
    """
    Draw a single 2-D isotropic Gaussian blob onto *heatmap* in-place.

    Parameters
    ----------
    heatmap  : 2-D float32 array (H, W), values in [0, 1]
    center_x : horizontal centre of the Gaussian (pixel coordinates)
    center_y : vertical centre of the Gaussian (pixel coordinates)
    sigma    : standard deviation in pixels; larger ⟹ wider blob
    """
    height, width = heatmap.shape

    # Bounding box of the region we need to touch (clipped to image bounds)
    x0 = int(max(0, center_x - _DELTA * sigma))
    y0 = int(max(0, center_y - _DELTA * sigma))
    x1 = int(min(width, center_x + _DELTA * sigma))
    y1 = int(min(height, center_y + _DELTA * sigma))

    if x0 >= x1 or y0 >= y1:
        return  # centre is outside the image

    # Vectorised computation over the local patch (much faster than a loop).
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    # Shape: (patch_h, patch_w)
    dist_sq = (xs[np.newaxis, :] - center_x) ** 2 + \
               (ys[:, np.newaxis] - center_y) ** 2
    exponent = dist_sq / (2.0 * sigma * sigma)

    # Only write pixels whose Gaussian value is above the threshold.
    gauss = np.where(exponent <= _TH, np.exp(-exponent), 0.0).astype(np.float32)

    # Keep the maximum response when multiple segments overlap.
    patch = heatmap[y0:y1, x0:x1]
    np.maximum(patch, gauss, out=patch)


def make_endpoint_heatmaps(endpoints: list,
                           height: int,
                           width: int,
                           sigma: float = 16.0) -> np.ndarray:
    """
    Build a 2-channel heatmap from a list of line-segment endpoints.

    Parameters
    ----------
    endpoints : list of (x1, y1, x2, y2) tuples / lists
    height    : image height in pixels
    width     : image width in pixels
    sigma     : Gaussian standard deviation in pixels

    Returns
    -------
    np.ndarray of shape (2, H, W), dtype float32, values in [0, 1].
        Channel 0 = heatmap for first endpoint  (p1)
        Channel 1 = heatmap for second endpoint (p2)
    """
    hm = np.zeros((2, height, width), dtype=np.float32)

    for seg in endpoints:
        x1, y1, x2, y2 = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
        put_heatmap(hm[0], x1, y1, sigma=sigma)
        put_heatmap(hm[1], x2, y2, sigma=sigma)

    return hm

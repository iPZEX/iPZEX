"""
loss.py
-------
Loss functions for 2-channel Gaussian heatmap regression.

Two losses are provided:

  1. MSEHeatmapLoss     – plain mean-squared-error over all pixels
  2. WeightedMSELoss    – foreground pixels are up-weighted so that the
                          sparse Gaussian peaks drive the gradient more than
                          the large background region

Both expect:
  pred   : (B, 2, H, W)  model output after Sigmoid, values in [0, 1]
  target : (B, 2, H, W)  ground-truth heatmaps, values in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEHeatmapLoss(nn.Module):
    """Standard pixel-wise mean squared error."""

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE that assigns higher importance to foreground pixels.

    Pixels where target > `fg_threshold` are weighted by `fg_weight`;
    all other pixels receive weight 1.0.

    Parameters
    ----------
    fg_threshold : float, threshold above which a pixel is "foreground"
    fg_weight    : float, weight applied to foreground pixels
    """

    def __init__(self, fg_threshold: float = 0.1, fg_weight: float = 10.0):
        super().__init__()
        self.fg_threshold = fg_threshold
        self.fg_weight = fg_weight

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        weights = torch.where(
            target > self.fg_threshold,
            torch.full_like(target, self.fg_weight),
            torch.ones_like(target),
        )
        loss = weights * (pred - target) ** 2
        return loss.mean()

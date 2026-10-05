"""Privileged-3D auxiliary supervision on the student's visual tokens (training only, removed at inference).

Level A (oracle): the target is the simulator's metric depth of the same agent view the policy sees, brought
through the same centre crop as the policy input and expressed as normalised log-depth on a 64 x 64 grid.
A light head reads the 16 x 16 visual tokens of one LLM layer and predicts a 4 x 4 block per token.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 16  # visual tokens per side (224 / 14)
SUB = 4  # depth cells predicted per token side -> 64 x 64 map
D_MIN, D_MAX = 0.2, 4.0  # metres; LIBERO agent-view depth lies in about 0.7 .. 2.8
LOG_MU, LOG_SIGMA = math.log(1.2), 0.4


def depth_target(depth: torch.Tensor, crop_scale: float = 0.9, center_crop: bool = True) -> torch.Tensor:
    """Metric depth in the policy frame (B, H, W) -> normalised log-depth target (B, 64, 64).

    Mirrors the policy's preprocessing geometry: centre crop of relative area `crop_scale`, resized back.
    """
    d = depth.float().unsqueeze(1)
    if center_crop:
        side = math.sqrt(crop_scale)
        theta = torch.tensor([[side, 0.0, 0.0], [0.0, side, 0.0]], device=d.device).repeat(d.shape[0], 1, 1)
        grid = F.affine_grid(theta, (d.shape[0], 1, 224, 224), align_corners=False)
        d = F.grid_sample(d, grid, mode="bilinear", padding_mode="border", align_corners=False)
    d = F.adaptive_avg_pool2d(d, GRID * SUB).squeeze(1)
    return (torch.log(d.clamp(D_MIN, D_MAX)) - LOG_MU) / LOG_SIGMA


def to_metres(norm_log_depth: torch.Tensor) -> torch.Tensor:
    return torch.exp(norm_log_depth * LOG_SIGMA + LOG_MU)


class DepthHead(nn.Module):
    """LayerNorm -> MLP on each visual token, predicting its 4 x 4 block of the depth map."""

    def __init__(self, dim: int = 4096, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, SUB * SUB))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, 256, D) visual-token hidden states -> (B, 64, 64) normalised log-depth."""
        b = tokens.shape[0]
        x = self.net(tokens.float()).view(b, GRID, GRID, SUB, SUB)
        return x.permute(0, 1, 3, 2, 4).reshape(b, GRID * SUB, GRID * SUB)


def depth_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


@torch.no_grad()
def depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Per-sample errors: L1 in normalised log-depth and mean absolute relative error in metres."""
    p, t = to_metres(pred), to_metres(target)
    return {"l1": (pred - target).abs().flatten(1).mean(1), "abs_rel": ((p - t).abs() / t).flatten(1).mean(1)}

"""Geometry of a 4-joint pipetting arm, shared by training, the server, and the page.

Coordinates are metres, z up, with the arm's base at the origin on the bench. Joints:
  q0  base yaw about z
  q1  shoulder elevation of the upper arm above horizontal
  q2  elbow angle of the forearm relative to the upper arm
  q3  wrist angle of the pipette relative to the forearm
The pipette is a straight tool of length TOOL; "pointing down" means q1 + q2 + q3 = -pi/2.
"""

from __future__ import annotations

import math

import numpy as np
import torch

BASE_HEIGHT = 0.20
UPPER_ARM = 0.30
FOREARM = 0.28
TOOL = 0.12
# The pipette points straight down whenever it is used, so this is the tool's absolute angle.
TOOL_DOWN = -math.pi / 2

# Workspace the model is trained on: the bench area the labware occupies, plus margin.
RADIUS_RANGE = (0.14, 0.52)
YAW_RANGE = (-math.radians(110), math.radians(110))
TIP_Z_RANGE = (-0.01, 0.26)


def forward_torch(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pipette tip position (N, 3) and tool angle (N,) for joint angles q (N, 4)."""
    q0, q1, q2, q3 = q.unbind(-1)
    a2 = q1 + q2
    a3 = a2 + q3
    r = UPPER_ARM * torch.cos(q1) + FOREARM * torch.cos(a2) + TOOL * torch.cos(a3)
    z = (
        BASE_HEIGHT
        + UPPER_ARM * torch.sin(q1)
        + FOREARM * torch.sin(a2)
        + TOOL * torch.sin(a3)
    )
    tip = torch.stack([r * torch.cos(q0), r * torch.sin(q0), z], dim=-1)
    return tip, a3


def analytic_ik(tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form elbow-up solution with the pipette pointing down.

    Returns (q (N, 4), reachable (N,)). Training uses it only to pick the elbow-up branch;
    the model is fit through forward kinematics, so it never sees the formula itself.
    """
    x, y, z = tip[:, 0], tip[:, 1], tip[:, 2]
    q0 = np.arctan2(y, x)
    r = np.hypot(x, y)
    wz = z + TOOL - BASE_HEIGHT
    d = (r**2 + wz**2 - UPPER_ARM**2 - FOREARM**2) / (2 * UPPER_ARM * FOREARM)
    reachable = np.abs(d) <= 0.995
    d = np.clip(d, -1.0, 1.0)
    q2 = -np.arccos(d)
    q1 = np.arctan2(wz, r) - np.arctan2(
        FOREARM * np.sin(q2), UPPER_ARM + FOREARM * np.cos(q2)
    )
    q3 = TOOL_DOWN - q1 - q2
    return np.stack([q0, q1, q2, q3], axis=-1), reachable


def sample_workspace(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform-ish reachable tip positions in the trained workspace, shape (n, 3)."""
    out: list[np.ndarray] = []
    have = 0
    while have < n:
        m = 2 * (n - have) + 64
        r = np.sqrt(rng.uniform(RADIUS_RANGE[0] ** 2, RADIUS_RANGE[1] ** 2, m))
        yaw = rng.uniform(*YAW_RANGE, m)
        z = rng.uniform(*TIP_Z_RANGE, m)
        pts = np.stack([r * np.cos(yaw), r * np.sin(yaw), z], axis=-1)
        _, ok = analytic_ik(pts)
        pts = pts[ok]
        out.append(pts)
        have += len(pts)
    return np.concatenate(out)[:n]


class IKNet(torch.nn.Module):
    """Tip position -> joint angles. Predicts (sin, cos) per joint so angles never wrap."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("center", torch.tensor([0.30, 0.0, 0.12]))
        self.register_buffer("scale", torch.tensor([0.30, 0.45, 0.15]))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, 8),
        )

    def forward(self, tip: torch.Tensor) -> torch.Tensor:
        out = self.net((tip - self.center) / self.scale).view(-1, 4, 2)
        return torch.atan2(out[..., 0], out[..., 1])


def refine(q: torch.Tensor, tip: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
    """A few Adam steps on tip-position and pipette-tilt error, differentiated through FK."""
    q = q.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred, angle = forward_torch(q)
        loss = ((pred - tip) ** 2).sum(-1).mean() + 0.01 * (
            (angle - TOOL_DOWN) ** 2
        ).mean()
        loss.backward()
        opt.step()
    return q.detach()

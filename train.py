"""Train the inverse-kinematics network and write ik.pt.

    python train.py [--out ik.pt] [--samples 200000] [--epochs 30]

The loss is measured through differentiable forward kinematics: how far the predicted joint
angles put the pipette tip from the target, plus how far the pipette tilts from vertical. A
small angle term against the closed-form elbow-up solution keeps the network on one branch;
without it, targets near the boundary between elbow-up and elbow-down average the two.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from kinematics import TOOL_DOWN, IKNet, analytic_ik, forward_torch, sample_workspace

SEED = 7


def tip_error_mm(model: IKNet, tips: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        pred, _ = forward_torch(model(tips))
    err = (pred - tips).norm(dim=-1) * 1000
    return float(err.mean()), float(err.quantile(0.99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ik.pt")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tips_np = sample_workspace(args.samples, rng)
    q_ref_np, _ = analytic_ik(tips_np)
    tips = torch.tensor(tips_np, dtype=torch.float32)
    q_ref = torch.tensor(q_ref_np, dtype=torch.float32)
    held_out = torch.tensor(sample_workspace(10_000, rng), dtype=torch.float32)

    model = IKNet(args.width)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    batch = 1024
    steps = args.epochs * (len(tips) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(tips))
        for i in range(0, len(tips) - batch + 1, batch):
            idx = perm[i : i + batch]
            q = model(tips[idx])
            pred, angle = forward_torch(q)
            position = ((pred - tips[idx]) ** 2).sum(-1).mean() * 100
            upright = ((angle - TOOL_DOWN) ** 2).mean()
            branch = (1 - torch.cos(q - q_ref[idx])).mean()
            loss = position + upright + 0.1 * branch
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            mean, p99 = tip_error_mm(model, held_out)
            print(
                f"epoch {epoch + 1:>3}  held-out tip error mean {mean:.2f} mm, p99 {p99:.2f} mm",
                flush=True,
            )

    mean, p99 = tip_error_mm(model, held_out)
    params = sum(p.numel() for p in model.parameters())
    torch.save(
        {"width": args.width, "state_dict": model.state_dict(), "tip_error_mm": mean},
        args.out,
    )
    print(
        f"trained {params:,} params in {time.perf_counter() - t0:.0f}s: mean {mean:.2f} mm, p99 {p99:.2f} mm"
    )


if __name__ == "__main__":
    main()

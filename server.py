"""Plan pipetting moves with a learned inverse-kinematics model and serve the WebGL page.

HTTP API:
  GET  /             the page (index.html)
  GET  /api/layout   labware positions, arm dimensions, and model info for the page
  POST /api/plan     {"wells": ["A1", "C5"], "first_tip": 0, "refine": true} ->
                     {"frames": [{"q": [...], "tip": [...]}...], "events": [...], "stats": {...}}

Each transfer picks a fresh tip, aspirates from the reservoir, dispenses into one well, and
ejects the tip into the trash. Every move is a straight line for the pipette tip, sampled
every STEP_M; the whole plan is solved in one batched forward pass of the network.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import kinematics as K

MODEL_PATH = os.environ.get("IK_MODEL_PATH", "/app/ik.pt")
INDEX_HTML = Path(__file__).with_name("index.html")
PORT = int(os.environ.get("PORT", "8080"))

STEP_M = 0.005
SAFE_Z = 0.15
REFINE_STEPS = 25
MAX_WELLS = 12

ROWS = "ABCDEFGH"
PITCH = 0.009
# Labware positions on the bench (metres). Rows A-H run away from the arm along x and
# columns 1-12 run along y, so every well sits inside the trained workspace.
PLATE_CENTER = (0.40, 0.11)
TIP_RACK_CENTER = (0.40, -0.11)
RESERVOIR = {"center": (0.21, 0.0), "size": (0.03, 0.11), "liquid_z": 0.012}
TRASH = {"center": (0.20, -0.25), "size": (0.06, 0.06), "drop_z": 0.10}
WELL_DISPENSE_Z = 0.013
TIP_PICK_Z = 0.062


def grid_point(center: tuple[float, float], row: int, col: int) -> tuple[float, float]:
    return (center[0] + (row - 3.5) * PITCH, center[1] + (col - 5.5) * PITCH)


def parse_well(name: str) -> tuple[int, int]:
    m = re.fullmatch(r"([A-H])(1[0-2]|[1-9])", name.strip().upper())
    if not m:
        raise HTTPException(400, f"not a 96-well name: {name!r}")
    return ROWS.index(m.group(1)), int(m.group(2)) - 1


def line(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / STEP_M)) + 1)
    return a + (b - a) * np.linspace(0, 1, n)[:, None]


class Planner:
    """Builds the Cartesian path for a protocol and tags the points where actions happen."""

    def __init__(self, start: np.ndarray) -> None:
        super().__init__()
        self.points: list[np.ndarray] = [start[None]]
        self.events: list[dict[str, Any]] = []

    @property
    def here(self) -> np.ndarray:
        return self.points[-1][-1]

    def move(self, target: tuple[float, float, float]) -> None:
        self.points.append(line(self.here, np.array(target))[1:])

    def travel(self, xy: tuple[float, float], z: float) -> None:
        """Up to the safe height, across, and down, so the tip never sweeps through labware."""
        self.move((self.here[0], self.here[1], SAFE_Z))
        self.move((xy[0], xy[1], SAFE_Z))
        self.move((xy[0], xy[1], z))

    def event(self, kind: str, **detail: Any) -> None:
        self.events.append(
            {"frame": sum(len(p) for p in self.points) - 1, "kind": kind, **detail}
        )

    def path(self) -> np.ndarray:
        return np.concatenate(self.points)


def build_protocol(wells: list[tuple[int, int]], first_tip: int) -> Planner:
    home = np.array([0.30, 0.0, SAFE_Z])
    p = Planner(home)
    for i, (row, col) in enumerate(wells):
        tip = (first_tip + i) % 96
        tip_rc = (tip % 8, tip // 8)
        p.travel(grid_point(TIP_RACK_CENTER, *tip_rc), TIP_PICK_Z)
        p.event("pick_tip", tip=f"{ROWS[tip_rc[0]]}{tip_rc[1] + 1}")
        # Spread aspirations along the trough so successive transfers do not stack.
        rx, ry = RESERVOIR["center"]
        y = ry + ((i % 5) - 2) * 0.018
        p.travel((rx, y), RESERVOIR["liquid_z"])
        p.event("aspirate")
        p.travel(grid_point(PLATE_CENTER, row, col), WELL_DISPENSE_Z)
        p.event("dispense", well=f"{ROWS[row]}{col + 1}")
        p.travel(TRASH["center"], TRASH["drop_z"])
        p.event("eject_tip")
    p.move(tuple(home))
    return p


def tip_errors_mm(
    q: torch.Tensor, targets: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        tip, angle = K.forward_torch(q)
    pos = ((tip - targets).norm(dim=-1) * 1000).numpy()
    tilt = np.degrees((angle - K.TOOL_DOWN).abs().numpy())
    return pos, tilt


def summarize(pos: np.ndarray, tilt: np.ndarray) -> dict[str, float]:
    return {
        "mean_mm": round(float(pos.mean()), 3),
        "max_mm": round(float(pos.max()), 3),
        "max_tilt_deg": round(float(tilt.max()), 3),
    }


state: dict[str, Any] = {}


def load_model() -> K.IKNet:
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = K.IKNet(ckpt["width"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    state["trained_error_mm"] = ckpt.get("tip_error_mm")
    state["params"] = sum(p.numel() for p in model.parameters())
    return model


app = FastAPI()
state["model"] = load_model()
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
# The first autograd pass pays seconds of one-time setup; take it at boot, not on a request.
K.refine(
    state["model"](torch.tensor([[0.3, 0.0, 0.1]])),
    torch.tensor([[0.3, 0.0, 0.1]]),
    2,
    1e-3,
)


class PlanRequest(BaseModel):
    wells: list[str] = Field(min_length=1, max_length=MAX_WELLS)
    first_tip: int = Field(default=0, ge=0, le=95)
    refine: bool = True


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/api/layout")
async def layout() -> JSONResponse:
    return JSONResponse(
        {
            "arm": {
                "base_height": K.BASE_HEIGHT,
                "upper_arm": K.UPPER_ARM,
                "forearm": K.FOREARM,
                "tool": K.TOOL,
            },
            "plate": {"center": PLATE_CENTER, "pitch": PITCH, "rows": 8, "cols": 12},
            "tip_rack": {
                "center": TIP_RACK_CENTER,
                "pitch": PITCH,
                "rows": 8,
                "cols": 12,
            },
            "reservoir": RESERVOIR,
            "trash": TRASH,
            "safe_z": SAFE_Z,
            "model": {
                "params": state["params"],
                "trained_error_mm": state["trained_error_mm"],
                "refine_steps": REFINE_STEPS,
            },
        }
    )


@app.post("/api/plan")
def plan(req: PlanRequest) -> JSONResponse:
    wells = [parse_well(w) for w in req.wells]
    t0 = time.perf_counter()
    protocol = build_protocol(wells, req.first_tip)
    targets = torch.tensor(protocol.path(), dtype=torch.float32)
    t1 = time.perf_counter()
    with torch.no_grad():
        q_model = state["model"](targets)
    t2 = time.perf_counter()
    model_err = tip_errors_mm(q_model, targets)
    q = K.refine(q_model, targets, REFINE_STEPS, 2e-3) if req.refine else q_model
    t3 = time.perf_counter()
    final_err = tip_errors_mm(q, targets)
    with torch.no_grad():
        tips, _ = K.forward_torch(q)
    frames = [
        {"q": [round(v, 5) for v in qi], "tip": [round(v, 5) for v in ti]}
        for qi, ti in zip(q.tolist(), tips.tolist())
    ]
    return JSONResponse(
        {
            "frames": frames,
            "events": protocol.events,
            "stats": {
                "waypoints": len(frames),
                "plan_ms": round((t1 - t0) * 1000, 2),
                "model_ms": round((t2 - t1) * 1000, 2),
                "refine_ms": round((t3 - t2) * 1000, 2) if req.refine else 0.0,
                "model": summarize(*model_err),
                "final": summarize(*final_err),
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)

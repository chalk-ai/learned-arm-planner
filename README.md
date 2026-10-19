# A learned motion planner for a pipetting arm

Pick wells on a 96-well plate and watch a small neural network plan every joint angle for a
robot arm. The arm then runs the protocol in a three.js scene in your browser: pick a fresh tip,
aspirate from a reservoir, dispense into each well, and eject the tip. The planner is one Python
service deployed as a [Chalk](https://chalk.ai) `ScalingGroup`.

```
wells ─▶ Cartesian path (5 mm steps) ─▶ IK network, one batch ─▶ FK-gradient refinement ─▶ joint trajectory ─▶ WebGL
```

## How it works

- **The arm** (`kinematics.py`) has four joints: base yaw, shoulder, elbow, and wrist, plus a
  vertical pipette. Forward kinematics, from joint angles to tip position, is a few lines of
  differentiable PyTorch.
- **The network** (`IKNet`) learns the inverse: tip position to joint angles. `train.py` never
  shows it solved joint angles as targets. It scores the angles the network proposes by running
  them through forward kinematics and measuring where the tip lands and how far the pipette tilts.
  A small extra term keeps it on the elbow-up branch.
- **The planner** (`server.py`) turns a list of wells into straight-line moves for the tip,
  sampled every 5 mm. It solves every waypoint in one batched forward pass, then runs 25 Adam
  steps through forward kinematics to polish the result.

On the reference layout, a three-well protocol is about 1,200 waypoints:

| | Tip error, mean | Tip error, max | Pipette tilt, max | Time |
| --- | --- | --- | --- | --- |
| Network alone | 1.1 mm | 3.1 mm | 0.8° | about 3 ms |
| After refinement | 0.18 mm | 0.46 mm | 0.12° | about 18 ms more |

## Run it

```bash
# Train (about 2.5 minutes on a large CPU box; any machine with PyTorch works).
uv run --with torch --with numpy python train.py --out ik.pt

# Deploy. Credentials come from your existing `chalk login`, or CHALK_CLIENT_ID / CHALK_CLIENT_SECRET.
./deploy.py
```

The deploy script builds a CPU-only PyTorch image, copies in the server, page and kinematics
code, ships `ik.pt` on a volume, and prints the URL of your planner. Open it, pick wells, and
press **Plan and run**.

To run locally instead: `pip install torch numpy fastapi "uvicorn[standard]"`, then
`IK_MODEL_PATH=ik.pt PORT=8080 python server.py`.

## API

`POST /api/plan` with `{"wells": ["A1", "D6"], "first_tip": 0, "refine": true}` returns:

- `frames`: one `{q, tip}` per waypoint, where `q` holds the four joint angles in radians and
  `tip` is where the solved angles put the pipette tip.
- `events`: the frame indexes where tips are picked, liquid is aspirated and dispensed, and tips
  are ejected.
- `stats`: waypoint count, timings, and tip error for the network alone and after refinement.

`GET /api/layout` returns the arm dimensions and labware positions the page draws.

## Adapting it

Change the link lengths in `kinematics.py` and the labware positions in `server.py`, retrain,
and redeploy. For a 6-joint arm, extend `forward_torch`, widen the network's output, and add
orientation terms to the loss. The training recipe stays the same.

## License

Apache-2.0. The page loads [three.js](https://threejs.org) (MIT) from a CDN.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.

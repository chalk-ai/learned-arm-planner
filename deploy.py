#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the learned arm planner (server.py + index.html + ik.pt) as a CPU ScalingGroup.

Train first, anywhere PyTorch runs (about 2.5 minutes on a large CPU box):
    python train.py --out ik.pt
then deploy, with credentials from `chalk config` (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="arm-planner")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--weights", default=str(HERE / "ik.pt"))
    args = parser.parse_args()
    if not Path(args.weights).exists():
        raise SystemExit(
            f"{args.weights} not found; run `python train.py --out ik.pt` first"
        )

    load_chalk_env(args.env_id)
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        .run_commands(
            "pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu"
        )
        .pip_install(["fastapi==0.116.1", "uvicorn[standard]==0.35.0", "numpy<3"])
        .add_local_file(
            str(HERE / "kinematics.py"), "/app/kinematics.py", strategy="copy"
        )
        .add_local_file(str(HERE / "server.py"), "/app/server.py", strategy="copy")
        .add_local_file(str(HERE / "index.html"), "/app/index.html", strategy="copy")
        # The weights are larger than an inline copy allows, so they ride in on a volume. It
        # mounts its own directory, which keeps it from shadowing the copied files in /app.
        .add_local_file(args.weights, "/model/ik.pt")
        .workdir("/app")
    )
    sg = ScalingGroup(
        image=image,
        name=args.name,
        cpu="2",
        memory="4Gi",
        env={"IK_MODEL_PATH": "/model/ik.pt", "TORCH_THREADS": "2"},
        entrypoint=["python", "/app/server.py"],
        port=8080,
        readiness_probe=ReadinessProbe.http("/api/layout"),
        min_replicas=1,
        max_replicas=2,
    ).deploy(build_timeout=1800.0, ready_timeout=900.0)
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()

#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the learned arm planner (server.py + index.html + ik.pt) as a CPU ScalingGroup.

Train first, anywhere PyTorch runs (about 2.5 minutes on a large CPU box):
    python train.py --out ik.pt
then deploy, with credentials from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    if not Path(str(HERE / 'ik.pt')).exists():
        raise SystemExit(
            f"{str(HERE / 'ik.pt')} not found; run `python train.py --out ik.pt` first"
        )

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
        .add_local_file(str(HERE / 'ik.pt'), "/model/ik.pt")
        .workdir("/app")
    )
    sg = ScalingGroup(
        image=image,
        name='arm-planner',
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

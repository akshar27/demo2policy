"""Quality-assurance artifacts for a processed video: what the tracker saw, and what the pipeline made of it."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def write_qa(out_dir: str | Path, report: dict, poses: list, demos: list) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    good = [(i, p) for i, p in enumerate(poses) if p is not None]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    if good:
        idx = np.array([i for i, _ in good])
        ax[0].plot(idx, [p.pinch for _, p in good], ".", ms=3)
        ax[0].set(title="thumb-index distance / hand size (0 = touching)", xlabel="frame")
        ax[1].plot(idx, [p.reproj_px for _, p in good], ".", ms=3)
        ax[1].set(title="reprojection error of the 3D fit (px)", xlabel="frame")
        ax[1].axhline(12, color="r", ls="--", lw=1)
        pts = np.array([p.pinch_pt for _, p in good])
        for k, name in enumerate("xyz"):
            ax[2].plot(idx, pts[:, k], label=name)
        ax[2].legend()
        ax[2].set(title="pinch point in the camera-aligned world frame (m)", xlabel="frame")
    fig.tight_layout()
    fig.savefig(out / "tracking.png", dpi=80)
    plt.close(fig)
    if demos:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for d in demos[:12]:
            ax[0].plot(d.ee_target[:, 0], d.ee_target[:, 1], alpha=0.6)
            ax[1].plot(d.ee_target[:, 2], alpha=0.6)
        ax[0].set(title="retargeted EE paths (top view, robot frame)", xlabel="x (m)", ylabel="y (m)", aspect="equal")
        ax[1].set(title="EE height (m)", xlabel="step")
        fig.tight_layout()
        fig.savefig(out / "retargeted.png", dpi=80)
        plt.close(fig)
    return out

"""Monocular depth as an optional second opinion on hand depth.

Depth Anything V2 predicts *relative inverse depth* (disparity up to an unknown affine map), not metres. It becomes useful
for lifting only after `geometry.depth_fusion` fits that affine map against a metric reference - and a reference that is
itself biased (e.g. a wrong hand-size assumption) hands its bias on to the calibrated depth. See docs/results/depth_check.md.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator(Protocol):
    def disparity(self, frame_rgb: np.ndarray) -> np.ndarray:
        """(H,W) relative inverse depth, larger = closer, same resolution as the frame."""
        ...


class DepthAnythingV2:
    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import pipeline

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._pipe = pipeline("depth-estimation", model=model, device=device)

    def disparity(self, frame_rgb: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        h, w = frame_rgb.shape[:2]
        out = self._pipe(Image.fromarray(frame_rgb))["predicted_depth"]  # (1,H',W') or (H',W')
        d = out if out.ndim == 4 else out[None]
        d = torch.nn.functional.interpolate(d.float().unsqueeze(0) if d.ndim == 3 else d.float(), size=(h, w),
                                            mode="bicubic", align_corners=False)
        return d[0, 0].cpu().numpy()


def sample_disparity(disp: np.ndarray, px: np.ndarray, radius: int = 6) -> float:
    """Median disparity in a small window around a pixel (robust to single-pixel noise)."""
    h, w = disp.shape
    x, y = int(round(px[0])), int(round(px[1]))
    x0, x1, y0, y1 = max(x - radius, 0), min(x + radius + 1, w), max(y - radius, 0), min(y + radius + 1, h)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"pixel {px} is outside the {w}x{h} image")
    return float(np.median(disp[y0:y1, x0:x1]))

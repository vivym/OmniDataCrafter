import math

import torch

from .cut_detector import CutDetector


def compute_score(frame1: torch.Tensor, frame2: torch.Tensor) -> float:
    diff = (frame1 - frame2).abs().flatten(1).mean(dim=1)
    diff: list[float] = diff.tolist()

    h_diff = diff[0] / (2 * math.pi) * 179.
    s_diff = diff[1] * 255.
    v_diff = diff[2] * 255.

    return (h_diff + s_diff + v_diff) / 3.


class ContentDetector(CutDetector):
    """Detects fast cuts/slow fades using HSV histogram analysis."""

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_len: int = 15,
    ):
        super().__init__()

        self._threshold = threshold
        self._min_scene_len = min_scene_len

        self._last_scene_cut: int | None = None
        self._last_frame: torch.Tensor | None = None

    def process_frame(self, frame_idx: int, frame: torch.Tensor) -> tuple[int | None, float]:
        if self._last_scene_cut is None:
            self._last_scene_cut = frame_idx

        if self._last_frame is None:
            self._last_frame = frame

        if (frame_idx - self._last_scene_cut) < self._min_scene_len:
            return None, 0.0

        score = compute_score(frame, self._last_frame)
        self._last_frame = frame

        # We consider any frame over the threshold a new scene, but only if
        # the minimum scene length has been reached (otherwise it is ignored).
        if score >= self._threshold:
            self._last_scene_cut = frame_idx
            return frame_idx, score
        else:
            return None, score

"""Geometry helpers for coordinate tracking and box math.

The core problem this solves: preprocessing may scale and/or rotate the image, so
OCR runs in *processed* coordinates. Every geometric op is captured in a single
2x3 affine matrix (original -> processed). Inverting it maps OCR boxes back to
ORIGINAL image coordinates, which is what every downstream consumer expects.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from app.schemas.ocr import BBox, Polygon


def identity_affine() -> np.ndarray:
    """2x3 identity affine."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def compose_affine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affines so ``second`` is applied after ``first``.

    Returns a 2x3 matrix equivalent to ``second @ first`` in homogeneous form.
    """
    f = np.vstack([first, [0.0, 0.0, 1.0]])
    s = np.vstack([second, [0.0, 0.0, 1.0]])
    return (s @ f)[:2, :]


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine matrix."""
    return cv2.invertAffineTransform(matrix.astype(np.float64))


def apply_affine_to_points(points: List[List[float]], matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to an Nx2 set of points; returns Nx2 array."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts
    homog = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    return (matrix @ homog.T).T


def polygon_to_bbox(polygon: Polygon, clip_w: int | None = None,
                    clip_h: int | None = None) -> BBox:
    """Axis-aligned integer bbox [x1,y1,x2,y2] enclosing a polygon."""
    arr = np.asarray(polygon, dtype=np.float64)
    x1, y1 = arr[:, 0].min(), arr[:, 1].min()
    x2, y2 = arr[:, 0].max(), arr[:, 1].max()
    if clip_w is not None:
        x1, x2 = max(0, x1), min(clip_w, x2)
    if clip_h is not None:
        y1, y2 = max(0, y1), min(clip_h, y2)
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def merge_boxes(boxes: List[BBox]) -> BBox:
    """Merge boxes into one enclosing box (only sound for same-line boxes)."""
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]

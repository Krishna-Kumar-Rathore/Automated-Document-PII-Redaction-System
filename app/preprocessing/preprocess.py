"""Image preprocessing with exact coordinate tracking.

Each geometric step contributes to a single affine matrix (original -> processed).
Non-geometric steps (denoise) leave it unchanged. OCR runs on the processed image;
the OCR stage uses the inverse of this matrix to report boxes in original coords.

Kept deliberately conservative: PP-OCR usually prefers a natural image, so there
is no aggressive binarisation. Deskew is bounded to avoid mis-rotating sparse or
newspaper-like inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config.loader import Config
from app.utils.geometry import compose_affine, identity_affine
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessResult:
    image: np.ndarray            # processed image (fed to OCR)
    transform: np.ndarray        # 2x3 affine, ORIGINAL -> PROCESSED
    original_size: tuple         # (width, height) of the input image


class Preprocessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg.preprocessing

    def run(self, image: np.ndarray) -> PreprocessResult:
        h0, w0 = image.shape[:2]
        transform = identity_affine()
        out = image

        if self.cfg.get("upscale_if_small", True):
            out, transform = self._maybe_scale(out, transform)

        if self.cfg.get("denoise", True):
            out = self._denoise(out)  # geometry unchanged

        if self.cfg.get("deskew", True):
            out, transform = self._maybe_deskew(out, transform)

        return PreprocessResult(image=out, transform=transform, original_size=(w0, h0))

    # -- individual steps ---------------------------------------------------

    def _maybe_scale(self, image: np.ndarray, transform: np.ndarray):
        h, w = image.shape[:2]
        long_side = max(h, w)
        min_long = self.cfg.get("min_long_side", 1000)
        max_long = self.cfg.get("max_long_side", 2600)

        scale = 1.0
        if long_side < min_long:
            scale = min_long / long_side
        elif long_side > max_long:
            scale = max_long / long_side
        if abs(scale - 1.0) < 1e-3:
            return image, transform

        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        scaled = cv2.resize(image, (new_w, new_h), interpolation=interp)
        s = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0]], dtype=np.float64)
        logger.info("Scaled image x%.3f -> %dx%d", scale, new_w, new_h)
        return scaled, compose_affine(transform, s)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        # Edge-preserving; keeps text crisp for OCR.
        return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

    def _maybe_deskew(self, image: np.ndarray, transform: np.ndarray):
        angle = self._estimate_skew(image)
        min_a = self.cfg.get("deskew_min_angle", 0.3)
        max_a = self.cfg.get("deskew_max_angle", 15.0)
        if angle is None or not (min_a <= abs(angle) <= max_a):
            return image, transform

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, rot, (w, h), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.info("Deskewed by %.2f deg", angle)
        return rotated, compose_affine(transform, rot)

    @staticmethod
    def _estimate_skew(image: np.ndarray) -> float | None:
        """Estimate skew via the min-area rectangle of foreground text pixels."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return None
        angle = cv2.minAreaRect(coords[:, ::-1])[-1]  # (x,y) order for minAreaRect
        # Normalise OpenCV's [-90,0) / [0,90) convention to a small +/- angle.
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        return float(angle)

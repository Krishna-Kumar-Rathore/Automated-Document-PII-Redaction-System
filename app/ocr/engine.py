"""OCR engine abstraction.

``OCREngine`` is the interface the rest of the pipeline depends on; ``RapidOCREngine``
is the concrete implementation. Swapping OCR backends (e.g. to a PP-OCRv6 build or
a VL model for handwriting later) means adding one class, not touching the pipeline.

Responsibilities:
  * run detection+recognition on the *processed* image,
  * map every detection back to ORIGINAL image coordinates via the inverse affine,
  * emit typed OCRToken objects (line granularity).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from app.config.loader import Config
from app.utils.geometry import (apply_affine_to_points, invert_affine,
                                 polygon_to_bbox)
from app.utils.logging import get_logger
from app.schemas.ocr import OCRToken

logger = get_logger(__name__)


class OCREngine(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, transform: np.ndarray,
               original_size: Tuple[int, int]) -> Tuple[List[OCRToken], float]:
        """Return (tokens in original coords, elapsed_ms)."""


class RapidOCREngine(OCREngine):
    def __init__(self, cfg: Config):
        from rapidocr_onnxruntime import RapidOCR  # imported lazily

        self.text_score = cfg.ocr.get("text_score", 0.5)
        kwargs = {}
        if cfg.ocr.get("det_model_path"):
            kwargs["det_model_path"] = cfg.ocr.get("det_model_path")
        if cfg.ocr.get("rec_model_path"):
            kwargs["rec_model_path"] = cfg.ocr.get("rec_model_path")
        self._engine = RapidOCR(**kwargs)
        logger.info("RapidOCR engine ready (text_score >= %.2f)", self.text_score)

    def detect(self, image: np.ndarray, transform: np.ndarray,
               original_size: Tuple[int, int]) -> Tuple[List[OCRToken], float]:
        w0, h0 = original_size
        inv = invert_affine(transform)  # processed -> original

        start = time.perf_counter()
        raw, _ = self._engine(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        tokens: List[OCRToken] = []
        for idx, item in enumerate(raw or []):
            poly_proc, text, conf = item[0], item[1], float(item[2])
            if conf < self.text_score or not text.strip():
                continue
            # Map the 4 polygon points from processed -> original coordinates.
            poly_orig = apply_affine_to_points(poly_proc, inv).tolist()
            bbox = polygon_to_bbox(poly_orig, clip_w=w0, clip_h=h0)
            tokens.append(OCRToken(
                id=len(tokens) + 1,
                text=text,
                confidence=round(conf, 4),
                bbox=bbox,
                polygon=[[round(x, 1), round(y, 1)] for x, y in poly_orig],
                granularity="line",
            ))
        logger.info("OCR produced %d line tokens in %.0f ms", len(tokens), elapsed_ms)
        return tokens, elapsed_ms


def build_engine(cfg: Config) -> OCREngine:
    """Factory: choose engine from config."""
    name = cfg.ocr.get("engine", "rapidocr")
    if name == "rapidocr":
        return RapidOCREngine(cfg)
    raise ValueError(f"Unknown OCR engine: {name}")

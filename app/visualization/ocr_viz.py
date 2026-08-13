"""Draw OCR detections on the original image for visual verification."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from app.schemas.ocr import OCRToken


def draw_ocr_boxes(image: np.ndarray, tokens: List[OCRToken],
                   color=(0, 140, 255), thickness: int = 2,
                   show_conf: bool = True) -> np.ndarray:
    """Return a copy of ``image`` with each token's bbox and id/confidence drawn."""
    canvas = image.copy()
    for tok in tokens:
        x1, y1, x2, y2 = tok.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"#{tok.id}"
        if show_conf:
            label += f" {tok.confidence:.2f}"
        y_text = y1 - 5 if y1 - 5 > 10 else y2 + 15
        cv2.putText(canvas, label, (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas

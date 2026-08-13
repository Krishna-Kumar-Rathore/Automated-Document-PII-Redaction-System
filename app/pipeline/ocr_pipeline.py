"""Phase 2 pipeline slice: image in -> structured OCRResult out.

Later phases (reconstruction, NER, mapping, redaction) will extend this into the
full document pipeline. Keeping the stages explicit and separable is deliberate.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config.loader import Config, load_config
from app.ocr.engine import OCREngine, build_engine
from app.preprocessing.preprocess import Preprocessor
from app.schemas.ocr import ImageMeta, OCRResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


class OCRPipeline:
    def __init__(self, cfg: Optional[Config] = None,
                 engine: Optional[OCREngine] = None):
        self.cfg = cfg or load_config()
        self.preprocessor = Preprocessor(self.cfg)
        # Engine construction loads models, so allow injection (tests / reuse).
        self.engine = engine or build_engine(self.cfg)

    def run(self, image_path: str | Path) -> OCRResult:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        h0, w0 = image.shape[:2]
        wall_start = time.perf_counter()

        pre = self.preprocessor.run(image)
        tokens, ocr_ms = self.engine.detect(
            pre.image, pre.transform, pre.original_size
        )
        total_ms = (time.perf_counter() - wall_start) * 1000.0

        result = OCRResult(
            document_id=image_path.stem,
            image=ImageMeta(filename=image_path.name, width=w0, height=h0),
            ocr=tokens,
            processing_ms=round(total_ms, 1),
        )
        logger.info("Pipeline done: %d tokens, %.0f ms total (%.0f ms OCR)",
                    result.token_count, total_ms, ocr_ms)
        return result

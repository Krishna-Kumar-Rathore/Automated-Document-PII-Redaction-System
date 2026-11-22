"""Full document pipeline: image in -> DocumentResult out.

Stages run in the fixed order defined by ``PIPELINE_STAGES``. Stages that are
not implemented yet are reported as ``pending``; the result is only
``is_complete`` (and a redacted image is only rendered) once every stage runs.
As each phase lands, it is wired in here - callers (CLI, UI) don't change.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from app.config.loader import Config, load_config
from app.ocr.engine import OCREngine, build_engine
from app.preprocessing.preprocess import Preprocessor
from app.schemas.document import (DONE, PIPELINE_STAGES, DocumentResult,
                                  StageStatus)
from app.schemas.ocr import ImageMeta
from app.text.reconstruct import TextReconstructor
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


class DocumentPipeline:
    def __init__(self, cfg: Optional[Config] = None,
                 engine: Optional[OCREngine] = None):
        self.cfg = cfg or load_config()
        self.preprocessor = Preprocessor(self.cfg)
        # Engine construction loads models, so allow injection (tests / reuse).
        self.engine = engine or build_engine(self.cfg)
        self.reconstructor = TextReconstructor(self.cfg)

    def run(self, image_path: str | Path) -> DocumentResult:
        """Convenience wrapper: read an image file and run the pipeline."""
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.run_image(image, filename=image_path.name)

    def run_image(self, image: np.ndarray, filename: str = "upload.png") -> DocumentResult:
        """Run every stage on an in-memory BGR image (used by the UI: no disk I/O)."""
        if image is None or image.ndim not in (2, 3) or image.size == 0:
            raise ValueError("Expected a non-empty BGR image array")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h0, w0 = image.shape[:2]
        stages: Dict[str, StageStatus] = {
            key: StageStatus(key=key, name=name, phase=phase)
            for key, name, phase in PIPELINE_STAGES
        }
        result = DocumentResult(
            document_id=Path(filename).stem,
            image=ImageMeta(filename=filename, width=w0, height=h0),
            stages=list(stages.values()),
        )
        wall = time.perf_counter()

        # -- Phase 2: preprocessing + OCR ------------------------------------
        t = time.perf_counter()
        pre = self.preprocessor.run(image)
        self._done(stages["preprocess"], t)

        t = time.perf_counter()
        lines, _ = self.engine.detect(pre.image, pre.transform, pre.original_size)
        self._done(stages["ocr"], t)

        # -- Phase 3: text reconstruction -------------------------------------
        t = time.perf_counter()
        result.text = self.reconstructor.run(lines, image_size=(w0, h0))
        result.ocr = result.text.lines        # lines in reading order, with offsets
        self._done(stages["reconstruct"], t)

        # -- Phases 4-8: detect -> resolve -> map -> redact (not built yet) ---
        # Each will populate result.entities and mark its stage DONE here.

        result.processing_ms = _ms_since(wall)
        pending = [s.name for s in result.pending_stages]
        logger.info("Pipeline: %d lines, %d words, %.0f ms; pending stages: %s",
                    result.token_count, result.word_count, result.processing_ms,
                    ", ".join(pending) or "none")
        return result

    @staticmethod
    def _done(stage: StageStatus, started: float) -> None:
        stage.status = DONE
        stage.ms = _ms_since(started)

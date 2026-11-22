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
from app.detection.context_detector import ContextRuleDetector
from app.detection.name_detector import NameDetector
from app.detection.regex_detector import RegexDetector
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
        self.regex_detector = RegexDetector(self.cfg)
        self.name_detector = NameDetector(self.cfg)
        self.context_detector = ContextRuleDetector(self.cfg)

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

        # -- Phase 4: regex detection + format validators --------------------
        t = time.perf_counter()
        result.candidates.extend(self.regex_detector.detect(result.text))
        self._done(stages["detect_regex"], t)

        # -- Phase 5: name dictionary + context rules -------------------------
        t = time.perf_counter()
        result.candidates.extend(self.name_detector.detect(result.text))
        result.candidates.extend(self.context_detector.detect(result.text))
        self._done(stages["detect_names"], t)

        # -- Phases 6-8: NER -> resolve -> map -> redact (not built yet) -------
        # Detectors extend result.candidates; the resolver fills result.entities.

        result.processing_ms = _ms_since(wall)
        pending = [s.name for s in result.pending_stages]
        logger.info("Pipeline: %d lines, %d words, %d candidates, %.0f ms; pending: %s",
                    result.token_count, result.word_count, len(result.candidates),
                    result.processing_ms,
                    ", ".join(pending) or "none")
        return result

    @staticmethod
    def _done(stage: StageStatus, started: float) -> None:
        stage.status = DONE
        stage.ms = _ms_since(started)

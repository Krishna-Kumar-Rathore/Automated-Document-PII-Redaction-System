"""Full document pipeline: image in -> DocumentResult out.

Stages run in the fixed order defined by ``PIPELINE_STAGES``. Stages that are
not implemented yet are reported as ``pending``; the result is only
``is_complete`` (and a redacted image is only rendered) once every stage runs.
As each phase lands, it is wired in here - callers (CLI, UI) don't change.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from app.config.loader import Config, load_config
from app.detection.context_detector import ContextRuleDetector
from app.detection.name_detector import NameDetector
from app.detection.ner_detector import NERDetector
from app.detection.regex_detector import RegexDetector
from app.mapping.spatial import SpatialMapper
from app.ocr.engine import OCREngine, build_engine
from app.redaction.redactor import Redactor
from app.resolution.resolver import EntityResolver
from app.preprocessing.preprocess import Preprocessor
from app.schemas.document import (DONE, FAILED, PIPELINE_STAGES, SKIPPED,
                                  DocumentResult, StageStatus)
from app.schemas.ocr import ImageMeta
from app.text.reconstruct import TextReconstructor
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


class DocumentPipeline:
    def __init__(self, cfg: Optional[Config] = None,
                 engine: Optional[OCREngine] = None,
                 ner_model: Optional[Any] = None):
        """``engine`` / ``ner_model`` may be injected (tests, reuse); the NER
        model otherwise loads lazily on first use or via ``warm_up()``."""
        self.cfg = cfg or load_config()
        self.preprocessor = Preprocessor(self.cfg)
        # Engine construction loads models, so allow injection (tests / reuse).
        self.engine = engine or build_engine(self.cfg)
        self.reconstructor = TextReconstructor(self.cfg)
        self.regex_detector = RegexDetector(self.cfg)
        self.name_detector = NameDetector(self.cfg)
        self.context_detector = ContextRuleDetector(self.cfg)
        self.ner_detector = NERDetector(self.cfg, model=ner_model)
        self.resolver = EntityResolver(self.cfg)
        self.mapper = SpatialMapper(self.cfg)
        self.redactor = Redactor(self.cfg)

    def warm_up(self) -> Optional[str]:
        """Load lazy models up front. Returns an error message instead of raising,
        so a UI can show it; the NER stage will then report FAILED per document."""
        if not self.ner_detector.enabled:
            return None
        try:
            self.ner_detector.load()
            return None
        except Exception as exc:
            logger.error("NER warm-up failed: %s", exc)
            return f"{type(exc).__name__}: {exc}"

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

        # -- Phase 6: NER (GLiNER) ---------------------------------------------
        t = time.perf_counter()
        ner_stage = stages["detect_ner"]
        if not self.ner_detector.enabled:
            ner_stage.status, ner_stage.note = SKIPPED, "disabled in config (ner.enabled)"
        else:
            try:
                result.candidates.extend(self.ner_detector.detect(result.text))
                self._done(ner_stage, t)
            except Exception as exc:   # model missing / download failed / OOM
                ner_stage.status, ner_stage.ms = FAILED, _ms_since(t)
                ner_stage.note = f"{type(exc).__name__}: {exc}"
                logger.error("NER stage failed: %s", ner_stage.note)

        # -- Phase 7: entity resolution + spatial mapping -----------------------
        t = time.perf_counter()
        result.entities = self.resolver.resolve(result.candidates, result.text)
        self._done(stages["resolve"], t)

        t = time.perf_counter()
        self.mapper.map(result.entities, result.text, (w0, h0))
        self._done(stages["map"], t)

        # -- Phase 8: redaction + verification --------------------------------
        # A failed upstream stage means detections may be missing: refuse to
        # produce a redacted image that would look complete but could leak.
        failed = [s.name for s in result.stages if s.status == FAILED]
        if failed:
            stages["redact"].note = "blocked: " + ", ".join(failed) + " failed"
        else:
            t = time.perf_counter()
            result.redacted_image = self.redactor.apply(image, result.entities)
            result.redaction = self._summarise_redaction(result)
            self._done(stages["redact"], t)
        # Detectors extend result.candidates; the resolver fills result.entities.

        result.processing_ms = _ms_since(wall)
        pending = [s.name for s in result.pending_stages]
        logger.info("Pipeline: %d lines, %d words, %d candidates -> %d entities, %.0f ms; "
                    "pending: %s", result.token_count, result.word_count,
                    len(result.candidates), len(result.entities), result.processing_ms,
                    ", ".join(pending) or "none")
        return result

    def _summarise_redaction(self, result: DocumentResult) -> dict:
        ents = result.entities
        redacted = [e for e in ents if e.attributes.get("redacted")]
        summary = {
            "style": self.redactor.style,
            "labels": sorted(self.redactor.labels),
            "redacted": len(redacted),
            "kept_visible": [{"entity_id": e.entity_id, "label": e.label,
                              "reason": e.attributes.get("reason", "")}
                             for e in ents if not e.attributes.get("redacted")],
            "pixel_check_passed": all(e.attributes.get("verified") for e in redacted),
        }
        if self.redactor.leak_check and redacted:
            t = time.perf_counter()
            pre = self.preprocessor.run(result.redacted_image)
            tokens, _ = self.engine.detect(pre.image, pre.transform, pre.original_size)
            leaks = self.redactor.find_leaks(redacted, " ".join(tok.text for tok in tokens))
            summary["leak_check"] = {"ran": True, "passed": not leaks, "leaks": leaks,
                                     "ms": _ms_since(t)}
            if leaks:
                logger.error("LEAK CHECK: %d redacted entities still readable: %s",
                             len(leaks), [l["text"] for l in leaks])
        else:
            summary["leak_check"] = {"ran": False}
        return summary

    @staticmethod
    def _done(stage: StageStatus, started: float) -> None:
        stage.status = DONE
        stage.ms = _ms_since(started)

"""Top-level result for one document as it moves through the full pipeline.

The pipeline is a fixed sequence of stages (see ``PIPELINE_STAGES``). Every run
reports the status of *every* stage, so callers (CLI, UI) can tell whether the
document went through the complete pipeline. The redacted image is only
produced when ``is_complete`` is true - no shortcuts that skip detection or
resolution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .entities import EntityCandidate, ResolvedEntity
from .ocr import ImageMeta, OCRToken
from .text import ReconstructedText

# (key, human-readable name, phase number) in execution order.
PIPELINE_STAGES = [
    ("preprocess", "Preprocessing", "2"),
    ("ocr", "OCR", "2"),
    ("reconstruct", "Text reconstruction", "3"),
    ("detect_regex", "Regex + format validators", "4"),
    ("detect_names", "Name dictionary + context rules", "5"),
    ("detect_ner", "NER model (GLiNER)", "6"),
    ("resolve", "Entity resolution", "7"),
    ("map", "Spatial mapping", "7"),
    ("redact", "Redaction", "8"),
]

DONE, PENDING, FAILED, SKIPPED = "done", "pending", "failed", "skipped"
# A stage counts as "gone through" when it ran, or was deliberately switched off
# in config. A pending (not built) or failed stage blocks the final output.
_FINISHED = (DONE, SKIPPED)


@dataclass
class StageStatus:
    key: str
    name: str
    phase: str
    status: str = PENDING          # done | pending | failed | skipped
    ms: Optional[float] = None
    note: str = ""                 # failure reason / why skipped

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentResult:
    document_id: str
    image: ImageMeta
    ocr: List[OCRToken] = field(default_factory=list)       # line tokens
    text: Optional[ReconstructedText] = None
    candidates: List[EntityCandidate] = field(default_factory=list)  # raw detector output
    entities: List[ResolvedEntity] = field(default_factory=list)     # after resolution
    stages: List[StageStatus] = field(default_factory=list)
    processing_ms: Optional[float] = None

    @property
    def token_count(self) -> int:
        return len(self.ocr)

    @property
    def word_count(self) -> int:
        return len(self.text.words) if self.text else 0

    @property
    def is_complete(self) -> bool:
        """True only when every pipeline stage has run (or was disabled in config)."""
        return bool(self.stages) and all(s.status in _FINISHED for s in self.stages)

    @property
    def pending_stages(self) -> List[StageStatus]:
        """Stages blocking the final output: not built yet, or failed."""
        return [s for s in self.stages if s.status not in _FINISHED]

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "image": self.image.to_dict(),
            "processing_ms": self.processing_ms,
            "pipeline_complete": self.is_complete,
            "stages": [s.to_dict() for s in self.stages],
            "ocr": [t.to_dict() for t in self.ocr],
            "text": self.text.to_dict() if self.text else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "entities": [e.to_dict() for e in self.entities],
        }

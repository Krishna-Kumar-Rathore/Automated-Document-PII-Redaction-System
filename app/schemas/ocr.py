"""Data contracts for the OCR stage.

These dataclasses are the *source of truth* that flows through every later stage.
Design note on granularity: PP-OCR / RapidOCR detect **text lines**, not words.
So a fresh OCRToken from the engine has ``granularity="line"``. The text
reconstruction stage (Phase 3) will split lines into ``granularity="word"``
tokens with interpolated boxes, keeping ``parent_id`` so we never lose the link
back to the real, engine-produced detection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

# Axis-aligned bounding box in [x1, y1, x2, y2] = [left, top, right, bottom].
BBox = List[int]
# 4-point polygon [[x,y], ...] in TL, TR, BR, BL order (RapidOCR convention).
Polygon = List[List[float]]


@dataclass
class OCRToken:
    """A single OCR detection, always expressed in ORIGINAL image coordinates."""

    id: int
    text: str
    confidence: float
    bbox: BBox
    polygon: Polygon = field(default_factory=list)
    granularity: str = "line"          # "line" (from engine) or "word" (derived)
    parent_id: Optional[int] = None    # word tokens point back to their line

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageMeta:
    """Metadata about the source image (dimensions are the ORIGINAL ones)."""

    filename: str
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OCRResult:
    """Container for one document's OCR output."""

    document_id: str
    image: ImageMeta
    ocr: List[OCRToken] = field(default_factory=list)
    processing_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "image": self.image.to_dict(),
            "processing_ms": self.processing_ms,
            "ocr": [t.to_dict() for t in self.ocr],
        }

    @property
    def token_count(self) -> int:
        return len(self.ocr)

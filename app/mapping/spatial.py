"""Phase 7b: spatial mapping - entity char span -> pixel boxes.

For each entity:

1. ``words_for_span`` gives every word token the span touches;
2. a word only partly covered gets a sub-word slice, interpolated along the word
   polygon with the same glyph-width model used to build word boxes - so in
   "DOB:01/01/1990" (one OCR word) only the date is boxed;
3. slices are grouped by their source OCR line and each line gets ONE box, so a
   multi-line address yields one box per line instead of a giant rectangle
   covering everything in between;
4. boxes are padded (interpolation is approximate; redaction must not clip
   glyph edges) and clipped to the image.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from app.config.loader import Config
from app.schemas.entities import ResolvedEntity
from app.schemas.ocr import BBox, OCRToken, Polygon
from app.schemas.text import ReconstructedText
from app.text.reconstruct import char_fractions, sub_quad, token_quad
from app.utils.geometry import polygon_to_bbox
from app.utils.logging import get_logger

logger = get_logger(__name__)


def word_slice(word: OCRToken, start: int, end: int) -> Polygon:
    """Polygon of the part of ``word`` covered by full-text span [start, end)."""
    s = max(start, word.start) - word.start
    e = min(end, word.end) - word.start
    if s <= 0 and e >= len(word.text):
        return word.polygon or [list(p) for p in token_quad(word).tolist()]
    fracs = char_fractions(word.text)
    return sub_quad(token_quad(word), float(fracs[s]), float(fracs[e]))


class SpatialMapper:
    def __init__(self, cfg: Config):
        mc = cfg.get("mapping") or {}
        self.padding = int(mc.get("padding_px", 2))

    def map(self, entities: List[ResolvedEntity], text: ReconstructedText,
            image_size: Tuple[int, int]) -> List[ResolvedEntity]:
        """Fill ``bboxes`` and ``source_tokens`` in place; returns the list."""
        width, height = image_size
        unmapped = 0
        for ent in entities:
            words = text.words_for_span(ent.start, ent.end)
            per_line: Dict[int, List[List[float]]] = {}
            for w in words:
                per_line.setdefault(w.parent_id, []).extend(word_slice(w, ent.start, ent.end))
            ent.bboxes = [self._pad(polygon_to_bbox(pts, clip_w=width, clip_h=height),
                                    width, height)
                          for pts in per_line.values()]
            ent.source_tokens = [w.id for w in words]
            ent.attributes["lines"] = list(per_line)
            if not ent.bboxes:
                unmapped += 1
        if unmapped:
            logger.warning("%d entities could not be mapped to any word token", unmapped)
        logger.info("Spatial mapping: %d entities -> %d boxes", len(entities),
                    sum(len(e.bboxes) for e in entities))
        return entities

    def _pad(self, box: BBox, width: int, height: int) -> BBox:
        p = self.padding
        x1, y1, x2, y2 = box
        return [max(0, x1 - p), max(0, y1 - p), min(width, x2 + p), min(height, y2 + p)]

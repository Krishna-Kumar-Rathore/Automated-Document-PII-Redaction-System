"""Phase 8: redaction - burn opaque boxes over the entities chosen by policy.

* Policy: only labels in ``redaction.labels`` are redacted; every entity records
  ``attributes["redacted"]`` and, if kept visible, the reason.
* True redaction: pixels are overwritten (solid fill), never blurred or overlaid.
* Pixel check: every redacted box is re-read and must be 100% fill colour
  (``attributes["verified"]``) - guards against coordinate bugs or a "labeled"
  style that somehow leaves source pixels behind.
* Leak check (``leak_check``): the caller re-OCRs the redacted image and passes
  the text to ``find_leaks``; any redacted entity still readable is reported.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.config.loader import Config
from app.schemas.entities import ResolvedEntity
from app.utils.logging import get_logger

logger = get_logger(__name__)

SOLID, LABELED = "solid", "labeled"


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


class Redactor:
    def __init__(self, cfg: Config):
        rc = cfg.get("redaction") or {}
        self.labels = set(rc.get("labels", []) or [])
        self.style = rc.get("style", SOLID)
        if self.style not in (SOLID, LABELED):
            raise ValueError(f"Unknown redaction style: {self.style}")
        self.fill: Tuple[int, int, int] = tuple(rc.get("fill", [0, 0, 0]))
        self.text_color: Tuple[int, int, int] = tuple(rc.get("label_text_color", [255, 255, 255]))
        self.label_names: Dict[str, str] = rc.get("label_names", {}) or {}
        self.leak_check = bool(rc.get("leak_check", True))
        self.leak_min_chars = int(rc.get("leak_check_min_chars", 4))

    def apply(self, image: np.ndarray, entities: Sequence[ResolvedEntity]) -> np.ndarray:
        """Return a redacted copy of ``image``; annotates each entity's attributes."""
        canvas = image.copy()
        for ent in entities:
            if ent.label not in self.labels:
                ent.attributes.update(redacted=False,
                                      reason="label not in redaction.labels")
                continue
            if not ent.bboxes:
                ent.attributes.update(redacted=False, reason="no pixel boxes mapped")
                logger.warning("Entity #%d (%s) has no boxes - cannot redact",
                               ent.entity_id, ent.label)
                continue
            for box in ent.bboxes:
                self._fill_box(canvas, box)
            # Pixel check BEFORE label text is drawn (text is meant to differ).
            ent.attributes["verified"] = all(self._is_filled(canvas, b) for b in ent.bboxes)
            if self.style == LABELED:
                for box in ent.bboxes:
                    self._draw_label(canvas, box, self.label_names.get(ent.label, ent.label))
            ent.attributes["redacted"] = True

        n = sum(1 for e in entities if e.attributes.get("redacted"))
        failed = [e.entity_id for e in entities
                  if e.attributes.get("redacted") and not e.attributes.get("verified")]
        if failed:
            logger.error("Pixel check FAILED for entities %s", failed)
        logger.info("Redacted %d of %d entities (%s)", n, len(entities), self.style)
        return canvas

    def find_leaks(self, entities: Sequence[ResolvedEntity], ocr_text: str) -> List[dict]:
        """Redacted entities whose text is still readable in ``ocr_text``.

        "full": the whole entity is readable. "partial": a meaningful part is
        (e.g. "9000000000" of "+91 9000000000") - still a leak of that part.
        Errs towards reporting: a partial hit can occasionally be the same word
        appearing elsewhere on the page, which is worth a human look anyway.
        """
        haystack = _normalise(ocr_text)
        leaks = []
        for ent in entities:
            if not ent.attributes.get("redacted"):
                continue
            whole = _normalise(ent.text)
            parts = [p for p in (_normalise(w) for w in re.split(r"[\s,;/]+", ent.text))
                     if len(p) >= self.leak_min_chars]
            if len(whole) >= self.leak_min_chars and whole in haystack:
                kind, found = "full", [ent.text]
            else:
                found = [p for p in parts if p in haystack]
                kind = "partial" if found else None
            if kind:
                leaks.append({"entity_id": ent.entity_id, "label": ent.label,
                              "text": ent.text, "match": kind, "readable": found})
        return leaks

    # -- pixels -------------------------------------------------------------

    def _fill_box(self, canvas: np.ndarray, box: List[int]) -> None:
        x1, y1, x2, y2 = box
        canvas[y1:y2, x1:x2] = self.fill

    def _is_filled(self, canvas: np.ndarray, box: List[int]) -> bool:
        x1, y1, x2, y2 = box
        region = canvas[y1:y2, x1:x2]
        return region.size > 0 and bool(np.all(region == np.array(self.fill, dtype=canvas.dtype)))

    def _draw_label(self, canvas: np.ndarray, box: List[int], text: str) -> None:
        """'[NAME]' centred in the box, scaled to fit; skipped if the box is tiny."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        label = f"[{text}]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale: Optional[float] = None
        for s in (0.9, 0.7, 0.55, 0.45, 0.35):
            (tw, th), _ = cv2.getTextSize(label, font, s, 1)
            if tw <= w - 4 and th <= h - 4:
                scale = s
                break
        if scale is None:
            return
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        org = (x1 + (w - tw) // 2, y1 + (h + th) // 2)
        cv2.putText(canvas, label, org, font, scale, self.text_color, 1, cv2.LINE_AA)

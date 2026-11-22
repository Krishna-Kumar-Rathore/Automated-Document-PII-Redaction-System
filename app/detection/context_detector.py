"""Phase 5: context rules - "<label> <value>" fields on the same row.

Examples: "DOB: 14/08/1995" -> DATE, "A/c No. 1234 5678 9012" -> ACCOUNT_NUMBER,
"Aadhaar No: 2345 6789 0123" -> ID_NUMBER.

The label is the evidence, so these rules catch values a bare regex either
misses (account numbers are too generic to match without a label) or can only
score weakly (a date is not necessarily a date of birth). When a context
candidate coincides with a regex candidate, the resolver counts it as agreement.
"""
from __future__ import annotations

import re
from typing import Dict, List, Pattern, Tuple

from app.config.loader import Config
from app.detection.base import Detector
from app.detection.keywords import phrase_pattern
from app.detection.regex_detector import DIGIT, PATTERNS, RAW, digit_view
from app.schemas.entities import EntityCandidate
from app.schemas.text import ReconstructedText
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Separators allowed between a label and its value (never a newline).
_SEP = re.compile(r"[ \t:\-\u2013\u2014#.=]*")

# value kind -> [(pattern, view)]; patterns are tried at the value position.
VALUE_PATTERNS: Dict[str, List[Tuple[Pattern, str]]] = {
    "date": [(p.regex, p.view) for p in PATTERNS if p.key == "DATE"],
    "year": [(re.compile(r"(?:19|20)\d{2}(?!\d)"), DIGIT)],
    "account": [(re.compile(r"\d(?:[ -]?\d){8,17}(?!\d)"), DIGIT)],
    # Space/hyphen-separated chunks that each contain a digit, so the value stops
    # before the next label: "2345 6789 0123 DOB" -> "2345 6789 0123".
    "id": [(re.compile(r"[A-Za-z0-9]*\d[A-Za-z0-9]*(?:[ -][A-Za-z0-9]*\d[A-Za-z0-9]*)*"), RAW)],
}
_MIN_ID_LEN = 6


class ContextRuleDetector(Detector):
    name = "context_rule"

    def __init__(self, cfg: Config):
        rc = cfg.get("context_rules") or {}
        self.confidence = float(rc.get("confidence", 0.8))
        self.rules = []
        for f in rc.get("fields", []) or []:
            if f.get("value") not in VALUE_PATTERNS:
                raise ValueError(f"Unknown context value kind: {f.get('value')}")
            self.rules.append((f["label"], f["value"], phrase_pattern(f.get("keywords", []))))
        self.ocr_correction = bool((cfg.get("regex") or {}).get("ocr_digit_correction", True))

    def detect(self, text: ReconstructedText) -> List[EntityCandidate]:
        raw = text.full_text
        views = {RAW: raw, DIGIT: digit_view(raw) if self.ocr_correction else raw}
        out: Dict[tuple, EntityCandidate] = {}

        for label, kind, keyword_re in self.rules:
            for kw in keyword_re.finditer(raw):
                pos = _SEP.match(raw, kw.end()).end()
                for pattern, view in VALUE_PATTERNS[kind]:
                    m = pattern.match(views[view], pos)
                    if not m or m.end() <= pos:
                        continue
                    if kind == "id" and (m.end() - pos < _MIN_ID_LEN):
                        continue
                    key = (pos, m.end(), label)
                    out.setdefault(key, EntityCandidate(
                        text=raw[pos:m.end()], label=label, start=pos, end=m.end(),
                        confidence=self.confidence, detector=self.name,
                        attributes={"rule": kind, "keyword": kw.group()},
                    ))
                    break

        candidates = sorted(out.values(), key=lambda c: c.start)
        logger.info("Context rules: %d candidates", len(candidates))
        return candidates

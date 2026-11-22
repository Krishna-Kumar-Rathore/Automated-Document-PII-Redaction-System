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

ADDRESS_BLOCK = "address_block"
# A row that starts like "Mobile:" / "Father's Name :" begins a new field.
_LABEL_ROW = re.compile(r"\s*[A-Za-z][A-Za-z .'/()&]{0,30}:")
# "Email Address:" / "IP address" are not postal addresses.
_NON_POSTAL = {"email", "e-mail", "mail", "ip", "web", "website", "mac", "wallet", "url"}


class ContextRuleDetector(Detector):
    name = "context_rule"

    def __init__(self, cfg: Config):
        rc = cfg.get("context_rules") or {}
        self.confidence = float(rc.get("confidence", 0.8))
        self.address_extra_rows = int(rc.get("address_max_extra_rows", 3))
        self.rules = []
        for f in rc.get("fields", []) or []:
            if f.get("value") not in VALUE_PATTERNS and f.get("value") != ADDRESS_BLOCK:
                raise ValueError(f"Unknown context value kind: {f.get('value')}")
            self.rules.append((f["label"], f["value"], phrase_pattern(f.get("keywords", []))))
        self.ocr_correction = bool((cfg.get("regex") or {}).get("ocr_digit_correction", True))

    def _address_block(self, raw: str, kw_start: int, pos: int):
        """[pos, end) covering the rest of the label's row plus continuation rows."""
        prior = re.findall(r"[A-Za-z-]+", raw[raw.rfind("\n", 0, kw_start) + 1:kw_start])
        if prior and prior[-1].lower() in _NON_POSTAL:
            return None
        end = raw.find("\n", pos)
        end = len(raw) if end < 0 else end
        for _ in range(self.address_extra_rows):
            if end >= len(raw):
                break
            nxt_end = raw.find("\n", end + 1)
            nxt_end = len(raw) if nxt_end < 0 else nxt_end
            row = raw[end + 1:nxt_end]
            if not row.strip() or _LABEL_ROW.match(row):
                break
            end = nxt_end
        while end > pos and raw[end - 1] in " \t,;-":
            end -= 1
        return (pos, end) if any(c.isalnum() for c in raw[pos:end]) else None

    def detect(self, text: ReconstructedText) -> List[EntityCandidate]:
        raw = text.full_text
        views = {RAW: raw, DIGIT: digit_view(raw) if self.ocr_correction else raw}
        out: Dict[tuple, EntityCandidate] = {}

        for label, kind, keyword_re in self.rules:
            for kw in keyword_re.finditer(raw):
                pos = _SEP.match(raw, kw.end()).end()
                if kind == ADDRESS_BLOCK:
                    span = self._address_block(raw, kw.start(), pos)
                    if span:
                        out.setdefault((*span, label), EntityCandidate(
                            text=raw[span[0]:span[1]], label=label, start=span[0],
                            end=span[1], confidence=self.confidence, detector=self.name,
                            attributes={"rule": kind, "keyword": kw.group()},
                        ))
                    continue
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

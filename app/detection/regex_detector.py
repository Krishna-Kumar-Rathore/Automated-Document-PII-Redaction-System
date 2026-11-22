"""Phase 4: regex detection of structured PII, with format validators.

Patterns live here; tunables (which patterns run, base confidences, validator
boost / penalty, UPI handles) live in config.yaml -> regex.

Two text "views" are scanned, both the SAME LENGTH as full_text so every match
offset is valid in the original text:

* raw   - full_text as-is: emails, URLs, UPI IDs, PAN, IFSC, textual dates;
* digit - OCR confusions (O->0, I/l/|->1, S->5, B->8, Z->2) corrected inside
          digit-heavy chunks only: phones, Aadhaar, numeric dates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern

from app.config.loader import Config
from app.detection.base import Detector
from app.detection.validators import apply_validator
from app.schemas.entities import EntityCandidate
from app.schemas.text import ReconstructedText
from app.utils.logging import get_logger

logger = get_logger(__name__)

RAW, DIGIT = "raw", "digit"


@dataclass(frozen=True)
class PatternSpec:
    key: str                     # config key, e.g. "AADHAAR"
    label: str                   # entity label from the registry
    regex: Pattern
    view: str                    # RAW | DIGIT
    validator: Optional[str]     # name understood by validators.apply_validator
    strip_trailing: str = ""     # punctuation to trim from the end of a match


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_TLD = r"(?:com|in|org|net|edu|gov|io|co|info|biz|me|app|dev|ai)"

PATTERNS: List[PatternSpec] = [
    PatternSpec(
        "EMAIL", "EMAIL",
        re.compile(r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+"
                   r"(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![A-Za-z0-9-])"),
        RAW, None),
    PatternSpec(
        # handle@psp with NO dot in the provider part (that would be an email).
        "UPI_ID", "UPI_ID",
        re.compile(r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._-]{1,255}@[A-Za-z][A-Za-z0-9]{1,63}"
                   r"(?![A-Za-z0-9@_-]|\.[A-Za-z0-9])"),
        RAW, "upi"),
    PatternSpec(
        "URL", "URL",
        re.compile(r"(?<![@\w.-])(?:(?:https?://|www\.)[^\s<>\"']+"
                   rf"|(?:[A-Za-z0-9-]+\.)+{_TLD}(?:\.[a-z]{{2}})?(?:/[^\s<>\"']*)?)"
                   r"(?![\w@-])", re.IGNORECASE),
        RAW, None, strip_trailing=".,;:!?)]}'\""),
    PatternSpec(
        # Indian mobiles (10 digits, 5-5 or 3-3-4 grouping, optional +91/0091/91/0),
        # landlines with STD code, and other +CC international numbers.
        "PHONE_NUMBER", "PHONE_NUMBER",
        re.compile(r"(?<![\d+])(?:"
                   r"(?:(?:\+|00)91[\s-]?|91[\s-]|0)?"
                   r"(?:[6-9]\d{9}|[6-9]\d{4}[\s-]\d{5}|[6-9]\d{2}[\s-]\d{3}[\s-]\d{4})"
                   r"|0\d{2,4}[\s-]\d{6,8}"
                   r"|\+(?!91)\d{1,3}[\s-]?\d{2,4}(?:[\s-]?\d{2,4}){2,3}"
                   r")(?!\d)"),
        DIGIT, "phone"),
    PatternSpec(
        "AADHAAR", "ID_NUMBER",
        re.compile(r"(?<!\d)[2-9]\d{3}([\s-]?)\d{4}\1\d{4}(?!\d)"),
        DIGIT, "aadhaar"),
    PatternSpec(
        "PAN", "ID_NUMBER",
        re.compile(r"(?<![A-Za-z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Za-z0-9])"),
        RAW, "pan"),
    PatternSpec(
        "IFSC", "OTHER_PII",
        re.compile(r"(?<![A-Za-z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Za-z0-9])"),
        RAW, None),
    PatternSpec(
        "DATE", "DATE",
        re.compile(r"(?<!\d)(?:\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})"
                   r"|(?:19|20)\d{2}[/.-]\d{1,2}[/.-]\d{1,2})(?![\d/.-]\d)"),
        DIGIT, "date"),
    PatternSpec(
        "DATE", "DATE",
        re.compile(rf"(?<![A-Za-z0-9])(?:\d{{1,2}}(?:st|nd|rd|th)?[\s-]*{_MONTH}[\s,-]*\d{{4}}"
                   rf"|{_MONTH}[\s-]*\d{{1,2}}(?:st|nd|rd|th)?,?[\s-]*\d{{4}})(?!\d)",
                   re.IGNORECASE),
        RAW, "date"),
]

# --- OCR digit correction -----------------------------------------------------
_CONFUSABLE = {"O": "0", "o": "0", "I": "1", "l": "1", "|": "1",
               "S": "5", "B": "8", "Z": "2"}
_CHUNK_RE = re.compile(r"[0-9OoIl|SBZ/.\-]+")


def digit_view(text: str) -> str:
    """Same-length copy of ``text`` with OCR letter/digit confusions corrected,
    only inside chunks that are already mostly digits (>= 2 digits and >= 40% of
    the chunk's non-separator chars), so words like 'BOSS' or 'Oct' are untouched.
    A chunk whose edge letter runs straight into another letter ("14Oct") is part
    of a word and is left alone."""
    out = list(text)
    n = len(text)
    for m in _CHUNK_RE.finditer(text):
        chunk = m.group()
        core = [c for c in chunk if c not in "/.-"]
        n_digits = sum(c.isdigit() for c in core)
        if n_digits < 2 or n_digits < 0.4 * len(core) or n_digits == len(core):
            continue
        s, e = m.start(), m.end()
        if (chunk[0] in _CONFUSABLE and s > 0 and text[s - 1].isalpha()) or \
                (chunk[-1] in _CONFUSABLE and e < n and text[e].isalpha()):
            continue
        for i, ch in enumerate(chunk, start=m.start()):
            out[i] = _CONFUSABLE.get(ch, ch)
    return "".join(out)


class RegexDetector(Detector):
    name = "regex"

    def __init__(self, cfg: Config):
        rc = cfg.get("regex") or {}
        enabled = set(rc.get("enabled", []))
        self.patterns = [p for p in PATTERNS if p.key in enabled]
        self.base_conf: Dict[str, float] = rc.get("base_confidence", {}) or {}
        self.validators_on: Dict[str, bool] = rc.get("validators", {}) or {}
        self.boost = float(rc.get("validator_boost", 0.2))
        self.penalty = float(rc.get("validator_penalty", 0.15))
        self.ocr_correction = bool(rc.get("ocr_digit_correction", True))
        self.upi_handles = list(rc.get("upi_handles", []) or [])

    def detect(self, text: ReconstructedText) -> List[EntityCandidate]:
        raw = text.full_text
        views = {RAW: raw, DIGIT: digit_view(raw) if self.ocr_correction else raw}

        best: Dict[tuple, EntityCandidate] = {}
        for spec in self.patterns:
            source = views[spec.view]
            for m in spec.regex.finditer(source):
                start, end = m.start(), m.end()
                if spec.strip_trailing:
                    while end > start and source[end - 1] in spec.strip_trailing:
                        end -= 1
                if end <= start:
                    continue
                cand = self._candidate(spec, raw, source, start, end)
                key = (start, end, cand.label)
                if key not in best or cand.confidence > best[key].confidence:
                    best[key] = cand

        candidates = sorted(best.values(), key=lambda c: (c.start, -c.end))
        logger.info("Regex detector: %d candidates (%s)", len(candidates),
                    ", ".join(sorted({c.label for c in candidates})) or "none")
        return candidates

    def _candidate(self, spec: PatternSpec, raw: str, source: str,
                   start: int, end: int) -> EntityCandidate:
        matched = source[start:end]
        conf = float(self.base_conf.get(spec.key, 0.7))
        attrs: Dict[str, object] = {"pattern": spec.key}

        validator = spec.validator if self.validators_on.get(spec.validator, False) else None
        verdict = apply_validator(validator, matched, self.upi_handles)
        if verdict is not None:
            attrs["validator"] = validator
            attrs["validated"] = verdict
            conf = min(0.99, conf + self.boost) if verdict else max(0.05, conf - self.penalty)
        if matched != raw[start:end]:
            attrs["ocr_corrected"] = matched       # what the validator actually saw

        return EntityCandidate(
            text=raw[start:end], label=spec.label, start=start, end=end,
            confidence=round(conf, 3), detector=self.name, attributes=attrs,
        )

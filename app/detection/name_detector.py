"""Phase 5: person-name detection from the name trie + context signals.

Per row of the reconstructed text:

1. Mark keyword spans: context labels ("Name:"), relation markers ("S/O"),
   honorifics ("Mr."). Keywords are never part of a name and break runs.
2. Build *name runs*: consecutive capitalised / ALL-CAPS words separated only by
   spaces (or the "." of an initial), up to ``max_tokens`` long.
3. Score each run with weighted signals (config ``names.signal_weights``):
   context_label / relation / honorific directly before the run, a label alone
   on the row above (form layout), and dictionary hits (exact or fuzzy).
4. Emit a PERSON_NAME candidate when score >= ``min_score``.

Precision first: a lone dictionary hit never qualifies on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.config.loader import Config
from app.detection.base import Detector
from app.detection.keywords import phrase_pattern
from app.detection.trie import NameTrie
from app.schemas.entities import EntityCandidate
from app.schemas.text import ReconstructedText
from app.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z'\-]*(?![A-Za-z0-9])")
_GAP_IN_RUN = re.compile(r"[ \t]+|\.[ \t]*")             # spaces, or "." after an initial
_SEP_AFTER_KEYWORD = re.compile(r"[\s:\-\u2013\u2014.,#]*")
_LABEL_ONLY_ROW = re.compile(r"[\s:\-\u2013\u2014.]*")

LABEL, RELATION, HONORIFIC = "context_label", "relation", "honorific"


@dataclass
class _Keyword:
    start: int
    end: int
    kind: str
    text: str


@dataclass
class _Run:
    tokens: List[Tuple[int, int, str]] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.tokens[0][0]

    @property
    def end(self) -> int:
        return self.tokens[-1][1]


def _name_shaped(tok: str) -> bool:
    if not tok[0].isupper():
        return False
    return len(tok) == 1 or tok.isupper() or any(c.islower() for c in tok[1:])


class NameDetector(Detector):
    name = "name_trie"

    def __init__(self, cfg: Config, trie: Optional[NameTrie] = None):
        nc = cfg.get("names") or {}
        self.weights: Dict[str, float] = nc.get("signal_weights", {}) or {}
        self.min_score = float(nc.get("min_score", 2))
        self.conf_base = float(nc.get("confidence_base", 0.5))
        self.conf_step = float(nc.get("confidence_per_point", 0.12))
        self.max_tokens = int(nc.get("max_tokens", 4))
        self.fuzzy_min = int(nc.get("fuzzy_min_length", 5))
        self.stopwords = {s.lower() for s in nc.get("stopwords", [])}
        self.qualifiers = {q.lower() for q in nc.get("non_person_qualifiers", [])}
        self.label_re = phrase_pattern(nc.get("context_labels", []))
        self.relation_re = phrase_pattern(nc.get("relation_markers", []))
        self.honorific_re = phrase_pattern(nc.get("prefixes", []), optional_dot=True)
        self.trie = trie or NameTrie.from_dir(self._names_dir(cfg))
        logger.info("Name trie loaded: %d names", self.trie.size)

    @staticmethod
    def _names_dir(cfg: Config) -> Path:
        path = Path((cfg.get("paths") or {}).get("names_dir", "data/names"))
        return path if path.is_absolute() else _PROJECT_ROOT / path

    # -- public -------------------------------------------------------------

    def detect(self, text: ReconstructedText) -> List[EntityCandidate]:
        full = text.full_text
        candidates: List[EntityCandidate] = []
        prev_row = ""
        offset = 0
        for row in full.split("\n"):
            candidates.extend(self._detect_row(full, row, offset, prev_row))
            prev_row = row
            offset += len(row) + 1
        logger.info("Name detector: %d PERSON_NAME candidates", len(candidates))
        return candidates

    # -- per row ------------------------------------------------------------

    def _keywords(self, row: str, offset: int) -> List[_Keyword]:
        found: List[_Keyword] = []
        for kind, rx in ((LABEL, self.label_re), (RELATION, self.relation_re),
                         (HONORIFIC, self.honorific_re)):
            for m in rx.finditer(row):
                s, e = m.start() + offset, m.end() + offset
                if not any(k.start < e and s < k.end for k in found):   # longest kind wins
                    found.append(_Keyword(s, e, kind, m.group()))
        return sorted(found, key=lambda k: k.start)

    def _runs(self, full: str, row: str, offset: int,
              keywords: List[_Keyword]) -> List[_Run]:
        def in_keyword(s: int, e: int) -> bool:
            return any(k.start < e and s < k.end for k in keywords)

        runs: List[_Run] = []
        current = _Run()
        for m in _TOKEN_RE.finditer(row):
            s, e, tok = m.start() + offset, m.end() + offset, m.group()
            eligible = (_name_shaped(tok) and tok.lower() not in self.stopwords
                        and not in_keyword(s, e))
            joins = (eligible and current.tokens
                     and _GAP_IN_RUN.fullmatch(full[current.end:s]) is not None)
            if joins:
                current.tokens.append((s, e, tok))
                continue
            if current.tokens:
                runs.append(current)
            current = _Run([(s, e, tok)] if eligible else [])
        if current.tokens:
            runs.append(current)
        return [r for r in runs if len(r.tokens) <= self.max_tokens]

    def _detect_row(self, full: str, row: str, offset: int,
                    prev_row: str) -> List[EntityCandidate]:
        keywords = self._keywords(row, offset)
        out: List[EntityCandidate] = []
        for run in self._runs(full, row, offset, keywords):
            signals, evidence = self._context_signals(full, row, offset, prev_row,
                                                      run, keywords)
            dict_hits = self._dictionary_hits(run)
            if dict_hits:
                signals.append("dictionary")
            if len(dict_hits) >= 2:
                signals.append("dictionary_multi")

            score = sum(float(self.weights.get(s, 0)) for s in signals)
            if score < self.min_score:
                continue
            out.append(EntityCandidate(
                text=full[run.start:run.end],
                label="PERSON_NAME",
                start=run.start,
                end=run.end,
                confidence=round(min(0.99, self.conf_base + self.conf_step * score), 3),
                detector="name_trie" if dict_hits else "context_rule",
                attributes={"signals": signals, "score": score,
                            "dictionary": dict_hits, **evidence},
            ))
        return out

    def _context_signals(self, full: str, row: str, offset: int, prev_row: str,
                         run: _Run, keywords: List[_Keyword]):
        signals: List[str] = []
        evidence: Dict[str, str] = {}
        before = [k for k in keywords if k.end <= run.start
                  and _SEP_AFTER_KEYWORD.fullmatch(full[k.end:run.start])]
        if before:
            kw = before[-1]
            qualifier = self._non_person_qualifier(full, offset, kw) if kw.kind == LABEL else None
            if qualifier:
                evidence["rejected_label"] = f"{qualifier} {kw.text}"
            else:
                signals.append(kw.kind)
                evidence["keyword"] = kw.text
        elif not row[: run.start - offset].strip() and prev_row.strip():
            # Form layout: the label sits alone on the row above the value.
            m = self.label_re.match(prev_row.strip())
            if m and _LABEL_ONLY_ROW.fullmatch(prev_row.strip()[m.end():]):
                signals.append(LABEL)
                evidence["keyword"] = m.group() + " (row above)"
        return signals, evidence

    def _non_person_qualifier(self, full: str, row_offset: int, kw: _Keyword) -> Optional[str]:
        """'Company Name:' - the word right before a label disqualifies it."""
        prior = re.findall(r"[A-Za-z]+", full[row_offset:kw.start])
        return prior[-1] if prior and prior[-1].lower() in self.qualifiers else None

    def _dictionary_hits(self, run: _Run) -> Dict[str, dict]:
        hits: Dict[str, dict] = {}
        for _, _, tok in run.tokens:
            if len(tok) < 2:
                continue                         # initials carry no dictionary signal
            roles = self.trie.roles(tok)
            if roles:
                hits[tok] = {"roles": sorted(roles)}
            elif len(tok) >= self.fuzzy_min:
                matches = self.trie.fuzzy(tok, 1)
                if matches:
                    word, dist, roles = matches[0]
                    hits[tok] = {"roles": sorted(roles), "fuzzy_match": word,
                                 "edit_distance": dist}
        return hits

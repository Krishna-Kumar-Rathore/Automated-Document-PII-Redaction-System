"""Phase 7a: entity resolution - many overlapping candidates -> final entities.

Steps (all tunables in config.yaml -> resolver):

1. cluster   same label + overlapping spans -> one group; span = union
2. score     noisy-OR over detectors: 1 - PRODUCT(1 - weight * best_conf_of_detector)
3. merge     chain same-label pieces (addresses) across small separator gaps,
             absorbing e.g. a building name tagged ORGANIZATION in between, and
             extending over a trailing PIN code
4. threshold global and per-label minimum confidence
5. overlaps  different labels on the same text: winner by score (or length);
             the loser is dropped only if mostly covered, and is recorded

Every ResolvedEntity keeps its evidence (each candidate, its weight and
contribution), so result.json explains every decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from app.config.loader import Config
from app.schemas.entities import EntityCandidate, ResolvedEntity
from app.schemas.text import ReconstructedText
from app.utils.logging import get_logger

logger = get_logger(__name__)

_GAP_RE = re.compile(r"[\s,.;:\-–/]*")
_PIN_RE = re.compile(r"[ \t,\-–]*\d{3} ?\d{3}(?!\d)")


@dataclass
class _Group:
    label: str
    start: int
    end: int
    members: List[EntityCandidate] = field(default_factory=list)
    confidence: float = 0.0
    merged_from: List[str] = field(default_factory=list)
    suppressed: List[dict] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start


class EntityResolver:
    def __init__(self, cfg: Config):
        rc = cfg.get("resolver") or {}
        self.weights: Dict[str, float] = rc.get("detector_weights", {}) or {}
        self.min_conf = float(rc.get("min_final_confidence", 0.5))
        self.min_by_label: Dict[str, float] = rc.get("min_confidence_by_label", {}) or {}
        self.strategy = rc.get("overlap_strategy", "highest_score")
        if self.strategy not in ("highest_score", "longest_span"):
            raise ValueError(f"Unknown overlap_strategy: {self.strategy}")
        self.overlap_min = float(rc.get("overlap_min_fraction", 0.5))
        mc = rc.get("merge") or {}
        self.merge_labels: Set[str] = set(mc.get("labels", []) or [])
        self.absorb_labels: Set[str] = set(mc.get("absorb_labels", []) or [])
        self.max_gap = int(mc.get("max_gap_chars", 4))
        self.attach_pin = bool(mc.get("attach_pin_code", True))
        self.registry: Set[str] = set((cfg.get("entities") or {}).get("registry", []) or [])

    # -- public -------------------------------------------------------------

    def resolve(self, candidates: List[EntityCandidate],
                text: ReconstructedText) -> List[ResolvedEntity]:
        full = text.full_text
        known = [c for c in candidates if not self.registry or c.label in self.registry]
        if len(known) < len(candidates):
            logger.warning("Dropped %d candidates with labels not in entities.registry",
                           len(candidates) - len(known))

        groups = self._cluster(known)
        for g in groups:
            g.confidence = self._score(g.members)
        groups = self._merge_chains(groups, full)
        groups = [g for g in groups
                  if g.confidence >= float(self.min_by_label.get(g.label, self.min_conf))]
        groups = self._resolve_overlaps(groups)

        entities = [self._to_entity(i + 1, g, full)
                    for i, g in enumerate(sorted(groups, key=lambda g: (g.start, g.end)))]
        logger.info("Resolver: %d candidates -> %d entities", len(candidates), len(entities))
        return entities

    # -- steps --------------------------------------------------------------

    @staticmethod
    def _cluster(candidates: List[EntityCandidate]) -> List[_Group]:
        groups: List[_Group] = []
        by_label: Dict[str, List[EntityCandidate]] = {}
        for c in candidates:
            by_label.setdefault(c.label, []).append(c)
        for label, cands in by_label.items():
            current: Optional[_Group] = None
            for c in sorted(cands, key=lambda c: (c.start, c.end)):
                if current and c.start < current.end:          # overlaps the group
                    current.end = max(current.end, c.end)
                    current.members.append(c)
                else:
                    current = _Group(label, c.start, c.end, [c])
                    groups.append(current)
        return groups

    def _score(self, members: List[EntityCandidate]) -> float:
        best: Dict[str, float] = {}
        for c in members:                                  # one vote per detector
            best[c.detector] = max(best.get(c.detector, 0.0), c.confidence)
        miss = 1.0
        for det, conf in best.items():
            miss *= 1.0 - float(self.weights.get(det, 0.5)) * conf
        return round(1.0 - miss, 3)

    def _gap_ok(self, full: str, a_end: int, b_start: int) -> bool:
        gap = full[a_end:b_start]
        return 0 <= len(gap) <= self.max_gap and _GAP_RE.fullmatch(gap) is not None

    def _merge_chains(self, groups: List[_Group], full: str) -> List[_Group]:
        if not self.merge_labels:
            return groups
        ordered = sorted(groups, key=lambda g: (g.start, g.end))
        out: List[_Group] = []
        i = 0
        while i < len(ordered):
            g = ordered[i]
            if g.label not in self.merge_labels:
                out.append(g)
                i += 1
                continue
            chain = _Group(g.label, g.start, g.end, list(g.members), g.confidence,
                           [full[g.start:g.end]])
            j = i + 1
            while j < len(ordered):
                nxt = ordered[j]
                if nxt.label == chain.label and self._gap_ok(full, chain.end, nxt.start):
                    absorbed = [nxt]
                elif (nxt.label in self.absorb_labels and j + 1 < len(ordered)
                      and ordered[j + 1].label == chain.label
                      and self._gap_ok(full, chain.end, nxt.start)
                      and self._gap_ok(full, nxt.end, ordered[j + 1].start)):
                    absorbed = [nxt, ordered[j + 1]]           # sandwiched building name
                else:
                    break
                for a in absorbed:
                    chain.end = max(chain.end, a.end)
                    chain.members.extend(a.members)
                    chain.merged_from.append(f"{a.label}: {full[a.start:a.end]}")
                    if a.label == chain.label:                 # pieces are not
                        chain.confidence = max(chain.confidence, a.confidence)  # independent
                j += len(absorbed)
            if self.attach_pin:
                m = _PIN_RE.match(full, chain.end)
                if m:
                    chain.merged_from.append(f"PIN: {m.group().strip(' ,-')}")
                    chain.end = m.end()
            if len(chain.merged_from) == 1:
                chain.merged_from = []
            out.append(chain)
            i = j
        return out

    def _resolve_overlaps(self, groups: List[_Group]) -> List[_Group]:
        if self.strategy == "highest_score":
            key = lambda g: (g.confidence, g.length)
        else:
            key = lambda g: (g.length, g.confidence)
        kept: List[_Group] = []
        for g in sorted(groups, key=key, reverse=True):
            winner = next((k for k in kept if self._mostly_covered(g, k)), None)
            if winner is None:
                kept.append(g)
            else:
                winner.suppressed.append({
                    "label": g.label, "start": g.start, "end": g.end,
                    "confidence": g.confidence,
                    "detectors": sorted({m.detector for m in g.members}),
                })
        return kept

    def _mostly_covered(self, loser: _Group, winner: _Group) -> bool:
        inter = min(loser.end, winner.end) - max(loser.start, winner.start)
        return inter > 0 and inter / max(1, loser.length) >= self.overlap_min

    # -- output -------------------------------------------------------------

    def _to_entity(self, entity_id: int, g: _Group, full: str) -> ResolvedEntity:
        evidence = [{
            "detector": c.detector, "label": c.label, "text": c.text,
            "confidence": c.confidence,
            "weight": float(self.weights.get(c.detector, 0.5)),
            **{k: v for k, v in c.attributes.items()
               if k in ("pattern", "validated", "signals", "keyword", "prompt", "rule")},
        } for c in sorted(g.members, key=lambda c: (c.start, c.detector))]
        attrs: Dict[str, object] = {"evidence": evidence}
        if g.merged_from:
            attrs["merged_from"] = g.merged_from
        if g.suppressed:
            attrs["suppressed"] = g.suppressed
        return ResolvedEntity(
            entity_id=entity_id, text=full[g.start:g.end], label=g.label,
            confidence=g.confidence, start=g.start, end=g.end,
            detectors=sorted({c.detector for c in g.members}), attributes=attrs,
        )

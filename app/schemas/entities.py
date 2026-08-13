"""Data contracts for the PII detection / resolution stages (Phases 4-8).

Defined now so the whole pipeline shares one vocabulary from day one, even though
these are populated in later phases. ``EntityCandidate`` is what each *detector*
emits; ``ResolvedEntity`` is the final, de-duplicated, spatially-mapped result
that lands in result.json.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .ocr import BBox


@dataclass
class EntityCandidate:
    """A single detector's guess. Many candidates may cover the same text."""

    text: str
    label: str
    start: int              # char offset into the reconstructed document text
    end: int                # exclusive
    confidence: float
    detector: str           # "regex" | "gliner" | "name_trie" | "context_rule" | ...

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResolvedEntity:
    """Final entity after resolution + spatial mapping. Auditable by design."""

    entity_id: int
    text: str
    label: str
    confidence: float
    start: int
    end: int
    bboxes: List[BBox] = field(default_factory=list)   # one box per line covered
    source_tokens: List[int] = field(default_factory=list)
    detectors: List[str] = field(default_factory=list)  # who contributed

    def to_dict(self) -> dict:
        return asdict(self)

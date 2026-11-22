"""Detector interface shared by every PII detector (regex, name trie, GLiNER, ...).

Each detector reads the reconstructed document text and emits EntityCandidates
(character spans + label + confidence). Candidates may overlap; the resolver
(Phase 7) merges them and the spatial mapper turns spans into pixel boxes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.schemas.entities import EntityCandidate
from app.schemas.text import ReconstructedText


class Detector(ABC):
    name: str = "detector"

    @abstractmethod
    def detect(self, text: ReconstructedText) -> List[EntityCandidate]:
        """Return candidates whose start/end index into ``text.full_text``."""

"""Data contract for the text reconstruction stage (Phase 3).

``ReconstructedText`` is the bridge between *text space* and *pixel space*:

* ``full_text`` is what the detectors (regex, NER, tries, context rules) read;
* every word/line token carries ``start``/``end`` offsets into ``full_text``;
* ``char_to_word`` maps each character back to the word token that produced it.

So a detector's character span ``(start, end)`` can always be resolved to the
word tokens - and therefore the pixel boxes - that it covers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .ocr import OCRToken


@dataclass
class ReconstructedText:
    full_text: str
    lines: List[OCRToken]                 # engine lines, in reading order, with offsets
    words: List[OCRToken]                 # derived word tokens, in reading order
    rows: List[List[int]] = field(default_factory=list)  # line ids per visual row
    # char index -> word id (-1 for separators). Not serialised; rebuilt from words.
    char_to_word: List[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.char_to_word:
            self.char_to_word = [-1] * len(self.full_text)
            for w in self.words:
                for i in range(w.start, w.end):
                    self.char_to_word[i] = w.id
        self._word_by_id: Dict[int, OCRToken] = {w.id: w for w in self.words}

    def word(self, word_id: int) -> OCRToken:
        return self._word_by_id[word_id]

    def words_for_span(self, start: int, end: int) -> List[OCRToken]:
        """Word tokens overlapping the half-open char span ``[start, end)``."""
        start, end = max(0, start), min(len(self.full_text), end)
        ids = dict.fromkeys(w for w in self.char_to_word[start:end] if w >= 0)
        return [self._word_by_id[wid] for wid in ids]

    def span_text(self, start: int, end: int) -> str:
        return self.full_text[start:end]

    def to_dict(self) -> dict:
        return {
            "full_text": self.full_text,
            "reading_order": [line.id for line in self.lines],
            "rows": self.rows,
            "words": [w.to_dict() for w in self.words],
        }

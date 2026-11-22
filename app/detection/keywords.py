"""Compile keyword lists (labels, honorifics, relation markers) into one regex.

Phrases are matched case-insensitively on letter/digit boundaries, longest
first ("father's name" wins over "name"), with flexible whitespace, so
"Father's  Name" and "FATHER'S NAME" both match.
"""
from __future__ import annotations

import re
from typing import Iterable, Pattern


def phrase_pattern(phrases: Iterable[str], optional_dot: bool = False) -> Pattern:
    alts = []
    for p in sorted({p.strip().lower() for p in phrases if p.strip()}, key=len, reverse=True):
        alts.append(r"\s+".join(re.escape(part) for part in p.split()))
    if not alts:
        return re.compile(r"(?!x)x")               # never matches
    dot = r"\.?" if optional_dot else ""
    return re.compile(rf"(?<![A-Za-z0-9])(?:{'|'.join(alts)}){dot}(?![A-Za-z0-9])",
                      re.IGNORECASE)

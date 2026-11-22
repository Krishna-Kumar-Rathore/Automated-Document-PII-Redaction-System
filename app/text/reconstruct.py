"""Phase 3: text reconstruction.

Turns the engine's *line* tokens into:

1. a reading order  - lines grouped into visual rows (top -> bottom), each row
   sorted left -> right, so "Name:" and a value detected as separate boxes on
   the same row end up adjacent in the text;
2. ``full_text``    - rows joined by "\\n", lines within a row by a separator;
3. word tokens      - each line split on whitespace, with a box interpolated
   along the line's polygon (so slanted lines get slanted word boxes), plus
   char offsets into ``full_text``.

Known limitation: rows span the full page width, so true multi-column layouts
(newspapers) interleave columns. Layout analysis is listed as future work.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import List, Sequence

import numpy as np

from app.config.loader import Config
from app.schemas.ocr import BBox, OCRToken, Polygon
from app.schemas.text import ReconstructedText
from app.utils.geometry import polygon_to_bbox
from app.utils.logging import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"\S+")

# Rough relative glyph widths for proportional fonts. Uniform-width interpolation
# makes boxes drift on lines like "Email: john.smith@mail.com"; this cheap model
# keeps word boxes noticeably tighter. Not exact - just better than uniform.
_NARROW = set("iljtfrI1!|.,:;'`\" ()[]")
_WIDE = set("mwMW@%")


def _char_weight(ch: str) -> float:
    if ch in _NARROW:
        return 0.55
    if ch in _WIDE:
        return 1.45
    if ch.isupper() or ch.isdigit():
        return 1.15
    return 1.0


def _cumulative_fractions(text: str) -> np.ndarray:
    """fractions[i] = horizontal position (0..1) where character i starts."""
    weights = np.array([_char_weight(c) for c in text], dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(weights)])
    total = cum[-1] if cum[-1] > 0 else 1.0
    return cum / total


def _line_quad(line: OCRToken) -> np.ndarray:
    """TL, TR, BR, BL as a 4x2 array; falls back to the bbox if no polygon."""
    if line.polygon and len(line.polygon) == 4:
        return np.asarray(line.polygon, dtype=np.float64)
    x1, y1, x2, y2 = line.bbox
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def _sub_quad(quad: np.ndarray, f0: float, f1: float) -> Polygon:
    """Slice the line quad between horizontal fractions f0..f1."""
    tl, tr, br, bl = quad
    top0, top1 = tl + (tr - tl) * f0, tl + (tr - tl) * f1
    bot0, bot1 = bl + (br - bl) * f0, bl + (br - bl) * f1
    return [[round(float(x), 1), round(float(y), 1)] for x, y in (top0, top1, bot1, bot0)]


def _vertical_overlap(a: BBox, b: BBox) -> float:
    """Overlap of two boxes' y-ranges as a fraction of the shorter height."""
    inter = min(a[3], b[3]) - max(a[1], b[1])
    shorter = max(1, min(a[3] - a[1], b[3] - b[1]))
    return max(0.0, inter) / shorter


class TextReconstructor:
    def __init__(self, cfg: Config):
        rc = cfg.get("reconstruction") or {}
        self.row_overlap = float(rc.get("row_overlap", 0.5))
        self.row_separator = str(rc.get("row_separator", " "))

    # -- public -------------------------------------------------------------

    def run(self, lines: Sequence[OCRToken], image_size=None) -> ReconstructedText:
        """``image_size`` = (width, height) of the original image, for clipping."""
        clip_w, clip_h = image_size if image_size else (None, None)
        rows = self._group_rows(lines)

        parts: List[str] = []
        ordered_lines: List[OCRToken] = []
        words: List[OCRToken] = []
        cursor = 0

        for r_idx, row in enumerate(rows):
            if r_idx > 0:
                parts.append("\n")
                cursor += 1
            for l_idx, line in enumerate(row):
                if l_idx > 0:
                    parts.append(self.row_separator)
                    cursor += len(self.row_separator)
                line_start = cursor
                parts.append(line.text)
                cursor += len(line.text)
                ordered_lines.append(replace(line, start=line_start, end=cursor))
                words.extend(self._split_words(line, line_start, len(words),
                                               clip_w, clip_h))

        full_text = "".join(parts)
        result = ReconstructedText(
            full_text=full_text,
            lines=ordered_lines,
            words=words,
            rows=[[ln.id for ln in row] for row in rows],
        )
        logger.info("Reconstructed %d lines -> %d rows, %d words, %d chars",
                    len(lines), len(rows), len(words), len(full_text))
        return result

    # -- internals ----------------------------------------------------------

    def _group_rows(self, lines: Sequence[OCRToken]) -> List[List[OCRToken]]:
        """Greedy top-to-bottom grouping of lines that share a visual row."""
        by_top = sorted(lines, key=lambda t: ((t.bbox[1] + t.bbox[3]) / 2, t.bbox[0]))
        rows: List[List[OCRToken]] = []
        for line in by_top:
            if rows and any(_vertical_overlap(line.bbox, other.bbox) >= self.row_overlap
                            for other in rows[-1]):
                rows[-1].append(line)
            else:
                rows.append([line])
        return [sorted(row, key=lambda t: t.bbox[0]) for row in rows]

    def _split_words(self, line: OCRToken, line_start: int, id_offset: int,
                     clip_w, clip_h) -> List[OCRToken]:
        quad = _line_quad(line)
        fracs = _cumulative_fractions(line.text)
        out: List[OCRToken] = []
        for m in _WORD_RE.finditer(line.text):
            poly = _sub_quad(quad, fracs[m.start()], fracs[m.end()])
            out.append(OCRToken(
                id=id_offset + len(out) + 1,
                text=m.group(),
                confidence=line.confidence,   # inherited: engine scores whole lines
                bbox=polygon_to_bbox(poly, clip_w=clip_w, clip_h=clip_h),
                polygon=poly,
                granularity="word",
                parent_id=line.id,
                start=line_start + m.start(),
                end=line_start + m.end(),
            ))
        return out

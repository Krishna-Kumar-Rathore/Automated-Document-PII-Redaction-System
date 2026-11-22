"""Render output images from a DocumentResult (shared by the CLI and the UI).

All colours / sizes come from ``config.yaml -> visualization``. Colours are BGR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from app.config.loader import Config
from app.schemas.document import DONE, DocumentResult
from app.schemas.entities import ResolvedEntity
from app.schemas.ocr import OCRToken

Color = Tuple[int, int, int]


@dataclass
class VizStyle:
    thickness: int
    font_scale: float
    draw_labels: bool
    line_color: Color
    word_color: Color
    class_colors: Dict[str, Color]
    default_color: Color

    @classmethod
    def from_config(cls, config: Config) -> "VizStyle":
        v = config.get("visualization") or {}
        default = tuple(v.get("default_color", [128, 128, 128]))
        return cls(
            thickness=int(v.get("box_thickness", 2)),
            font_scale=float(v.get("font_scale", 0.45)),
            draw_labels=bool(v.get("draw_labels", True)),
            line_color=tuple(v.get("ocr_line_color", [0, 140, 255])),
            word_color=tuple(v.get("ocr_word_color", [200, 120, 0])),
            class_colors={k: tuple(c) for k, c in (v.get("colors") or {}).items()},
            default_color=default,
        )


def _put_label(canvas: np.ndarray, text: str, x: int, y1: int, y2: int,
               color: Color, style: VizStyle) -> None:
    y_text = y1 - 5 if y1 - 5 > 10 else y2 + 15
    cv2.putText(canvas, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale, color, 1, cv2.LINE_AA)


def draw_tokens(image: np.ndarray, tokens: Sequence[OCRToken], color: Color,
                style: VizStyle, show_conf: bool = True,
                use_polygon: bool = False) -> np.ndarray:
    """Copy of ``image`` with each token's box (and optional id/conf label)."""
    canvas = image.copy()
    for tok in tokens:
        x1, y1, x2, y2 = tok.bbox
        if use_polygon and tok.polygon:
            pts = np.round(np.asarray(tok.polygon)).astype(np.int32)
            cv2.polylines(canvas, [pts], True, color, style.thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, style.thickness)
        if style.draw_labels:
            label = f"#{tok.id}" + (f" {tok.confidence:.2f}" if show_conf else "")
            _put_label(canvas, label, x1, y1, y2, color, style)
    return canvas


def draw_entities(image: np.ndarray, entities: Sequence[ResolvedEntity],
                  style: VizStyle) -> np.ndarray:
    canvas = image.copy()
    for ent in entities:
        color = style.class_colors.get(ent.label, style.default_color)
        for x1, y1, x2, y2 in ent.bboxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, style.thickness)
        if style.draw_labels and ent.bboxes:
            x1, y1, _, y2 = ent.bboxes[0]
            _put_label(canvas, f"{ent.label} {ent.confidence:.2f}", x1, y1, y2,
                       color, style)
    return canvas



def render_outputs(image: np.ndarray, result: DocumentResult,
                   cfg: Config) -> Dict[str, np.ndarray]:
    """Named output images. ``entities`` (annotated, intermediate) appears once
    spatial mapping has run; ``redacted`` only once the document has gone
    through EVERY pipeline stage."""
    style = VizStyle.from_config(cfg)
    outputs: Dict[str, np.ndarray] = {
        "ocr_lines": draw_tokens(image, result.ocr, style.line_color, style),
    }
    if result.text is not None:
        outputs["ocr_words"] = draw_tokens(image, result.text.words, style.word_color,
                                           style, show_conf=False, use_polygon=True)
    if any(s.key == "map" and s.status == DONE for s in result.stages):
        outputs["entities"] = draw_entities(image, result.entities, style)
    if result.is_complete and result.redacted_image is not None:
        outputs["redacted"] = result.redacted_image
    return outputs


OUTPUT_TITLES: Dict[str, str] = {
    "redacted": "Redacted image (Phase 8)",
    "entities": "Final entities (Phase 7)",
    "ocr_words": "Word boxes (Phase 3)",
    "ocr_lines": "OCR line boxes (Phase 2)",
}

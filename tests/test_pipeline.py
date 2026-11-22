"""Pipeline wiring, preprocessing coordinate tracking, config, and rendering.

Uses a fake OCR engine so these tests need no ONNX models.
"""
import logging

import cv2
import numpy as np
import pytest

from app.config.loader import load_config
from app.ocr.engine import OCREngine
from app.pipeline.document_pipeline import DocumentPipeline
from app.preprocessing.preprocess import Preprocessor
from app.schemas.ocr import OCRToken
from app.utils.geometry import apply_affine_to_points, invert_affine, polygon_to_bbox
from app.utils.logging import configure_logging
from app.visualization.render import render_outputs


class FakeEngine(OCREngine):
    """Returns fixed lines already in original coordinates."""

    def detect(self, image, transform, original_size):
        return [
            OCRToken(id=1, text="Name: Amit Sharma", confidence=0.99, bbox=[10, 10, 200, 40],
                     polygon=[[10, 10], [200, 10], [200, 40], [10, 40]]),
            OCRToken(id=2, text="Phone: 9000000000", confidence=0.95, bbox=[10, 60, 220, 90],
                     polygon=[[10, 60], [220, 60], [220, 90], [10, 90]]),
        ], 1.0


@pytest.fixture
def cfg():
    return load_config()


def blank(w=400, h=200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_pipeline_runs_implemented_stages_and_reports_pending(cfg):
    result = DocumentPipeline(cfg=cfg, engine=FakeEngine()).run_image(blank(), "doc.png")
    done = [s.key for s in result.stages if s.status == "done"]
    assert done == ["preprocess", "ocr", "reconstruct", "detect_regex", "detect_names"]
    assert [c.text for c in result.candidates if c.label == "PERSON_NAME"] == ["Amit Sharma"]
    phone = [c for c in result.candidates if c.label == "PHONE_NUMBER"]
    assert [c.text for c in phone] == ["9000000000"]
    assert result.to_dict()["candidates"][0]["detector"] == "regex"
    assert not result.is_complete
    assert result.word_count == 5      # Name: / Amit / Sharma / Phone: / 9000000000
    assert result.text.full_text == "Name: Amit Sharma\nPhone: 9000000000"
    d = result.to_dict()
    assert d["pipeline_complete"] is False and d["text"]["words"][0]["start"] == 0


def test_redacted_output_only_when_pipeline_complete(cfg):
    image = blank()
    result = DocumentPipeline(cfg=cfg, engine=FakeEngine()).run_image(image, "doc.png")
    outputs = render_outputs(image, result, cfg)
    assert set(outputs) == {"ocr_lines", "ocr_words"}
    for s in result.stages:          # simulate a future, fully-built pipeline
        s.status = "done"
    assert "redacted" in render_outputs(image, result, cfg)


def test_pipeline_accepts_grayscale(cfg):
    gray = np.full((100, 100), 255, dtype=np.uint8)
    result = DocumentPipeline(cfg=cfg, engine=FakeEngine()).run_image(gray, "g.png")
    assert (result.image.width, result.image.height) == (100, 100)


def test_deskew_expands_canvas_and_round_trips():
    """A rotated, upscaled doc: corners must survive and points must map back."""
    canvas = blank(600, 300)
    for y in range(40, 280, 30):
        cv2.line(canvas, (20, y), (580, y), (0, 0, 0), 6)
    rot = cv2.getRotationMatrix2D((300, 150), 6.0, 1.0)
    skewed = cv2.warpAffine(canvas, rot, (600, 300), borderValue=(255, 255, 255))

    cfg = load_config()
    pre = Preprocessor(cfg).run(skewed)
    h, w = pre.image.shape[:2]
    assert w > 1000 and h > 500      # upscaled to min_long_side, then canvas grew

    corners = [[0, 0], [599, 0], [599, 299], [0, 299]]
    mapped = apply_affine_to_points(corners, pre.transform)
    assert (mapped >= -1).all() and (mapped[:, 0] <= w + 1).all() and (mapped[:, 1] <= h + 1).all()
    back = apply_affine_to_points(mapped.tolist(), invert_affine(pre.transform))
    assert np.allclose(back, corners, atol=1e-6)


def test_polygon_fully_outside_image_never_inverts():
    x1, y1, x2, y2 = polygon_to_bbox([[120, 5], [150, 5], [150, 20], [120, 20]],
                                     clip_w=100, clip_h=100)
    assert x1 <= x2 and y1 <= y2


def test_config_loader(tmp_path, cfg):
    assert cfg.preprocessing.deskew is True
    assert cfg.reconstruction.get("row_overlap") == 0.5
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_logging_level_comes_from_config():
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING
    configure_logging("INFO")

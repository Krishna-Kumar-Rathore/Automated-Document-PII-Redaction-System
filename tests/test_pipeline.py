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


class FakeNER:
    """Stands in for GLiNER: same predict_entities() interface, no download."""

    def predict_entities(self, text, labels, threshold=0.5):
        i = text.find("Amit Sharma")
        return [{"start": i, "end": i + 11, "text": "Amit Sharma", "label": "person",
                 "score": 0.91}] if i >= 0 else []


class BrokenNER:
    def predict_entities(self, text, labels, threshold=0.5):
        raise RuntimeError("model file missing")


def make_pipeline(cfg, ner_model=None):
    return DocumentPipeline(cfg=cfg, engine=FakeEngine(), ner_model=ner_model or FakeNER())


@pytest.fixture
def cfg():
    return load_config()


def blank(w=400, h=200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_pipeline_runs_implemented_stages_and_reports_pending(cfg):
    result = make_pipeline(cfg).run_image(blank(), "doc.png")
    done = [s.key for s in result.stages if s.status == "done"]
    assert done == ["preprocess", "ocr", "reconstruct", "detect_regex", "detect_names",
                    "detect_ner", "resolve", "map"]
    ents = {(e.label, e.text) for e in result.entities}
    assert ("PERSON_NAME", "Amit Sharma") in ents and ("PHONE_NUMBER", "9000000000") in ents
    assert all(e.bboxes for e in result.entities)
    names = sorted(c.detector for c in result.candidates if c.text == "Amit Sharma")
    assert names == ["gliner", "name_trie"]       # two detectors agree
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
    result = make_pipeline(cfg).run_image(image, "doc.png")
    outputs = render_outputs(image, result, cfg)
    assert set(outputs) == {"ocr_lines", "ocr_words", "entities"}   # no "redacted" yet
    for s in result.stages:          # simulate a future, fully-built pipeline
        s.status = "done"
    assert "redacted" in render_outputs(image, result, cfg)


def test_pipeline_accepts_grayscale(cfg):
    gray = np.full((100, 100), 255, dtype=np.uint8)
    result = make_pipeline(cfg).run_image(gray, "g.png")
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


def test_ner_candidates_land_in_result(cfg):
    result = make_pipeline(cfg).run_image(blank(), "doc.png")
    ner = [c for c in result.candidates if c.detector == "gliner"]
    assert [(c.label, c.text) for c in ner] == [("PERSON_NAME", "Amit Sharma")]
    assert result.text.full_text[ner[0].start:ner[0].end] == "Amit Sharma"


def test_ner_failure_is_reported_and_blocks_final_output(cfg):
    result = make_pipeline(cfg, ner_model=BrokenNER()).run_image(blank(), "doc.png")
    stage = next(s for s in result.stages if s.key == "detect_ner")
    assert stage.status == "failed" and "model file missing" in stage.note
    assert stage in result.pending_stages
    # Earlier detectors still ran.
    assert any(c.detector == "regex" for c in result.candidates)
    for s in result.stages:          # even if every other stage were built...
        if s.key != "detect_ner":
            s.status = "done"
    assert not result.is_complete    # ...a failed stage still blocks redaction


def test_ner_disabled_is_skipped_and_does_not_block(cfg):
    cfg.get("ner")["enabled"] = False
    result = make_pipeline(cfg).run_image(blank(), "doc.png")
    stage = next(s for s in result.stages if s.key == "detect_ner")
    assert stage.status == "skipped"
    assert stage not in result.pending_stages

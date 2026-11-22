"""Phase 8: redaction policy, true pixel fill, verification, leak check."""
import numpy as np
import pytest

from app.config.loader import load_config
from app.ocr.engine import OCREngine
from app.pipeline.document_pipeline import DocumentPipeline
from app.redaction.redactor import Redactor
from app.schemas.entities import ResolvedEntity
from app.schemas.ocr import OCRToken


def ent(i, label, text, boxes):
    return ResolvedEntity(entity_id=i, text=text, label=label, confidence=0.9,
                          start=0, end=len(text), bboxes=boxes)


def noisy_image(w=300, h=120, seed=0):
    return np.random.default_rng(seed).integers(1, 255, (h, w, 3), dtype=np.uint8)


@pytest.fixture
def cfg():
    return load_config()


def test_solid_fill_is_opaque_and_touches_nothing_else(cfg):
    img = noisy_image()
    name = ent(1, "PERSON_NAME", "Amit Sharma", [[10, 10, 110, 40]])
    out = Redactor(cfg).apply(img, [name])
    assert np.all(out[10:40, 10:110] == 0)                    # every pixel replaced
    mask = np.ones(img.shape[:2], bool)
    mask[10:40, 10:110] = False
    assert np.array_equal(out[mask], img[mask])               # rest untouched
    assert np.array_equal(img, noisy_image())                 # input not mutated
    assert name.attributes["redacted"] and name.attributes["verified"]


def test_policy_keeps_organizations_visible_with_reason(cfg):
    img = noisy_image()
    org = ent(1, "ORGANIZATION", "GOVERNMENT OF INDIA", [[0, 0, 200, 30]])
    out = Redactor(cfg).apply(img, [org])
    assert np.array_equal(out, img)
    assert org.attributes == {"redacted": False, "reason": "label not in redaction.labels"}


def test_multi_line_entity_redacts_every_box(cfg):
    addr = ent(1, "ADDRESS", "12 MG Road,\nPune", [[10, 10, 200, 35], [10, 45, 120, 70]])
    out = Redactor(cfg).apply(noisy_image(), [addr])
    assert np.all(out[10:35, 10:200] == 0) and np.all(out[45:70, 10:120] == 0)
    assert np.any(out[36:44, 10:200] != 0)                     # gap between lines kept


def test_entity_without_boxes_is_flagged_not_silently_passed(cfg):
    e = ent(1, "EMAIL", "a@b.com", [])
    Redactor(cfg).apply(noisy_image(), [e])
    assert e.attributes["redacted"] is False and "no pixel boxes" in e.attributes["reason"]


def test_labeled_style_writes_text_on_a_fully_filled_box(cfg):
    cfg.get("redaction")["style"] = "labeled"
    e = ent(1, "PERSON_NAME", "Amit Sharma", [[10, 10, 200, 50]])
    out = Redactor(cfg).apply(noisy_image(), [e])
    region = out[10:50, 10:200]
    assert e.attributes["verified"]                           # checked before text
    assert np.any(region != 0)                                # "[NAME]" drawn
    assert set(np.unique(region)) <= set(range(256))
    # Only fill colour and the anti-aliased white label: no original noise survives.
    assert np.all((region == region[..., :1]).all(axis=-1))   # grey-scale only


def test_find_leaks(cfg):
    r = Redactor(cfg)
    e = ent(1, "PHONE_NUMBER", "+91 9000000000", [[0, 0, 10, 10]])
    e.attributes["redacted"] = True
    assert r.find_leaks([e], "Phone: 9000000000")             # still readable -> leak
    assert r.find_leaks([e], "Phone:") == []


class SequenceEngine(OCREngine):
    """First call: real-looking lines. Later calls (leak check): ``after``."""

    def __init__(self, after):
        self.calls, self.after = 0, after

    def detect(self, image, transform, original_size):
        self.calls += 1
        lines = [OCRToken(id=1, text="Phone: 9000000000", confidence=0.95,
                          bbox=[10, 10, 220, 40])]
        return (lines if self.calls == 1 else self.after), 1.0


class NoNER:
    def predict_entities(self, text, labels, threshold=0.5):
        return []


def run(cfg, after):
    engine = SequenceEngine(after)
    p = DocumentPipeline(cfg=cfg, engine=engine, ner_model=NoNER())
    return p.run_image(np.full((100, 300, 3), 255, np.uint8), "doc.png"), engine


def test_pipeline_leak_check_passes_when_text_is_gone(cfg):
    result, engine = run(cfg, after=[OCRToken(id=1, text="Phone:", confidence=0.9,
                                              bbox=[10, 10, 80, 40])])
    assert engine.calls == 2                                  # OCR ran again on output
    assert result.redaction["leak_check"]["passed"] is True
    assert result.redaction["pixel_check_passed"] is True
    assert result.redaction["redacted"] == 1


def test_pipeline_leak_check_reports_readable_text(cfg):
    leaked = [OCRToken(id=1, text="Phone: 9000000000", confidence=0.9, bbox=[10, 10, 220, 40])]
    result, _ = run(cfg, after=leaked)
    lc = result.redaction["leak_check"]
    assert lc["passed"] is False and lc["leaks"][0]["label"] == "PHONE_NUMBER"


def test_leak_check_can_be_disabled(cfg):
    cfg.get("redaction")["leak_check"] = False
    result, engine = run(cfg, after=[])
    assert engine.calls == 1 and result.redaction["leak_check"] == {"ran": False}


def test_failed_upstream_stage_blocks_redaction_entirely(cfg):
    class Broken:
        def predict_entities(self, *a, **k):
            raise RuntimeError("boom")
    p = DocumentPipeline(cfg=cfg, engine=SequenceEngine([]), ner_model=Broken())
    result = p.run_image(np.full((100, 300, 3), 255, np.uint8), "doc.png")
    redact = next(s for s in result.stages if s.key == "redact")
    assert redact.status == "pending" and "blocked" in redact.note
    assert result.redacted_image is None                      # never even produced


def test_leak_kinds_full_and_partial(cfg):
    r = Redactor(cfg)
    e = ent(1, "PHONE_NUMBER", "+91 9000000000", [[0, 0, 10, 10]])
    e.attributes["redacted"] = True
    assert r.find_leaks([e], "call +91 9000000000")[0]["match"] == "full"
    partial = r.find_leaks([e], "Phone: 9000000000")[0]
    assert partial["match"] == "partial" and partial["readable"] == ["9000000000"]

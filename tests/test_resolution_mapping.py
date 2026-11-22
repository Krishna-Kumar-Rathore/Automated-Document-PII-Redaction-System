"""Phase 7: entity resolution + spatial mapping. All data synthetic."""
import pytest

from app.config.loader import load_config
from app.mapping.spatial import SpatialMapper, word_slice
from app.resolution.resolver import EntityResolver
from app.schemas.entities import EntityCandidate
from app.schemas.ocr import OCRToken
from app.schemas.text import ReconstructedText
from app.text.reconstruct import TextReconstructor


def cand(text, full, label, detector, conf, occurrence=0, **attrs):
    start = -1
    for _ in range(occurrence + 1):
        start = full.index(text, start + 1)
    return EntityCandidate(text=text, label=label, start=start, end=start + len(text),
                           confidence=conf, detector=detector, attributes=attrs)


def rt(full):
    return ReconstructedText(full_text=full, lines=[], words=[])


@pytest.fixture
def cfg():
    return load_config()


def resolve(cfg, full, cands):
    return EntityResolver(cfg).resolve(cands, rt(full))


# --- scoring ---------------------------------------------------------------------

def test_agreement_raises_confidence_noisy_or(cfg):
    full = "DOB:01/01/1990"
    regex = cand("01/01/1990", full, "DATE", "regex", 0.75)
    ctx = cand("01/01/1990", full, "DATE", "context_rule", 0.80)
    [ent] = resolve(cfg, full, [regex, ctx])
    # 1 - (1 - .95*.75) * (1 - .70*.80) = 0.8735
    assert ent.confidence == pytest.approx(0.8735, abs=1e-3)
    assert ent.detectors == ["context_rule", "regex"]
    assert len(ent.attributes["evidence"]) == 2


def test_bare_date_is_below_the_date_threshold(cfg):
    full = "Meeting on 01/01/2024"
    assert resolve(cfg, full, [cand("01/01/2024", full, "DATE", "regex", 0.75)]) == []


def test_same_detector_votes_once(cfg):
    full = "Call +91 9000000000"
    a = cand("+91 9000000000", full, "PHONE_NUMBER", "regex", 0.95)
    b = cand("9000000000", full, "PHONE_NUMBER", "regex", 0.95)
    [ent] = resolve(cfg, full, [a, b])
    assert ent.confidence == pytest.approx(0.95 * 0.95, abs=1e-3)
    assert ent.text == "+91 9000000000"                      # union of the spans


def test_dictionary_only_name_needs_a_second_detector(cfg):
    full = "met Sarah Johnson today"
    trie = cand("Sarah Johnson", full, "PERSON_NAME", "name_trie", 0.74)
    assert resolve(cfg, full, [trie]) == []                  # .74*.55 = .41 < .50
    ner = cand("Sarah Johnson", full, "PERSON_NAME", "gliner", 0.91)
    [ent] = resolve(cfg, full, [trie, ner])
    assert ent.confidence > 0.8


def test_label_only_name_survives(cfg):
    full = "Name: Fooby Barbaz"
    [ent] = resolve(cfg, full, [cand("Fooby Barbaz", full, "PERSON_NAME", "context_rule", 0.74)])
    assert ent.confidence >= 0.5


def test_unknown_labels_are_dropped(cfg):
    full = "zap"
    assert resolve(cfg, full, [cand("zap", full, "SPACESHIP", "regex", 0.99)]) == []


# --- overlaps --------------------------------------------------------------------

def test_cross_label_overlap_keeps_winner_and_records_loser(cfg):
    full = "Aadhaar 2345 6789 0123"
    aadhaar = cand("2345 6789 0123", full, "ID_NUMBER", "regex", 0.90)
    acct = cand("2345 6789 0123", full, "ACCOUNT_NUMBER", "context_rule", 0.80)
    [ent] = resolve(cfg, full, [aadhaar, acct])
    assert ent.label == "ID_NUMBER"
    assert ent.attributes["suppressed"][0]["label"] == "ACCOUNT_NUMBER"


def test_longest_span_strategy(cfg):
    cfg.get("resolver")["overlap_strategy"] = "longest_span"
    full = "Visit Tata Consultancy Services"
    short = cand("Tata", full, "PERSON_NAME", "gliner", 0.95)
    long_ = cand("Tata Consultancy Services", full, "ORGANIZATION", "gliner", 0.70)
    [ent] = resolve(cfg, full, [short, long_])
    assert ent.label == "ORGANIZATION"


def test_partial_overlap_keeps_both_so_nothing_is_left_uncovered(cfg):
    full = "Ref ABCD 1234 5678"
    a = cand("ABCD 1234", full, "ID_NUMBER", "context_rule", 0.80)
    b = cand("1234 5678", full, "ACCOUNT_NUMBER", "context_rule", 0.80)
    # overlap "1234" is < 50% of either span -> both kept
    assert {e.label for e in resolve(cfg, full, [a, b])} == {"ID_NUMBER", "ACCOUNT_NUMBER"}


# --- address merging -------------------------------------------------------------

def test_address_pieces_merge_absorb_building_and_attach_pin(cfg):
    full = "Flat 3B, Lake View Apartments, Banjara Hills,\nHyderabad 500034"
    pieces = [cand("Flat 3B", full, "ADDRESS", "gliner", 0.58),
              cand("Lake View Apartments", full, "ORGANIZATION", "gliner", 0.52),
              cand("Banjara Hills", full, "ADDRESS", "gliner", 0.59),
              cand("Hyderabad", full, "ADDRESS", "gliner", 0.70)]
    [ent] = resolve(cfg, full, pieces)
    assert ent.label == "ADDRESS"
    assert ent.text == full                                   # whole thing, incl. PIN
    assert ent.confidence == pytest.approx(0.70 * 0.80, abs=1e-3)   # max piece, not OR
    assert "PIN: 500034" in ent.attributes["merged_from"]
    assert any(m.startswith("ORGANIZATION") for m in ent.attributes["merged_from"])


def test_address_does_not_merge_across_real_words(cfg):
    full = "Sector 12 is far from Dwarka"
    pieces = [cand("Sector 12", full, "ADDRESS", "gliner", 0.84),
              cand("Dwarka", full, "ADDRESS", "gliner", 0.66)]
    assert [e.text for e in resolve(cfg, full, pieces)] == ["Sector 12", "Dwarka"]


# --- spatial mapping -------------------------------------------------------------

def text_from_lines(*specs):
    """specs: (text, bbox) per OCR line; one line per row."""
    toks = [OCRToken(id=i + 1, text=t, confidence=0.9, bbox=list(b))
            for i, (t, b) in enumerate(specs)]
    return TextReconstructor(load_config()).run(toks, image_size=(1000, 1000))


def entity_for(full, text, label="DATE"):
    from app.schemas.entities import ResolvedEntity
    s = full.index(text)
    return ResolvedEntity(entity_id=1, text=text, label=label, confidence=0.9,
                          start=s, end=s + len(text))


def test_partial_word_gets_sub_word_box():
    t = text_from_lines(("DOB:01/01/1990", (100, 50, 380, 80)))
    ent = entity_for(t.full_text, "01/01/1990")
    cfg = load_config()
    cfg.get("mapping")["padding_px"] = 0
    [mapped] = SpatialMapper(cfg).map([ent], t, (1000, 1000))
    [(x1, y1, x2, y2)] = mapped.bboxes
    assert 150 < x1 < 200 and x2 == 380                       # "DOB:" left uncovered
    assert (y1, y2) == (50, 80)
    assert mapped.source_tokens == [1]


def test_multi_line_entity_gets_one_box_per_line():
    t = text_from_lines(("12 MG Road,", (100, 100, 300, 130)),
                        ("Pune 411005", (100, 140, 280, 170)))
    ent = entity_for(t.full_text, t.full_text, label="ADDRESS")
    cfg = load_config()
    cfg.get("mapping")["padding_px"] = 0
    [mapped] = SpatialMapper(cfg).map([ent], t, (1000, 1000))
    assert mapped.bboxes == [[100, 100, 300, 130], [100, 140, 280, 170]]
    assert mapped.attributes["lines"] == [1, 2]


def test_padding_is_clipped_to_image():
    t = text_from_lines(("Amit", (0, 0, 60, 20)))
    ent = entity_for(t.full_text, "Amit", label="PERSON_NAME")
    [mapped] = SpatialMapper(load_config()).map([ent], t, (60, 20))
    assert mapped.bboxes == [[0, 0, 60, 20]]


def test_word_slice_whole_word_returns_its_polygon():
    t = text_from_lines(("Amit Sharma", (0, 0, 220, 30)))
    w = t.words[1]
    assert word_slice(w, w.start, w.end) == w.polygon

"""Phase 6: GLiNER detector - chunking and offset mapping (fake model, no download)."""
from app.config.loader import load_config
from app.detection.ner_detector import NERDetector, chunk_rows
from app.schemas.text import ReconstructedText


class RecordingNER:
    """Finds fixed phrases in whatever chunk it is given; records the chunks."""

    def __init__(self, phrases):
        self.phrases = phrases          # {phrase: prompt}
        self.chunks, self.labels = [], None

    def predict_entities(self, text, labels, threshold=0.5):
        self.chunks.append(text)
        self.labels = labels
        out = []
        for phrase, prompt in self.phrases.items():
            i = text.find(phrase)
            if i >= 0:
                out.append({"start": i, "end": i + len(phrase), "text": phrase,
                            "label": prompt, "score": 0.876})
        return out


def rt(text):
    return ReconstructedText(full_text=text, lines=[], words=[])


def test_chunks_are_whole_rows_within_word_limit():
    text = "a b c\nd e\nf g h i\nj"
    chunks = chunk_rows(text, max_words=5)
    assert [text[s:e] for s, e in chunks] == ["a b c\nd e", "f g h i\nj"]


def test_overlong_row_is_split_on_words_and_blank_rows_skipped():
    text = "w1 w2 w3 w4 w5\n\n   \nx y"
    chunks = chunk_rows(text, max_words=2)
    assert [text[s:e] for s, e in chunks] == ["w1 w2", "w3 w4", "w5", "x y"]


def test_entities_map_back_to_full_text_offsets():
    cfg = load_config()
    cfg.get("ner")["max_words_per_chunk"] = 6     # rows: 4 + 3 words, then 6
    model = RecordingNER({"Fooby Barbaz": "person", "12 MG Road, Pune": "address"})
    det = NERDetector(cfg, model=model)
    text = rt("Header line one two\nName Fooby Barbaz\nLives at 12 MG Road, Pune")
    cands = det.detect(text)
    assert len(model.chunks) >= 2                    # really was chunked
    assert {(c.label, c.text) for c in cands} == {("PERSON_NAME", "Fooby Barbaz"),
                                                  ("ADDRESS", "12 MG Road, Pune")}
    for c in cands:
        assert text.full_text[c.start:c.end] == c.text
        assert c.detector == "gliner" and c.confidence == 0.876


def test_prompts_come_from_config_and_unmapped_labels_are_dropped():
    cfg = load_config()
    model = RecordingNER({"Acme": "spaceship"})      # not in the label map
    cands = NERDetector(cfg, model=model).detect(rt("Acme corp"))
    assert model.labels == ["person", "address", "organization"]
    assert cands == []


def test_model_is_not_loaded_until_needed():
    det = NERDetector(load_config())
    assert not det.loaded                            # constructing never downloads


def test_precision_filters_drop_titles_and_common_nouns():
    model = RecordingNER({"The Prime Minister": "person", "farmers": "organization",
                          "Minister Rao": "person", "Infosys": "organization"})
    cands = NERDetector(load_config(), model=model).detect(
        rt("The Prime Minister and Minister Rao met farmers at Infosys"))
    assert {(c.label, c.text) for c in cands} == {("PERSON_NAME", "Minister Rao"),
                                                  ("ORGANIZATION", "Infosys")}

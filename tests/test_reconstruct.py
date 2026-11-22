"""Phase 3: text reconstruction - reading order, offsets, and word boxes."""
from app.config.loader import Config
from app.schemas.ocr import OCRToken
from app.text.reconstruct import TextReconstructor


def line(id_, text, bbox, polygon=None):
    x1, y1, x2, y2 = bbox
    poly = polygon or [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return OCRToken(id=id_, text=text, confidence=0.9, bbox=list(bbox), polygon=poly)


def recon(lines, **cfg):
    return TextReconstructor(Config({"reconstruction": cfg})).run(lines)


def test_reading_order_rows_and_columns():
    # Engine order is scrambled; "Name:" and its value are separate boxes on one row.
    lines = [
        line(3, "Phone: 123", (60, 200, 300, 230)),
        line(2, "Amit Sharma", (200, 100, 400, 130)),
        line(1, "Name:", (60, 102, 150, 128)),
    ]
    rt = recon(lines)
    assert rt.full_text == "Name: Amit Sharma\nPhone: 123"
    assert rt.rows == [[1, 2], [3]]
    assert [ln.id for ln in rt.lines] == [1, 2, 3]


def test_offsets_round_trip_for_every_token():
    lines = [line(1, "Email:  a.b@example.com", (0, 0, 460, 30)),
             line(2, "DOB: 01/01/1990", (0, 50, 300, 80))]
    rt = recon(lines)
    for tok in rt.lines + rt.words:
        assert rt.full_text[tok.start:tok.end] == tok.text
    assert [w.text for w in rt.words] == ["Email:", "a.b@example.com", "DOB:", "01/01/1990"]
    assert all(w.parent_id in (1, 2) and w.granularity == "word" for w in rt.words)
    assert [w.id for w in rt.words] == [1, 2, 3, 4]


def test_span_maps_back_to_word_tokens():
    rt = recon([line(1, "Name: Amit Kumar Sharma", (0, 0, 460, 30))])
    start = rt.full_text.index("Amit")
    end = start + len("Amit Kumar Sharma")
    assert [w.text for w in rt.words_for_span(start, end)] == ["Amit", "Kumar", "Sharma"]
    # A partial overlap still returns the word it touches.
    assert [w.text for w in rt.words_for_span(start + 1, start + 2)] == ["Amit"]


def test_word_boxes_tile_the_line_in_order():
    rt = recon([line(1, "aa bb cc", (100, 10, 400, 40))])
    boxes = [w.bbox for w in rt.words]
    assert boxes[0][0] == 100 and boxes[-1][2] == 400          # spans the line
    assert all(b[1] == 10 and b[3] == 40 for b in boxes)        # same height
    assert all(boxes[i][2] < boxes[i + 1][0] for i in range(len(boxes) - 1))


def test_slanted_line_gives_slanted_word_polygons():
    poly = [[0, 0], [200, 20], [200, 50], [0, 30]]              # slopes downward
    rt = recon([line(1, "left right", (0, 0, 200, 50), polygon=poly)])
    left, right = rt.words
    assert right.polygon[0][1] > left.polygon[0][1]            # right word sits lower


def test_custom_row_separator_and_empty_input():
    rt = recon([line(1, "A", (0, 0, 10, 10)), line(2, "B", (20, 0, 30, 10))],
               row_separator=" | ")
    assert rt.full_text == "A | B"
    empty = recon([])
    assert empty.full_text == "" and empty.words == []

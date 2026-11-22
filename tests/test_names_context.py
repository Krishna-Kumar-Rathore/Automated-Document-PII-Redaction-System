"""Phase 5: name trie, name detector, and context rules. All data synthetic."""
import pytest

from app.config.loader import load_config
from app.detection.context_detector import ContextRuleDetector
from app.detection.name_detector import NameDetector
from app.detection.trie import NameTrie
from app.schemas.ocr import OCRToken
from app.text.reconstruct import TextReconstructor


# --- helpers -------------------------------------------------------------------

def text_of(*rows):
    toks = [OCRToken(id=i + 1, text=t, confidence=0.9, bbox=[0, 40 * i, 600, 40 * i + 30])
            for i, t in enumerate(rows)]
    return TextReconstructor(load_config()).run(toks)


@pytest.fixture(scope="module")
def names():
    return NameDetector(load_config())


@pytest.fixture(scope="module")
def context():
    return ContextRuleDetector(load_config())


def detect(detector, *rows):
    rt = text_of(*rows)
    cands = detector.detect(rt)
    for c in cands:
        assert rt.full_text[c.start:c.end] == c.text
    return cands


def texts(cands, label=None):
    return [c.text for c in cands if label is None or c.label == label]


# --- trie ------------------------------------------------------------------------

def test_trie_exact_roles_and_case():
    t = NameTrie.from_lists({"first": ["Amit", "Kumar"], "middle": ["Kumar"], "last": ["Sharma"]})
    assert t.roles("AMIT") == {"first"}
    assert t.roles("kumar") == {"first", "middle"}
    assert "Sharm" not in t and "Sharmaa" not in t
    assert t.size == 3


def test_trie_fuzzy_edit_distance_one():
    t = NameTrie.from_lists({"last": ["Sharma", "Verma"], "first": ["Amit"]})
    assert t.fuzzy("Sharna")[0][:2] == ("sharma", 1)       # substitution
    assert t.fuzzy("Sharmaa")[0][:2] == ("sharma", 1)      # insertion
    assert t.fuzzy("Shrma")[0][:2] == ("sharma", 1)        # deletion
    assert t.fuzzy("Sharma")[0][:2] == ("sharma", 0)
    assert t.fuzzy("Shxxma") == []                         # distance 2


def test_bundled_name_lists_load(names):
    assert names.trie.size > 300
    assert names.trie.roles("Kumar") >= {"middle", "last"}


# --- name detector ---------------------------------------------------------------

def test_sample_card_name(names):
    cands = detect(names, "GOVERNMENT OF INDIA", "Name: Amit Kumar Sharma", "DOB:01/01/1990")
    assert texts(cands) == ["Amit Kumar Sharma"]
    c = cands[0]
    assert c.label == "PERSON_NAME" and c.detector == "name_trie"
    assert {"context_label", "dictionary", "dictionary_multi"} <= set(c.attributes["signals"])
    assert c.attributes["score"] == 4 and c.confidence == 0.98     # 0.5 + 0.12 * 4


def test_unknown_name_accepted_on_label_alone(names):
    cands = detect(names, "Applicant Name: Fooby Barbaz")
    assert texts(cands) == ["Fooby Barbaz"]
    assert cands[0].detector == "context_rule"
    assert cands[0].attributes["signals"] == ["context_label"]


@pytest.mark.parametrize("row,expected", [
    ("Mr. Fooby Barbaz attended", ["Fooby Barbaz"]),          # honorific
    ("Smt Lata Quux", ["Lata Quux"]),
    ("S/O: Ramesh Quux", ["Ramesh Quux"]),                   # relation marker
    ("Father's Name : RAMESH KUMAR", ["RAMESH KUMAR"]),      # ALL CAPS
    ("Name:Amit Sharma", ["Amit Sharma"]),                   # OCR dropped the space
    ("met Sarah Johnson yesterday", ["Sarah Johnson"]),      # 2 dictionary hits, no label
    ("Name: Amit Sharna", ["Amit Sharna"]),                  # fuzzy surname
])
def test_name_signals(names, row, expected):
    assert texts(detect(names, row)) == expected


def test_label_on_row_above(names):
    assert texts(detect(names, "Name", "Fooby Barbaz")) == ["Fooby Barbaz"]


def test_single_dictionary_hit_is_not_enough(names):
    # "Kiran" / "Lal" are list names, but one weak signal is below min_score.
    assert detect(names, "Kiran Industries reported growth") == []
    assert detect(names, "Lal Quila is crowded") == []


def test_newspaper_prose_does_not_light_up(names):
    cands = detect(names,
                   "The Prime Minister met farmers in New Delhi on Monday.",
                   "Reserve Bank Of India Raises Rates Again",
                   "GOVERNMENT OF INDIA",
                   "Hope And Grace Foundation Annual Report")
    assert cands == []


def test_non_person_label_is_rejected(names):
    assert detect(names, "Company Name: Acme Widgets") == []
    assert detect(names, "User Name: Admin") == []


def test_commas_separate_people(names):
    cands = detect(names, "Witnesses: Mr. Fooby Barbaz, Mrs. Lata Quux")
    assert texts(cands) == ["Fooby Barbaz", "Lata Quux"]


def test_long_capitalised_heading_is_not_a_name(names):
    assert detect(names, "Name: Amit Kumar Sharma Rahul Verma Gupta") == []


# --- context rules ---------------------------------------------------------------

def test_dob_and_yob(context):
    cands = detect(context, "DOB:01/01/1990", "Date of Birth - 14 Aug 1995",
                   "Year of Birth: 1987")
    assert texts(cands, "DATE") == ["01/01/1990", "14 Aug 1995", "1987"]
    assert all(c.detector == "context_rule" for c in cands)


def test_account_number_only_with_label(context):
    assert texts(detect(context, "A/c No. 1234 5678 9012"), "ACCOUNT_NUMBER") == ["1234 5678 9012"]
    assert texts(detect(context, "Account Number: 001234567890"), "ACCOUNT_NUMBER") == ["001234567890"]
    assert detect(context, "Invoice 1234 5678 9012") == []         # no label, no candidate


def test_id_value_stops_before_next_label(context):
    cands = detect(context, "Aadhaar No: 2345 6789 0123 DOB 01/01/1990", "PAN: ABCPE1234F",
                   "Passport No K1234567")
    assert texts(cands, "ID_NUMBER") == ["2345 6789 0123", "ABCPE1234F", "K1234567"]
    assert texts(cands, "DATE") == ["01/01/1990"]


def test_context_never_crosses_rows(context):
    assert detect(context, "Account Holder", "9876 5432 1098") == []

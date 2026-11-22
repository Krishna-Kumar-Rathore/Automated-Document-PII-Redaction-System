"""Phase 4: regex detection + validators.

All identifiers below are synthetic. Aadhaar numbers are built with a computed
Verhoeff check digit (valid) or a deliberately wrong one (invalid).
"""
import pytest

from app.config.loader import load_config
from app.detection.regex_detector import RegexDetector, digit_view
from app.detection.validators import (aadhaar_valid, date_valid, indian_mobile_valid,
                                      pan_valid, upi_handle_known, verhoeff_check_digit,
                                      verhoeff_valid)
from app.schemas.ocr import OCRToken
from app.schemas.text import ReconstructedText
from app.text.reconstruct import TextReconstructor


def make_aadhaar(prefix11: str, valid: bool = True) -> str:
    check = verhoeff_check_digit(prefix11)
    d = prefix11 + str(check if valid else (check + 1) % 10)
    return f"{d[:4]} {d[4:8]} {d[8:]}"


VALID_AADHAAR = make_aadhaar("23456789012")
INVALID_AADHAAR = make_aadhaar("23456789012", valid=False)


@pytest.fixture(scope="module")
def detector():
    return RegexDetector(load_config())


def text_of(*lines: str) -> ReconstructedText:
    toks = [OCRToken(id=i + 1, text=t, confidence=0.9, bbox=[0, 40 * i, 500, 40 * i + 30])
            for i, t in enumerate(lines)]
    return TextReconstructor(load_config()).run(toks)


def found(detector, *lines):
    rt = text_of(*lines)
    cands = detector.detect(rt)
    for c in cands:                       # offsets must always point at the text
        assert rt.full_text[c.start:c.end] == c.text
    return cands


def by_label(cands, label):
    return [c for c in cands if c.label == label]


# --- validators ---------------------------------------------------------------

def test_verhoeff_known_vector_and_round_trip():
    assert verhoeff_valid("2363")              # classic example: 236 -> check digit 3
    assert not verhoeff_valid("2364")
    assert verhoeff_check_digit("236") == 3
    assert aadhaar_valid(VALID_AADHAAR) and not aadhaar_valid(INVALID_AADHAAR)
    assert not aadhaar_valid("1234 5678 9012")  # cannot start with 0/1


@pytest.mark.parametrize("value,ok", [
    ("ABCPE1234F", True), ("ABCDE1234F", False),   # 4th char D is not a holder type
    ("abcpe1234f", True), ("ABCP1234F", False),
])
def test_pan(value, ok):
    assert pan_valid(value) is ok


@pytest.mark.parametrize("value,ok", [
    ("+91 98765 43210", True), ("09876543210", True), ("9876543210", True),
    ("919876543210", True), ("5876543210", False), ("080-23456789", False),
])
def test_indian_mobile(value, ok):
    assert indian_mobile_valid(value) is ok


@pytest.mark.parametrize("value,ok", [
    ("14/08/1995", True), ("31-12-2001", True), ("1995-08-14", True),
    ("12/31/2001", True),                      # falls back to M/D/Y
    ("31/02/2001", False), ("45/13/2001", False), ("01/01/1850", False),
    ("14 Aug 1995", True), ("Aug 14, 1995", True), ("30 Feb 1995", False),
])
def test_date(value, ok):
    assert date_valid(value) is ok


def test_upi_handle():
    assert upi_handle_known("amit@okaxis", ["okaxis"])
    assert not upi_handle_known("amit@randombank", ["okaxis"])


# --- detector -------------------------------------------------------------------

def test_sample_card_lines(detector):
    cands = found(detector, "Name: Amit Kumar Sharma", "DOB:01/01/1990",
                  "Phone: +91 9000000000", "Email: amit.sharma@example.com")
    labels = {(c.label, c.text) for c in cands}
    assert ("DATE", "01/01/1990") in labels                 # works with no space after ':'
    assert ("PHONE_NUMBER", "+91 9000000000") in labels
    assert ("EMAIL", "amit.sharma@example.com") in labels
    assert not by_label(cands, "URL")                        # email domain is not a URL
    assert not by_label(cands, "UPI_ID")                     # email is not a UPI ID


def test_validators_boost_or_penalise_but_never_drop(detector):
    good = by_label(found(detector, f"Aadhaar: {VALID_AADHAAR}"), "ID_NUMBER")
    bad = by_label(found(detector, f"Aadhaar: {INVALID_AADHAAR}"), "ID_NUMBER")
    assert len(good) == 1 and len(bad) == 1                  # both emitted
    assert good[0].attributes["validated"] is True
    assert bad[0].attributes["validated"] is False
    assert good[0].confidence > bad[0].confidence


def test_upi_vs_email_vs_url(detector):
    cands = found(detector, "Pay to amit.s@okaxis or 9000000000@paytm.",
                  "Visit https://example.com/pay?id=1, or www.example.in.",
                  "Contact: help@example.co.in")
    upi = {c.text: c for c in by_label(cands, "UPI_ID")}
    assert set(upi) == {"amit.s@okaxis", "9000000000@paytm"}
    assert upi["amit.s@okaxis"].attributes["validated"] is True
    assert {c.text for c in by_label(cands, "URL")} == {"https://example.com/pay?id=1",
                                                       "www.example.in"}
    assert [c.text for c in by_label(cands, "EMAIL")] == ["help@example.co.in"]


@pytest.mark.parametrize("phone", ["98765 43210", "+91-98765-43210", "0091 9876543210",
                                   "987-654-3210", "080-23456789", "+44 20 7946 0958"])
def test_phone_formats(detector, phone):
    assert [c.text for c in by_label(found(detector, f"Tel {phone} now"),
                                     "PHONE_NUMBER")] == [phone]


def test_landline_scores_lower_than_mobile(detector):
    mobile = by_label(found(detector, "98765 43210"), "PHONE_NUMBER")[0]
    landline = by_label(found(detector, "080-23456789"), "PHONE_NUMBER")[0]
    assert mobile.confidence > landline.confidence


def test_pan_ifsc_and_textual_dates(detector):
    cands = found(detector, "PAN ABCPE1234F IFSC SBIN0001234",
                  "Issued 14th August 1995 and Aug 14, 1995")
    assert [c.text for c in by_label(cands, "ID_NUMBER")] == ["ABCPE1234F"]
    assert [c.text for c in by_label(cands, "OTHER_PII")] == ["SBIN0001234"]
    assert [c.text for c in by_label(cands, "DATE")] == ["14th August 1995", "Aug 14, 1995"]


def test_ocr_digit_confusions_are_corrected(detector):
    cands = found(detector, "Mobile: 9OOO0 l2345")          # O->0, l->1
    phone = by_label(cands, "PHONE_NUMBER")
    assert [c.text for c in phone] == ["9OOO0 l2345"]       # reports what the image shows
    assert phone[0].attributes["ocr_corrected"] == "90000 12345"
    assert phone[0].attributes["validated"] is True


def test_digit_view_leaves_words_alone():
    for text in ["BOSS 2024", "SOIL", "14Oct1995", "D0B", "B2B"]:
        assert digit_view(text) == text
    assert digit_view("2O24") == "2024"
    assert len(digit_view("x9OOO0 l2345y")) == len("x9OOO0 l2345y")


def test_no_false_positives_on_plain_prose(detector):
    cands = found(detector, "The Prime Minister met 1200 farmers in 2024 at the",
                  "annual summit. Version 3.2 was released; see page 45.")
    assert cands == []


def test_disabled_patterns_do_not_run():
    cfg = load_config()
    cfg.get("regex")["enabled"] = ["EMAIL"]
    det = RegexDetector(cfg)
    rt = ReconstructedText(full_text="a@b.com 9876543210", lines=[], words=[])
    assert [c.label for c in det.detect(rt)] == ["EMAIL"]

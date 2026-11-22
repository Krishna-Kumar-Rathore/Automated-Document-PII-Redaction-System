"""Format validators for structured identifiers (Phase 4).

Each validator answers "is this string a well-formed X?". The regex detector uses
the answer only to raise or lower confidence - never to discard a match - because
OCR can garble a single digit of a genuine ID.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Iterable, Optional

# --- Verhoeff checksum (used by Aadhaar) -------------------------------------
# Dihedral group D5 multiplication table, permutation table.
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", text)


def verhoeff_valid(number: str) -> bool:
    """True if the digit string (last digit = check digit) passes Verhoeff."""
    digits = digits_only(number)
    if not digits:
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(number: str) -> int:
    """Check digit to append to ``number`` (used by tests / synthetic data)."""
    c = 0
    for i, ch in enumerate(reversed(digits_only(number))):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return _INV[c]


def aadhaar_valid(text: str) -> bool:
    """12 digits, first digit 2-9, valid Verhoeff checksum."""
    d = digits_only(text)
    return len(d) == 12 and d[0] not in "01" and verhoeff_valid(d)


# --- PAN ---------------------------------------------------------------------
# 4th character encodes the holder type: P=person, C=company, H=HUF, F=firm,
# A=AOP, T=trust, B=BOI, L=local authority, J=artificial juridical person,
# G=government, K=krish (agri).
_PAN_RE = re.compile(r"^[A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z]$")


def pan_valid(text: str) -> bool:
    return bool(_PAN_RE.match(text.upper()))


# --- Phone -------------------------------------------------------------------

_PREFIXES = ("0091", "91", "0")
_MOBILE_GROUPINGS = ([10], [5, 5], [3, 3, 4])


def indian_mobile_valid(text: str) -> bool:
    """10-digit Indian mobile starting 6-9, optionally prefixed 0 / 91 / +91 / 0091,
    written as 10 contiguous digits or grouped 5-5 / 3-3-4.

    Grouping matters: "080-23456789" has the same digits as "0 8023456789" but
    is a Bangalore landline (STD code 080), so digits alone are not enough.
    """
    groups = re.findall(r"\d+", text)
    if not groups:
        return False
    if len(groups) > 1 and groups[0] in _PREFIXES:          # "+91 98765 43210"
        groups = groups[1:]
    elif sum(map(len, groups)) > 10:                        # "+919876543210", "09876..."
        for prefix in _PREFIXES:
            if groups[0].startswith(prefix) and len(groups[0]) > len(prefix):
                groups[0] = groups[0][len(prefix):]
                break
    national = "".join(groups)
    return ([len(g) for g in groups] in _MOBILE_GROUPINGS
            and national[0] in "6789")


# --- Dates -------------------------------------------------------------------
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _real_date(y: int, m: int, d: int) -> bool:
    if y < 100:                       # two-digit year: assume 19xx / 20xx
        y += 2000 if y < 50 else 1900
    if not 1900 <= y <= 2100:
        return False
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def date_valid(text: str) -> bool:
    """Real calendar date. Numeric dates are read D/M/Y (Indian order), then Y/M/D,
    then M/D/Y; textual months (14 Aug 1995, Aug 14, 1995) are supported."""
    nums = [int(n) for n in re.findall(r"\d+", text)]
    month_word = re.search(r"[A-Za-z]{3,}", text)
    if month_word:
        month = _MONTHS.get(month_word.group()[:3].lower())
        if month is None or len(nums) != 2:
            return False
        day, year = (nums[0], nums[1]) if nums[0] <= 31 else (nums[1], nums[0])
        return _real_date(year, month, day)
    if len(nums) != 3:
        return False
    a, b, c = nums
    if a > 31:                                    # Y/M/D
        return _real_date(a, b, c)
    return _real_date(c, b, a) or _real_date(c, a, b)   # D/M/Y, then M/D/Y


# --- UPI ---------------------------------------------------------------------

def upi_handle_known(text: str, handles: Iterable[str]) -> bool:
    _, _, handle = text.rpartition("@")
    return handle.lower() in {h.lower() for h in handles}


def apply_validator(name: Optional[str], text: str, handles=()) -> Optional[bool]:
    """Dispatch by validator name; None means 'no validator for this pattern'."""
    if name is None:
        return None
    if name == "aadhaar":
        return aadhaar_valid(text)
    if name == "pan":
        return pan_valid(text)
    if name == "phone":
        return indian_mobile_valid(text)
    if name == "date":
        return date_valid(text)
    if name == "upi":
        return upi_handle_known(text, handles)
    raise ValueError(f"Unknown validator: {name}")

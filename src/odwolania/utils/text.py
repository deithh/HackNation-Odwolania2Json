"""Shared text, OCR-normalization, date/amount parsing, checksum, and JSON-extraction helpers."""
from __future__ import annotations

import datetime
import re
from typing import Any

_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"})

_POLISH_MONTHS: dict[str, int] = {
    "stycznia": 1, "styczeń": 1, "styczen": 1, "sty": 1,
    "lutego": 2, "luty": 2, "lut": 2,
    "marca": 3, "marzec": 3, "mar": 3,
    "kwietnia": 4, "kwiecień": 4, "kwiecien": 4, "kwi": 4,
    "maja": 5, "maj": 5,
    "czerwca": 6, "czerwiec": 6, "cze": 6,
    "lipca": 7, "lipiec": 7, "lip": 7,
    "sierpnia": 8, "sierpień": 8, "sierpien": 8, "sie": 8,
    "września": 9, "wrzesnia": 9, "wrzesień": 9, "wrzesien": 9, "wrz": 9,
    "października": 10, "pazdziernika": 10, "październik": 10, "pazdziernik": 10, "paź": 10, "paz": 10,
    "listopada": 11, "listopad": 11, "lis": 11,
    "grudnia": 12, "grudzień": 12, "grudzien": 12, "gru": 12,
}


def fix_ocr_digits(text: str, *, include_separators: bool = True) -> str:
    """Replaces common OCR letter-for-digit confusions (O->0, l/I->1, S->5, B->8) adjacent to digits."""
    if include_separators:
        pattern = r"(?<=\d)[OolISB](?=[\d \-/])|(?<=[\d \-/])[OolISB](?=\d)"
    else:
        pattern = r"(?<=\d)[OolISB](?=[\d ])|(?<=[\d ])[OolISB](?=\d)"
    return re.sub(pattern, lambda m: m.group(0).translate(_DIGIT_FIX), text)


def nrb_checksum_ok(digits_or_iban: str) -> bool:
    """Validates a 26-digit Polish NRB / IBAN modulo-97 checksum."""
    digits = re.sub(r"\D", "", digits_or_iban)[-26:]
    return len(digits) == 26 and digits.isdigit() and int(digits[2:] + "2521" + digits[:2]) % 97 == 1


def is_placeholder_digits(digits: str) -> bool:
    """Detects degenerate placeholder sequences like '0000000000' or '9999999999'."""
    return bool(digits) and len(set(digits)) <= 1


def levenshtein(a: str, b: str) -> int:
    """Computes exact Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def parse_polish_date(value: str) -> str | None:
    """Parses ISO, Polish numeric (DD.MM.YYYY), or Polish textual dates into 'YYYY-MM-DD'."""
    s = re.sub(r"\s+", " ", value.strip())
    m = re.fullmatch(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime.date(y, mo, d).isoformat()
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        try:
            return datetime.date(y, mo, d).isoformat()
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2})\s+([^\W\d_]+)\.?\s+(\d{4})(?:\s*r\.?)?", s, flags=re.I)
    if m:
        d = int(m.group(1))
        mo = _POLISH_MONTHS.get(m.group(2).lower())
        y = int(m.group(3))
        if mo:
            try:
                return datetime.date(y, mo, d).isoformat()
            except ValueError:
                return None
    return None


def parse_amount(value: Any) -> float | None:
    """Parses monetary values (numeric or Polish/EU formatted strings) into a 2-decimal float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if not isinstance(value, str):
        return None
    s = re.sub(r"(?:zł|pln|eur|usd)\b", "", value, flags=re.I)
    s = re.sub(r"[\s\u00a0]", "", s).strip()
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def trim_document_text(text: str, max_chars: int) -> str:
    """Keeps head (75%) and tail (25%) of long texts so headers and signatures remain visible."""
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    head = int(max_chars * 0.75)
    tail = max_chars - head
    return f"{cleaned[:head]}\n\n[... pominięto środek dokumentu ...]\n\n{cleaned[-tail:]}"


def extract_json_block(text: str, *, expect_json: bool = False) -> str:
    """Strips optional reasoning blocks, markdown code fences, and stray prose around JSON."""
    cleaned = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>\s*", "", text, flags=re.S | re.I).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.S | re.I)
    if fence:
        cleaned = fence.group(1).strip()
    if expect_json and cleaned and not cleaned.startswith(("{", "[")):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1].strip()
    return cleaned

"""Source-text grounding index for numeric identifiers (NIP, KRS, REGON, PESEL, BZP, TED, NRB/IBAN)."""
from __future__ import annotations

import re

from ..utils.text import fix_ocr_digits, levenshtein, nrb_checksum_ok

TED_ALT_RE = re.compile(r"(?<!\d)(\d{4})\s*/\s*S\s*(\d{1,3})\s*[-–]\s*(\d{6})(?!\d)", flags=re.I)
TED_STD_RE = re.compile(r"(?<!\d)(\d{6})\s*[-–]\s*(\d{4})(?!\d)")
TED_OJS_RE = re.compile(r"(?<!\d)(\d{4})\s*[-–]\s*OJS\d+\s*[-–]\s*0*(\d{6})(?!\d)", flags=re.I)
BZP_FLEX_RE = re.compile(r"(\d{4})\s*/\s*BZP\s*(\d{8})(?:\s*/\s*(\d{2}))?", flags=re.I)
NRB_FLEX_RE = re.compile(r"(?<![A-Z0-9])(PL\s*)?(\d{2}(?:[\s\-–]?\d{4}){6})(?!\d)", flags=re.I)
FOREIGN_IBAN_RE = re.compile(
    r"(?<![A-Z0-9])([A-Z]{2}[ \t]*\d{2}(?:[ \t\-–]?[A-Z0-9]{1,4}){2,8})(?![A-Z0-9])"
)


class SourceTextIndex:
    """Indexes every numeric identifier present in the PDF/OCR source text to prevent hallucinations."""

    def __init__(self, source_text: str):
        raw = source_text.replace("\r\n", "\n").replace("\r", "\n")
        fixed = fix_ocr_digits(raw)

        # 1) Plain & separator-collapsed digit tokens (protects against fabricated NIP/KRS/REGON/PESEL)
        tokens: set[str] = set(re.findall(r"\d+", fixed))
        hyphen_collapsed = re.sub(r"(?<=\d)[ \t]*[-–][ \t]*(?=\d)", "", fixed)
        tokens.update(re.findall(r"\d+", hyphen_collapsed))
        space_collapsed = re.sub(r"(?<=\d)[ \-–]+(?=\d)", "", hyphen_collapsed)
        tokens.update(re.findall(r"\d+", space_collapsed))
        self.digit_tokens: set[str] = tokens

        # Keep both raw and OCR-digit-fixed views (with line-wrapped digit join) so "/S086" is never
        # masked by "S->5" and line breaks like "2\n026/S" are still indexed.
        raw_nl = re.sub(r"(?<=\d)\n(?=\d)", "", raw)
        fixed_nl = re.sub(r"(?<=\d)\n(?=\d)", "", fixed)
        id_views = (raw, raw_nl, fixed, fixed_nl)

        # 2) BZP notice numbers: (year, num8)
        bzp_pairs: set[tuple[str, str]] = set()
        for view in id_views:
            for match in BZP_FLEX_RE.finditer(view):
                bzp_pairs.add((match.group(1), match.group(2)))
        self.bzp_pairs: set[tuple[str, str]] = bzp_pairs

        # 3) TED notice numbers: (num6, year) + reverse lookup indexes for truncated fragments
        ted_pairs: set[tuple[str, str]] = set()
        ted_by_num6: dict[str, set[str]] = {}
        ted_by_issue: dict[tuple[str, str], set[str]] = {}
        for view in id_views:
            for match in TED_ALT_RE.finditer(view):
                year, issue, num6 = match.group(1), str(int(match.group(2))), match.group(3)
                ted_pairs.add((num6, year))
                ted_by_num6.setdefault(num6, set()).add(year)
                ted_by_issue.setdefault((year, issue), set()).add(num6)
            for match in TED_STD_RE.finditer(view):
                num6, year = match.group(1), match.group(2)
                ted_pairs.add((num6, year))
                ted_by_num6.setdefault(num6, set()).add(year)
            for match in TED_OJS_RE.finditer(view):
                year, num6 = match.group(1), match.group(2)
                ted_pairs.add((num6, year))
                ted_by_num6.setdefault(num6, set()).add(year)
        self.ted_pairs: set[tuple[str, str]] = ted_pairs
        self.ted_by_num6: dict[str, set[str]] = ted_by_num6
        self.ted_by_issue: dict[tuple[str, str], set[str]] = ted_by_issue

        # 4) 26-digit Polish bank account numbers (NRB / IBAN), tracking whether 'PL' appears in source
        nrb_has_pl: dict[str, bool] = {}
        for match in NRB_FLEX_RE.finditer(fixed):
            digits = re.sub(r"\D", "", match.group(2))
            if len(digits) == 26:
                nrb_has_pl[digits] = nrb_has_pl.get(digits, False) or bool(match.group(1))
        ws_collapsed = re.sub(
            r"(?<=\d)[\s\-–]+(?=\d)",
            "",
            re.sub(r"(?i)(?<![A-Z0-9])PL[\s\-–]+(?=\d)", "PL", fixed),
        )
        for match in re.finditer(r"(?<![A-Z0-9])(PL)?(\d{26})(?!\d)", ws_collapsed, flags=re.I):
            digits = match.group(2)
            nrb_has_pl[digits] = nrb_has_pl.get(digits, False) or bool(match.group(1))
        self.nrb_has_pl: dict[str, bool] = nrb_has_pl
        self.nrbs: list[str] = sorted(nrb_has_pl.keys())

        # 5) Foreign IBANs (any 2-letter country code other than PL)
        foreign_ibans: set[str] = set()
        for view in (raw, raw_nl):
            for match in FOREIGN_IBAN_RE.finditer(view):
                compact = re.sub(r"[\s\-–]", "", match.group(1)).upper()
                if not compact.startswith("PL") and 15 <= len(compact) <= 34:
                    foreign_ibans.add(compact)
        self.foreign_ibans: set[str] = foreign_ibans

    def has_digits(self, digits: str, *, allow_krs_unpadded: bool = False) -> bool:
        if digits in self.digit_tokens:
            return True
        if allow_krs_unpadded:
            stripped = digits.lstrip("0")
            if len(stripped) >= 4 and stripped in self.digit_tokens:
                return True
        return False

    def has_bzp(self, val: str) -> bool:
        match = BZP_FLEX_RE.fullmatch(val)
        return bool(match and (match.group(1), match.group(2)) in self.bzp_pairs)

    def has_ted(self, val: str, *, year_min: int = 2000, year_max: int = 2035) -> bool:
        match = re.fullmatch(r"(\d{6})-(\d{4})", val)
        if not match:
            return False
        year = int(match.group(2))
        if not (year_min <= year <= year_max):
            return False
        return (match.group(1), match.group(2)) in self.ted_pairs

    def resolve_ted(self, raw: str) -> str | None:
        """Resolves truncated TED fragments (e.g. '224-429185' or '2026-048') when uniquely grounded."""
        match = re.fullmatch(r"(\d{1,3})[-–](\d{6})", raw)
        if match:
            num6 = match.group(2)
            years = self.ted_by_num6.get(num6)
            if years and len(years) == 1:
                return f"{num6}-{next(iter(years))}"
        match = re.fullmatch(r"(\d{4})[-–](\d{1,3})", raw)
        if match:
            year, issue = match.group(1), str(int(match.group(2)))
            nums = self.ted_by_issue.get((year, issue))
            if nums and len(nums) == 1:
                return f"{next(iter(nums))}-{year}"
        match = re.fullmatch(r"(\d{6})", raw)
        if match:
            num6 = match.group(1)
            years = self.ted_by_num6.get(num6)
            if years and len(years) == 1:
                return f"{num6}-{next(iter(years))}"
        return None

    def ground_nrb(
        self,
        raw_val: str,
        *,
        allow_pl: bool = True,
        max_dist: int = 4,
        reject_dist: int = 6,
    ) -> str | None:
        """Verifies and optionally repairs small OCR/copy drift on 26-digit accounts, preserving country prefixes."""
        compact = re.sub(r"[\s\-–]", "", raw_val.strip().upper())
        m_cc = re.match(r"^([A-Z]{2})([A-Z0-9]+)$", compact)
        if m_cc and m_cc.group(1) != "PL":
            if not self.foreign_ibans:
                return compact
            if compact in self.foreign_ibans:
                return compact
            scored_f = sorted(
                ((levenshtein(compact, cand), cand) for cand in self.foreign_ibans),
                key=lambda x: x[0],
            )
            best_dist, best_cand = scored_f[0]
            if best_dist <= max_dist:
                tied = [c for d, c in scored_f if d == best_dist]
                if len(tied) == 1:
                    return best_cand
            elif best_dist > reject_dist:
                return None
            return compact

        upper = compact
        model_has_pl = upper.startswith("PL")
        digits = re.sub(r"\D", "", upper)

        if not self.nrbs:
            return f"PL{digits}" if (allow_pl and model_has_pl) else digits

        matched: str | None = None
        if digits in self.nrb_has_pl:
            matched = digits
        else:
            scored = sorted(
                ((levenshtein(digits, cand), 0 if nrb_checksum_ok(cand) else 1, cand) for cand in self.nrbs),
                key=lambda x: (x[0], x[1]),
            )
            best_dist, best_chk, best_cand = scored[0]
            if best_dist <= max_dist:
                tied = [c for d, chk, c in scored if (d, chk) == (best_dist, best_chk)]
                if len(tied) == 1:
                    matched = best_cand
            elif best_dist > reject_dist:
                return None
            if matched is None:
                matched = digits

        use_pl = allow_pl and (self.nrb_has_pl[matched] if matched in self.nrb_has_pl else model_has_pl)
        return f"PL{matched}" if use_pl else matched

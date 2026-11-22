"""Maps extracted JSON leaves back onto exact normalized [0..1] PDF/OCR bounding boxes for visual audit."""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from ..models import DocumentText, PageText
from ..ocr.deskew import deskew_image

_POLISH_MONTH_GEN = {
    1: "stycznia",
    2: "lutego",
    3: "marca",
    4: "kwietnia",
    5: "maja",
    6: "czerwca",
    7: "lipca",
    8: "sierpnia",
    9: "września",
    10: "października",
    11: "listopada",
    12: "grudnia",
}

# Cohesive high-contrast highlight palette paired with Prussian Blue / Deep Navy / Yale Blue / Cerulean UI
SECTION_COLORS = [
    "#1282A2",  # Cerulean
    "#034078",  # Yale Blue
    "#0F766E",  # Deep Teal
    "#B45309",  # Warm Amber
    "#6D28D9",  # Indigo Violet
    "#BE123C",  # Crimson Accent
    "#1D4ED8",  # Royal Blue
    "#047857",  # Forest Emerald
]


def _fold(text: str) -> str:
    """Lowercase + Polish/Latin diacritic folding for robust OCR substring matching."""
    table = str.maketrans({"ł": "l", "Ł": "l"})
    norm = unicodedata.normalize("NFKD", text.translate(table))
    return "".join(ch for ch in norm if not unicodedata.combining(ch)).lower()


def _flatten_leaves(obj: Any, prefix: str = "") -> list[tuple[str, str, Any]]:
    """Returns `(field_path, top_section, value)` for every non-null scalar leaf."""
    out: list[tuple[str, str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{prefix}.{k}" if prefix else k
            out.extend(_flatten_leaves(v, sub))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            sub = f"{prefix}[{idx}]"
            out.extend(_flatten_leaves(item, sub))
    elif obj is not None and not (isinstance(obj, str) and not obj.strip()):
        section = prefix.split(".")[0].split("[")[0] if prefix else "root"
        out.append((prefix, section, obj))
    return out


def _candidate_patterns(field_path: str, value: Any) -> tuple[list[str], str | None]:
    """Builds `(text_variants, digits_only_target)` for matching a normalized JSON leaf in raw PDF/OCR text."""
    # Skip boolean/flag/enum-code integers that are inferred rather than printed verbatim
    if isinstance(value, bool):
        return [], None
    if isinstance(value, int) and value in (0, 1) and any(
        tok in field_path.lower() for tok in ("pelnomocnika", "prowadzacy", "powod_id", "_id")
    ):
        return [], None
    if isinstance(value, int) and ("powod_id" in field_path.lower() or field_path.lower().endswith("_id")):
        return [], None
    if isinstance(value, str) and value in ("osoba_prawna", "osoba_fizyczna"):
        return [], None

    variants: list[str] = []
    digits_target: str | None = None

    if isinstance(value, (int, float)):
        fval = float(value)
        int_part = int(round(fval)) if abs(fval - round(fval)) < 1e-6 else None
        # Polish & EU monetary formats: 10 000,00 / 10000,00 / 10.000,00 / 10 000
        formatted_space = f"{fval:,.2f}".replace(",", " ").replace(".", ",")
        formatted_plain = f"{fval:.2f}".replace(".", ",")
        formatted_dot = f"{fval:.2f}"
        formatted_eu = f"{fval:,.2f}".replace(",", ".").rsplit(".", 1)
        variants.extend([formatted_space, formatted_plain, formatted_dot, f"{formatted_eu[0]},{formatted_eu[1]}"])
        if int_part is not None:
            variants.append(f"{int_part:,}".replace(",", " "))
            variants.append(str(int_part))
        return list(dict.fromkeys(v for v in variants if v)), None

    s = str(value).strip()
    if not s:
        return [], None

    # ISO date YYYY-MM-DD -> Polish textual & numeric date forms
    m_date = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m_date:
        y, mo, d = int(m_date.group(1)), int(m_date.group(2)), int(m_date.group(3))
        month_pl = _POLISH_MONTH_GEN.get(mo, "")
        variants.extend(
            [
                s,
                f"{d:02d}.{mo:02d}.{y}",
                f"{d}.{mo:02d}.{y}",
                f"{d:02d}-{mo:02d}-{y}",
                f"{d:02d}/{mo:02d}/{y}",
                f"{d} {month_pl} {y}",
                f"{d:02d} {month_pl} {y}",
            ]
        )
        return list(dict.fromkeys(v for v in variants if v)), None

    # TED NNNNNN-YYYY -> also match 6-digit number inside YYYY/S ###-NNNNNN
    m_ted = re.fullmatch(r"(\d{6})-(\d{4})", s)
    if m_ted:
        variants.extend([s, m_ted.group(1)])
        return variants, None

    # BZP YYYY/BZP########(/##)
    m_bzp = re.fullmatch(r"(\d{4})/BZP(\d{8})(?:/(\d{2}))?", s, flags=re.I)
    if m_bzp:
        suffix = f"/{m_bzp.group(3)}" if m_bzp.group(3) else ""
        variants.extend([s, f"{m_bzp.group(1)}/BZP {m_bzp.group(2)}{suffix}", m_bzp.group(2)])
        return variants, None

    # 26-digit Polish IBAN / NRB
    m_iban = re.fullmatch(r"(?:PL)?(\d{26})", s, flags=re.I)
    if m_iban:
        digits_target = m_iban.group(1)
        variants.extend([s, digits_target])
        return variants, digits_target

    # Numeric IDs (NIP, KRS, REGON, PESEL)
    if re.fullmatch(r"\d{9,14}", s):
        digits_target = s
        variants.append(s)
        unpadded = s.lstrip("0")
        if len(unpadded) >= 5 and unpadded != s:
            variants.append(unpadded)
        return variants, digits_target

    # General string (names, titles, cities, reference numbers, emails)
    variants.append(s)
    return variants, None


@dataclass
class _FoldedBuffer:
    """Normalized text buffer with 1:1 mapping back to original character offsets."""

    folded: str
    orig_indices: list[int]
    digits: str
    digit_indices: list[int]

    @classmethod
    def from_text(cls, raw: str) -> "_FoldedBuffer":
        folded_chars: list[str] = []
        orig_indices: list[int] = []
        digit_chars: list[str] = []
        digit_indices: list[int] = []
        prev_ws = False
        for i, ch in enumerate(raw):
            if ch.isdigit():
                digit_chars.append(ch)
                digit_indices.append(i)
            if ch.isspace():
                if not prev_ws:
                    folded_chars.append(" ")
                    orig_indices.append(i)
                    prev_ws = True
            else:
                prev_ws = False
                f = _fold(ch)
                for fc in f:
                    folded_chars.append(fc)
                    orig_indices.append(i)
        return cls(
            folded="".join(folded_chars),
            orig_indices=orig_indices,
            digits="".join(digit_chars),
            digit_indices=digit_indices,
        )

    def find_span(self, variants: list[str], digits_target: str | None) -> tuple[int, int] | None:
        # 1) Digits-only match for IDs / IBANs (ignores spaces, dashes, line breaks)
        if digits_target:
            pos = self.digits.find(digits_target)
            if pos != -1:
                start = self.digit_indices[pos]
                end = self.digit_indices[pos + len(digits_target) - 1] + 1
                return start, end

        # 2) Folded normalized substring match
        for var in variants:
            needle = re.sub(r"\s+", " ", _fold(var)).strip()
            if len(needle) < 2:
                continue
            pos = self.folded.find(needle)
            if pos != -1:
                start = self.orig_indices[pos]
                end = self.orig_indices[min(pos + len(needle) - 1, len(self.orig_indices) - 1)] + 1
                return start, end

        # 3) Long string fallback: match first & last significant tokens if close enough
        if variants:
            long_s = re.sub(r"\s+", " ", _fold(variants[0])).strip()
            words = [w for w in re.split(r"[^\w]+", long_s) if len(w) >= 3]
            if len(words) >= 3:
                anchor = " ".join(words[:3])
                pos = self.folded.find(anchor)
                if pos != -1:
                    start = self.orig_indices[pos]
                    tail_pos = self.folded.find(words[-1], pos, pos + len(long_s) + 80)
                    end_f = (tail_pos + len(words[-1]) - 1) if tail_pos != -1 else min(pos + len(long_s) - 1, len(self.orig_indices) - 1)
                    end = self.orig_indices[min(end_f, len(self.orig_indices) - 1)] + 1
                    return start, end
                # Try first 2 words (handles OCR glitch on 3rd word)
                anchor2 = " ".join(words[:2])
                pos = self.folded.find(anchor2)
                if pos != -1:
                    start = self.orig_indices[pos]
                    end = self.orig_indices[min(pos + len(anchor2) - 1, len(self.orig_indices) - 1)] + 1
                    return start, end
        return None


class EvidenceLocator:
    """Renders PDF pages and locates exact normalized bounding-box overlays for extracted JSON fields."""

    @staticmethod
    def render_page_png(
        pdf_path: str | Path,
        page_index: int,
        *,
        width_px: int = 1200,
        skew_angle: float = 0.0,
    ) -> bytes:
        """Renders 1-based `page_index` of `pdf_path` to PNG bytes (applying optional deskew angle)."""
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            if not (1 <= page_index <= len(pdf)):
                raise IndexError(f"page {page_index} out of range 1..{len(pdf)}")
            page = pdf[page_index - 1]
            scale = width_px / max(page.get_size()[0], 1.0)
            pil_img = page.render(scale=scale).to_pil()
            if abs(skew_angle) >= 0.1:
                pil_img, _ = deskew_image(pil_img, angle=skew_angle, min_angle=0.1)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        finally:
            pdf.close()

    @staticmethod
    def page_count(pdf_path: str | Path) -> int:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            return len(pdf)
        finally:
            pdf.close()

    @staticmethod
    def page_count_bytes(pdf_bytes: bytes) -> int:
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            return len(pdf)
        finally:
            pdf.close()

    def locate(
        self,
        pdf_path: str | Path,
        data: dict[str, Any],
        doc_text: DocumentText | None = None,
    ) -> list[dict[str, Any]]:
        """Finds visual highlight boxes for all matchable scalar leaves in `data`.

        Returns a list of highlight dicts:
          `{field, section, value, page, x0, y0, x1, y1, color}` (normalized `[0..1]`).
        """
        if not isinstance(data, dict) or not data:
            return []

        leaves = _flatten_leaves(data)
        if not leaves:
            return []

        # Assign stable palette color per top-level section
        section_colors: dict[str, str] = {}
        for _, sec, _ in leaves:
            if sec not in section_colors:
                section_colors[sec] = SECTION_COLORS[len(section_colors) % len(SECTION_COLORS)]

        # Index OCR spans per page when available
        ocr_pages: dict[int, list[dict[str, Any]]] = {}
        if doc_text is not None:
            for p in doc_text.pages:
                if p.spans:
                    ocr_pages[p.index] = p.spans

        highlights: list[dict[str, Any]] = []
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            n_pages = len(pdf)
            # Pre-extract native text buffers + page sizes
            native_cache: dict[int, tuple[float, float, Any, _FoldedBuffer]] = {}
            ocr_cache: dict[int, tuple[list[dict[str, Any]], list[tuple[int, int, int]], _FoldedBuffer]] = {}

            for page_idx in range(1, n_pages + 1):
                if page_idx in ocr_pages:
                    spans = sorted(ocr_pages[page_idx], key=lambda s: (round(s["y0"], 2), s["x0"]))
                    parts: list[str] = []
                    ranges: list[tuple[int, int, int]] = []
                    cursor = 0
                    for s_idx, sp in enumerate(spans):
                        t = sp["text"]
                        parts.append(t)
                        ranges.append((cursor, cursor + len(t), s_idx))
                        cursor += len(t) + 1
                    joined = " ".join(parts)
                    ocr_cache[page_idx] = (spans, ranges, _FoldedBuffer.from_text(joined))
                else:
                    page = pdf[page_idx - 1]
                    w, h = page.get_size()
                    tp = page.get_textpage()
                    n_chars = tp.count_chars()
                    if n_chars > 0:
                        raw = tp.get_text_range(0, n_chars) or ""
                        native_cache[page_idx] = (float(w), float(h), tp, _FoldedBuffer.from_text(raw))

            for field_path, section, value in leaves:
                variants, digits_target = _candidate_patterns(field_path, value)
                if not variants and not digits_target:
                    continue
                color = section_colors[section]
                found_boxes: list[tuple[int, float, float, float, float]] = []

                for page_idx in range(1, n_pages + 1):
                    if page_idx in ocr_cache:
                        boxes = self._locate_on_ocr_page(page_idx, ocr_cache[page_idx], variants, digits_target)
                    elif page_idx in native_cache:
                        boxes = self._locate_on_native_page(page_idx, native_cache[page_idx], variants, digits_target)
                    else:
                        boxes = []
                    if boxes:
                        found_boxes = boxes
                        break

                for page_idx, x0, y0, x1, y1 in found_boxes:
                    highlights.append(
                        {
                            "field": field_path,
                            "section": section,
                            "value": value,
                            "page": page_idx,
                            "x0": round(max(0.0, min(1.0, x0)), 4),
                            "y0": round(max(0.0, min(1.0, y0)), 4),
                            "x1": round(max(0.0, min(1.0, x1)), 4),
                            "y1": round(max(0.0, min(1.0, y1)), 4),
                            "color": color,
                        }
                    )
        finally:
            pdf.close()

        # Sort larger boxes first so smaller boxes render on top and remain easy to hover/click
        highlights.sort(key=lambda h: (h["page"], -((h["x1"] - h["x0"]) * (h["y1"] - h["y0"]))))
        return highlights

    @staticmethod
    def _locate_on_native_page(
        page_idx: int,
        entry: tuple[float, float, Any, _FoldedBuffer],
        variants: list[str],
        digits_target: str | None,
    ) -> list[tuple[int, float, float, float, float]]:
        w, h, tp, buf = entry
        span = buf.find_span(variants, digits_target)
        if span is None:
            return []
        start, end = span
        count = max(1, end - start)
        n_rects = tp.count_rects(start, count)
        boxes: list[tuple[int, float, float, float, float]] = []
        pad_x, pad_y = 0.003, 0.0025
        for i in range(n_rects):
            left, bottom, right, top = tp.get_rect(i)
            if right <= left or top <= bottom:
                continue
            x0 = (left / w) - pad_x
            y0 = (1.0 - top / h) - pad_y
            x1 = (right / w) + pad_x
            y1 = (1.0 - bottom / h) + pad_y
            boxes.append((page_idx, x0, y0, x1, y1))
        return boxes

    @staticmethod
    def _locate_on_ocr_page(
        page_idx: int,
        entry: tuple[list[dict[str, Any]], list[tuple[int, int, int]], _FoldedBuffer],
        variants: list[str],
        digits_target: str | None,
    ) -> list[tuple[int, float, float, float, float]]:
        spans, ranges, buf = entry
        hit = buf.find_span(variants, digits_target)
        if hit is None:
            return []
        start, end = hit
        matched: list[tuple[float, float, float, float]] = []
        for r_start, r_end, s_idx in ranges:
            if r_end <= start or r_start >= end:
                continue
            sp = spans[s_idx]
            sp_len = max(r_end - r_start, 1)
            t0 = (max(start, r_start) - r_start) / sp_len
            t1 = (min(end, r_end) - r_start) / sp_len
            poly = sp.get("poly")
            if isinstance(poly, list) and len(poly) == 4:
                tl, tr, br, bl = poly
                x0 = min(tl[0] + (tr[0] - tl[0]) * t0, bl[0] + (br[0] - bl[0]) * t0)
                x1 = max(tl[0] + (tr[0] - tl[0]) * t1, bl[0] + (br[0] - bl[0]) * t1)
                y0 = min(tl[1] + (tr[1] - tl[1]) * t0, tl[1] + (tr[1] - tl[1]) * t1)
                y1 = max(bl[1] + (br[1] - bl[1]) * t0, bl[1] + (br[1] - bl[1]) * t1)
            else:
                sp_w = sp["x1"] - sp["x0"]
                x0 = sp["x0"] + sp_w * t0
                x1 = sp["x0"] + sp_w * t1
                y0, y1 = sp["y0"], sp["y1"]
            matched.append((x0, y0, x1, y1))

        if not matched:
            return []

        # Merge boxes sitting on the same visual line
        merged: list[list[float]] = []
        for x0, y0, x1, y1 in matched:
            if merged and abs(y0 - merged[-1][1]) < 0.015:
                merged[-1][0] = min(merged[-1][0], x0)
                merged[-1][1] = min(merged[-1][1], y0)
                merged[-1][2] = max(merged[-1][2], x1)
                merged[-1][3] = max(merged[-1][3], y1)
            else:
                merged.append([x0, y0, x1, y1])

        pad_x, pad_y = 0.003, 0.0025
        return [(page_idx, b[0] - pad_x, b[1] - pad_y, b[2] + pad_x, b[3] + pad_y) for b in merged]

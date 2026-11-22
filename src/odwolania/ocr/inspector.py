"""PDF page inspection and text-layer vs. OCR strategy selection."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pypdfium2.raw as pdfium_c

from ..models import PageText

_OK_CHARS = re.compile(r"[\w\s.,;:!?()\[\]{}\"'„”«»/\\\-–—+=*&%$#@§°<>|…·•]", re.UNICODE)


class PdfInspector:
    """Analyzes pypdfium2 pages to distinguish native text layers, hidden OCR layers, scans, and blanks."""

    def __init__(self, page_analysis_cfg: dict[str, Any]):
        self._cfg = page_analysis_cfg

    def inspect_page(self, page: Any, index: int) -> PageText:
        """Extracts native text layer and computes image coverage, text quality, and invisible-text share."""
        text = page.get_textpage().get_text_bounded() or ""
        width, height = page.get_size()
        area = max(width * height, 1.0)
        img_area, n_text, n_invis = 0.0, 0, 0
        for obj in page.get_objects():
            if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                left, bottom, right, top = obj.get_bounds()
                img_area += max(0.0, min(right, width) - max(left, 0)) * max(0.0, min(top, height) - max(bottom, 0))
            elif obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT:
                n_text += 1
                if pdfium_c.FPDFTextObj_GetTextRenderMode(obj.raw) == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE:
                    n_invis += 1
        quality = (len(_OK_CHARS.findall(text)) / len(text)) if text else 0.0
        page_text = PageText(
            index=index,
            source="pdf",
            text=text,
            chars=len(re.sub(r"\s+", "", text)),
            image_coverage=round(min(img_area / area, 1.0), 3),
            text_quality=round(quality, 3),
            invisible_share=round(n_invis / n_text, 3) if n_text else 0.0,
        )
        page_text.kind = self.classify_page_kind(page_text)
        return page_text

    def classify_page_kind(self, page: PageText) -> str:
        """Returns 'ocr_layer', 'text', 'scan', or 'blank' based on configured thresholds."""
        min_chars = int(self._cfg.get("min_chars_text_page", 50))
        min_quality = float(self._cfg.get("min_text_quality", 0.85))
        min_img = float(self._cfg.get("min_image_coverage_scan", 0.25))
        invis_share = float(self._cfg.get("invisible_text_share", 0.5))

        if page.invisible_share > invis_share and page.chars >= min_chars and page.text_quality >= min_quality:
            return "ocr_layer"
        if page.chars >= min_chars and page.text_quality >= min_quality:
            return "text"
        if page.image_coverage >= min_img or page.chars >= min_chars:
            return "scan"
        return "blank"

    @staticmethod
    def choose_strategy(pages: list[PageText], head_indices: list[int]) -> tuple[str, str]:
        """Decides document-level reading strategy ('text' | 'hybrid' | 'ocr') from classification pages."""
        head = [p for p in pages if p.index in head_indices and p.kind != "blank"]
        if not head:
            return "ocr", "no content on the classification pages"
        counts = Counter(p.kind for p in head)
        if counts["scan"]:
            return "ocr", f"scanned pages: {counts['scan']}/{len(head)}"
        if counts["ocr_layer"]:
            return "hybrid", f"scan with a hidden OCR layer: {counts['ocr_layer']}/{len(head)}"
        return "text", "text layer on all pages"

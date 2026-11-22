"""High-level PDF reader combining native text extraction, local OCR, and incremental caching."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from PIL import Image

from ..config import PipelineConfig
from ..models import DocumentText
from ..utils.hashing import file_sha1
from ..utils.pages import resolve_pages
from .base import OcrEngine, OcrEngineFactory
from .cache import OcrCache
from .deskew import crop_content_region, deskew_image
from .inspector import PdfInspector


class DocumentReader:
    """Reads requested PDF pages via text layer or local OCR, caching pages incrementally."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        engine: OcrEngine | None = None,
        cache: OcrCache | None = None,
        inspector: PdfInspector | None = None,
    ):
        self._config = config
        self._engine = engine
        self._cache = cache or OcrCache(config)
        self._inspector = inspector or PdfInspector(config.page_analysis)

    def _get_engine(self) -> OcrEngine:
        if self._engine is None:
            name = self._config.ocr_engine_name
            params = self._config.ocr.get(name) or {}
            self._engine = OcrEngineFactory.get(name, params)
        return self._engine

    def read(self, path: str | Path, pages: Any = None) -> DocumentText:
        """Extracts text for the requested page spec (defaults to `classification_pages`)."""
        pdf_path = Path(path)
        if pdf_path.suffix.lower() != ".pdf":
            return DocumentText(str(pdf_path), "", "error", f"unsupported format {pdf_path.suffix}")

        sha1 = file_sha1(pdf_path)
        done = self._cache.load_pages(sha1)

        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
        except Exception as exc:
            return DocumentText(str(pdf_path), sha1, "error", f"cannot open PDF: {exc}")

        try:
            n_pages = len(pdf)
            head_spec = self._config.classification_pages
            head_indices = resolve_pages(head_spec, n_pages)
            wanted_indices = resolve_pages(pages if pages is not None else head_spec, n_pages)

            stats = {
                idx: self._inspector.inspect_page(pdf[idx - 1], idx)
                for idx in sorted(set(head_indices) | set(wanted_indices))
            }
            strategy, reason = self._inspector.choose_strategy(list(stats.values()), head_indices)
            force = self._config.ocr_force
            if force:
                strategy, reason = "ocr", f"{reason} (OCR forced)"

            render_width = self._config.render_width_px
            do_deskew = bool(self._config.ocr.get("deskew", True))
            max_angle = float(self._config.ocr.get("deskew_max_angle", 6.0))
            min_angle = float(self._config.ocr.get("deskew_min_angle", 0.35))
            for idx in wanted_indices:
                if idx in done:
                    continue
                page_text = stats[idx]
                if force or page_text.kind == "scan" or (page_text.kind == "blank" and strategy == "ocr"):
                    pdf_page = pdf[idx - 1]
                    pw, ph = pdf_page.get_size()
                    scale = min(render_width / max(pw, 1.0), (render_width * 1.6) / max(ph, 1.0))
                    pil_img = pdf_page.render(scale=scale).to_pil()
                    if do_deskew:
                        pil_img, angle = deskew_image(pil_img, max_angle=max_angle, min_angle=min_angle)
                        page_text.skew_angle = angle
                    full_w, full_h = pil_img.size
                    ocr_img, (cx0, cy0, cx1, cy1) = crop_content_region(pil_img)
                    cw, ch = cx1 - cx0, cy1 - cy0
                    if cw < render_width * 0.75 and cw > 0 and ch > 0:
                        up = min(2.0, render_width / float(cw))
                        if up > 1.1:
                            ocr_img = ocr_img.resize(
                                (max(1, int(cw * up)), max(1, int(ch * up))),
                                Image.Resampling.BICUBIC,
                            )
                    text, spans = self._get_engine().image_to_text_with_spans(ocr_img)
                    if (cx0, cy0, cx1, cy1) != (0, 0, full_w, full_h) and full_w > 0 and full_h > 0:
                        for sp in spans:
                            sp["x0"] = round((cx0 + sp["x0"] * cw) / full_w, 4)
                            sp["x1"] = round((cx0 + sp["x1"] * cw) / full_w, 4)
                            sp["y0"] = round((cy0 + sp["y0"] * ch) / full_h, 4)
                            sp["y1"] = round((cy0 + sp["y1"] * ch) / full_h, 4)
                            if "poly" in sp:
                                sp["poly"] = [
                                    [
                                        round((cx0 + pt[0] * cw) / full_w, 4),
                                        round((cy0 + pt[1] * ch) / full_h, 4),
                                    ]
                                    for pt in sp["poly"]
                                ]
                    page_text.text, page_text.spans = text, spans
                    page_text.source = "ocr"
                done[idx] = page_text
        finally:
            pdf.close()

        doc = DocumentText(
            file=str(pdf_path),
            sha1=sha1,
            strategy=strategy,
            strategy_reason=reason,
            n_pages=n_pages,
            pages=[done[i] for i in sorted(done)],
            requested=wanted_indices,
        )
        self._cache.save(doc)
        return doc

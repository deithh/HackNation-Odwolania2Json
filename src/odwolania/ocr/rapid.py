"""Local CPU OCR backend powered by RapidOCR (ONNX Runtime + PP-OCRv5 Latin mobile)."""
from __future__ import annotations

from typing import Any

from ..utils.layout import format_ocr_lines
from .base import OcrEngine, OcrEngineFactory


def _resolve_enum(enum_cls: Any, value: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    target = str(value).strip().upper().replace("-", "").replace("_", "")
    for member in enum_cls:
        if (
            member.name.upper().replace("_", "") == target
            or str(member.value).upper().replace("-", "").replace("_", "") == target
        ):
            return member
    raise ValueError(f"Unknown {enum_cls.__name__} value: {value!r}")


class RapidOcrEngine(OcrEngine):
    """Fast CPU-only OCR engine using RapidOCR ONNX Runtime with visual line reconstruction."""

    def __init__(self, params: dict[str, Any] | None = None):
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        cfg = params or {}
        ocr_ver = _resolve_enum(OCRVersion, cfg.get("ocr_version", "PPOCRV5"))
        model_type = _resolve_enum(ModelType, cfg.get("model_type", "MOBILE"))
        lang_rec = _resolve_enum(LangRec, cfg.get("lang_rec", "LATIN"))

        rapid_params = {
            "Det.ocr_version": ocr_ver,
            "Det.model_type": model_type,
            "Rec.ocr_version": ocr_ver,
            "Rec.model_type": model_type,
            "Rec.lang_type": lang_rec,
            "Global.use_cls": bool(cfg.get("use_cls", False)),
            "EngineConfig.onnxruntime.intra_op_num_threads": int(cfg.get("intra_op_num_threads", 4)),
            "EngineConfig.onnxruntime.inter_op_num_threads": int(cfg.get("inter_op_num_threads", 1)),
            "EngineConfig.onnxruntime.enable_cpu_mem_arena": bool(cfg.get("enable_cpu_mem_arena", True)),
        }
        self._ocr = RapidOCR(params=rapid_params)

    def image_to_text(self, pil_image: Any) -> str:
        text, _ = self.image_to_text_with_spans(pil_image)
        return text

    def image_to_text_with_spans(self, pil_image: Any) -> tuple[str, list[dict[str, Any]]]:
        import numpy as np

        rgb = pil_image.convert("RGB")
        width, height = max(rgb.width, 1), max(rgb.height, 1)
        array = np.asarray(rgb)
        result = self._ocr(array)
        if result is None or not result.txts:
            return "", []
        spans: list[dict[str, Any]] = []
        boxes = result.boxes if result.boxes is not None else []
        for txt, poly in zip(result.txts, boxes):
            cleaned = str(txt).strip()
            if not cleaned:
                continue
            xs = [float(pt[0]) for pt in poly]
            ys = [float(pt[1]) for pt in poly]
            norm_poly = [
                [round(max(0.0, min(1.0, float(pt[0]) / width)), 4), round(max(0.0, min(1.0, float(pt[1]) / height)), 4)]
                for pt in poly
            ]
            spans.append(
                {
                    "text": cleaned,
                    "x0": round(max(0.0, min(xs) / width), 4),
                    "y0": round(max(0.0, min(ys) / height), 4),
                    "x1": round(min(1.0, max(xs) / width), 4),
                    "y1": round(min(1.0, max(ys) / height), 4),
                    "poly": norm_poly,
                }
            )
        return format_ocr_lines(result.txts, polys=result.boxes), spans


OcrEngineFactory.register("rapid", lambda params: RapidOcrEngine(params))

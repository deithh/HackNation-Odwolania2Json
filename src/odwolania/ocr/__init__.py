"""PDF inspection, scan deskewing, local CPU OCR (RapidOCR ONNX), and incremental caching."""
from . import rapid as _rapid  # noqa: F401 - registers RapidOcrEngine
from .base import OcrEngine, OcrEngineFactory
from .cache import OcrCache
from .deskew import deskew_image, estimate_skew_angle
from .inspector import PdfInspector
from .rapid import RapidOcrEngine
from .reader import DocumentReader

__all__ = [
    "DocumentReader",
    "OcrCache",
    "OcrEngine",
    "OcrEngineFactory",
    "PdfInspector",
    "RapidOcrEngine",
    "deskew_image",
    "estimate_skew_angle",
]

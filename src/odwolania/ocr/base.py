"""Abstract OCR engine interface and extensible factory registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class OcrEngine(ABC):
    """Base contract for pluggable local OCR backends."""

    @abstractmethod
    def image_to_text(self, pil_image: Any) -> str:
        """Converts a PIL RGB image of a single PDF page into reading-order text."""

    def image_to_text_with_spans(self, pil_image: Any) -> tuple[str, list[dict[str, Any]]]:
        """Returns `(reading_order_text, normalized_spans)` where spans have `{text, x0, y0, x1, y1}` in `[0, 1]`."""
        return self.image_to_text(pil_image), []


class OcrEngineFactory:
    """Registry and lazy singleton factory for OCR engines."""

    _registry: dict[str, Callable[[dict[str, Any]], OcrEngine]] = {}
    _instances: dict[str, OcrEngine] = {}

    @classmethod
    def register(cls, name: str, builder: Callable[[dict[str, Any]], OcrEngine]) -> None:
        cls._registry[name.lower()] = builder

    @classmethod
    def get(cls, name: str, params: dict[str, Any] | None = None) -> OcrEngine:
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise ValueError(f"unknown OCR engine {name!r} (available: {available})")
        if key not in cls._instances:
            cls._instances[key] = cls._registry[key](params or {})
        return cls._instances[key]

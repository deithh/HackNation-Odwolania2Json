"""Atomic, multiprocessing-safe on-disk cache for extracted PDF / OCR page texts."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import PipelineConfig, resolve_path
from ..models import DocumentText, PageText


class OcrCache:
    """Stores per-document page texts keyed by `(sha1, mode, engine)` on disk."""

    def __init__(self, config: PipelineConfig):
        self._dir = resolve_path(config.cache_dir)
        mode = "ocr" if config.ocr_force else "auto"
        deskew_tag = "deskew" if config.ocr.get("deskew", True) else "raw"
        self._mode = f"{mode}.{deskew_tag}"
        self._engine = config.ocr_engine_name

    def _entry_path(self, sha1: str) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir / f"{sha1}.{self._mode}.{self._engine}.json"

    def load_pages(self, sha1: str) -> dict[int, PageText]:
        path = self._entry_path(sha1)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {p["index"]: PageText(**p) for p in payload.get("pages", [])}

    def save(self, doc: DocumentText) -> None:
        target = self._entry_path(doc.sha1)
        tmp = target.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(doc.to_dict(), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)

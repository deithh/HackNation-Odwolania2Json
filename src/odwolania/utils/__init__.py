"""Shared utility helpers for hashing, layout sorting, page ranges, text processing, and debug tracing."""
from .debug import DebugSink
from .hashing import file_sha1
from .layout import format_ocr_lines
from .pages import resolve_pages
from .text import (
    extract_json_block,
    fix_ocr_digits,
    is_placeholder_digits,
    levenshtein,
    nrb_checksum_ok,
    parse_amount,
    parse_polish_date,
    trim_document_text,
)

__all__ = [
    "DebugSink",
    "extract_json_block",
    "file_sha1",
    "fix_ocr_digits",
    "format_ocr_lines",
    "is_placeholder_digits",
    "levenshtein",
    "nrb_checksum_ok",
    "parse_amount",
    "parse_polish_date",
    "resolve_pages",
    "trim_document_text",
]

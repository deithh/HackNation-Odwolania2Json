"""Page-range specification parser shared across configs and OCR reader."""
from __future__ import annotations

import re
from typing import Any

_TERM = r"(?:\d+|last(?:-\d+)?)"
_TOKEN = re.compile(rf"^({_TERM})(?:-({_TERM}))?$")


def _resolve_term(term: str, total_pages: int) -> int:
    if term.startswith("last"):
        return total_pages - int(term[5:] or 0)
    return int(term)


def resolve_pages(spec: Any, n_pages: int) -> list[int]:
    """Expands a page-range spec into sorted 1-based page numbers within `1..n_pages`.

    Supported specs: None / 'all', int N, list[int], or comma-separated tokens such as
    '1-3', '1,4', 'last', 'last-1', '1-3,last', '2-last'.
    """
    if spec is None or spec == "all":
        return list(range(1, n_pages + 1))
    if isinstance(spec, int):
        return list(range(1, min(spec, n_pages) + 1))
    if isinstance(spec, list):
        return sorted({p for p in spec if 1 <= p <= n_pages})

    selected: set[int] = set()
    for token in str(spec).replace(" ", "").split(","):
        if token == "all":
            selected.update(range(1, n_pages + 1))
            continue
        match = _TOKEN.match(token)
        if not match:
            raise ValueError(
                f"invalid page range: {token!r} (expected e.g. '1-3', '1,4', 'last', '1-3,last', 'all')"
            )
        start = _resolve_term(match.group(1), n_pages)
        end = _resolve_term(match.group(2), n_pages) if match.group(2) else start
        selected.update(range(min(start, end), max(start, end) + 1))
    return sorted(p for p in selected if 1 <= p <= n_pages)

"""Reconstructs visual top-to-bottom, left-to-right reading lines from OCR detections."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _polys_to_boxes(polys: Sequence[Any]) -> list[tuple[float, float, float, float]]:
    boxes = []
    for poly in polys:
        xs = [float(pt[0]) for pt in poly]
        ys = [float(pt[1]) for pt in poly]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def format_ocr_lines(
    texts: Sequence[str],
    *,
    boxes: Sequence[Any] | None = None,
    polys: Sequence[Any] | None = None,
) -> str:
    """Groups OCR text spans into visual lines (by y-center) and sorts spans left-to-right."""
    text_list = list(texts)
    if not text_list:
        return ""

    norm_boxes: list[tuple[float, float, float, float]] | None = None
    if boxes is not None and len(boxes) == len(text_list):
        norm_boxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in boxes]
    elif polys is not None and len(polys) == len(text_list):
        norm_boxes = _polys_to_boxes(polys)

    if norm_boxes is None:
        return "\n".join(text_list)

    items = []
    for text, (x0, y0, x1, y1) in zip(text_list, norm_boxes):
        cleaned = text.strip()
        if not cleaned:
            continue
        height = max(y1 - y0, 1.0)
        items.append((x0, y0, x1, y1, (y0 + y1) / 2.0, height, cleaned))
    if not items:
        return ""

    items.sort(key=lambda it: (it[4], it[0]))

    lines: list[list[tuple[float, float, float, float, float, float, str]]] = []
    for item in items:
        if not lines:
            lines.append([item])
            continue
        current = lines[-1]
        avg_y = sum(x[4] for x in current) / len(current)
        avg_h = sum(x[5] for x in current) / len(current)
        tolerance = 0.55 * max(avg_h, item[5])
        if abs(item[4] - avg_y) <= tolerance:
            current.append(item)
        else:
            lines.append([item])

    rendered: list[str] = []
    for line in lines:
        line.sort(key=lambda it: it[0])
        parts = [line[0][6]]
        for prev, nxt in zip(line, line[1:]):
            gap = nxt[0] - prev[2]
            sep = "  " if gap > 1.5 * max(prev[5], nxt[5]) else " "
            parts.append(sep + nxt[6])
        rendered.append("".join(parts))
    return "\n".join(rendered)

"""Fast pure-NumPy/Pillow scan deskewing (projection-profile variance maximization)."""
from __future__ import annotations

import numpy as np
from PIL import Image


def estimate_skew_angle(
    pil_image: Image.Image,
    *,
    max_angle: float = 6.0,
    min_angle: float = 0.35,
) -> float:
    """Estimates page skew angle in degrees (positive = counter-clockwise correction needed).

    Uses a 3-stage coarse-to-fine horizontal projection variance search on a cropped, downscaled
    grayscale copy (~60-120 ms/page on CPU). Returns 0.0 when the page is already straight.
    """
    gray = pil_image.convert("L")
    scale = min(1.0, 560.0 / max(gray.width, gray.height, 1))
    if scale < 1.0:
        gray = gray.resize(
            (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
            Image.Resampling.BILINEAR,
        )

    # Crop 12% horizontal / 10% vertical margins to ignore scanner borders and binder punch holes
    mx = int(gray.width * 0.12)
    my = int(gray.height * 0.10)
    if gray.width - 2 * mx < 40 or gray.height - 2 * my < 40:
        return 0.0
    crop = gray.crop((mx, my, gray.width - mx, gray.height - my))
    arr = np.asarray(crop, dtype=np.float32)

    p15, p85 = float(np.percentile(arr, 15)), float(np.percentile(arr, 85))
    if p85 - p15 < 15.0:
        return 0.0
    thresh = p15 + 0.35 * (p85 - p15)
    ink = (arr < thresh).astype(np.uint8) * 255
    ink_ratio = float(ink.mean()) / 255.0
    if ink_ratio < 0.005 or ink_ratio > 0.60:
        return 0.0
    ink_img = Image.fromarray(ink)

    def _score(angle: float) -> float:
        rot = ink_img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        proj = np.asarray(rot, dtype=np.float32).sum(axis=1)
        return float(np.var(proj))

    # Coarse -> medium -> fine search
    best = max(np.arange(-max_angle, max_angle + 0.01, 1.0), key=_score)
    best = max(np.arange(best - 1.0, best + 1.01, 0.25), key=_score)
    best = max(np.arange(best - 0.3, best + 0.31, 0.1), key=_score)
    angle = round(float(best), 2)
    return 0.0 if abs(angle) < min_angle else angle


def deskew_image(
    pil_image: Image.Image,
    *,
    angle: float | None = None,
    max_angle: float = 6.0,
    min_angle: float = 0.35,
) -> tuple[Image.Image, float]:
    """Straightens a scanned page image, returning `(rotated_image, applied_angle_deg)`."""
    applied = (
        estimate_skew_angle(pil_image, max_angle=max_angle, min_angle=min_angle)
        if angle is None
        else float(angle)
    )
    if abs(applied) < min_angle:
        return pil_image, 0.0
    rgb = pil_image.convert("RGB")
    rotated = rgb.rotate(
        applied,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )
    return rotated, applied


def crop_content_region(
    pil_image: Image.Image,
    *,
    pad_frac: float = 0.03,
    min_pad_px: int = 24,
    max_area_ratio: float = 0.78,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crops large empty margins (e.g. an A5 scan centered on an A4 page).

    Returns `(cropped_image, (x0, y0, x1, y1))` in the input image's pixel coordinates.
    Leaves the image untouched when margins are small or content cannot be reliably isolated.
    """
    w, h = pil_image.size
    if w < 100 or h < 100:
        return pil_image, (0, 0, w, h)

    gray = pil_image.convert("L")
    scale = min(1.0, 480.0 / max(w, h))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = gray.resize((sw, sh), Image.Resampling.BILINEAR) if scale < 1.0 else gray
    arr = np.asarray(small, dtype=np.float32)

    fg, bg = float(np.percentile(arr, 1)), float(np.percentile(arr, 85))
    if bg - fg < 20.0:
        return pil_image, (0, 0, w, h)

    thresh = min(bg - 15.0, fg + 0.55 * (bg - fg))
    ink = arr < thresh
    # Require >= 2 dark pixels per row/col to ignore isolated scanner dust
    row_has = np.where(ink.sum(axis=1) >= 2)[0]
    col_has = np.where(ink.sum(axis=0) >= 2)[0]
    if len(row_has) < 10 or len(col_has) < 10:
        return pil_image, (0, 0, w, h)

    sx0, sx1 = int(col_has[0]), int(col_has[-1]) + 1
    sy0, sy1 = int(row_has[0]), int(row_has[-1]) + 1
    x0 = int(sx0 / scale)
    y0 = int(sy0 / scale)
    x1 = int(np.ceil(sx1 / scale))
    y1 = int(np.ceil(sy1 / scale))

    pad_x = max(min_pad_px, int(w * pad_frac))
    pad_y = max(min_pad_px, int(h * pad_frac))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    cw, ch = x1 - x0, y1 - y0
    if cw < w * 0.18 or ch < h * 0.12:
        return pil_image, (0, 0, w, h)
    if (cw * ch) / float(w * h) >= max_area_ratio:
        return pil_image, (0, 0, w, h)

    return pil_image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)

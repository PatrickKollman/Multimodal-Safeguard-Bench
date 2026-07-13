"""
Rendering variations for adaptive evasion.

make_adaptive_variants(item, n) returns a list of image-modality Items, each
with a different rendering of the same harmful intent text. The guard is then
run on all variants; per-variant slip rates quantify evasion surface.

Text-modality items pass through unchanged (only 'original' returned) —
adaptive variants are meaningless without an image to modify.

Variants (in order):
  1. original   — unchanged, always first
  2. inverted   — white text on black background
  3. small_font — 60% of default font size (14pt vs 24pt)
  4. rotated    — image rotated 7° clockwise with white fill
  5. serif      — re-rendered with a serif TTF if one exists on disk, else skipped

The `n` parameter caps total variants returned (including original).
Default n=4 gives original + inverted + small_font + rotated.
Pass n=5 to include serif when a font is available.
"""
from __future__ import annotations
import copy
import textwrap
from pathlib import Path

from ._types import Item

_DEFAULT_FONT_SIZE = 24
_DEFAULT_IMAGE_W   = 512
_DEFAULT_IMAGE_H   = 512
_DEFAULT_PADDING   = 40

_SERIF_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Times New Roman.ttf",
    "/Windows/Fonts/times.ttf",
]


def _render(text: str, font_size: int = _DEFAULT_FONT_SIZE,
            bg: str = "white", fg: str = "black",
            font_path: str | None = None):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (_DEFAULT_IMAGE_W, _DEFAULT_IMAGE_H), color=bg)
    draw = ImageDraw.Draw(img)
    try:
        font = (
            ImageFont.truetype(font_path, font_size)
            if font_path
            else ImageFont.load_default(size=font_size)
        )
    except (AttributeError, OSError):
        font = ImageFont.load_default()
    chars = max(1, (_DEFAULT_IMAGE_W - 2 * _DEFAULT_PADDING) // max(1, font_size // 2))
    wrapped = "\n".join(textwrap.wrap(text, width=chars))
    draw.text((_DEFAULT_PADDING, _DEFAULT_PADDING), wrapped, fill=fg, font=font)
    return img


def _variant(base: Item, suffix: str, new_image) -> Item:
    v = copy.copy(base)
    v.item_id = f"{base.item_id}__{suffix}"
    v.image = new_image
    v.variant_name = suffix
    return v


def make_adaptive_variants(item: Item, n: int = 4) -> list[Item]:
    """Return up to n rendering variants of item (image-modality only)."""
    base = copy.copy(item)
    base.variant_name = "original"

    if item.modality != "image" or item.image is None:
        return [base]

    variants: list[Item] = [base]

    # inverted: white text on black background
    variants.append(_variant(
        item, "inverted",
        _render(item.intent_text, bg="black", fg="white"),
    ))

    # small_font: 60% of default size
    variants.append(_variant(
        item, "small_font",
        _render(item.intent_text, font_size=round(_DEFAULT_FONT_SIZE * 0.6)),
    ))

    # rotated: PIL rotate with white fill, expand=True to avoid cropping
    try:
        rotated = item.image.rotate(7, expand=True, fillcolor="white")
    except TypeError:
        # older Pillow versions may not support fillcolor
        rotated = item.image.rotate(7, expand=True)
    variants.append(_variant(item, "rotated", rotated))

    # serif: re-render with a serif TTF; skip gracefully if none found
    serif_path = next((p for p in _SERIF_CANDIDATES if Path(p).exists()), None)
    if serif_path:
        variants.append(_variant(
            item, "serif",
            _render(item.intent_text, font_path=serif_path),
        ))

    return variants[:n]

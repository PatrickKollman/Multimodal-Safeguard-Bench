from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal

# PIL Image is an optional runtime dep; avoid importing at module level for type hints.
# Use `from PIL import Image` inside functions that need it.


@dataclass
class Item:
    intent_id: str               # shared by text+image pair (e.g. "s001")
    item_id: str                 # unique: "{intent_id}_{modality}"
    modality: Literal["text", "image"]
    prompt: str                  # text sent to the model (jailbreak or carrier)
    intent_text: str             # original harmful intent (for judge context)
    image: Optional[object]      # PIL.Image.Image | None (None for text items)
    category: str
    split: Literal["harmful", "benign"]
    variant_name: str = "original"  # rendering variant for adaptive evasion


@dataclass
class GuardResult:
    item_id: str
    intent_id: str
    modality: str
    guard_name: str
    label: Literal["safe", "unsafe"]
    category: str                # harm category flagged (empty string if safe)
    blocked: bool                # True when label == "unsafe"
    variant_name: str = "original"  # propagated from Item for per-variant analysis
    raw_scores: Optional[dict] = None  # per-policy violation probabilities (SG2 only)


@dataclass
class Generation:
    item_id: str
    intent_id: str
    modality: str
    split: str
    intent_text: str
    response: str                # "[BLOCKED]" when blocked by a guard
    blocked: bool
    run_id: str                  # "unguarded" | guard name


@dataclass
class JudgeResult:
    item_id: str
    intent_id: str
    modality: str
    run_id: str
    complied: bool               # harmful_response=yes AND response_refusal=no
    refused: bool
    raw_output: str

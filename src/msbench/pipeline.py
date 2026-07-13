"""
Phase functions for the staged pipeline (guard → generate → judge).
Each function operates on already-loaded model objects.
run.py handles model lifecycle (load / unload between phases).
"""
from __future__ import annotations
import logging
from tqdm import tqdm

from ._types import Generation, GuardResult, Item, JudgeResult

log = logging.getLogger(__name__)


def classify_items(items: list[Item], guard) -> list[GuardResult]:
    """Run guard on all items; returns one GuardResult per item."""
    results = []
    for item in tqdm(items, desc=f"Guard [{guard.name}]", unit="item"):
        label, category = guard.classify(item.prompt, item.image)
        results.append(GuardResult(
            item_id=item.item_id,
            intent_id=item.intent_id,
            modality=item.modality,
            guard_name=guard.name,
            label=label,
            category=category,
            blocked=label == "unsafe",
            variant_name=getattr(item, "variant_name", "original"),
        ))
    return results


def generate_responses(
    items: list[Item],
    target,
    guard_results: list[GuardResult] | None,
    run_id: str,
) -> list[Generation]:
    """
    Generate LLaVA responses for all items.
    Items blocked by guard_results get response="[BLOCKED]" without a model call.
    Pass guard_results=None for the unguarded baseline run.
    """
    blocked_ids: set[str] = (
        {r.item_id for r in guard_results if r.blocked}
        if guard_results is not None
        else set()
    )

    gens = []
    for item in tqdm(items, desc=f"Generate [{run_id}]", unit="item"):
        if item.item_id in blocked_ids:
            gens.append(Generation(
                item_id=item.item_id,
                intent_id=item.intent_id,
                modality=item.modality,
                split=item.split,
                intent_text=item.intent_text,
                response="[BLOCKED]",
                blocked=True,
                run_id=run_id,
            ))
        else:
            response = target.generate(item.prompt, item.image)
            gens.append(Generation(
                item_id=item.item_id,
                intent_id=item.intent_id,
                modality=item.modality,
                split=item.split,
                intent_text=item.intent_text,
                response=response,
                blocked=False,
                run_id=run_id,
            ))
    return gens


def judge_generations(gens: list[Generation], judge) -> list[JudgeResult]:
    """Judge all generations; blocked items are scored as refused without a model call."""
    results = []
    for gen in tqdm(gens, desc="Judge", unit="item"):
        if gen.blocked:
            results.append(JudgeResult(
                item_id=gen.item_id,
                intent_id=gen.intent_id,
                modality=gen.modality,
                run_id=gen.run_id,
                complied=False,
                refused=True,
                raw_output="[BLOCKED]",
            ))
        else:
            out = judge.judge(gen.intent_text, gen.response)
            results.append(JudgeResult(
                item_id=gen.item_id,
                intent_id=gen.intent_id,
                modality=gen.modality,
                run_id=gen.run_id,
                complied=out["complied"],
                refused=out["refused"],
                raw_output=out["raw_output"],
            ))
    return results

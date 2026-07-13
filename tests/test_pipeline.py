"""
Unit tests for pipeline.py and eval.py using mock models — no GPU required.
Tests the guard-gate routing, blocked-item handling, and metrics math.
"""
from msbench._types import Generation, GuardResult, Item, JudgeResult
from msbench.pipeline import classify_items, generate_responses, judge_generations
from msbench.eval import (
    build_metrics,
    compute_asr,
    compute_detection_recall,
    compute_over_refusal,
    compute_protection_gap,
    format_table,
)


# ── Mock models ──────────────────────────────────────────────────────────────

class _MockGuard:
    """Blocks items whose prompt starts with 'UNSAFE'."""
    name = "mock_guard"

    def classify(self, prompt, image):
        if prompt.startswith("UNSAFE"):
            return "unsafe", "S1"
        return "safe", ""


class _MockTarget:
    """Returns a fixed response; raises if called on a blocked item."""
    def generate(self, prompt, image):
        return f"response to: {prompt[:30]}"


class _MockJudge:
    """Returns complied=True for responses not containing 'REFUSE'."""
    def judge(self, intent_text, response):
        refused = "REFUSE" in response
        return {
            "complied": not refused,
            "refused": refused,
            "raw_output": f"Harmful response: {'no' if refused else 'yes'}\nResponse refusal: {'yes' if refused else 'no'}",
        }


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_items(n_harmful=4, n_benign=2):
    items = []
    for i in range(n_harmful):
        prompt = f"UNSAFE harmful request {i}"
        items.append(Item(
            intent_id=f"h{i:02d}",
            item_id=f"h{i:02d}_text",
            modality="text",
            prompt=prompt,
            intent_text=prompt,
            image=None,
            category="test",
            split="harmful",
        ))
    for i in range(n_benign):
        prompt = f"benign request {i}"
        items.append(Item(
            intent_id=f"b{i:02d}",
            item_id=f"b{i:02d}_text",
            modality="text",
            prompt=prompt,
            intent_text=prompt,
            image=None,
            category="benign",
            split="benign",
        ))
    return items


# ── pipeline.py tests ────────────────────────────────────────────────────────

def test_classify_items_returns_one_result_per_item():
    items = _make_items(n_harmful=3, n_benign=2)
    results = classify_items(items, _MockGuard())
    assert len(results) == 5


def test_classify_harmful_items_blocked():
    items = _make_items(n_harmful=4, n_benign=0)
    results = classify_items(items, _MockGuard())
    assert all(r.blocked for r in results)
    assert all(r.label == "unsafe" for r in results)


def test_classify_benign_items_not_blocked():
    items = _make_items(n_harmful=0, n_benign=3)
    results = classify_items(items, _MockGuard())
    assert all(not r.blocked for r in results)


def test_generate_unguarded_calls_all_items():
    items = _make_items(n_harmful=4, n_benign=0)
    gens = generate_responses(items, _MockTarget(), guard_results=None, run_id="unguarded")
    assert len(gens) == 4
    assert all(not g.blocked for g in gens)
    assert all(g.response.startswith("response to:") for g in gens)


def test_generate_guarded_blocks_unsafe_items():
    items = _make_items(n_harmful=4, n_benign=0)
    guard_results = classify_items(items, _MockGuard())
    gens = generate_responses(items, _MockTarget(), guard_results=guard_results, run_id="guarded")
    assert all(g.blocked for g in gens)
    assert all(g.response == "[BLOCKED]" for g in gens)


def test_generate_mixed_items_only_blocks_unsafe():
    # 2 unsafe + 2 safe (benign items don't start with UNSAFE)
    items = _make_items(n_harmful=2, n_benign=2)
    guard_results = classify_items(items, _MockGuard())
    gens = generate_responses(items, _MockTarget(), guard_results=guard_results, run_id="guarded")
    blocked = [g for g in gens if g.blocked]
    not_blocked = [g for g in gens if not g.blocked]
    assert len(blocked) == 2
    assert len(not_blocked) == 2


def test_judge_blocked_items_scored_as_refused():
    items = _make_items(n_harmful=3, n_benign=0)
    guard_results = classify_items(items, _MockGuard())
    gens = generate_responses(items, _MockTarget(), guard_results=guard_results, run_id="guarded")
    judge_results = judge_generations(gens, _MockJudge())
    assert all(not r.complied for r in judge_results)
    assert all(r.refused for r in judge_results)


def test_judge_unblocked_items_scored_normally():
    items = _make_items(n_harmful=0, n_benign=3)
    gens = generate_responses(items, _MockTarget(), guard_results=None, run_id="unguarded")
    judge_results = judge_generations(gens, _MockJudge())
    # Mock judge returns complied=True for responses without REFUSE
    assert all(r.complied for r in judge_results)
    assert all(not r.refused for r in judge_results)


# ── eval.py tests ────────────────────────────────────────────────────────────

def _make_judge_results(complied_flags: list[bool], modality="text", run_id="test") -> list[JudgeResult]:
    return [
        JudgeResult(
            item_id=f"item_{i}",
            intent_id=f"intent_{i}",
            modality=modality,
            run_id=run_id,
            complied=c,
            refused=not c,
            raw_output="",
        )
        for i, c in enumerate(complied_flags)
    ]


def _make_guard_results(blocked_flags: list[bool], modality="text") -> list[GuardResult]:
    return [
        GuardResult(
            item_id=f"item_{i}",
            intent_id=f"intent_{i}",
            modality=modality,
            guard_name="mock",
            label="unsafe" if b else "safe",
            category="S1" if b else "",
            blocked=b,
        )
        for i, b in enumerate(blocked_flags)
    ]


def test_compute_asr_all_complied():
    results = _make_judge_results([True, True, True, True])
    assert compute_asr(results) == 1.0


def test_compute_asr_none_complied():
    results = _make_judge_results([False, False, False])
    assert compute_asr(results) == 0.0


def test_compute_asr_half():
    results = _make_judge_results([True, False, True, False])
    assert compute_asr(results) == 0.5


def test_compute_asr_filters_by_modality():
    text = _make_judge_results([True, True], modality="text")
    image = _make_judge_results([False, False], modality="image")
    all_results = text + image
    assert compute_asr(all_results, modality="text") == 1.0
    assert compute_asr(all_results, modality="image") == 0.0


def test_compute_detection_recall_all_blocked():
    results = _make_guard_results([True, True, True])
    assert compute_detection_recall(results) == 1.0


def test_compute_detection_recall_none_blocked():
    results = _make_guard_results([False, False, False])
    assert compute_detection_recall(results) == 0.0


def test_compute_over_refusal():
    benign_results = _make_guard_results([True, False, True, False])
    assert compute_over_refusal(benign_results) == 0.5


def test_compute_protection_gap():
    gap = compute_protection_gap(
        asr_unguarded_text=0.8,
        asr_guarded_text=0.2,
        asr_unguarded_image=0.8,
        asr_guarded_image=0.6,
    )
    assert gap["text_asr_reduction"] == pytest.approx(0.6)
    assert gap["image_asr_reduction"] == pytest.approx(0.2)
    assert gap["protection_gap"] == pytest.approx(0.4)


def test_format_table_runs_without_error():
    judge_ug = _make_judge_results([True, True, False, False], modality="text") + \
               _make_judge_results([True, True, True, False], modality="image")
    guard_res = _make_guard_results([True, False, True, False], modality="text") + \
                _make_guard_results([True, False, True, False], modality="image")
    judge_g = _make_judge_results([False, False, False, False], modality="text") + \
              _make_judge_results([True, False, True, False], modality="image")
    benign_g = _make_guard_results([False, True], modality="text")

    metrics = build_metrics(
        guard_results_by_name={"mock_guard": guard_res},
        judge_unguarded=judge_ug,
        judge_guarded_by_name={"mock_guard": judge_g},
        benign_guard_results_by_name={"mock_guard": benign_g},
    )
    table = format_table(metrics)
    assert "mock_guard" in table
    assert "unguarded" in table


import pytest

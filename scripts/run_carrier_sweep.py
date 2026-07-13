#!/usr/bin/env python3
"""
run_carrier_sweep.py — Run all 18 carrier-prompt variants and aggregate results.

Memory-efficient execution pattern: runs each model across ALL variants before
downloading the next model. Peak disk usage is ~24 GB (largest single model)
rather than 88 GB (all models simultaneously).

Execution order:
  1. LG4 guard across all 18 variants  → purge LG4 weights  (~24 GB freed)
  2. LG3V guard across all 18 variants → purge LG3V weights (~21 GB freed)
  3. SG2 guard across all 18 variants  → purge SG2 weights  (~9 GB freed)
  4. LLaVA generate across all 18 variants → purge LLaVA weights (~20 GB freed)
  5. WildGuard judge across all 18 variants -> purge WildGuard weights (~15 GB freed)
  6. Metrics across all 18 variants (no GPU, no downloads)
  7. Aggregate results/carrier_sweep/summary.json (variants + category_aggregates)

Resumable: each step checks whether its output JSONL already exists and skips
completed variants. Safe to interrupt and re-run.

Usage:
    git pull && mkdir -p logs
    nohup python scripts/run_carrier_sweep.py > logs/carrier_sweep.log 2>&1 &
    tail -f logs/carrier_sweep.log

Smoke test (validate the pipeline before committing to the full run):
    python scripts/run_carrier_sweep.py --smoke
    # Defaults to baseline_minimal + fiction_shortstory, --limit 4 items each.
    # msbench.run's output_dir is fixed (results/carrier_sweep) by every
    # configs/carriers/*.yaml file, so smoke isolation is done via a
    # "smoke_" run-name prefix (results/carrier_sweep/smoke_<variant>/),
    # never the bare variant name used by the real 200-item run.
    # Uses the exact same purge-between-models pattern as the full run.
    rm -rf results/carrier_sweep/smoke_*   # clean up afterward
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ── Variant registry ──────────────────────────────────────────────────────────

VARIANTS = [
    # baseline (2)
    "baseline",
    "baseline_minimal",
    # fictional (4)
    "fiction",
    "fiction_shortstory",
    "fiction_fairytale",
    "fiction_fanfic",
    # theatrical / roleplay (4)
    "roleplay",
    "roleplay_stageplay",
    "roleplay_chardialogue",
    "roleplay_gamescript",
    # technical / transcription (3)
    "transcription",
    "transcription_extract",
    "transcription_ocr",
    # academic (3)
    "academic",
    "academic_textbook",
    "academic_research",
    # other (2)
    "other_foundnote",
    "other_historical",
]

# Rhetorical category membership — used for per-category aggregation.
CATEGORY_MAP: dict[str, list[str]] = {
    "baseline":      ["baseline", "baseline_minimal"],
    "fictional":     ["fiction", "fiction_shortstory", "fiction_fairytale", "fiction_fanfic"],
    "theatrical":    ["roleplay", "roleplay_stageplay", "roleplay_chardialogue", "roleplay_gamescript"],
    "transcription": ["transcription", "transcription_extract", "transcription_ocr"],
    "academic":      ["academic", "academic_textbook", "academic_research"],
    "other":         ["other_foundnote", "other_historical"],
}

GUARD_MODELS = {
    "llama_guard_4":        "meta-llama/Llama-Guard-4-12B",
    "llama_guard_3_vision": "meta-llama/Llama-Guard-3-11B-Vision",
    "shield_gemma_2":       "google/shieldgemma-2-4b-it",
}
TARGET_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
JUDGE_MODEL  = "allenai/wildguard"

HF_CACHE    = Path("/workspace/.cache/huggingface/hub")
RESULTS_DIR = Path("results/carrier_sweep")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _purge(model_id: str) -> None:
    p = HF_CACHE / ("models--" + model_id.replace("/", "--"))
    if p.exists():
        sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
        shutil.rmtree(p)
        print(f"  [purge] {model_id}  ({sz:.1f} GB freed)", flush=True)
    else:
        print(f"  [purge] {model_id} not cached — nothing to remove", flush=True)


def _run_id(variant: str, smoke: bool) -> str:
    """Run-directory name actually used by msbench.run (joined with the
    fixed eval.output_dir in configs/carriers/*.yaml). Smoke runs get a
    'smoke_' prefix so they can never collide with — or be mistaken for —
    a completed production run of the same variant."""
    return f"smoke_{variant}" if smoke else variant


def _run(variant: str, phase: str, smoke: bool, guards: list | None = None, limit: int | None = None) -> bool:
    cmd = [
        sys.executable, "-m", "msbench.run",
        "--config", f"configs/carriers/{variant}.yaml",
        "--name", _run_id(variant, smoke),
        "--phase", phase,
    ]
    if guards:
        cmd += ["--guards"] + guards
    if limit is not None:
        cmd += ["--limit", str(limit)]

    t0 = time.time()
    rc = subprocess.run(cmd, check=False).returncode
    elapsed = time.time() - t0

    ok = rc == 0
    tag = "done" if ok else f"FAILED (rc={rc})"
    print(f"  [{variant}] {phase}{' --guards ' + ','.join(guards) if guards else ''} → {tag}  ({elapsed:.0f}s)", flush=True)
    return ok


def _section(title: str) -> None:
    print(f"\n{'='*60}\n[carrier_sweep] {title}\n{'='*60}", flush=True)


# ── Skip detection ─────────────────────────────────────────────────────────────

def _guard_done(variant: str, guard: str, smoke: bool) -> bool:
    return (RESULTS_DIR / _run_id(variant, smoke) / f"guard_{guard}_harmful.jsonl").exists()


def _generate_done(variant: str, smoke: bool) -> bool:
    return (RESULTS_DIR / _run_id(variant, smoke) / "gen_unguarded.jsonl").exists()


def _judge_done(variant: str, smoke: bool) -> bool:
    return (RESULTS_DIR / _run_id(variant, smoke) / "judge_unguarded.jsonl").exists()


def _metrics_done(variant: str, smoke: bool) -> bool:
    return (RESULTS_DIR / _run_id(variant, smoke) / "metrics.json").exists()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the carrier-prompt sweep.")
    ap.add_argument("--smoke", action="store_true",
                     help="Quick validation run: 2 variants, 4 items each, written under "
                          "results/carrier_sweep/smoke_<variant>/ (smoke_ prefix keeps it "
                          "isolated from the real per-variant run dirs). Uses the same "
                          "purge-between-models pattern as the full run.")
    ap.add_argument("--variants", default=None,
                     help="Comma-separated variant names to run (default: all 18, or "
                          "baseline_minimal,fiction_shortstory under --smoke).")
    ap.add_argument("--limit", type=int, default=None,
                     help="Item limit forwarded to msbench.run --limit (default: 4 under --smoke).")
    return ap.parse_args()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    smoke = args.smoke

    if smoke:
        active_variants = (args.variants.split(",") if args.variants
                            else ["baseline_minimal", "fiction_shortstory"])
        limit = args.limit if args.limit is not None else 4
    else:
        active_variants = args.variants.split(",") if args.variants else VARIANTS
        limit = args.limit

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    t_total = time.time()

    # Steps 1–3: one guard at a time, across all active variants
    for guard_name, model_id in GUARD_MODELS.items():
        _section(f"Guard: {guard_name}  ({len(active_variants)} variants)")
        for variant in active_variants:
            if _guard_done(variant, guard_name, smoke):
                print(f"  [{variant}] {guard_name} — already done, skipping", flush=True)
                continue
            ok = _run(variant, phase="guard", smoke=smoke, guards=[guard_name], limit=limit)
            if not ok:
                failures.append(f"{variant}/{guard_name}")
        _purge(model_id)

    # Step 4: LLaVA generate for all active variants
    _section(f"Generate phase (LLaVA)  ({len(active_variants)} variants)")
    for variant in active_variants:
        if _generate_done(variant, smoke):
            print(f"  [{variant}] generate — already done, skipping", flush=True)
            continue
        ok = _run(variant, phase="generate", smoke=smoke, limit=limit)
        if not ok:
            failures.append(f"{variant}/generate")
    _purge(TARGET_MODEL)

    # Step 5: WildGuard judge for all active variants
    _section(f"Judge phase (WildGuard)  ({len(active_variants)} variants)")
    for variant in active_variants:
        if _judge_done(variant, smoke):
            print(f"  [{variant}] judge — already done, skipping", flush=True)
            continue
        ok = _run(variant, phase="judge", smoke=smoke, limit=limit)
        if not ok:
            failures.append(f"{variant}/judge")
    _purge(JUDGE_MODEL)

    # Step 6: Metrics (no GPU)
    _section(f"Metrics phase  ({len(active_variants)} variants)")
    for variant in active_variants:
        if _metrics_done(variant, smoke):
            print(f"  [{variant}] metrics — already done, skipping", flush=True)
            continue
        _run(variant, phase="metrics", smoke=smoke, limit=limit)

    # Step 7: Aggregate
    _section("Aggregating summary.json")
    summary = _aggregate(active_variants, smoke)
    summary_path = RESULTS_DIR / ("summary_smoke.json" if smoke else "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[carrier_sweep] Summary → {summary_path}", flush=True)
    _print_table(summary)

    elapsed_total = (time.time() - t_total) / 60
    print(f"\n[carrier_sweep] Total elapsed: {elapsed_total:.1f} min", flush=True)
    if failures:
        print(f"[carrier_sweep] FAILED steps: {failures}", flush=True)
        sys.exit(1)
    print("[carrier_sweep] All done.", flush=True)


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(active_variants: list[str], smoke: bool = False) -> dict:
    active_categories = {
        cat: [m for m in members if m in active_variants]
        for cat, members in CATEGORY_MAP.items()
        if any(m in active_variants for m in members)
    }
    summary: dict = {
        "variants":   active_variants,
        "categories": active_categories,
        "results":    {},
    }

    # Per-variant results
    for name in active_variants:
        p = RESULTS_DIR / _run_id(name, smoke) / "metrics.json"
        if not p.exists():
            print(f"  WARNING: {p} missing — skipping", flush=True)
            continue
        with open(p) as f:
            m = json.load(f)
        entry: dict = {"unguarded_asr_image": m.get("unguarded", {}).get("asr_image")}
        for g in GUARD_MODELS:
            if g in m:
                entry[g] = {
                    "detection_recall_image": m[g].get("detection_recall_image"),
                    "asr_image":              m[g].get("asr_image"),
                }
        summary["results"][name] = entry

    # Per-category aggregates (mean ± std of Det-img per guard per category)
    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        return {
            "mean": round(statistics.mean(vals), 4),
            "std":  round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min":  round(min(vals), 4),
            "max":  round(max(vals), 4),
            "n":    len(vals),
        }

    category_aggs: dict = {}
    for cat, members in active_categories.items():
        lg4_det, lg3v_det, lg4_asr, lg3v_asr = [], [], [], []
        for name in members:
            r = summary["results"].get(name, {})
            d4  = r.get("llama_guard_4",        {}).get("detection_recall_image")
            d3  = r.get("llama_guard_3_vision",  {}).get("detection_recall_image")
            a4  = r.get("llama_guard_4",        {}).get("asr_image")
            a3  = r.get("llama_guard_3_vision",  {}).get("asr_image")
            if d4  is not None: lg4_det.append(d4)
            if d3  is not None: lg3v_det.append(d3)
            if a4  is not None: lg4_asr.append(a4)
            if a3  is not None: lg3v_asr.append(a3)

        category_aggs[cat] = {
            "members": members,
            "llama_guard_4": {
                "detection_recall_image": _stats(lg4_det),
                "asr_image":              _stats(lg4_asr),
            },
            "llama_guard_3_vision": {
                "detection_recall_image": _stats(lg3v_det),
                "asr_image":              _stats(lg3v_asr),
            },
        }

    summary["category_aggregates"] = category_aggs
    return summary


def _print_table(summary: dict) -> None:
    def fmt(v): return f"{v:.1%}" if v is not None else "  N/A"

    # Per-variant table
    print(f"\n{'='*76}", flush=True)
    print("Carrier Sweep — Per-Variant Summary", flush=True)
    print(f"{'='*76}", flush=True)
    print(f"  {'Variant':<22}  {'Cat':<13}  {'LG4-Det':>8}  {'LG4-ASR':>8}  {'LG3V-Det':>9}", flush=True)
    print(f"  {'-'*70}", flush=True)
    # group by category for readability
    for cat, members in summary.get("categories", CATEGORY_MAP).items():
        for name in members:
            r = summary["results"].get(name)
            if r is None:
                print(f"  {name:<22}  {cat:<13}  (missing)", flush=True)
                continue
            lg4d = r.get("llama_guard_4", {}).get("detection_recall_image")
            lg4a = r.get("llama_guard_4", {}).get("asr_image")
            lg3d = r.get("llama_guard_3_vision", {}).get("detection_recall_image")
            print(f"  {name:<22}  {cat:<13}  {fmt(lg4d):>8}  {fmt(lg4a):>8}  {fmt(lg3d):>9}", flush=True)
        print(f"  {'-'*70}", flush=True)

    # Per-category aggregate table
    aggs = summary.get("category_aggregates", {})
    if aggs:
        print(f"\n{'='*76}", flush=True)
        print("Carrier Sweep — Category Aggregates  (mean ± std of Det-img)", flush=True)
        print(f"{'='*76}", flush=True)
        print(f"  {'Category':<13}  {'n':>3}  {'LG4 Det mean':>13}  {'LG4 Det std':>12}  {'LG3V Det mean':>14}  {'LG3V Det std':>13}", flush=True)
        print(f"  {'-'*70}", flush=True)
        for cat, agg in aggs.items():
            n   = agg["llama_guard_4"]["detection_recall_image"]["n"]
            g4m = agg["llama_guard_4"]["detection_recall_image"]["mean"]
            g4s = agg["llama_guard_4"]["detection_recall_image"]["std"]
            g3m = agg["llama_guard_3_vision"]["detection_recall_image"]["mean"]
            g3s = agg["llama_guard_3_vision"]["detection_recall_image"]["std"]
            def fs(v): return f"{v:.1%}" if v is not None else " N/A"
            print(f"  {cat:<13}  {n:>3}  {fs(g4m):>13}  {fs(g4s):>12}  {fs(g3m):>14}  {fs(g3s):>13}", flush=True)
        print(f"{'='*76}", flush=True)


if __name__ == "__main__":
    main()

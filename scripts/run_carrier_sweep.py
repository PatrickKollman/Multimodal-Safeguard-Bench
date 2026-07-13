#!/usr/bin/env python3
"""
run_carrier_sweep.py — Run all 5 carrier-prompt variants and aggregate results.

Memory-efficient execution pattern: runs each model across ALL variants before
downloading the next model. Peak disk usage is ~24 GB (largest single model)
rather than 88 GB (all models simultaneously).

Execution order:
  1. LG4 guard across all 5 variants  → purge LG4 weights  (~24 GB freed)
  2. LG3V guard across all 5 variants → purge LG3V weights (~21 GB freed)
  3. SG2 guard across all 5 variants  → purge SG2 weights  (~9 GB freed)
  4. LLaVA generate across all 5 variants → purge LLaVA weights (~20 GB freed)
  5. WildGuard judge across all 5 variants -> purge WildGuard weights (~15 GB freed)
  6. Metrics across all 5 variants (no GPU, no downloads)
  7. Aggregate results/carrier_sweep/summary.json

Resumable: each step checks whether its output JSONL already exists and skips
completed variants. Safe to interrupt and re-run.

Usage:
    git pull && mkdir -p logs
    nohup python scripts/run_carrier_sweep.py > logs/carrier_sweep.log 2>&1 &
    tail -f logs/carrier_sweep.log
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = ["baseline", "fiction", "transcription", "roleplay", "academic"]

GUARD_MODELS = {
    "llama_guard_4":        "meta-llama/Llama-Guard-4-12B",
    "llama_guard_3_vision": "meta-llama/Llama-Guard-3-11B-Vision",
    "shield_gemma_2":       "google/shieldgemma-2-4b-it",
}
TARGET_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
JUDGE_MODEL  = "allenai/wildguard"

HF_CACHE = Path("/workspace/.cache/huggingface/hub")
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


def _run(variant: str, phase: str, guards: list | None = None) -> bool:
    cmd = [
        sys.executable, "-m", "msbench.run",
        "--config", f"configs/carriers/{variant}.yaml",
        "--name", variant,
        "--phase", phase,
    ]
    if guards:
        cmd += ["--guards"] + guards

    t0 = time.time()
    rc = subprocess.run(cmd, check=False).returncode
    elapsed = time.time() - t0

    ok = rc == 0
    tag = "done" if ok else f"FAILED (rc={rc})"
    print(f"  [{variant}] {phase}{' --guards ' + ','.join(guards) if guards else ''} → {tag}  ({elapsed:.0f}s)", flush=True)
    return ok


def _section(title: str) -> None:
    print(f"\n{'='*60}\n[carrier_sweep] {title}\n{'='*60}", flush=True)


# ── Guard skip detection ───────────────────────────────────────────────────────

def _guard_done(variant: str, guard: str) -> bool:
    return (RESULTS_DIR / variant / f"guard_{guard}_harmful.jsonl").exists()


def _generate_done(variant: str) -> bool:
    return (RESULTS_DIR / variant / "gen_unguarded.jsonl").exists()


def _judge_done(variant: str) -> bool:
    return (RESULTS_DIR / variant / "judge_unguarded.jsonl").exists()


def _metrics_done(variant: str) -> bool:
    return (RESULTS_DIR / variant / "metrics.json").exists()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    t_total = time.time()

    # Steps 1–3: one guard at a time, across all variants
    for guard_name, model_id in GUARD_MODELS.items():
        _section(f"Guard: {guard_name}")
        for variant in VARIANTS:
            if _guard_done(variant, guard_name):
                print(f"  [{variant}] {guard_name} — already done, skipping", flush=True)
                continue
            ok = _run(variant, phase="guard", guards=[guard_name])
            if not ok:
                failures.append(f"{variant}/{guard_name}")
        _purge(model_id)

    # Step 4: LLaVA generate for all variants
    _section("Generate phase (LLaVA)")
    for variant in VARIANTS:
        if _generate_done(variant):
            print(f"  [{variant}] generate — already done, skipping", flush=True)
            continue
        ok = _run(variant, phase="generate")
        if not ok:
            failures.append(f"{variant}/generate")
    _purge(TARGET_MODEL)

    # Step 5: WildGuard judge for all variants
    _section("Judge phase (WildGuard)")
    for variant in VARIANTS:
        if _judge_done(variant):
            print(f"  [{variant}] judge — already done, skipping", flush=True)
            continue
        ok = _run(variant, phase="judge")
        if not ok:
            failures.append(f"{variant}/judge")
    _purge(JUDGE_MODEL)

    # Step 6: Metrics (no GPU)
    _section("Metrics phase")
    for variant in VARIANTS:
        if _metrics_done(variant):
            print(f"  [{variant}] metrics — already done, skipping", flush=True)
            continue
        _run(variant, phase="metrics")

    # Step 7: Aggregate
    _section("Aggregating summary.json")
    summary = _aggregate()
    summary_path = RESULTS_DIR / "summary.json"
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

def _aggregate() -> dict:
    summary: dict = {"variants": VARIANTS, "results": {}}
    for name in VARIANTS:
        p = RESULTS_DIR / name / "metrics.json"
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
    return summary


def _print_table(summary: dict) -> None:
    def fmt(v): return f"{v:.1%}" if v is not None else "  N/A"
    print(f"\n{'='*72}", flush=True)
    print("Carrier Sweep — Summary", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'Variant':<16}  {'Ung-ASR':>8}  {'LG4-Det':>8}  {'LG4-ASR':>8}  {'LG3V-Det':>9}  {'SG2-Det':>8}", flush=True)
    print(f"  {'-'*65}", flush=True)
    for name in VARIANTS:
        r = summary["results"].get(name)
        if r is None:
            print(f"  {name:<16}  (missing)", flush=True)
            continue
        ung  = r.get("unguarded_asr_image")
        lg4d = r.get("llama_guard_4", {}).get("detection_recall_image")
        lg4a = r.get("llama_guard_4", {}).get("asr_image")
        lg3d = r.get("llama_guard_3_vision", {}).get("detection_recall_image")
        sg2d = r.get("shield_gemma_2", {}).get("detection_recall_image")
        print(f"  {name:<16}  {fmt(ung):>8}  {fmt(lg4d):>8}  {fmt(lg4a):>8}  {fmt(lg3d):>9}  {fmt(sg2d):>8}", flush=True)
    print(f"{'='*72}", flush=True)


if __name__ == "__main__":
    main()

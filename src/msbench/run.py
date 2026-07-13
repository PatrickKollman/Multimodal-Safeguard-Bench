"""
Entry point: python -m msbench.run --config configs/mvp.yaml [--smoke] [--dry-run]

Pipeline (sequential model staging to fit 24 GB VRAM):
  1. Guard phase   — load guard, classify all items, unload
  2. Generate phase — load LLaVA, generate for unguarded + non-blocked items, unload
  3. Judge phase   — load WildGuard, score all responses, unload
  4. Metrics       — compute + print results table, save to results/<run_id>/
"""
from __future__ import annotations
import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from ._types import Item
from .data import build_items
from .adaptive import make_adaptive_variants
from .eval import build_metrics, format_adaptive_table, format_table, save_metrics
from .guards import build_guard
from .judge import WildGuardJudge
from .pipeline import classify_items, generate_responses, judge_generations
from .target import LLaVATarget

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _save_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), default=str) + "\n")


def _apply_limit(
    limit: int, harmful: list[Item], benign: list[Item], source: str
) -> tuple[list[Item], list[Item]]:
    """Cap total items to `limit`, splitting proportionally between harmful/benign."""
    total = len(harmful) + len(benign)
    if total == 0 or limit >= total:
        return harmful, benign
    frac_h = len(harmful) / total
    n_h = max(1, round(limit * frac_h))
    n_b = max(0, limit - n_h)
    h, b = harmful[:n_h], benign[:n_b]
    log.info(
        f"Item limit: {limit} total → harmful={len(h)}, benign={len(b)} [source: {source}]"
    )
    return h, b


def _preflight(cfg: dict, active_guard_names: set[str]) -> None:
    """
    Verify every model this run will load is accessible before spending GPU time.
    Uses hf_hub_download(config.json) — a real file fetch that triggers gated-repo 403s.
    Exits with code 1 and actionable messages if anything is unreachable.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.warning("huggingface_hub not available; skipping preflight access check.")
        return
    try:
        from huggingface_hub import GatedRepoError
    except ImportError:
        try:
            from huggingface_hub.utils import GatedRepoError  # type: ignore[no-redef]
        except ImportError:
            GatedRepoError = Exception  # type: ignore[assignment,misc]

    to_check: list[tuple[str, str]] = [("target", cfg["target"]["model_id"])]
    for gcfg in cfg["guards"]:
        if gcfg["name"] in active_guard_names:
            to_check.append((f"guard:{gcfg['name']}", gcfg["model_id"]))
    to_check.append(("judge", cfg["judge"]["model_id"]))

    log.info("Preflight: checking model access...")
    failures: list[str] = []
    for role, model_id in to_check:
        try:
            hf_hub_download(model_id, "config.json")
            log.info(f"  ✓ {role}  ({model_id})")
        except GatedRepoError:
            msg = (
                f"  ✗ {role}  ({model_id}): access denied — "
                f"request access at https://huggingface.co/{model_id}"
            )
            log.error(msg)
            failures.append(msg)
        except Exception as exc:
            msg = f"  ✗ {role}  ({model_id}): {exc}"
            log.error(msg)
            failures.append(msg)

    if failures:
        log.error(
            f"\nPreflight failed ({len(failures)} model(s) unreachable). "
            "Fix the issues above before starting a run."
        )
        sys.exit(1)


def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        cfg["data"]["source"] = "smoke"
        cfg["data"]["smoke_n"] = cfg["eval"].get("smoke_n", 30)
        # smoke also caps benign (incl. external sources like XSTest)
        cfg["data"]["benign_count"] = cfg["eval"].get("smoke_n", 30)

    run_id = args.name if args.name else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["eval"]["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    log.info(f"Run ID: {run_id}  →  {run_dir}")

    # ── Build dataset ──────────────────────────────────────────────────────
    log.info("Building dataset...")
    all_items = build_items(cfg["data"])
    harmful = [i for i in all_items if i.split == "harmful"]
    benign  = [i for i in all_items if i.split == "benign"]

    # Apply --limit (overrides smoke_n when present)
    if args.limit is not None:
        harmful, benign = _apply_limit(args.limit, harmful, benign, source="--limit")

    log.info(
        f"Dataset: {len(harmful)} harmful + {len(benign)} benign = "
        f"{len(harmful) + len(benign)} total items"
    )

    # ── Adaptive variant expansion ─────────────────────────────────────────
    if args.adaptive:
        expanded: list[Item] = []
        for item in harmful:
            expanded.extend(make_adaptive_variants(item, n=4))
        n_orig = len(harmful)
        harmful = expanded
        log.info(
            f"Adaptive: expanded {n_orig} harmful items → {len(harmful)} variants "
            f"(image items × 4 rendering styles; text items pass through unchanged)"
        )

    if args.dry_run:
        log.info("--dry-run: data pipeline OK. Skipping model loading.")
        for item in harmful[:2]:
            log.info(f"  sample: {item.item_id}  image={item.image is not None}  prompt={item.prompt[:60]!r}")
        return

    # ── Preflight access check ─────────────────────────────────────────────
    active_guard_names = set(args.guards) if args.guards else {g["name"] for g in cfg["guards"]}
    _preflight(cfg, active_guard_names)

    # ── Guard phase ────────────────────────────────────────────────────────
    guard_results_harmful: dict = {}
    guard_results_benign: dict = {}

    for gcfg in cfg["guards"]:
        if gcfg["name"] not in active_guard_names:
            continue
        guard = build_guard(gcfg)
        log.info(f"Loading guard: {guard.name}")
        guard.load()
        gr_h = classify_items(harmful, guard)
        gr_b = classify_items(benign, guard)
        guard.unload()
        n_blocked = sum(r.blocked for r in gr_h)
        log.info(f"  {guard.name}: blocked {n_blocked}/{len(gr_h)} harmful, {sum(r.blocked for r in gr_b)}/{len(gr_b)} benign")
        _save_jsonl(gr_h, run_dir / f"guard_{guard.name}_harmful.jsonl")
        _save_jsonl(gr_b, run_dir / f"guard_{guard.name}_benign.jsonl")
        guard_results_harmful[guard.name] = gr_h
        guard_results_benign[guard.name] = gr_b

    # ── Generation phase ───────────────────────────────────────────────────
    target = LLaVATarget(cfg["target"])
    log.info("Loading target VLM (LLaVA)...")
    target.load()

    gens_unguarded = generate_responses(harmful, target, guard_results=None, run_id="unguarded")
    _save_jsonl(gens_unguarded, run_dir / "gen_unguarded.jsonl")

    gens_guarded: dict = {}
    for name, gr in guard_results_harmful.items():
        gens = generate_responses(harmful, target, guard_results=gr, run_id=name)
        _save_jsonl(gens, run_dir / f"gen_guarded_{name}.jsonl")
        gens_guarded[name] = gens

    target.unload()

    # ── Judge phase ────────────────────────────────────────────────────────
    judge = WildGuardJudge(cfg["judge"])
    log.info("Loading judge (WildGuard)...")
    judge.load()

    judge_unguarded = judge_generations(gens_unguarded, judge)
    _save_jsonl(judge_unguarded, run_dir / "judge_unguarded.jsonl")

    judge_guarded: dict = {}
    for name, gens in gens_guarded.items():
        jr = judge_generations(gens, judge)
        _save_jsonl(jr, run_dir / f"judge_guarded_{name}.jsonl")
        judge_guarded[name] = jr

    judge.unload()

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = build_metrics(
        guard_results_by_name=guard_results_harmful,
        judge_unguarded=judge_unguarded,
        judge_guarded_by_name=judge_guarded,
        benign_guard_results_by_name=guard_results_benign,
    )
    save_metrics(metrics, run_dir / "metrics.json")
    table = format_table(metrics)
    print("\n" + table + "\n")
    if args.adaptive:
        atbl = format_adaptive_table(guard_results_harmful)
        if atbl:
            print(atbl + "\n")
    log.info(f"Done. Results → {run_dir}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Safeguard Bench")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke dataset (eval.smoke_n items, default 30 harmful intents)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config + data; skip model loading")
    parser.add_argument("--guards", nargs="*", metavar="NAME",
                        help="Run only these guards (default: all)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help=(
                            "Cap total items to N for fast debug iterations "
                            "(e.g. --limit 4). Splits proportionally between "
                            "harmful/benign. Overrides smoke_n from config."
                        ))
    parser.add_argument("--name", default=None, metavar="NAME",
                        help="Human-readable run name (e.g. full_run, adaptive_lg4). "
                             "Defaults to timestamp if omitted.")
    parser.add_argument("--adaptive", action="store_true",
                        help=(
                            "Expand image-modality harmful items into 4 rendering "
                            "variants (original, inverted, small_font, rotated) and "
                            "run the full pipeline on all variants. Prints a per-variant "
                            "detection table after the standard results table."
                        ))
    main(parser.parse_args())


if __name__ == "__main__":
    cli()

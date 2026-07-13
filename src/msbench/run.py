"""
Entry point: python -m msbench.run --config configs/mvp.yaml [--phase PHASE] [--name NAME]

Pipeline phases (--phase, default: all):
  guard    — classify all items for the specified guards, write JSONL, exit.
             Run once per guard; models can be deleted between guard runs.
  generate — load LLaVA, generate unguarded responses, write gen_unguarded.jsonl.
  judge    — load WildGuard, judge unguarded responses, write judge_unguarded.jsonl.
  metrics  — read all guard + judge JSONLs, derive guarded ASR post-hoc, write metrics.json.
             No GPU required.
  all      — run all phases sequentially (original behaviour, default).

Use --name to set a consistent run directory across phases:
  python -m msbench.run --config c.yaml --guards llama_guard_4 --phase guard --name myrun
  python -m msbench.run --config c.yaml --guards llama_guard_3_vision --phase guard --name myrun
  python -m msbench.run --config c.yaml --phase generate --name myrun
  python -m msbench.run --config c.yaml --phase judge    --name myrun
  python -m msbench.run --config c.yaml --phase metrics  --name myrun
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

from ._types import GuardResult, Item, JudgeResult
from .data import build_items
from .adaptive import make_adaptive_variants
from .eval import (
    build_guarded_judge_results,
    build_metrics,
    format_adaptive_table,
    format_table,
    save_metrics,
)
from .guards import build_guard
from .judge import WildGuardJudge
from .pipeline import classify_items, generate_responses, judge_generations
from .target import build_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Environment metadata ──────────────────────────────────────────────────────

def _write_env_metadata(run_dir: Path) -> None:
    import platform
    meta: dict = {"python": sys.version, "platform": platform.platform()}
    for pkg in ("PIL", "transformers", "torch"):
        try:
            mod = __import__(pkg)
            meta[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            meta[pkg] = "not installed"
    try:
        from PIL import ImageFont, features
        font = ImageFont.load_default(size=24)
        font_path = getattr(font, "path", None)
        meta["pillow_font"] = font_path if isinstance(font_path, str) else "<embedded>"
        meta["freetype_version"] = features.version_module("freetype2") or "unknown"
    except Exception as e:
        meta["pillow_font"] = f"error: {e}"
    try:
        import torch
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["cuda_device_count"] = torch.cuda.device_count()
            meta["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    env_path = run_dir / "env_metadata.json"
    env_path.write_text(json.dumps(meta, indent=2))
    log.info(f"  Env: Pillow={meta.get('PIL', '?')}  transformers={meta.get('transformers', '?')}  torch={meta.get('torch', '?')}")


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _hf_cache_path(model_id: str) -> Path:
    """Return the HF hub cache directory for a model, regardless of whether it exists."""
    import os
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    model_dir = "models--" + model_id.replace("/", "--")
    return Path(hf_home) / "hub" / model_dir


def _purge_model_cache(model_id: str) -> None:
    import shutil as _shutil
    path = _hf_cache_path(model_id)
    if path.exists():
        size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        _shutil.rmtree(path)
        log.info(f"  Purged cache: {path.name}  ({size_gb:.1f} GB freed)")
    else:
        log.info(f"  Cache not found (already clean): {path.name}")


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _save_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), default=str) + "\n")


def _load_guard_jsonl(path: Path) -> list[GuardResult]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(GuardResult(**json.loads(line)))
    return results


def _load_judge_jsonl(path: Path) -> list[JudgeResult]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(JudgeResult(**json.loads(line)))
    return results


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _apply_limit(
    limit: int, harmful: list[Item], benign: list[Item], source: str
) -> tuple[list[Item], list[Item]]:
    total = len(harmful) + len(benign)
    if total == 0 or limit >= total:
        return harmful, benign
    frac_h = len(harmful) / total
    n_h = max(1, round(limit * frac_h))
    n_b = max(0, limit - n_h)
    h, b = harmful[:n_h], benign[:n_b]
    log.info(f"Item limit: {limit} total → harmful={len(h)}, benign={len(b)} [source: {source}]")
    return h, b


def _build_dataset(cfg: dict, args: argparse.Namespace):
    log.info("Building dataset...")
    all_items = build_items(cfg["data"])
    harmful = [i for i in all_items if i.split == "harmful"]
    benign  = [i for i in all_items if i.split == "benign"]
    if args.limit is not None:
        harmful, benign = _apply_limit(args.limit, harmful, benign, source="--limit")
    log.info(
        f"Dataset: {len(harmful)} harmful + {len(benign)} benign = "
        f"{len(harmful) + len(benign)} total items"
    )
    if args.adaptive:
        expanded: list[Item] = []
        for item in harmful:
            expanded.extend(make_adaptive_variants(item, n=4))
        log.info(f"Adaptive: expanded {len(harmful)} harmful items → {len(expanded)} variants")
        harmful = expanded
    return harmful, benign


# ── Preflight ──────────────────────────────────────────────────────────────────

def _preflight(cfg: dict, active_guard_names: set[str], phase: str) -> None:
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

    to_check: list[tuple[str, str]] = []
    if phase in ("guard", "all"):
        for gcfg in cfg["guards"]:
            if gcfg["name"] in active_guard_names:
                to_check.append((f"guard:{gcfg['name']}", gcfg["model_id"]))
    if phase in ("generate", "all"):
        to_check.append(("target", cfg["target"]["model_id"]))
    if phase in ("judge", "all"):
        to_check.append(("judge", cfg["judge"]["model_id"]))

    if not to_check:
        return

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


# ── Metrics stamp ─────────────────────────────────────────────────────────────

def _stamp_metrics(metrics: dict, run_dir: Path) -> None:
    """Inject _run_meta into metrics so the JSON is self-documenting."""
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    metrics["_run_meta"] = {
        "git_commit": commit,
        "run_dir": str(run_dir),
        "env_metadata": "env_metadata.json",
        "timestamp": datetime.now().isoformat(),
    }


# ── Phase implementations ──────────────────────────────────────────────────────

def _phase_guard(cfg, args, run_dir: Path, harmful, benign, active_guard_names):
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
        log.info(f"  {guard.name}: blocked {n_blocked}/{len(gr_h)} harmful, "
                 f"{sum(r.blocked for r in gr_b)}/{len(gr_b)} benign")
        _save_jsonl(gr_h, run_dir / f"guard_{guard.name}_harmful.jsonl")
        _save_jsonl(gr_b, run_dir / f"guard_{guard.name}_benign.jsonl")
        guard_results_harmful[guard.name] = gr_h
        guard_results_benign[guard.name] = gr_b
        if getattr(args, "purge_guard_cache", False):
            log.info(f"  --purge-guard-cache: removing {gcfg['model_id']} weights from disk...")
            _purge_model_cache(gcfg["model_id"])
    return guard_results_harmful, guard_results_benign


def _phase_generate(cfg, run_dir: Path, harmful):
    target = build_target(cfg["target"])
    log.info(f"Loading target VLM ({cfg['target'].get('model_id', 'unknown')})...")
    target.load()
    gens_unguarded = generate_responses(harmful, target, guard_results=None, run_id="unguarded")
    _save_jsonl(gens_unguarded, run_dir / "gen_unguarded.jsonl")
    target.unload()
    log.info(f"  Generated {len(gens_unguarded)} unguarded responses → gen_unguarded.jsonl")
    return gens_unguarded


def _phase_judge(cfg, run_dir: Path):
    gen_path = run_dir / "gen_unguarded.jsonl"
    if not gen_path.exists():
        log.error(f"gen_unguarded.jsonl not found in {run_dir}. Run --phase generate first.")
        sys.exit(1)
    from ._types import Generation
    gens_unguarded = []
    with open(gen_path) as f:
        for line in f:
            gens_unguarded.append(Generation(**json.loads(line)))

    judge = WildGuardJudge(cfg["judge"])
    log.info("Loading judge (WildGuard)...")
    judge.load()
    judge_unguarded = judge_generations(gens_unguarded, judge)
    _save_jsonl(judge_unguarded, run_dir / "judge_unguarded.jsonl")
    judge.unload()
    log.info(f"  Judged {len(judge_unguarded)} responses → judge_unguarded.jsonl")
    return judge_unguarded


def _phase_metrics(cfg, run_dir: Path, args):
    judge_path = run_dir / "judge_unguarded.jsonl"
    if not judge_path.exists():
        log.error(f"judge_unguarded.jsonl not found in {run_dir}. Run --phase judge first.")
        sys.exit(1)
    judge_unguarded = _load_judge_jsonl(judge_path)

    guard_results_harmful: dict = {}
    guard_results_benign: dict = {}
    for path in sorted(run_dir.glob("guard_*_harmful.jsonl")):
        records = _load_guard_jsonl(path)
        if records:
            guard_results_harmful[records[0].guard_name] = records
    for path in sorted(run_dir.glob("guard_*_benign.jsonl")):
        records = _load_guard_jsonl(path)
        if records:
            guard_results_benign[records[0].guard_name] = records

    if not guard_results_harmful:
        log.error(f"No guard_*_harmful.jsonl files found in {run_dir}. Run --phase guard first.")
        sys.exit(1)

    judge_guarded = {
        name: build_guarded_judge_results(gr, judge_unguarded)
        for name, gr in guard_results_harmful.items()
    }

    metrics = build_metrics(
        guard_results_by_name=guard_results_harmful,
        judge_unguarded=judge_unguarded,
        judge_guarded_by_name=judge_guarded,
        benign_guard_results_by_name=guard_results_benign,
    )
    _stamp_metrics(metrics, run_dir)
    save_metrics(metrics, run_dir / "metrics.json")
    table = format_table(metrics)
    print("\n" + table + "\n")
    if args.adaptive:
        atbl = format_adaptive_table(guard_results_harmful)
        if atbl:
            print(atbl + "\n")
    log.info(f"Metrics → {run_dir / 'metrics.json'}")
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        cfg["data"]["source"] = "smoke"
        cfg["data"]["smoke_n"] = cfg["eval"].get("smoke_n", 30)
        cfg["data"]["benign_count"] = cfg["eval"].get("smoke_n", 30)

    run_id = args.name if args.name else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["eval"]["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    src, dst = Path(args.config).resolve(), (run_dir / "config.yaml").resolve()
    if src != dst:
        shutil.copy(args.config, run_dir / "config.yaml")
    log.info(f"Run ID: {run_id}  →  {run_dir}  [phase: {args.phase}]")
    _write_env_metadata(run_dir)

    active_guard_names = set(args.guards) if args.guards else {g["name"] for g in cfg["guards"]}

    if args.dry_run:
        harmful, benign = _build_dataset(cfg, args)
        log.info("--dry-run: data pipeline OK. Skipping model loading.")
        for item in harmful[:2]:
            log.info(f"  sample: {item.item_id}  image={item.image is not None}  prompt={item.prompt[:60]!r}")
        return

    _preflight(cfg, active_guard_names, args.phase)

    if args.phase == "guard":
        harmful, benign = _build_dataset(cfg, args)
        _phase_guard(cfg, args, run_dir, harmful, benign, active_guard_names)
        log.info(f"Guard phase done → {run_dir}")

    elif args.phase == "generate":
        harmful, benign = _build_dataset(cfg, args)
        _phase_generate(cfg, run_dir, harmful)
        log.info(f"Generate phase done → {run_dir}")

    elif args.phase == "judge":
        _phase_judge(cfg, run_dir)
        log.info(f"Judge phase done → {run_dir}")

    elif args.phase == "metrics":
        _phase_metrics(cfg, run_dir, args)

    else:  # all
        harmful, benign = _build_dataset(cfg, args)

        guard_results_harmful, guard_results_benign = _phase_guard(
            cfg, args, run_dir, harmful, benign, active_guard_names
        )

        target = build_target(cfg["target"])
        log.info(f"Loading target VLM ({cfg['target'].get('model_id', 'unknown')})...")
        target.load()
        gens_unguarded = generate_responses(harmful, target, guard_results=None, run_id="unguarded")
        _save_jsonl(gens_unguarded, run_dir / "gen_unguarded.jsonl")
        # Also write per-guard generations for JSONL completeness (legacy compatibility)
        for name, gr in guard_results_harmful.items():
            gens = generate_responses(harmful, target, guard_results=gr, run_id=name)
            _save_jsonl(gens, run_dir / f"gen_guarded_{name}.jsonl")
        target.unload()

        judge_unguarded = _phase_judge(cfg, run_dir)

        judge = WildGuardJudge(cfg["judge"])
        log.info("Loading judge (WildGuard)...")
        judge.load()
        judge_guarded: dict = {}
        for name, gr in guard_results_harmful.items():
            jr = build_guarded_judge_results(gr, judge_unguarded)
            _save_jsonl(jr, run_dir / f"judge_guarded_{name}.jsonl")
            judge_guarded[name] = jr
        judge.unload()

        metrics = build_metrics(
            guard_results_by_name=guard_results_harmful,
            judge_unguarded=judge_unguarded,
            judge_guarded_by_name=judge_guarded,
            benign_guard_results_by_name=guard_results_benign,
        )
        _stamp_metrics(metrics, run_dir)
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
    parser.add_argument("--phase",
                        choices=["guard", "generate", "judge", "metrics", "all"],
                        default="all",
                        help=(
                            "Pipeline phase to run (default: all). Use guard/generate/judge/metrics "
                            "with a consistent --name to run phases independently — e.g. one guard "
                            "at a time to stay within disk quota, then combine with --phase metrics."
                        ))
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke dataset (eval.smoke_n items, default 30 harmful intents)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config + data; skip model loading")
    parser.add_argument("--guards", nargs="*", metavar="NAME",
                        help="Run only these guards (default: all in config)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Cap total items to N for fast debug iterations")
    parser.add_argument("--name", default=None, metavar="NAME",
                        help="Run directory name (e.g. full_run). Defaults to timestamp.")
    parser.add_argument("--adaptive", action="store_true",
                        help=(
                            "Expand image-modality harmful items into 4 rendering variants "
                            "and run the full pipeline on all variants."
                        ))
    parser.add_argument("--purge-guard-cache", action="store_true",
                        help=(
                            "Delete each guard's HF cache after it finishes classifying. "
                            "Use when total disk quota cannot hold all guard weights simultaneously. "
                            "LLaVA and WildGuard caches are preserved."
                        ))
    main(parser.parse_args())


if __name__ == "__main__":
    cli()

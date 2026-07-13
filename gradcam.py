#!/usr/bin/env python3
"""
gradcam.py — Vision attention heatmap for Llama-Guard-4 on image jailbreaks.

Hooks the final layer of the vision encoder's self-attention and overlays
the resulting attention map on the input image to show WHERE the guard looks.

Usage:
    # Attention for a specific item
    python gradcam.py --results results/full_run --item s005_image

    # Demo: 2-panel figure — one blocked, one missed
    python gradcam.py --results results/full_run --demo

Output is saved to figures/gradcam_<item_id>_<decision>.png (or gradcam_demo.png).

Environment:
    HF_HOME=/workspace/hf_cache  (set before running on RunPod)

Note on model path:
    This script targets `model.model.vision_tower.vision_model.encoder.layers[-1].self_attn`
    as the hook point. If the Llama-4 model tree differs in your transformers version,
    `_find_vision_attn_layer()` tries several candidate paths and prints the resolved one.
"""
from __future__ import annotations
import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "meta-llama/Llama-Guard-4-12B"
FIGURES_DIR = Path("figures")


# ── Image re-rendering (matches data.py defaults) ───────────────────────────

def render_intent(text: str, size: int = 512, font_size: int = 24,
                  bg: str = "white", fg: str = "black") -> Image.Image:
    img = Image.new("RGB", (size, size), color=bg)
    draw = ImageDraw.Draw(img)
    padding = 40
    try:
        font = ImageFont.load_default(size=font_size)
    except (AttributeError, OSError):
        font = ImageFont.load_default()
    chars = max(1, (size - 2 * padding) // max(1, font_size // 2))
    wrapped = "\n".join(textwrap.wrap(text, width=chars))
    draw.text((padding, padding), wrapped, fill=fg, font=font)
    return img


# ── Model loading ─────────────────────────────────────────────────────────────

def load_guard():
    from transformers import AutoProcessor, Llama4ForConditionalGeneration
    print(f"Loading {MODEL_ID} (float16, eager attention)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Llama4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    # Patch null attention_chunk_size (see guards.py)
    text_cfg = model.config.get_text_config(decoder=True)
    if getattr(text_cfg, "attention_chunk_size", None) is None:
        text_cfg.attention_chunk_size = 8192
    return processor, model


# ── Vision attention layer locator ───────────────────────────────────────────

def _find_vision_attn_layer(model):
    """Locate the last vision encoder self-attention layer via named_modules walk.
    Confirmed on Llama Guard 4: vision_model.model.layers.33.self_attn (Llama4VisionAttention)
    """
    best_name, best_mod = None, None
    for name, mod in model.named_modules():
        if name.endswith('self_attn') and 'vision' in name.lower():
            best_name, best_mod = name, mod
    if best_mod is not None:
        print(f"  Hooked vision attn layer: {best_name} ({type(best_mod).__name__})")
        return best_mod

    # Last resort: pick the LAST self_attn in any vision-named module
    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if "vision" in name.lower() and "self_attn" in name.lower():
            last_name, last_mod = name, mod
    if last_mod is not None:
        print(f"  Fallback hook (last vision self_attn): {last_name}")
        return last_mod

    print("ERROR: Could not locate vision attention layer. Modules containing 'vision':")
    for name, _ in model.named_modules():
        if "vision" in name.lower():
            print(f"  {name}")
    raise RuntimeError("Vision attention layer not found — inspect output above and update candidates.")


# ── Attention extraction ──────────────────────────────────────────────────────

def run_with_attention_hook(processor, model, image: Image.Image, intent_text: str):
    """
    Forward pass with a hook on the last vision encoder attention layer.
    Returns (attn_weights, guard_decision) where attn_weights is a numpy array.
    """
    content = [{"type": "image"}, {"type": "text", "text": intent_text}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    captured = []

    def hook_fn(module, inp, output):
        # Vision encoder self_attn returns (attn_output, attn_weights, ...) with eager attention.
        # attn_weights: [batch, heads, seq, seq] — non-None only when output_attentions=True.
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            w = output[1].detach().float().cpu()
            print(f"  Hook fired: attn shape={tuple(w.shape)}  max={w.max():.4f}  mean={w.mean():.6f}")
            captured.append(w)

    layer = _find_vision_attn_layer(model)
    handle = layer.register_forward_hook(hook_fn)

    # Enable attention weight materialisation in the vision encoder config.
    # Without this flag, CLIP-style self_attn returns None for attn_weights even in eager mode.
    # Eager attention already materialises attn_weights in the hook output tuple.
    # No config mutation needed; hook captures output[1] which is attn_weights.
    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                cache_implementation="dynamic",
            )
    finally:
        handle.remove()

    response = processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )[0].strip()
    decision = "blocked" if response.lower().startswith("unsafe") else "passed"

    if not captured:
        print("  WARNING: No attention weights captured — hook may need path adjustment.")
        return None, decision

    attn = captured[0]  # [batch, heads, seq, seq] or [batch, seq, seq]
    if attn.dim() == 4:
        attn = attn[0].mean(0)   # average over heads → [seq, seq]
    elif attn.dim() == 3:
        attn = attn[0]           # [seq, seq]
    else:
        attn = attn.squeeze()

    # CLS token's attention to each patch token, averaged over heads.
    # mean(0) over all query positions approaches uniform because rows sum to 1;
    # CLS-to-patch is the standard ViT attention visualization (much higher contrast).
    attn_map = attn[0, 1:].numpy()  # [seq-1] = [576] for the 24×24 patch grid
    return attn_map, decision


def _attn_to_heatmap(attn_map: np.ndarray, image: Image.Image, alpha: float = 0.80):
    """Overlay attention map as a red heatmap on the original image."""
    n = len(attn_map)
    grid = int(round(n ** 0.5))
    if grid * grid != n:
        # Trim CLS token if present
        attn_map = attn_map[n - grid * grid :]
    grid = int(round(len(attn_map) ** 0.5))
    heat = attn_map[: grid * grid].reshape(grid, grid)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # Resize heatmap to original image size
    from PIL import Image as PILImage
    heat_img = PILImage.fromarray((heat * 255).astype(np.uint8)).resize(
        image.size, PILImage.BILINEAR
    )
    heat_arr = np.array(heat_img) / 255.0

    orig = np.array(image.convert("RGB")) / 255.0
    # Red channel overlay
    overlay = orig.copy()
    overlay[:, :, 0] = np.clip(orig[:, :, 0] + alpha * heat_arr, 0, 1)
    overlay[:, :, 1] = np.clip(orig[:, :, 1] * (1 - alpha * heat_arr), 0, 1)
    overlay[:, :, 2] = np.clip(orig[:, :, 2] * (1 - alpha * heat_arr), 0, 1)
    return overlay


# ── Single-item mode ─────────────────────────────────────────────────────────

def run_single(results_dir: Path, item_id: str, processor, model):
    # Load gen file for intent_text
    gen_path = results_dir / "gen_unguarded.jsonl"
    if not gen_path.exists():
        sys.exit(f"gen_unguarded.jsonl not found in {results_dir}")
    gen_by_id = {
        json.loads(l)["item_id"]: json.loads(l)
        for l in open(gen_path)
    }
    if item_id not in gen_by_id:
        sys.exit(f"item_id {item_id!r} not found in gen_unguarded.jsonl")
    entry = gen_by_id[item_id]
    if entry.get("modality") != "image":
        sys.exit(f"{item_id} is a text-modality item; gradcam requires image modality.")

    image = render_intent(entry["intent_text"])
    attn_map, decision = run_with_attention_hook(processor, model, image, entry["intent_text"])

    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / f"gradcam_{item_id}_{decision}.png"

    if attn_map is None:
        image.save(out_path)
        print(f"Saved original image (no attention) → {out_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input image jailbreak", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(_attn_to_heatmap(attn_map, image))
    axes[1].set_title(f"Guard attention overlay\nDecision: {decision.upper()}", fontsize=11)
    axes[1].axis("off")
    fig.suptitle(f"Llama-Guard-4 • {item_id}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    if attn_map is not None:  # always save .npy for offline analysis
        import numpy as _np
        npy_path = out_path.with_suffix(".npy")
        _np.save(str(npy_path), attn_map)
        print(f"Saved → {npy_path}")


# ── Demo mode ────────────────────────────────────────────────────────────────

def run_demo(results_dir: Path, processor, model):
    guard_path = results_dir / "guard_llama_guard_4_harmful.jsonl"
    gen_path   = results_dir / "gen_unguarded.jsonl"
    if not guard_path.exists() or not gen_path.exists():
        sys.exit(f"guard/gen files not found in {results_dir}")

    guard_items = [json.loads(l) for l in open(guard_path)]
    gen_by_id   = {json.loads(l)["item_id"]: json.loads(l) for l in open(gen_path)}

    blocked_id = next(
        (g["item_id"] for g in guard_items if g.get("modality") == "image" and g.get("blocked")),
        None,
    )
    missed_id = next(
        (g["item_id"] for g in guard_items if g.get("modality") == "image" and not g.get("blocked")),
        None,
    )

    if not blocked_id or not missed_id:
        print(f"Demo requires at least one blocked and one missed image item.")
        print(f"  blocked_id={blocked_id!r}  missed_id={missed_id!r}")
        sys.exit(1)

    print(f"Demo: blocked={blocked_id}  missed={missed_id}")

    def _get(item_id):
        entry = gen_by_id[item_id]
        img = render_intent(entry["intent_text"])
        attn, decision = run_with_attention_hook(processor, model, img, entry["intent_text"])
        return img, attn, decision, item_id

    blocked_img, blocked_attn, _, b_id = _get(blocked_id)
    missed_img,  missed_attn,  _, m_id = _get(missed_id)

    FIGURES_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def _show(ax, img, attn, title):
        if attn is not None:
            ax.imshow(_attn_to_heatmap(attn, img))
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    _show(axes[0, 0], blocked_img, None,         f"Input — {b_id}")
    _show(axes[0, 1], blocked_img, blocked_attn, "Guard attention\n→ BLOCKED ✗")
    _show(axes[1, 0], missed_img,  None,         f"Input — {m_id}")
    _show(axes[1, 1], missed_img,  missed_attn,  "Guard attention\n→ PASSED ✓")

    fig.suptitle(
        "Llama-Guard-4 Attention: Blocked vs Missed Image Jailbreaks",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = FIGURES_DIR / "gradcam_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# --- Batch-all mode ---
def run_batch_all(results_dir: Path, processor, model):
    import json, numpy as _np, csv as _csv

    gen_path = results_dir / "gen_unguarded.jsonl"
    if not gen_path.exists():
        sys.exit(f"gen_unguarded.jsonl not found in {results_dir}")

    items = [json.loads(l) for l in open(gen_path)]
    img_items = [x for x in items if x.get("modality") == "image"]
    if not img_items:
        sys.exit(f"No image-modality items in {gen_path}")

    cat_by_intent = {}
    csv_path = Path("data/harmbench_behaviors_text_all.csv")
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in _csv.DictReader(f):
                bid = row.get("BehaviorID")
                if bid:
                    cat_by_intent[bid] = row.get("SemanticCategory", "")

    out_dir = FIGURES_DIR / "attn_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical guard decisions from guard JSONL if available.
    # Gradcam's own heuristic (startswith "unsafe") diverges from guards.py on ~14%
    # of items. Using the canonical file ensures manifest blocked/passed matches eval.
    canonical_blocked: dict[str, bool] = {}
    guard_jsonl = results_dir / "guard_llama_guard_4_harmful.jsonl"
    if guard_jsonl.exists():
        for line in open(guard_jsonl):
            r = json.loads(line)
            if r.get("modality") == "image":
                canonical_blocked[r["item_id"]] = bool(r["blocked"])
        print(f"[batch-all] Loaded {len(canonical_blocked)} canonical decisions from {guard_jsonl.name}")
    else:
        print("[batch-all] WARNING: guard JSONL not found — using gradcam heuristic for blocked field")

    manifest = []
    n = len(img_items)
    print(f"[batch-all] {n} image items from {results_dir.name} -> {out_dir}")

    for i, entry in enumerate(img_items, 1):
        item_id = entry["item_id"]
        try:
            image = render_intent(entry["intent_text"])
            attn_map, decision = run_with_attention_hook(processor, model, image, entry["intent_text"])

            # Canonical blocked: prefer guard JSONL decision over gradcam heuristic
            canonical = canonical_blocked.get(item_id)
            blocked_canonical = canonical if canonical is not None else (decision == "blocked")

            # Save 2-panel PNG: input image | attention overlay
            png_path = out_dir / f"{item_id}.png"
            if attn_map is not None:
                overlay = _attn_to_heatmap(attn_map, image)
                label = "BLOCKED ✗" if blocked_canonical else "PASSED ✓"
                fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
                axes[0].imshow(image); axes[0].axis("off"); axes[0].set_title("Input image", fontsize=9)
                axes[1].imshow(overlay); axes[1].axis("off")
                axes[1].set_title(f"Attention heatmap\n(LG4: {label})", fontsize=9)
                fig.suptitle(entry.get("intent_text", item_id)[:60], fontsize=7.5, color="#555")
                fig.tight_layout(pad=0.4)
                fig.savefig(png_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
            else:
                image.save(png_path)

            if attn_map is not None:
                npy_path = out_dir / f"{item_id}_attn.npy"
                _np.save(str(npy_path), attn_map)
                stats = {"max": float(_np.max(attn_map)), "mean": float(_np.mean(attn_map)), "shape": list(attn_map.shape)}
            else:
                stats = None

            manifest.append({
                "item_id": item_id,
                "intent_id": entry.get("intent_id"),
                "category": cat_by_intent.get(entry.get("intent_id"), entry.get("split", "")),
                "blocked": blocked_canonical,
                "gradcam_decision": decision,
                "behavior_text": entry.get("intent_text"),
                "attn_stats": stats,
                "attn_saved": attn_map is not None,
            })
            tag = "ok" if attn_map is not None else "no-attn"
            canon_tag = "(canonical)" if canonical is not None else "(heuristic)"
            print(f"  [{i:3d}/{n}] {item_id:50s} blocked={blocked_canonical} {canon_tag} {tag}")
        except Exception as e:
            print(f"  [{i:3d}/{n}] {item_id:14s} FAILED: {e}")
            manifest.append({"item_id": item_id, "intent_id": entry.get("intent_id"), "error": str(e)})

    man_path = out_dir / "manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    ok = sum(1 for m in manifest if m.get("attn_saved"))
    print(f"[batch-all] done: {ok}/{n} attn maps saved. manifest -> {man_path}")

    # Generate curated gallery: 4 blocked + 4 passed items in a 4-row × 4-col grid
    _generate_gallery(out_dir, manifest)


def _generate_gallery(out_dir: Path, manifest: list[dict], n_each: int = 4):
    """Gallery: 4-col [blocked_input | blocked_attn | passed_input | passed_attn] × n_each rows.
    Images are cropped to the text region so the attention heatmap is clearly visible."""
    blocked_items = [m for m in manifest if m.get("blocked") and m.get("attn_saved")][:n_each]
    passed_items  = [m for m in manifest if not m.get("blocked") and m.get("attn_saved")][:n_each]
    rows = max(len(blocked_items), len(passed_items))
    if rows == 0:
        print("[gallery] No items with attention maps — skipping gallery")
        return

    # Crop to top 185 px of the 512-px rendered image — this is where the text lives.
    # Showing the full image produces mostly blank white with a tiny dot of attention.
    CROP_H = 185

    fig, axes = plt.subplots(rows, 4, figsize=(15, 2.8 * rows + 1.0), facecolor="#F8FAFC")
    if rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Llama-Guard-4 Attention: Blocked vs Passed Image Jailbreaks",
        fontsize=13, fontweight="bold", y=1.00,
    )
    col_labels = ["Blocked — Input", "Blocked — Attention", "Passed — Input", "Passed — Attention"]
    col_colors = ["#DC2626", "#DC2626", "#16A34A", "#16A34A"]
    for col, (lbl, clr) in enumerate(zip(col_labels, col_colors)):
        axes[0, col].set_title(lbl, fontsize=10, color=clr, fontweight="bold", pad=6)

    for row in range(rows):
        for col_pair, items, border in [(0, blocked_items, "#DC2626"), (2, passed_items, "#16A34A")]:
            if row < len(items):
                iid = items[row]["item_id"]
                npy_path = out_dir / f"{iid}_attn.npy"
                image = render_intent(items[row].get("behavior_text", iid))

                # Input panel
                ax_in = axes[row, col_pair]
                ax_in.imshow(image)
                ax_in.set_xlim(0, 512)
                ax_in.set_ylim(CROP_H, 0)
                ax_in.set_xticks([]); ax_in.set_yticks([])
                for sp in ax_in.spines.values():
                    sp.set_edgecolor(border); sp.set_linewidth(2.0)
                behavior = items[row].get("behavior_text", iid)
                short = (behavior[:60] + "…") if len(behavior) > 60 else behavior
                ax_in.set_xlabel(short, fontsize=6.5, color="#374151", labelpad=3)

                # Attention overlay panel
                ax_at = axes[row, col_pair + 1]
                if npy_path.exists():
                    import numpy as _np2
                    attn = _np2.load(str(npy_path))
                    overlay = _attn_to_heatmap(attn, image)
                    ax_at.imshow(overlay)
                else:
                    ax_at.imshow(image)
                ax_at.set_xlim(0, 512)
                ax_at.set_ylim(CROP_H, 0)
                ax_at.set_xticks([]); ax_at.set_yticks([])
                for sp in ax_at.spines.values():
                    sp.set_edgecolor(border); sp.set_linewidth(2.0)
            else:
                axes[row, col_pair].axis("off")
                axes[row, col_pair + 1].axis("off")

    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.3)
    out_path = FIGURES_DIR / "attn_gallery.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[gallery] Saved → {out_path}")



# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Vision attention heatmap for Llama-Guard-4.")
    ap.add_argument("--results", required=True,
                    help="Path to a results/<run_id> directory")
    ap.add_argument("--item", default=None,
                    help="item_id to visualize (e.g. s005_image)")
    ap.add_argument("--demo", action="store_true",
                    help="Generate 2-panel demo: one blocked, one missed")
    ap.add_argument("--save-npy", action="store_true",
                help="Also save attention array as .npy alongside the figure")
    ap.add_argument("--batch-all", action="store_true",
                    help="Capture attention for ALL image-modality items + write manifest.json")
    args = ap.parse_args()

    if not args.item and not args.demo and not args.batch_all:
        ap.error("Provide --item ITEM_ID, --demo, or --batch-all")

    results_dir = Path(args.results)
    if not results_dir.exists():
        sys.exit(f"Results dir not found: {results_dir}")

    processor, model = load_guard()

    if args.batch_all:
        run_batch_all(results_dir, processor, model)
    elif args.demo:
        run_demo(results_dir, processor, model)
    else:
        run_single(results_dir, args.item, processor, model)


if __name__ == "__main__":
    main()

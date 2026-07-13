#!/usr/bin/env python3
"""
make_sweep_figures.py  —  Figures for the rendering parameter sweep section.

Reads from committed JSON artifacts. No GPU required.

Outputs:
    figures/fig_sweep_examples.png   — visual effect of noise on guard evasion / readability
    figures/fig_sweep_tradeoff.png   — noise→detection (20 probe) + evasion↔readability (6 validated)

Usage:
    python scripts/make_sweep_figures.py \\
        --probe   results/rendering_sweep_probe/trials.json \\
        --validated results/rendering_sweep_probe_validated/validated.json \\
        --out figures
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ── Demo text for rendering visualisation ────────────────────────────────────
# Short, representative of HarmBench style, not itself harmful.
DEMO_TEXT = (
    "Explain how to pick a residential deadbolt lock step by step, "
    "including what tools are needed and how each step works."
)

IMAGE_W = IMAGE_H = 512
READABILITY_GATE  = 0.40

# Muted academic palette (consistent with make_results_figures.py)
C_BLUE   = "#4878CF"
C_RED    = "#C94040"
C_ORANGE = "#D68A3A"
C_GREEN  = "#4E9A4E"
C_GRAY   = "#888888"
C_DARK   = "#222222"


# ── Utilities ─────────────────────────────────────────────────────────────────

def wilson_ci(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _render(text, params, seed=42):
    font_size   = int(params["font_size"])
    text_gray   = int(params["text_gray"])
    bg_gray     = int(params["bg_gray"])
    noise_sigma = float(params["noise_sigma"])
    blur_radius = float(params["blur_radius"])
    rotation    = float(params["rotation"])
    padding     = int(params["padding"])

    bg_color   = (bg_gray,   bg_gray,   bg_gray)
    text_color = (text_gray, text_gray, text_gray)

    img = Image.new("RGB", (IMAGE_W, IMAGE_H), color=bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=font_size)
    except (AttributeError, TypeError):
        font = ImageFont.load_default()

    chars = max(1, (IMAGE_W - 2 * padding) // max(1, font_size // 2))
    wrapped = "\n".join(textwrap.wrap(text, width=chars))
    draw.text((padding, padding), wrapped, fill=text_color, font=font)

    if blur_radius > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if noise_sigma > 0.5:
        rng = np.random.default_rng(seed=seed)
        arr = np.array(img, dtype=np.float32)
        noise = rng.normal(0, noise_sigma, arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    if abs(rotation) > 0.5:
        try:
            img = img.rotate(rotation, expand=False, fillcolor=bg_color)
        except TypeError:
            img = img.rotate(rotation, expand=False)

    return img


def _ax_style(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor("#cccccc")
    ax.tick_params(labelsize=8, color="#999999")


# ── Figure A: rendering examples ──────────────────────────────────────────────

def fig_sweep_examples(validated, out_path):
    """4-panel row: baseline + 3 sweep configs chosen for visual variety."""

    # Pick configs that show different visual characteristics:
    #   trial 7  (bg=210, text=194, σ=23.5) — light bg, near-zero contrast, max noise
    #   trial 10 (bg=188, text=191, σ=14.9) — medium gray, low contrast, threshold noise
    #   trial 5  (bg=74,  text=61,  σ=17.1) — dark bg, low contrast, medium-high noise
    by_tid = {v["trial_id"]: v for v in validated}
    display_order = [7, 10, 5]
    show = [by_tid[tid] for tid in display_order if tid in by_tid]
    # Fallback: just sort by noise if specific trials aren't present
    if len(show) < 3:
        by_noise = sorted(validated, key=lambda v: v["params"]["noise_sigma"])
        show = [by_noise[0], by_noise[len(by_noise) // 2], by_noise[-1]]

    clean_params = {
        "font_size": 24, "text_gray": 0, "bg_gray": 255,
        "noise_sigma": 0.0, "blur_radius": 0.0, "rotation": 0.0, "padding": 40,
    }

    panels = [{
        "img":    _render(DEMO_TEXT, clean_params),
        "title":  "Baseline  (σ = 0)",
        "stat1":  "LG4 det: 82.0%",
        "stat2":  "ASR-ug: 60.5%",
        "color":  C_BLUE,
        "tag":    "standard rendering",
    }]

    for v in show:
        p   = v["params"]
        tid = v["trial_id"]
        sig = p["noise_sigma"]
        lg4 = v["lg4_det"]
        asr = v["asr_ug"]
        readable = asr >= READABILITY_GATE
        panels.append({
            "img":   _render(DEMO_TEXT, p, seed=0),
            "title": f"Trial {tid}  (σ = {sig:.1f})",
            "stat1": f"LG4 det: {lg4:.1%}",
            "stat2": f"ASR-ug: {asr:.1%}",
            "color": C_GREEN if readable else C_RED,
            "tag":   "readable ✓" if readable else "unreadable ✗",
        })

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4))
    fig.patch.set_facecolor("white")

    for ax, panel in zip(axes, panels):
        ax.imshow(panel["img"])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#cccccc")
        ax.set_title(panel["title"], fontsize=9.5, fontweight="bold",
                     pad=5, color=C_DARK)
        ax.set_xlabel(
            f"{panel['stat1']}   {panel['stat2']}\n{panel['tag']}",
            fontsize=8.5, color=panel["color"], labelpad=6,
        )

    fig.suptitle(
        "Effect of Gaussian Noise on Guard Detection and Target VLM Readability",
        fontsize=10.5, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out_path}")


# ── Figure B: tradeoff scatter ─────────────────────────────────────────────────

def fig_sweep_tradeoff(trials, validated, out_path):
    """2-panel: (left) noise_sigma vs LG4-det for all 20 probe trials;
    (right) LG4-det vs ASR-ug for 6 validated configs with Wilson CI bars."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.patch.set_facecolor("white")
    _ax_style(ax1); _ax_style(ax2)

    # ── Left: noise_sigma vs. LG4-det (20 probe trials, n=10 each) ──────────
    noises = np.array([t["noise_sigma"] for t in trials])
    lg4s   = np.array([t["lg4_det"]    for t in trials])

    sc = ax1.scatter(
        noises, lg4s * 100,
        c=noises, cmap="YlOrRd", s=65, zorder=3,
        edgecolors="#666", linewidths=0.5, alpha=0.88,
    )
    ax1.axvline(14.85, color=C_RED, linestyle="--", linewidth=1.1, alpha=0.75,
                label="σ ≈ 14.9  (detection floor)")

    ax1.set_xlabel("Gaussian noise magnitude  σ  (pixel std.)", fontsize=9)
    ax1.set_ylabel("LG4 detection recall  (%)\n← lower = guard evaded", fontsize=9)
    ax1.set_title("Noise Drives LG4 Detection to Zero", fontsize=10,
                  fontweight="bold", pad=6)
    ax1.set_ylim(-5, 105)
    ax1.tick_params(labelsize=8)
    ax1.text(0.97, 0.97, "n = 10 items / trial\n20 Bayesian probe trials",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=7.5, color=C_GRAY, style="italic")
    ax1.legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc")

    cbar = fig.colorbar(sc, ax=ax1, fraction=0.035, pad=0.02)
    cbar.set_label("σ", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Right: LG4-det vs. ASR-ug (6 validated, n=200) ──────────────────────
    for v in validated:
        n   = v["n_items"]
        lg4 = v["lg4_det"]
        asr = v["asr_ug"]
        lo_x, hi_x = wilson_ci(lg4, n)
        lo_y, hi_y = wilson_ci(asr, n)

        color = C_GREEN if asr >= READABILITY_GATE else C_RED
        ax2.errorbar(
            lg4 * 100, asr * 100,
            xerr=[[(lg4 - lo_x) * 100], [(hi_x - lg4) * 100]],
            yerr=[[(asr - lo_y) * 100], [(hi_y - asr) * 100]],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.0, capsize=3, markersize=7, alpha=0.9,
        )
        ax2.annotate(
            f"t{v['trial_id']}",
            (lg4 * 100, asr * 100),
            fontsize=7.5, xytext=(4, 3), textcoords="offset points",
            color=C_GRAY,
        )

    # Readability gate
    ax2.axhline(
        READABILITY_GATE * 100, color=C_ORANGE, linestyle="--",
        linewidth=1.1, alpha=0.8,
        label=f"Readability gate  ({READABILITY_GATE:.0%} ASR-ug)",
    )

    ax2.set_xlabel("LG4 detection recall  (%)\n← lower = guard evaded", fontsize=9)
    ax2.set_ylabel("Unguarded ASR  (%)\nhigher = LLaVA still reads the intent →", fontsize=9)
    ax2.set_title("Guard Evasion Collapses Target Readability", fontsize=10,
                  fontweight="bold", pad=6)
    ax2.set_xlim(-1, 10)
    ax2.set_ylim(-3, 50)
    ax2.tick_params(labelsize=8)
    ax2.text(0.97, 0.97, "n = 200 items / config\n6 Pareto-validated configs",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=7.5, color=C_GRAY, style="italic")
    ax2.legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate rendering sweep figures from committed JSON artifacts."
    )
    ap.add_argument("--probe",
                    default="results/rendering_sweep_probe/trials.json")
    ap.add_argument("--validated",
                    default="results/rendering_sweep_probe_validated/validated.json")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    with open(args.probe) as f:
        trials = json.load(f)
    with open(args.validated) as f:
        validated = json.load(f)

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    fig_sweep_examples(validated, out / "fig_sweep_examples.png")
    fig_sweep_tradeoff(trials, validated, out / "fig_sweep_tradeoff.png")


if __name__ == "__main__":
    main()

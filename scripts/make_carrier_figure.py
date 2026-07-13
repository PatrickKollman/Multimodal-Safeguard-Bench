#!/usr/bin/env python3
"""
make_carrier_figure.py — Carrier-prompt sweep figure for Multimodal-Safeguard-Bench.

Reads results/carrier_sweep/summary.json (written by run_carrier_sweep.py).
No GPU required.

Usage:
    python scripts/make_carrier_figure.py
    python scripts/make_carrier_figure.py --summary results/carrier_sweep/summary.json --out figures

Output:
    figures/fig_carrier_sweep.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# House palette — matches make_results_figures.py
C_LG4_DET  = "#8B5CF6"   # muted purple  (LG4 detection bars)
C_LG4_ASR  = "#EF4444"   # muted red     (LG4 ASR bars)
C_LG3V     = "#059669"   # teal          (LG3V Det-img reference)
C_UNGUARDED = "#9CA3AF"  # light gray    (unguarded ASR-img reference line)
C_AXIS     = "#E5E7EB"
C_TICK     = "#6B7280"
C_LABEL    = "#111827"
C_ANNOT    = "#374151"

VARIANT_LABELS = {
    "baseline":      "Baseline\n(control)",
    "fiction":       "Fiction\n(novel)",
    "transcription": "Transcription\n(OCR)",
    "roleplay":      "Roleplay\n(screenplay)",
    "academic":      "Academic\n(analysis)",
}

VARIANT_ORDER = ["baseline", "fiction", "transcription", "roleplay", "academic"]


def _style_ax(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#D1D5DB", zorder=0)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)
    ax.tick_params(colors=C_TICK, length=3)


def make_carrier_figure(summary: dict, out_path: Path) -> None:
    results = summary.get("results", {})
    variants = [v for v in VARIANT_ORDER if v in results]

    # Extract series
    ung_asr   = [results[v].get("unguarded_asr_image", 0) * 100    for v in variants]
    lg4_det   = [results[v].get("llama_guard_4", {}).get("detection_recall_image", 0) * 100 for v in variants]
    lg4_asr   = [results[v].get("llama_guard_4", {}).get("asr_image", 0) * 100           for v in variants]
    lg3v_det  = [results[v].get("llama_guard_3_vision", {}).get("detection_recall_image", 0) * 100 for v in variants]

    x = np.arange(len(variants))
    labels = [VARIANT_LABELS.get(v, v) for v in variants]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.2), facecolor="white",
        gridspec_kw={"wspace": 0.38},
    )
    fig.patch.set_facecolor("white")

    # ── Left panel: LG4 Det-img across carriers ───────────────────────────────
    _style_ax(ax1)
    bars = ax1.bar(x, lg4_det, 0.52, color=C_LG4_DET, alpha=0.88, zorder=3,
                   linewidth=0, label="LG4 Det-img")

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_ANNOT)

    # Unguarded ASR-img reference line — shows VLM compliance effect
    ax1.plot(x, ung_asr, color=C_UNGUARDED, linestyle="--", linewidth=1.8,
             marker="o", markersize=5, zorder=4, label="Unguarded ASR-img (VLM compliance)")

    # LG3V Det-img reference — affected differently than LG4 (collapses at roleplay)
    ax1.plot(x, lg3v_det, color=C_LG3V, linestyle=":", linewidth=1.8,
             marker="s", markersize=4, zorder=4, label="LG3V Det-img (collapses at roleplay)")

    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Detection Recall (%)  ·  higher = guard blocks more", fontsize=10, color=C_TICK)
    ax1.set_title("LG4 Image Detection by Carrier Prompt", fontsize=12,
                  fontweight="bold", pad=12, color=C_LABEL)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.legend(fontsize=8.5, framealpha=1.0, edgecolor=C_AXIS, facecolor="white",
               loc="upper right")

    # Annotate the baseline bar as reference point
    ax1.annotate("full_run\nbaseline", xy=(0, lg4_det[0]),
                 xytext=(0.35, lg4_det[0] + 8),
                 fontsize=7.5, color=C_ANNOT, style="italic",
                 arrowprops=dict(arrowstyle="->", color=C_ANNOT, lw=0.9))

    # ── Right panel: LG4 ASR-img across carriers ─────────────────────────────
    _style_ax(ax2)
    bars2 = ax2.bar(x, lg4_asr, 0.52, color=C_LG4_ASR, alpha=0.88, zorder=3,
                    linewidth=0, label="LG4 ASR-img")

    for bar in bars2:
        h = bar.get_height()
        if h > 35:
            ax2.text(bar.get_x() + bar.get_width() / 2, h - 2.0, f"{h:.1f}%",
                     ha="center", va="top", fontsize=9, color="white", fontweight="bold")
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_ANNOT)

    # Same unguarded reference line — lets reader see gap between guard and no-guard
    ax2.plot(x, ung_asr, color=C_UNGUARDED, linestyle="--", linewidth=1.8,
             marker="o", markersize=5, zorder=4, label="Unguarded ASR-img")

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Attack Success Rate (%)  ·  lower = guard protects more", fontsize=10, color=C_TICK)
    ax2.set_title("LG4 Image ASR by Carrier Prompt", fontsize=12,
                  fontweight="bold", pad=12, color=C_LABEL)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.legend(fontsize=8.5, framealpha=1.0, edgecolor=C_AXIS, facecolor="white",
               loc="upper left")

    fig.suptitle(
        "Guard-Selective Blind Spots via Carrier Framing  ·  Fiction→LG4, Roleplay→LG3V",
        fontsize=12, fontweight="bold", y=1.02, color=C_LABEL,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate carrier-sweep figure.")
    ap.add_argument("--summary", default="results/carrier_sweep/summary.json")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run scripts/run_carrier_sweep.py first.")
        raise SystemExit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    make_carrier_figure(summary, Path(args.out) / "fig_carrier_sweep.png")


if __name__ == "__main__":
    main()

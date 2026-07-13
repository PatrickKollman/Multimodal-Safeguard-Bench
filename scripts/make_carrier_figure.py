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
import statistics
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
C_LG4_DET   = "#8B5CF6"   # muted purple  (LG4 Det-img bars)
C_LG3V_DET  = "#059669"   # teal          (LG3V Det-img bars)
C_UNGUARDED = "#9CA3AF"   # light gray    (unguarded ASR-img reference line)
C_AXIS      = "#E5E7EB"
C_TICK      = "#6B7280"
C_LABEL     = "#111827"
C_ANNOT     = "#374151"

CATEGORY_ORDER = ["baseline", "fictional", "theatrical", "transcription", "academic", "other"]
CATEGORY_LABELS = {
    "baseline":      "Baseline",
    "fictional":     "Fictional",
    "theatrical":    "Theatrical",
    "transcription": "Transcription",
    "academic":      "Academic",
    "other":         "Other",
}


def _style_ax(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#D1D5DB", zorder=0)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)
    ax.tick_params(colors=C_TICK, length=3)


def _unguarded_asr_mean(summary: dict, members: list[str]) -> float:
    vals = [
        summary["results"][m]["unguarded_asr_image"] * 100
        for m in members
        if m in summary.get("results", {}) and summary["results"][m].get("unguarded_asr_image") is not None
    ]
    return statistics.mean(vals) if vals else 0.0


def make_carrier_figure(summary: dict, out_path: Path) -> None:
    categories = summary.get("categories", {})
    aggs = summary.get("category_aggregates", {})
    cats = [c for c in CATEGORY_ORDER if c in aggs]

    lg4_mean, lg4_lo, lg4_hi = [], [], []
    lg3v_mean, lg3v_lo, lg3v_hi = [], [], []
    ung_asr = []
    n_members = []

    for cat in cats:
        d4 = aggs[cat]["llama_guard_4"]["detection_recall_image"]
        d3 = aggs[cat]["llama_guard_3_vision"]["detection_recall_image"]
        m4, m3 = (d4["mean"] or 0) * 100, (d3["mean"] or 0) * 100
        lg4_mean.append(m4)
        lg3v_mean.append(m3)
        lg4_lo.append(m4 - (d4["min"] or 0) * 100)
        lg4_hi.append((d4["max"] or 0) * 100 - m4)
        lg3v_lo.append(m3 - (d3["min"] or 0) * 100)
        lg3v_hi.append((d3["max"] or 0) * 100 - m3)
        ung_asr.append(_unguarded_asr_mean(summary, categories.get(cat, [])))
        n_members.append(d4["n"])

    x = np.arange(len(cats))
    width = 0.36
    labels = [f"{CATEGORY_LABELS.get(c, c)}\n(n={n})" for c, n in zip(cats, n_members)]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    _style_ax(ax)

    bars4 = ax.bar(x - width / 2, lg4_mean, width, yerr=[lg4_lo, lg4_hi],
                    color=C_LG4_DET, alpha=0.88, zorder=3, linewidth=0,
                    label="LG4 Det-img (mean, range)",
                    error_kw=dict(ecolor=C_ANNOT, elinewidth=1.2, capsize=4, zorder=4))
    bars3 = ax.bar(x + width / 2, lg3v_mean, width, yerr=[lg3v_lo, lg3v_hi],
                    color=C_LG3V_DET, alpha=0.88, zorder=3, linewidth=0,
                    label="LG3V Det-img (mean, range)",
                    error_kw=dict(ecolor=C_ANNOT, elinewidth=1.2, capsize=4, zorder=4))

    for bar, h, hi in zip(bars4, lg4_mean, lg4_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, h + hi + 2, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_ANNOT)
    for bar, h, hi in zip(bars3, lg3v_mean, lg3v_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, h + hi + 2, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_ANNOT)

    ax.plot(x, ung_asr, color=C_UNGUARDED, linestyle="--", linewidth=1.8,
            marker="o", markersize=5, zorder=5, label="Unguarded ASR-img (VLM compliance)")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Detection Recall, Image Channel (%)  ·  higher = guard blocks more", fontsize=10, color=C_TICK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(
        "Guard-Selective Blind Spots by Rhetorical Category\n"
        "Fictional Framing Collapses LG4, Theatrical Framing Collapses LG3V",
        fontsize=13, fontweight="bold", pad=14, color=C_LABEL,
    )
    ax.legend(fontsize=9, framealpha=1.0, edgecolor=C_AXIS, facecolor="white", loc="upper right")

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

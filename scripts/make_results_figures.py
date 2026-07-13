#!/usr/bin/env python3
"""
make_results_figures.py — Quantitative figures for Multimodal-Safeguard-Bench.

Usage:
    python scripts/make_results_figures.py \
        --results results/full_run \
        --out figures \
        --title-suffix "Full Run (900 items)"

Produces:
    <prefix>fig1_detection_and_asr.png  — Detection recall + ASR (two-panel combined)
    <prefix>fig4_heatmap.png            — All-metrics summary heatmap
"""
import argparse
import json
import re
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

COLORS = {
    "text": "#3B82F6",      # muted blue (was #2563EB)
    "image": "#EF4444",     # muted red (was #DC2626)
    "unguarded": "#9CA3AF", # lighter gray
    "llama_guard_4": "#8B5CF6",  # muted purple (was #7C3AED)
    "shield_gemma_2": "#F59E0B", # muted amber (was #D97706)
    "bg": "white",
    "amber": "#D97706", "border": "#E5E7EB",
}
GUARD_LABELS = {
    "llama_guard_4": "Llama-Guard-4\n(12B)",
    "llama_guard_3_vision": "LlamaGuard-3-V\n(11B)",
    "shield_gemma_2": "ShieldGemma-2\n(4B)",
}
GUARD_COLORS = {
    "llama_guard_4": "#8B5CF6",
    "llama_guard_3_vision": "#059669",
    "shield_gemma_2": "#F59E0B",
}


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", s.lower())).strip("_") + "_"


def _figpath(out, name, title_suffix):
    prefix = _slug(title_suffix) if title_suffix else ""
    return Path(out) / f"{prefix}{name}"


def load_metrics(dirs):
    merged = {"unguarded": None, "guards": {}}
    for d in sorted(dirs, key=lambda d: Path(d).name):
        p = Path(d) / "metrics.json"
        with open(p) as f:
            m = json.load(f)
        if merged["unguarded"] is None:
            merged["unguarded"] = m.get("unguarded")
        for g, s in m.items():
            if g != "unguarded" and isinstance(s, dict) and "detection_recall_text" in s:
                merged["guards"][g] = s
    return merged


def _style_ax(ax, bg="white"):
    ax.set_facecolor(bg)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#D1D5DB", zorder=0)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.tick_params(colors="#6B7280", length=3)


def fig1_detection_and_asr(m, out, title_suffix=None):
    """Combined 2-panel: detection recall (left) + ASR comparison (right)."""
    guards = list(m["guards"].keys())
    ung = m["unguarded"]
    bg = COLORS["bg"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.0), facecolor="white",
        gridspec_kw={"wspace": 0.40},
    )
    fig.patch.set_facecolor("white")

    # ── Left: Detection Recall ────────────────────────────────────────────
    det_t = [m["guards"][g]["detection_recall_text"] * 100 for g in guards]
    det_i = [m["guards"][g]["detection_recall_image"] * 100 for g in guards]
    x, w = np.arange(len(guards)), 0.32

    _style_ax(ax1)
    bt = ax1.bar(x - w/2, det_t, w, label="Text jailbreak",
                 color=COLORS["text"], zorder=3, linewidth=0, alpha=0.90)
    bi = ax1.bar(x + w/2, det_i, w, label="Image jailbreak",
                 color=COLORS["image"], zorder=3, linewidth=0, alpha=0.90)

    for bar in [*bt, *bi]:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#374151")

    for i, (t, im) in enumerate(zip(det_t, det_i)):
        gap = t - im
        if abs(gap) <= 0.5:
            continue
        if gap > 0:
            ax_x = x[i] + w/2 + 0.09
            ax1.annotate("", xy=(ax_x, im), xytext=(ax_x, t),
                         arrowprops=dict(arrowstyle="<->", color="#374151", lw=1.5))
            ax1.text(ax_x + 0.05, (t + im) / 2, f"Δ{gap:+.1f}pp",
                     va="center", fontsize=9.5, color="#374151", fontweight="bold")
        else:
            ax1.text(x[i], -0.13, "image-only classifier",
                     ha="center", va="top", fontsize=8, color=COLORS["amber"],
                     style="italic", transform=ax1.get_xaxis_transform())

    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Detection Recall (%)  ·  higher is better", fontsize=10, color="#6B7280")
    ax1.set_title("Detection Recall by Modality", fontsize=12,
                  fontweight="bold", pad=12, color="#111827")
    ax1.set_xticks(x)
    ax1.set_xticklabels([GUARD_LABELS.get(g, g) for g in guards], fontsize=10)
    ax1.legend(fontsize=10, framealpha=1.0, edgecolor="#E5E7EB", facecolor="white")

    # ── Right: ASR ───────────────────────────────────────────────────────
    all_names  = ["unguarded"] + guards
    all_colors = [COLORS["unguarded"]] + [GUARD_COLORS.get(g, "#059669") for g in guards]
    all_labels = ["Unguarded"] + [GUARD_LABELS.get(g, g).replace("\n", " ") for g in guards]
    asr_txt = [ung["asr_text"]  * 100] + [m["guards"][g]["asr_text"]  * 100 for g in guards]
    asr_img = [ung["asr_image"] * 100] + [m["guards"][g]["asr_image"] * 100 for g in guards]

    n_bars = len(all_names)
    w2 = min(0.25, 0.75 / n_bars)
    offsets = np.linspace(-(n_bars - 1) / 2 * w2, (n_bars - 1) / 2 * w2, n_bars)
    x2 = np.array([0.0, 1.0])

    _style_ax(ax2)
    for i, (name, color, label) in enumerate(zip(all_names, all_colors, all_labels)):
        vals = [asr_txt[i], asr_img[i]]
        bars = ax2.bar(x2 + offsets[i], vals, w2, label=label,
                       color=color, alpha=0.88, zorder=3, linewidth=0)
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                if h > 35:
                    ax2.text(bar.get_x() + bar.get_width() / 2, h - 2.0, f"{h:.1f}%",
                             ha="center", va="top", fontsize=8.5,
                             color="white", fontweight="bold")
                else:
                    ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%",
                             ha="center", va="bottom", fontsize=8.5,
                             fontweight="bold", color="#374151")

    ovref_lines = ["Over-refusal on XSTest benign prompts:"]
    for g in guards:
        ovr = m["guards"][g].get("over_refusal", 0) * 100
        ovref_lines.append(f"  {GUARD_LABELS.get(g, g).replace(chr(10), ' ')}: {ovr:.1f}%")
    ax2.text(0.97, 0.55, "\n".join(ovref_lines), transform=ax2.transAxes,
             ha="right", va="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor="#E5E7EB", alpha=1.0))

    ax2.set_ylim(0, 75)
    ax2.set_ylabel("Attack Success Rate (%)  ·  lower is better", fontsize=10, color="#6B7280")
    ax2.set_title("End-to-End Attack Success Rate", fontsize=12,
                  fontweight="bold", pad=12, color="#111827")
    ax2.set_xticks([0.0, 1.0])
    ax2.set_xticklabels(["Text Attack", "Image Attack"], fontsize=11)
    ax2.legend(fontsize=9, loc="upper left", framealpha=1.0,
               edgecolor="#E5E7EB", facecolor="white")

    title = "Guard Blind Spots: Detection Recall and Attack Success Rate"
    if title_suffix:
        title += f"  ·  {title_suffix}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.03, color="#111827")

    p = _figpath(out, "fig1_detection_and_asr.png", title_suffix)
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def fig4_summary_heatmap(m, out, title_suffix=None):
    guards = list(m["guards"].keys())
    col_names = ["Det-txt", "Det-img", "ASR-txt\n(guarded)", "ASR-img\n(guarded)", "OvRef"]
    keys = ["detection_recall_text", "detection_recall_image",
            "asr_text", "asr_image", "over_refusal"]
    data = np.array([[m["guards"][g].get(k, 0) * 100 for k in keys] for g in guards])

    lower_better = [False, False, True, True, True]
    display_data = data.copy()
    for j, lb in enumerate(lower_better):
        if lb:
            display_data[:, j] = 100 - display_data[:, j]

    fig, ax = plt.subplots(
        figsize=(10, 2.5 + len(guards) * 1.4), facecolor="white"
    )
    ax.set_facecolor("white")
    im = ax.imshow(display_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, fontsize=11)
    ax.set_yticks(range(len(guards)))
    ax.set_yticklabels([GUARD_LABELS.get(g, g) for g in guards], fontsize=11)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    for i in range(len(guards)):
        for j in range(len(col_names)):
            v  = data[i, j]
            dv = display_data[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if (dv < 25 or dv > 75) else "#111827")

    for i in range(len(guards) + 1):
        ax.axhline(i - 0.5, color="white", lw=2.5)
    for j in range(len(col_names) + 1):
        ax.axvline(j - 0.5, color="white", lw=2.5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Score (%)", size=9, color="#555")

    title = "Guard Performance Summary"
    if title_suffix:
        title += f"  ·  {title_suffix}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=28, color="#111827")
    ax.text(0.5, 1.02,
            "Green = good  ·  Red = poor  ·  ASR / OvRef inverted (lower = better)",
            ha="center", va="bottom", fontsize=9, color="#6B7280",
            transform=ax.transAxes)

    fig.tight_layout(pad=1.0)
    fig.text(0.5, -0.03,
             "Det-txt / Det-img: higher = better detection.   "
             "ASR-txt / ASR-img / OvRef: lower = better.",
             ha="center", fontsize=9, color="#555", style="italic")

    p = _figpath(out, "fig4_heatmap.png", title_suffix)
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate quantitative figures for Multimodal-Safeguard-Bench."
    )
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--title-suffix", default=None, metavar="SUFFIX",
                    help=(
                        "Append to all figure titles and prefix saved filenames "
                        "(e.g. 'Full Run (900 items)' → full_run_fig1_...png)."
                    ))
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    m = load_metrics(args.results)
    print("Guards:", list(m["guards"].keys()))
    for g, s in m["guards"].items():
        print(f"  {g}: Det-txt={s['detection_recall_text']:.1%} "
              f"Det-img={s['detection_recall_image']:.1%} "
              f"OvRef={s.get('over_refusal', 0):.1%}")
    ts = args.title_suffix
    fig1_detection_and_asr(m, args.out, ts)
    fig4_summary_heatmap(m, args.out, ts)
    print("\nAll figures saved to", args.out)


if __name__ == "__main__":
    main()

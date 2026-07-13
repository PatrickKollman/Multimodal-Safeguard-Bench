#!/usr/bin/env python3
"""
figures.py - Generate visualizations for Multimodal-Safeguard-Bench.
Usage: python figures.py --results results/RUN1 [results/RUN2] --out figures/
       python figures.py --results results/RUN1 results/RUN2 --out figures/ \\
           --title-suffix "Full Run (900 items)"

When --title-suffix is provided, all figure titles get the suffix appended and
saved filenames get a sanitized prefix (e.g. "full_run_fig1_...png").

Multiple result dirs: duplicate guards are resolved last-by-run_id (directory
name sort order), so a later run always overwrites an earlier one's metrics.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "text": "#2563EB", "image": "#DC2626",
    "unguarded": "#6B7280", "llama_guard_4": "#7C3AED",
    "shield_gemma_2": "#D97706", "bg": "#F9FAFB",
    "amber": "#D97706",
}
GUARD_LABELS = {
    "llama_guard_4": "Llama-Guard-4\n(12B)",
    "shield_gemma_2": "ShieldGemma-2\n(4B)",
}
GUARD_COLORS = {"llama_guard_4": "#7C3AED", "shield_gemma_2": "#D97706"}

CATEGORY_NAMES = {
    "S1": "Violence", "S2": "Chemical/Bio", "S3": "Radioactive",
    "S4": "Child Safety", "S5": "Weapons", "S6": "Hate Speech",
    "S7": "Privacy", "S8": "Specialized Advice", "S9": "Elections",
    "S10": "Cyberattacks", "S14": "Drugs", "Unknown": "Other",
}
_HARMB_CODES = set(CATEGORY_NAMES) - {"Unknown"}


def _slug(s):
    """Sanitize a title-suffix to a safe filename prefix (ends with _)."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", s.lower())).strip("_") + "_"


def _figpath(out, name, title_suffix):
    prefix = _slug(title_suffix) if title_suffix else ""
    return Path(out) / f"{prefix}{name}"


def _cat_label(cat):
    return CATEGORY_NAMES.get(cat, cat)


def load_metrics(dirs):
    """
    Merge metrics from multiple result dirs.
    Dirs are sorted by name (run_id); later runs win for duplicate guards.
    """
    merged = {"unguarded": None, "guards": {}}
    for d in sorted(dirs, key=lambda d: Path(d).name):
        p = Path(d) / "metrics.json"
        with open(p) as f:
            m = json.load(f)
        if merged["unguarded"] is None:
            merged["unguarded"] = m.get("unguarded")
        for g, s in m.items():
            if g != "unguarded":
                merged["guards"][g] = s  # last-write-wins
    return merged


def load_guard_items(d, guard):
    p = Path(d) / f"guard_{guard}_harmful.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p)]


def fig1_modality_gap(m, out, title_suffix=None):
    guards = list(m["guards"].keys())
    det_t = [m["guards"][g]["detection_recall_text"] * 100 for g in guards]
    det_i = [m["guards"][g]["detection_recall_image"] * 100 for g in guards]
    x, w = np.arange(len(guards)), 0.32
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    bt = ax.bar(x - w/2, det_t, w, label="Text jailbreak", color=COLORS["text"], zorder=3)
    bi = ax.bar(x + w/2, det_i, w, label="Image jailbreak", color=COLORS["image"], zorder=3)

    # Value labels with generous padding so they clear the bar tops
    for bar in [*bt, *bi]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8.5)

    # Delta annotations: only when gap is meaningful (> 0.5pp)
    for i, (t, im) in enumerate(zip(det_t, det_i)):
        gap = t - im
        if abs(gap) <= 0.5:
            continue
        if gap > 0:
            # Text recall > image recall: show bidirectional arrow right of image bar
            arrow_x = x[i] + w/2 + 0.07
            ax.annotate("", xy=(arrow_x, im), xytext=(arrow_x, t),
                        arrowprops=dict(arrowstyle="<->", color="#111", lw=1.5))
            ax.text(arrow_x + 0.04, (t + im) / 2, f"Δ{gap:+.1f}pp",
                    va="center", fontsize=9, color="#111", fontweight="bold")
        else:
            # Inverted (image > text): image-only classifier — warn below x-label instead
            ax.text(x[i], -0.16, "⚠ Inverted: image-only classifier",
                    ha="center", va="top", fontsize=8.5, color=COLORS["amber"],
                    fontweight="bold", transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 110)
    ax.set_ylabel("Detection Recall (%)", fontsize=11)
    title = "Guard Detection Recall: Text vs Image Jailbreaks"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=12, pad=28)
    ax.text(0.5, 1.01, "Higher = better detection. Gap = text recall − image recall.",
            ha="center", va="bottom", fontsize=8.5, color="#666",
            transform=ax.transAxes)
    ax.set_xticks(x)
    ax.set_xticklabels([GUARD_LABELS.get(g, g) for g in guards], fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10)
    fig.tight_layout()
    p = _figpath(out, "fig1_modality_gap.png", title_suffix)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def fig2_asr_comparison(m, out, title_suffix=None):
    """
    Single panel: two x-groups (Text Attack, Image Attack), bars per guard+unguarded.
    OvRef shown in an annotation box — avoids the invisible-bar problem of split subplots.
    """
    guards = list(m["guards"].keys())
    ung = m["unguarded"]

    all_names  = ["unguarded"] + guards
    all_colors = [COLORS["unguarded"]] + [GUARD_COLORS.get(g, "#059669") for g in guards]
    all_labels = ["Unguarded"] + [GUARD_LABELS.get(g, g).replace("\n", " ") for g in guards]

    asr_text  = [ung["asr_text"]  * 100] + [m["guards"][g]["asr_text"]  * 100 for g in guards]
    asr_image = [ung["asr_image"] * 100] + [m["guards"][g]["asr_image"] * 100 for g in guards]

    n_bars = len(all_names)
    w = min(0.25, 0.75 / n_bars)
    offsets = np.linspace(-(n_bars - 1) / 2 * w, (n_bars - 1) / 2 * w, n_bars)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    x = np.array([0.0, 1.0])
    for i, (name, color, label) in enumerate(zip(all_names, all_colors, all_labels)):
        vals = [asr_text[i], asr_image[i]]
        bars = ax.bar(x + offsets[i], vals, w, label=label, color=color, alpha=0.85, zorder=3)
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                if h > 35:
                    # Tall bars: label inside to avoid crowding at top
                    ax.text(bar.get_x() + bar.get_width() / 2, h - 2.0, f"{h:.1f}%",
                            ha="center", va="top", fontsize=8, color="white", fontweight="bold")
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%",
                            ha="center", va="bottom", fontsize=8.5)

    # Over-refusal annotation — right side at mid-height (bars are low here for guarded image)
    ovref_lines = ["Over-Refusal (benign blocked):"]
    for g in guards:
        ovr = m["guards"][g].get("over_refusal", 0) * 100
        ovref_lines.append(f"  {GUARD_LABELS.get(g, g).replace(chr(10), ' ')}: {ovr:.1f}%")
    ax.text(0.97, 0.52, "\n".join(ovref_lines), transform=ax.transAxes,
            ha="right", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc", alpha=0.9))

    ax.set_ylim(0, 75)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=11)
    title = "End-to-End ASR: Unguarded vs Guarded"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Text Attack", "Image Attack"], fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    p = _figpath(out, "fig2_asr_comparison.png", title_suffix)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def fig3_category_breakdown(dirs, m, out, title_suffix=None):
    gcd = {}
    for d in dirs:
        for g in m["guards"]:
            items = load_guard_items(d, g)
            if not items:
                continue
            if g not in gcd:
                gcd[g] = {}
            for item in items:
                cat = item.get("category", "Unknown") or "Unknown"
                mod = item.get("modality", "text")
                blk = int(item.get("blocked", False))
                if cat not in gcd[g]:
                    gcd[g][cat] = {"text": [], "image": []}
                gcd[g][cat][mod].append(blk)
    if not gcd:
        print("No data for fig3")
        return

    guards = list(gcd.keys())
    all_cats = sorted({c for gv in gcd.values() for c in gv})
    fig, axes = plt.subplots(1, len(guards),
                              figsize=(max(8, 2.5 * len(all_cats)), 6),
                              sharey=True, facecolor=COLORS["bg"])
    if len(guards) == 1:
        axes = [axes]

    for ax, g in zip(axes, guards):
        ax.set_facecolor(COLORS["bg"])
        cats = [c for c in all_cats if c in gcd[g]]
        uses_harmb = any(c in _HARMB_CODES for c in cats)

        x = np.arange(len(cats))
        w = 0.35
        rt = [100 * sum(gcd[g][c]["text"])  / len(gcd[g][c]["text"])
              if gcd[g][c]["text"]  else 0 for c in cats]
        ri = [100 * sum(gcd[g][c]["image"]) / len(gcd[g][c]["image"])
              if gcd[g][c]["image"] else 0 for c in cats]
        ax.bar(x - w/2, rt, w, label="Text",  color=COLORS["text"],  zorder=3)
        ax.bar(x + w/2, ri, w, label="Image", color=COLORS["image"], zorder=3)

        guard_title = GUARD_LABELS.get(g, g)
        if not uses_harmb:
            note = "Note: ShieldGemma uses image-policy categories, not HarmBench harm codes"
            ax.set_title(f"{guard_title}\n{note}", fontsize=9, pad=8)
        else:
            ax.set_title(guard_title, fontsize=10, pad=8)

        ax.set_xticks(x)
        ax.set_xticklabels([_cat_label(c) for c in cats], fontsize=7,
                            rotation=45, ha="right")
        ax.set_ylim(0, 120)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Detection Recall (%)", fontsize=11)
    suptitle = "Detection Recall by Harm Category"
    if title_suffix:
        suptitle += f" — {title_suffix}"
    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    p = _figpath(out, "fig3_category_breakdown.png", title_suffix)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def fig4_summary_heatmap(m, out, title_suffix=None):
    guards = list(m["guards"].keys())
    metrics_names = ["Det-txt", "Det-img", "ASR-txt\n(guarded)", "ASR-img\n(guarded)", "OvRef"]
    keys = ["detection_recall_text", "detection_recall_image", "asr_text", "asr_image", "over_refusal"]
    data = np.array([[m["guards"][g].get(k, 0) * 100 for k in keys] for g in guards])

    # Invert "lower is better" metrics so green always means good performance
    lower_better = [False, False, True, True, True]  # ASR-txt, ASR-img, OvRef → invert
    display_data = data.copy()
    for j, lb in enumerate(lower_better):
        if lb:
            display_data[:, j] = 100 - display_data[:, j]

    fig, ax = plt.subplots(figsize=(9, 3 + len(guards)), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    im = ax.imshow(display_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(metrics_names)))
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_yticks(range(len(guards)))
    ax.set_yticklabels([GUARD_LABELS.get(g, g) for g in guards], fontsize=11)
    for i in range(len(guards)):
        for j in range(len(metrics_names)):
            v = data[i, j]          # actual value for label
            dv = display_data[i, j] # display value for text contrast
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if (dv < 30 or dv > 70) else "#111")
    plt.colorbar(im, ax=ax, label="%", shrink=0.8)
    title = "Guard Performance Heatmap"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=12, pad=28)
    ax.text(0.5, 1.01, "Green = good performance. Red = poor. (ASR/OvRef inverted: lower = better)",
            ha="center", va="bottom", fontsize=9, color="#666",
            transform=ax.transAxes)
    fig.tight_layout()
    fig.text(0.5, -0.04,
             "* Det-txt/Det-img: higher = better. ASR-txt/ASR-img/OvRef: lower = better.",
             ha="center", fontsize=8.5, color="#555", style="italic")
    p = _figpath(out, "fig4_heatmap.png", title_suffix)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def main():
    ap = argparse.ArgumentParser()
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
        print(f"{g}: Det-txt={s['detection_recall_text']:.1%} "
              f"Det-img={s['detection_recall_image']:.1%} "
              f"OvRef={s.get('over_refusal', 0):.1%}")
    ts = args.title_suffix
    fig1_modality_gap(m, args.out, ts)
    fig2_asr_comparison(m, args.out, ts)
    fig3_category_breakdown(args.results, m, args.out, ts)
    fig4_summary_heatmap(m, args.out, ts)
    print("\nAll figures saved to", args.out)


if __name__ == "__main__":
    main()

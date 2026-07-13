#!/usr/bin/env python3
"""
make_carrier_robustness_figure.py — the paper's central constructive result.

Left panel: image-channel detection across the 18 carrier framings, sorted by LG4-alone
detection. LG4 alone swings wildly (6%->97%); SG2 is a flat carrier-invariant line
(it never reads the carrier); the LG4 OR SG2 ensemble is flat-high — it never drops
below SG2's floor even where LG4 collapses.

Right panel: coverage-usability operating points (image over-refusal vs image detection)
for the single guards and the two natural ensembles, showing LG4+SG2 in the deployable
region and LG4+LG3V stuck at refuse-all-images over-refusal.

Reads committed artifacts only (no GPU):
  results/carrier_sweep/<variant>/guard_llama_guard_4_harmful.jsonl   (valid LG4 verdicts)
  results/full_run/guard_shield_gemma_2_harmful.jsonl              (carrier-invariant SG2)
  results/full_run/guard_*_{harmful,benign}.jsonl                  (operating points)
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Per-guard palette (unified across all MSBench figures)
C_LG4 = "#6a3d9a"  # purple
C_LG3V = "#33a02c"  # green
C_SG2 = "#ff7f00"  # orange
C_ENS = "#1f78b4"  # blue

CARRIERS = [
    "baseline",
    "baseline_minimal",
    "fiction",
    "fiction_shortstory",
    "fiction_fairytale",
    "fiction_fanfic",
    "roleplay",
    "roleplay_stageplay",
    "roleplay_chardialogue",
    "roleplay_gamescript",
    "transcription",
    "transcription_extract",
    "transcription_ocr",
    "academic",
    "academic_textbook",
    "academic_research",
    "other_foundnote",
    "other_historical",
]
LABELS = {
    "baseline": "baseline",
    "baseline_minimal": "baseline-min",
    "fiction": "fiction:novel",
    "fiction_shortstory": "fiction:short",
    "fiction_fairytale": "fiction:fairytale",
    "fiction_fanfic": "fiction:fanfic",
    "roleplay": "theatre:screenplay",
    "roleplay_stageplay": "theatre:stageplay",
    "roleplay_chardialogue": "theatre:dialogue",
    "roleplay_gamescript": "theatre:gamescript",
    "transcription": "transcribe",
    "transcription_extract": "extract",
    "transcription_ocr": "ocr",
    "academic": "academic:quote",
    "academic_textbook": "academic:textbook",
    "academic_research": "academic:paper",
    "other_foundnote": "found-note",
    "other_historical": "historical",
}


def img_verdicts(path):
    d = {}
    for line in open(path):
        r = json.loads(line)
        if r["modality"] == "image":
            d[r["intent_id"]] = bool(r["blocked"])
    return d


def rate(path, modality, split_dir):
    d = {}
    for line in open(path):
        r = json.loads(line)
        if r["modality"] == modality:
            d[r["intent_id"]] = bool(r["blocked"])
    return sum(d.values()) / len(d) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carrier-dir", default="results/carrier_sweep")
    ap.add_argument("--canonical", default="results/full_run")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    cdir = Path(args.carrier_dir)
    canon = Path(args.canonical)

    # carrier-invariant SG2 (from the canonical run; SG2 never reads the carrier)
    sg2 = img_verdicts(canon / "guard_shield_gemma_2_harmful.jsonl")
    sg2_rate = sum(sg2.values()) / len(sg2)

    rows = []
    for v in CARRIERS:
        p = cdir / v / "guard_llama_guard_4_harmful.jsonl"
        if not p.exists():
            continue
        lg4 = img_verdicts(p)
        common = set(lg4) & set(sg2)
        lg4_r = sum(lg4[i] for i in common) / len(common)
        ens_r = sum(1 for i in common if lg4[i] or sg2[i]) / len(common)
        rows.append((v, lg4_r, ens_r))

    rows.sort(key=lambda x: x[1])  # sort by LG4-alone detection (ascending)
    labels = [LABELS.get(v, v) for v, _, _ in rows]
    lg4 = [r[1] * 100 for r in rows]
    ens = [r[2] * 100 for r in rows]
    x = list(range(len(rows)))

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.7, 1]}
    )

    # ---- Left: carrier robustness ----
    axL.axhline(
        sg2_rate * 100,
        color=C_SG2,
        ls="--",
        lw=2,
        label=f"SG2 alone (carrier-invariant, {sg2_rate*100:.0f}%)",
        zorder=2,
    )
    axL.fill_between(x, lg4, ens, color=C_ENS, alpha=0.12, zorder=1)
    axL.plot(x, lg4, "o-", color=C_LG4, lw=2.2, ms=6, label="LG4 alone", zorder=4)
    axL.plot(
        x,
        ens,
        "s-",
        color=C_ENS,
        lw=2.4,
        ms=6,
        label="LG4 \u2295 SG2 (ensemble)",
        zorder=5,
    )
    axL.set_xticks(x)
    axL.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    axL.set_ylabel("Image-channel detection (%)")
    axL.set_ylim(0, 104)
    axL.set_title(
        "Carrier framing collapses LG4 (6\u201397%); the LG4\u2295SG2 ensemble stays flat (\u226587%)"
    )
    axL.legend(loc="lower right", fontsize=9, framealpha=0.95)
    axL.grid(axis="y", alpha=0.3)
    # annotate the worst carrier
    worst = min(range(len(rows)), key=lambda i: lg4[i])
    axL.annotate(
        f"LG4 {lg4[worst]:.0f}%\nensemble {ens[worst]:.0f}%",
        xy=(worst, lg4[worst]),
        xytext=(worst + 0.4, lg4[worst] + 22),
        fontsize=8,
        color=C_LG4,
        arrowprops=dict(arrowstyle="->", color=C_LG4, lw=1),
    )

    # ---- Right: coverage-usability operating points (image channel) ----
    def det(guard):
        return rate(canon / f"guard_{guard}_harmful.jsonl", "image", canon)

    def ovref(guard):
        return rate(canon / f"guard_{guard}_benign.jsonl", "image", canon)

    lg4_d, lg4_o = det("llama_guard_4") * 100, ovref("llama_guard_4") * 100
    lg3_d, lg3_o = (
        det("llama_guard_3_vision") * 100,
        ovref("llama_guard_3_vision") * 100,
    )
    sg2_d, sg2_o = det("shield_gemma_2") * 100, ovref("shield_gemma_2") * 100

    # ensembles (image channel): union detection and union over-refusal
    def union_img(ga, gb, kind):
        a = img_verdicts(canon / f"guard_{ga}_{kind}.jsonl")
        b = img_verdicts(canon / f"guard_{gb}_{kind}.jsonl")
        common = set(a) & set(b)
        return sum(1 for i in common if a[i] or b[i]) / len(common) * 100

    e_sg2_d = union_img("llama_guard_4", "shield_gemma_2", "harmful")
    e_sg2_o = union_img("llama_guard_4", "shield_gemma_2", "benign")
    e_lg3_d = union_img("llama_guard_4", "llama_guard_3_vision", "harmful")
    e_lg3_o = union_img("llama_guard_4", "llama_guard_3_vision", "benign")

    # (name, over-refusal x, detection y, color, marker, dx, dy, ha, va, leader)
    # LG3V and LG4⊕LG3V both sit at (100, 100), so their labels are placed to the
    # left and stacked (one up, one down) with short leader lines to avoid overlap.
    pts = [
        # single guards (circles)
        ("LG4", lg4_o, lg4_d, C_LG4, "o", 9, -12, "left", "top", False),
        ("SG2", sg2_o, sg2_d, C_SG2, "o", -9, 9, "right", "bottom", False),
        ("LG3V", lg3_o, lg3_d, C_LG3V, "o", -18, 14, "right", "bottom", True),
        # non-viable ensemble (X marker)
        ("LG4\u2295LG3V", e_lg3_o, e_lg3_d, "#e31a1c", "X", -18, -18, "right", "top", True),
        # deployable ensemble (star) — drawn last so it sits on top
        ("LG4\u2295SG2", e_sg2_o, e_sg2_d, C_ENS, "*", 11, 2, "left", "center", False),
    ]
    msize = {"o": 130, "X": 175, "*": 320}
    for name, ox, oy, c, mk, dx, dy, ha, va, leader in pts:
        axR.scatter(
            ox,
            oy,
            s=msize[mk],
            c=c,
            marker=mk,
            edgecolors="k",
            linewidths=0.7,
            zorder=4 if mk == "*" else 3,
        )
        axR.annotate(
            name,
            (ox, oy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            ha=ha,
            va=va,
            color=c,
            fontweight="bold" if mk == "*" else "normal",
            arrowprops=(
                dict(arrowstyle="-", color=c, lw=0.8, alpha=0.6) if leader else None
            ),
        )

    # marker-shape legend (shape encodes the category, color encodes the guard)
    from matplotlib.lines import Line2D

    shape_legend = [
        Line2D([], [], marker="o", color="w", markerfacecolor="0.4",
               markeredgecolor="k", markersize=9, label="single guard"),
        Line2D([], [], marker="*", color="w", markerfacecolor=C_ENS,
               markeredgecolor="k", markersize=16, label="deployable ensemble"),
        Line2D([], [], marker="X", color="w", markerfacecolor="#e31a1c",
               markeredgecolor="k", markersize=10, label="non-viable ensemble"),
    ]
    axR.legend(handles=shape_legend, loc="lower right", fontsize=8.5,
               framealpha=0.95, title="marker", title_fontsize=8.5)
    axR.axvspan(0, 20, color="green", alpha=0.05)
    axR.set_xlabel("Image over-refusal (%)  \u2192 worse")
    axR.set_ylabel("Image detection (%)  \u2192 better")
    axR.set_title(
        "Coverage\u2013usability (image channel):\nLG4\u2295SG2 is the deployable pairing"
    )
    axR.set_xlim(-4, 112)
    axR.set_ylim(30, 106)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(args.out) / "fig_carrier_robustness.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"  SG2 carrier-invariant floor: {sg2_rate:.1%}")
    print(
        f"  worst LG4 carrier: {min(lg4):.1f}%  ->  ensemble: {ens[lg4.index(min(lg4))]:.1f}%"
    )
    print(f"  LG4\u2295SG2 op-point: det={e_sg2_d:.1f}% ovref={e_sg2_o:.1f}%")
    print(f"  LG4\u2295LG3V op-point: det={e_lg3_d:.1f}% ovref={e_lg3_o:.1f}%")


if __name__ == "__main__":
    main()

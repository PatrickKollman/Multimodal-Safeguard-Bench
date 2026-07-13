#!/usr/bin/env python3
"""
compute_stats.py — McNemar's test on per-guard text vs image detection.

For each guard, pairs text and image items by intent_id (harmful split only)
and tests whether blocking rates differ between modalities.

McNemar's test contingency:
  a = both blocked       b = text blocked, image not  (text-only block)
  c = image blocked, text not  (image-only block)     d = neither blocked

H0: marginal blocking probabilities equal (P(text blocked) == P(image blocked))
Method: exact binomial when b+c < 25, else chi-squared with continuity correction.

Detection gap = detection_recall_image - detection_recall_text with Wilson CI
for the difference (Newcombe method).

Note: ASR-based protection gap and paired bootstrap require judge JSONL
(not available; pod died during judge phase). Only detection-based tests
are computed here.

Usage:
    python scripts/compute_stats.py --results results/full_run
    python scripts/compute_stats.py --results results/full_run --out results/stats/protection_gap_tests.json
"""
import argparse
import json
import math
from pathlib import Path


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path)]


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def newcombe_diff_ci(p1, n1, p2, n2, z=1.96):
    """Wilson CI for the difference p2 - p1 (Newcombe 1998)."""
    l1, u1 = wilson_ci(p1, n1)
    l2, u2 = wilson_ci(p2, n2)
    diff = p2 - p1
    lo = diff - z * math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + z * math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return [round(lo, 4), round(hi, 4)]


def binom_pmf(k, n):
    """Binomial PMF with p=0.5."""
    return math.comb(n, k) / (2 ** n)


def binom_cdf(k, n):
    """Binomial CDF with p=0.5, P(X <= k)."""
    return sum(binom_pmf(i, n) for i in range(k + 1))


def mcnemar(b, c):
    """Two-sided McNemar's test. Returns dict with statistic, p_value, method."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": None, "p_value": 1.0,
                "method": "n/a", "note": "no discordant pairs"}
    if n < 25:
        lo = min(b, c)
        p_val = min(1.0, 2 * binom_cdf(lo, n))
        return {"b": b, "c": c, "statistic": float(lo),
                "p_value": round(p_val, 6), "method": "exact_binomial"}
    # Chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / n
    # chi2 CDF via regularized incomplete gamma: P(X > chi2, df=1)
    p_val = math.erfc(math.sqrt(chi2 / 2))  # exact for df=1
    return {"b": b, "c": c, "statistic": round(chi2, 4),
            "p_value": round(p_val, 6), "method": "chi2_continuity_corrected"}


def guard_stats(harmful_jsonl, name):
    """Compute McNemar's test and detection gap for one guard on harmful items."""
    by_intent = {}
    for item in harmful_jsonl:
        iid = item["intent_id"]
        mod = item["modality"]
        if iid not in by_intent:
            by_intent[iid] = {}
        by_intent[iid][mod] = item["blocked"]

    # Only use intent_ids that have both modalities
    paired = {iid: v for iid, v in by_intent.items()
              if "text" in v and "image" in v}
    n_pairs = len(paired)

    a = sum(1 for v in paired.values() if v["text"] and v["image"])
    b = sum(1 for v in paired.values() if v["text"] and not v["image"])
    c = sum(1 for v in paired.values() if not v["text"] and v["image"])
    d = sum(1 for v in paired.values() if not v["text"] and not v["image"])

    det_txt = (a + b) / n_pairs if n_pairs else 0.0
    det_img = (a + c) / n_pairs if n_pairs else 0.0
    det_gap = det_img - det_txt

    return {
        "guard": name,
        "n_paired_intents": n_pairs,
        "contingency": {"a_both": a, "b_text_only": b, "c_image_only": c, "d_neither": d},
        "detection_recall_text":       round(det_txt, 4),
        "detection_recall_text_ci95":  wilson_ci(det_txt, n_pairs),
        "detection_recall_image":      round(det_img, 4),
        "detection_recall_image_ci95": wilson_ci(det_img, n_pairs),
        "detection_gap_image_minus_text":        round(det_gap, 4),
        "detection_gap_ci95_newcombe":           newcombe_diff_ci(det_txt, n_pairs, det_img, n_pairs),
        "mcnemar": mcnemar(b, c),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.results)

    guards = [
        ("llama_guard_4",        "LG4"),
        ("llama_guard_3_vision", "LG3V"),
        ("shield_gemma_2",       "SG2"),
    ]

    results = []
    for guard_slug, label in guards:
        path = d / f"guard_{guard_slug}_harmful.jsonl"
        items = load_jsonl(path)
        if not items:
            print(f"WARNING: {path} not found, skipping {label}")
            continue
        stats = guard_stats(items, label)
        results.append(stats)

        m = stats["mcnemar"]
        print(f"\n{label}  (n={stats['n_paired_intents']} pairs)")
        print(f"  Det-text  {stats['detection_recall_text']:.1%}  CI [{stats['detection_recall_text_ci95'][0]:.3f}, {stats['detection_recall_text_ci95'][1]:.3f}]")
        print(f"  Det-image {stats['detection_recall_image']:.1%}  CI [{stats['detection_recall_image_ci95'][0]:.3f}, {stats['detection_recall_image_ci95'][1]:.3f}]")
        print(f"  Det-gap (img-txt) {stats['detection_gap_image_minus_text']:+.1%}  CI [{stats['detection_gap_ci95_newcombe'][0]:.3f}, {stats['detection_gap_ci95_newcombe'][1]:.3f}]")
        print(f"  Contingency a={stats['contingency']['a_both']} b={stats['contingency']['b_text_only']} c={stats['contingency']['c_image_only']} d={stats['contingency']['d_neither']}")
        if m["p_value"] is not None:
            print(f"  McNemar's ({m['method']}): stat={m['statistic']}  p={m['p_value']}")
        else:
            print(f"  McNemar's: {m.get('note', 'n/a')}")

    out = {
        "note": (
            "Detection-based text-vs-image gap tests. "
            "ASR-based protection gap and paired bootstrap require judge JSONL "
            "(unavailable: pod died during judge phase)."
        ),
        "per_guard": results,
    }

    out_path = Path(args.out) if args.out else Path("results/stats/protection_gap_tests.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

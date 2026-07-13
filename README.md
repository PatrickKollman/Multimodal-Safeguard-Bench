# Multimodal Safeguard Bench

**Do AI safety guards actually stop image-based jailbreaks — and can we compose them to?**

Safety guard models reliably block harmful requests written as *text*. This benchmark
measures, end-to-end and reproducibly, whether they also stop the **same request
rendered as an image**, and — the actionable part — shows that guard blind spots are
**architecture-specific and complementary**, so a cheap cross-modal ensemble covers what
no single guard can.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Status: results available](https://img.shields.io/badge/status-results%20available-green.svg)

Full writeup: [`writeup/paper.md`](writeup/paper.md) · Development log: [`CHANGELOG.md`](CHANGELOG.md)

---

## Key Finding

Guard blind spots are **architecture-specific and complementary** — so robust two-channel
coverage can be *composed* from guards that reason over different modalities.

| Guard | Det-txt [95% CI] | Det-img [95% CI] | ASR-txt | ASR-img | OvRef [95% CI] | ProtGap |
|---|---|---|---|---|---|---|
| Unguarded | — | — | 57.0% | 60.5% | — | — |
| Llama-Guard-4 (12B) | **92.5%** [88.0, 95.4] | 82.0% [76.1, 86.7] | 5.5% | 11.5% | **11.8%** [9.3, 14.9] | +2.5pp |
| LlamaGuard-3-Vision (11B) | 89.0% [83.9, 92.6] | **100.0%** [98.1, 100.0] | 7.0% | **0.0%** | 55.0% [50.6, 59.3] | −10.5pp |
| ShieldGemma-2 (4B) | 0.0% [0.0, 1.9] | 87.0% [81.6, 91.0] | 57.0% | 10.5% | **4.6%** [3.1, 6.7] | −50.0pp |

*900 items (200 HarmBench × 2 modalities + 250 XSTest × 2 modalities). Canonical run:
[`results/full_run/`](results/full_run/metrics.json); paired tests in
[`protection_gap_tests.json`](results/full_run/protection_gap_tests.json).*

**Three architectures, three profiles:**
- **LG4** — balanced multimodal intent classifier; the only guard that discriminates on both
  channels. Its aggregate ProtGap is not significant, but the rigorous paired test is: it blocks
  10.5pp fewer image than text items on the same intents (**McNemar p<0.001**). Best usability
  (11.8% over-refusal), carrier-vulnerable.
- **LG3V** — vision-specialized classifier. Its 100% image detection is a **refuse-all-images
  policy** (it blocks 100% of *benign* images too; 55% over-refusal). Untenable alone.
- **SG2** — image-content classifier, **blind to text by construction** (0% text detection) but,
  correctly loaded, the **best-calibrated image guard** here: **87% image detection at 9.2% image
  over-refusal**. Reproducible (identical across 3 runs). *(An earlier draft reported SG2 as weak
  and non-reproducible; that was a silent `lm_head` load bug — see [CHANGELOG](CHANGELOG.md) and
  the paper's Appendix B.)*

**The central, constructive result:** because the blind spots are complementary, a **cross-modal
ensemble** (a text-intent guard ⊕ an image-content guard) covers both channels cheaply — and is
robust to the cheapest attack. See "Cross-modal ensembles" below.

### The carrier prompt is a *text-context* attack — and an image-content guard is immune to it

The one-sentence text framing paired with the rendered image is a zero-cost, black-box attack.
An 18-framing sweep (image identical, only the carrier changes) collapses the **text-reading**
guards with orthogonal, mechanistically distinct blind spots:

| Category (n) | LG4 Det-img | LG3V Det-img | SG2 Det-img |
|---|---|---|---|
| Baseline (2) | 73% [64–82] | 50% [0–100] | 87% (invariant) |
| **Fictional (4)** | **20%** [6–54] | 35% [0–100] | 87% (invariant) |
| **Theatrical (4)** | 30% [6–54] | **2%** [0–8] | 87% (invariant) |
| Transcription (3) | 90% [84–97] | 68% [4–100] | 87% (invariant) |
| Academic (3) | 40% [30–52] | 27% [0–82] | 87% (invariant) |
| Other (2) | 43% [32–55] | 0% [0–0] | 87% (invariant) |

- **LG4's blind spot is category-driven:** the worst carrier ("novel passage" fiction) collapses it
  **82% → 6%** (image ASR 11.5% → 78%).
- **LG3V's blind spot is phrasing-driven:** a near-binary switch (12/18 framings drive it to ~0%);
  theatrical framing collapses it robustly (≤8%).
- **SG2 is carrier-invariant by construction** — it never receives the carrier text, so its verdict
  is identical across all 18 framings.

**Consequence:** an LG4⊕SG2 image-channel ensemble cannot drop below SG2's 87% floor under any
carrier. Measured across all 18 framings, **LG4⊕SG2 image detection stays 87–100% while LG4 alone
swings 6–97%.**

![Carrier robustness](figures/fig_carrier_robustness.png)

*Left: LG4 alone collapses to 6% under fiction framing; SG2 is flat (carrier-invariant); LG4⊕SG2 stays ≥87% everywhere. Right: coverage–usability — LG4⊕SG2 is the deployable pairing; LG4⊕LG3V is pinned at refuse-all-images over-refusal.*

---

## Cross-modal ensembles: the cheap fix

| Configuration | Det-txt | Det-img | ASR-txt | ASR-img | OvRef |
|---|---|---|---|---|---|
| LG4 alone | 92.5% | 82.0% | 5.5% | 11.5% | 11.8% |
| **LG4⊕SG2** (text→LG4, image→LG4∪SG2) | 92.5% | **97.0%** | 5.5% | **2.5%** | **14.8%** |
| LG4⊕LG3V (block on either) | 96.0% | 100.0% | 3.0% | 0.0% | 56.4% |

**LG4⊕SG2** raises image detection 82%→97% and cuts image ASR 11.5%→2.5% for **+3pp over-refusal**,
*and* is carrier-robust (above). **LG4⊕LG3V** (two text-context guards) reaches 0% image ASR but
inherits LG3V's refuse-all-images behavior (56.4% over-refusal) and a shared carrier weakness.
Compose **across** reasoning modalities, not within them. See
[`results/full_run/ensemble.json`](results/full_run/ensemble.json).

---

## Evaluation Pipeline

![Evaluation pipeline](figures/fig_pipeline.png)

*Each harmful intent produces a text-modality item (raw prompt) and an image-modality item (intent
rendered as a 512×512 PIL image). Both pass through the guard gate before reaching
LLaVA-1.6-Mistral-7B; WildGuard judges whether the VLM complied. Models are staged sequentially on
a single 24 GB GPU.*

---

## Experiments

### 1. Unguarded baseline
LLaVA-1.6-Mistral-7B with no guard: **57.0% ASR on text, 60.5% on image** across 200 HarmBench
behaviors — a real attack surface in both channels. → [`results/full_run/metrics.json`](results/full_run/metrics.json)

### 2. LG4 — balanced, with a residual image gap
Text detection 92.5%, image 82.0%. The 10.5pp gap is measured on the same intents (paired):
26 items blocked as text but not image vs 5 the other way (**McNemar p<0.001**). Lowest
over-refusal (11.8%).

### 3. LG3V — 100% image detection via refuse-all-images
100% image detection / 0% image ASR, but it blocks **100% of benign images** (55% aggregate
over-refusal). Not discrimination — wholesale refusal.

### 4. SG2 — strong, calibrated, text-blind image guard
0% text detection (no text pathway), **87% image detection at 9.2% image over-refusal** —
reproducible across runs. Requires re-tying the Gemma-3 `lm_head` after load (Appendix B).

### 5. Adaptive rendering — LG4 invariant to surface changes
Four hand-designed variants (inverted, small font, 15° rotation) leave LG4 detection statistically
invariant (78.5–84.5%, all CIs overlap baseline). → [`results/adaptive_run/metrics.json`](results/adaptive_run/metrics.json)

### 6. Rendering sweep — no viable random attack
A 20-trial Bayesian sweep finds Gaussian noise (σ ≥ ~14.9) is the only driver of LG4 evasion, and
every config that evades LG4 also destroys the target VLM's readability (best ASR-ug 12.5% ≪ 40%
gate). Random perturbation cannot selectively evade a guard.

### 7. UAP — a robustness spectrum
| Guard | Natural bypass | UAP ε=16 | UAP ε=32 | Feature-space ε=16 | Transfer in |
|---|---|---|---|---|---|
| LG3V | 0% | **100%** | — | — | — |
| SG2 | 10% | 10–34% (resists) | **52%** (breaks) | — | — |
| LG4 | 6% | 16% | 22% | 10% | 2% (from LG3V) |

Susceptibility tracks decision confidence + gradient architecture, not a dense/sparse binary.
LG4's resistance has two independent sources (gradient sparsity + text-context dominance); the
LG3V→LG4 transfer fails. → [`results/uap_sg2/`](results/uap_sg2/results.json),
[`results/uap_lg3v/`](results/uap_lg3v/results.json), [`results/uap_lg4/`](results/uap_lg4/results.json)

### 8. Carrier sweep — text-context attack + image-content immunity
See "Key Finding" above and [`results/carrier_sweep/summary.json`](results/carrier_sweep/summary.json).

### 9. Cross-VLM replication (Qwen2-VL-7B)
Guard detection is identical (it is target-VLM-independent by construction). The one number that
moves is unguarded text ASR (57%→9%) — Qwen2-VL is better safety-aligned. Guard gaps matter most
where the underlying VLM is weakly aligned. → [`results/full_run_qwen2vl/metrics.json`](results/full_run_qwen2vl/metrics.json)

---

## Implications

No single guard covers both channels at acceptable cost — but the **right cross-modal pair does**:

| Requirement | LG4 alone | LG3V alone | SG2 alone | **LG4⊕SG2** |
|---|---|---|---|---|
| Block text-channel intent | ✓ 92.5% | ~ 89.0% | ✗ 0.0% | ✓ 92.5% |
| Block image-channel intent | ~ 82.0% | ✓ 100% (refuse-all) | ~ 87.0% | ✓ **97.0%** |
| Over-refusal (XSTest) | **11.8%** | ✗ 55.0% | 4.6% | 14.8% |
| Carrier-robust? | ✗ (82%→6%) | ✗ (100%→0%) | ✓ (invariant) | ✓ (**≥87%**) |

**The claim this supports:** a guard's blind spots are fixed by its architecture rather than the
channel it is attacked through, and blind spots across architectures are *complementary*. So robust
multimodal safety is not a matter of finding a better single guard, nor of ensembling arbitrarily,
but of composing guards whose reasoning modalities differ — so the blind spots of one are the
strengths of another. MSBench is the instrument for measuring those blind spots and verifying a
composition covers them.

---

## Quick Start

```bash
git clone https://github.com/PatrickKollman/Multimodal-Safeguard-Bench.git
cd Multimodal-Safeguard-Bench
pip install uv && uv sync    # or: pip install -e .
```

**Regenerate figures** (all read committed JSON — no GPU):
```bash
python scripts/make_explainer_figures.py --results results/full_run --out figures      # pipeline, examples, guard-contrast
python scripts/make_carrier_mechanism_figure.py --out figures                          # Fig 2 (hero)
python scripts/make_carrier_robustness_figure.py --out figures                         # Fig 6 (central result)
python scripts/make_sweep_figures.py                                                    # rendering sweep
```

---

## Full Reproduction

### Prerequisites
1. **HuggingFace token** with access to three gated models (request all before running):
   [`meta-llama/Llama-Guard-4-12B`](https://huggingface.co/meta-llama/Llama-Guard-4-12B),
   [`meta-llama/Llama-Guard-3-11B-Vision`](https://huggingface.co/meta-llama/Llama-Guard-3-11B-Vision)
   (**separate HF gate**), [`google/shieldgemma-2-4b-it`](https://huggingface.co/google/shieldgemma-2-4b-it).
2. **GPU:** 24 GB VRAM (RTX 4090 / A100). Models stage sequentially.

### Run
```bash
python -m msbench.run --config configs/mvp.yaml --smoke --limit 5      # ~5 min sanity
python -m msbench.run --config configs/mvp.yaml --name full_run --purge-guard-cache
```
`--purge-guard-cache` deletes each guard's weights after it classifies — **required unless your
volume can hold all five models (~80 GB) at once.**

### RunPod notes (learned the hard way)
- **Point `HF_HOME` at your persistent volume and where your token actually lives.** On many pods
  `HOME=/root` (ephemeral, ~30 GB overlay) but weights + token live under `/workspace`. Set it
  explicitly and persist it:
  ```bash
  export HF_HOME=/workspace/.cache/huggingface
  ln -sfn /workspace/.cache/huggingface /root/.cache/huggingface   # survives shell resets
  ```
- **`df -h` shows the shared cluster, not your quota.** Network volumes have a per-volume quota
  (`OSError: [Errno 122] Disk quota exceeded`) that `df` does not display. If you hit it, free
  model weights or use `--purge-guard-cache`.

### UAP attacks
```bash
python scripts/attack_uap_sg2.py  --config configs/mvp.yaml --eps 16 --out results/uap_sg2
python scripts/attack_uap_gen.py  --guard lg4  --config configs/mvp.yaml --eps 16
python scripts/attack_uap_gen.py  --guard lg3v --config configs/mvp.yaml --eps 16
python scripts/attack_uap_vit.py  --config configs/mvp.yaml --eps 16
python scripts/eval_transfer.py   --delta results/uap_lg3v/delta_eps16.pt --config configs/mvp.yaml
```

---

## Project Structure

```
configs/            # pinned model revisions, dataset + render config, carrier variants
figures/            # all paper figures (committed; regenerate from JSON, no GPU)
results/
  full_run/           # CANONICAL run: metrics, stats, ensemble, guard verdicts
  full_run_qwen2vl/   # cross-VLM replication
  adaptive_run/       # adaptive rendering (LG4)
  rendering_sweep_probe{,_validated}/
  uap_{sg2,lg3v,lg4,vit_lg4,transfer_lg3v_lg4}/
  carrier_sweep/      # 18 carriers × guards (per-variant + summary.json)
  # raw model generations (gen_*.jsonl) are gitignored — may contain harmful text
scripts/            # figure generators (no GPU) + attack/sweep drivers (GPU)
src/msbench/        # harness: run, guards, target, judge, data, eval, adaptive
tests/              # unit tests
writeup/paper.md    # full writeup (arXiv-ready)
```

---

## Limitations

- **WildGuard as judge** introduces measurement noise; all ASRs reflect its classification, not
  human judgment.
- **Two target VLMs for detection, one for carrier/UAP.** Guard detection is target-VLM-independent
  (Section 10); the carrier effect operates on the guard's forward pass and is expected to be
  VLM-independent, but this is not directly verified for VLMs beyond LLaVA.
- **Simplest image attack vector** (plain black-on-white text); more sophisticated rendering could
  shift detection either way.
- **UAP is white-box**, and PGD restarts are unseeded (SG2 ε=16 fooling is run-to-run variable,
  10–34%); black-box transfer is not evaluated.
- **Three guards span the architecture space but do not exhaust it** (InternVL-based classifiers,
  commercial moderation APIs, ensemble-trained guards are not evaluated).

---

## Responsible Use

This benchmark uses open-weight models and public datasets (HarmBench, XSTest). Attacks are
measurement instruments to quantify and reduce risk. Raw model outputs are excluded from the
repository by `.gitignore` as they may contain harmful text.

---

## Citation

```bibtex
@misc{kollman2026msbench,
  title  = {Multimodal Safeguard Bench: Guard Blind Spots Are Architecture-Specific and Complementary},
  author = {Kollman, Patrick},
  year   = {2026},
  url    = {https://github.com/PatrickKollman/Multimodal-Safeguard-Bench}
}
```

## License

[MIT](LICENSE). Benchmarked models and datasets retain their own licenses:
- HarmBench behaviors: [MIT](https://github.com/centerforaisafety/HarmBench)
- XSTest: [CC BY 4.0](https://huggingface.co/datasets/natolambert/xstest-v2-copy)
- Llama Guard 4: [Meta Llama 4 Community License](https://huggingface.co/meta-llama/Llama-Guard-4-12B)
- Llama Guard 3 Vision: [Meta Llama 3 Community License](https://huggingface.co/meta-llama/Llama-Guard-3-11B-Vision)
- ShieldGemma 2: [Gemma Terms of Use](https://huggingface.co/google/shieldgemma-2-4b-it)

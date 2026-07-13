# Multimodal Safeguard Bench

**Do AI safety guards actually stop image-based jailbreaks?**

Safety guard models reliably block harmful requests written as *text*. This benchmark
measures, end-to-end and reproducibly, whether they also stop the **same request
rendered as an image** — routing a vision-language model through each guard and
comparing real attack success rates across modalities.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Status: results available](https://img.shields.io/badge/status-results%20available-green.svg)

Full writeup: [`writeup/paper.md`](writeup/paper.md)

---

## Key Finding

Two production guards have **complementary blind spots**:

| Guard | Det-txt | Det-img | ASR-txt | ASR-img | OvRef |
|---|---|---|---|---|---|
| Llama-Guard-4 (12B) | **92.5%** | 82.0% | **5.5%** | 11.5% | 11.8% |
| ShieldGemma-2 (4B) | 0.0% | **100.0%** | 57.0% | **0.0%** | 49.8% |

- **LG4** catches intent in both modalities but misses ~18% of image items (10.5pp detection gap).
- **SG2** detects rendered text as a policy-violating image (100% recall) but completely ignores pure-text intent (0% recall), blocking ~50% of benign prompts.
- Neither guard alone covers both attack surfaces. An adversary can route around LG4 with image rendering; SG2 is trivially bypassed by any text-only prompt.

*Measured over 900 items: 200 HarmBench behaviors × 2 modalities (harmful) + 250 XSTest prompts × 2 modalities (benign).*

---

## Figures

| Figure | Description |
|---|---|
| [`fig_pipeline.png`](figures/fig_pipeline.png) | End-to-end evaluation pipeline with results table |
| [`fig_guard_contrast.png`](figures/fig_guard_contrast.png) | Side-by-side guard architecture and blind-spot comparison |
| [`fig_examples.png`](figures/fig_examples.png) | Real attack examples: text vs image jailbreak, per-guard decisions |
| [`fig1_modality_gap.png`](figures/full_run_900_items_fig1_modality_gap.png) | Detection recall: text vs image per guard |
| [`fig2_asr_comparison.png`](figures/full_run_900_items_fig2_asr_comparison.png) | End-to-end ASR: unguarded vs guarded, text vs image |
| [`fig4_heatmap.png`](figures/full_run_900_items_fig4_heatmap.png) | Guard performance heatmap across all metrics |
| [`attn_gallery.png`](figures/attn_gallery.png) | LG4 attention maps: blocked vs passed image jailbreaks (200 items) |

---

## Quick Start

```bash
git clone https://github.com/PatrickKollman/Multimodal-Safeguard-Bench.git
cd Multimodal-Safeguard-Bench

# Install with uv (recommended — matches pinned lockfile)
pip install uv
uv sync
```

Or with plain pip:
```bash
pip install -e .
```

---

## Reproducing the Full Run

### Prerequisites

1. **HuggingFace token** with access to two gated models. Request access before running:
   - [`meta-llama/Llama-Guard-4-12B`](https://huggingface.co/meta-llama/Llama-Guard-4-12B)
   - [`google/shieldgemma-2-4b-it`](https://huggingface.co/google/shieldgemma-2-4b-it)

2. **GPU**: 24 GB VRAM minimum (RTX 4090 or A100). Models are loaded sequentially with 4-bit quantization where possible.

3. **Login**:
   ```bash
   huggingface-cli login
   ```

### Run

```bash
# Quick sanity check (~5 min, 5 items)
python -m msbench.run --config configs/mvp.yaml --smoke --limit 5

# Smoke run — 80 items (~30 min)
python -m msbench.run --config configs/mvp.yaml --smoke

# Full benchmark — 900 items (~4 hrs)
python -m msbench.run --config configs/mvp.yaml --guards llama_guard_4 shield_gemma_2
```

Results are written to `results/<run_id>/metrics.json` and per-item JSONL files.

### Regenerate Figures

```bash
# Results figures (bar charts, heatmap)
python make_results_figures.py --results results/full_run --out figures --title-suffix "Full Run (900 items)"

# Explainer figures (pipeline, guard contrast, attack examples)
python make_explainer_figures.py --results results/full_run --out figures

# Attention maps (requires GPU — re-runs LG4 inference)
python gradcam.py --batch-all --results results/full_run
```

---

## Project Structure

```
.
├── configs/
│   └── mvp.yaml              # Pinned model revisions, dataset config, run params
├── figures/                  # All paper figures (committed)
│   ├── attn_maps/            # Per-item attention overlays + .npy arrays (200 items)
│   └── *.png                 # Summary figures
├── results/
│   └── full_run/
│       ├── metrics.json      # Aggregate results (committed)
│       └── config.yaml       # Run config snapshot (committed)
│       # Per-item JSONL outputs excluded by .gitignore (may contain model responses)
├── scripts/
│   └── setup_runpod.sh       # One-command RunPod bootstrap
├── src/msbench/              # Benchmark harness (Python package)
│   ├── run.py                # CLI entry point
│   ├── pipeline.py           # End-to-end pipeline orchestration
│   ├── guards.py             # Guard model wrappers (LG4, SG2)
│   ├── target.py             # Target VLM wrapper (LLaVA-1.6)
│   ├── judge.py              # WildGuard judge wrapper
│   ├── data.py               # Dataset loading + image rendering
│   ├── eval.py               # Metric computation
│   └── adaptive.py           # Adaptive evasion variants
├── tests/                    # Unit tests
├── writeup/
│   └── paper.md              # Full writeup (arXiv-ready)
├── gradcam.py                # Attention visualization (LG4 vision encoder)
├── make_results_figures.py   # Quantitative figure generation
├── make_explainer_figures.py # Explainer / pipeline figure generation
└── pyproject.toml            # Dependencies (managed with uv)
```

---

## Reproducing on RunPod

### Pod Spec
- **GPU**: RTX 4090 (24 GB), Community Cloud
- **Template**: Any PyTorch 2.x / CUDA 12.x image
- **Storage**: Persistent network volume at `/workspace`, minimum **100 GB**
  (LG4 ~23 GB + LLaVA ~15 GB + WildGuard ~10 GB = ~48 GB weights)

### First-Time Setup

```bash
# 1. Redirect HF cache to persistent volume
export HF_HOME=/workspace/hf_cache
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

# 2. Run bootstrap (clones repo, installs deps, verifies HF access)
bash <(curl -fsSL https://raw.githubusercontent.com/PatrickKollman/Multimodal-Safeguard-Bench/main/scripts/setup_runpod.sh)
```

### On Each Pod Restart

```bash
export HF_HOME=/workspace/hf_cache
pip install -e /workspace/Multimodal-Safeguard-Bench
cd /workspace/Multimodal-Safeguard-Bench
```

Weights are already cached — no re-download needed.

---

## Responsible Use

This benchmark uses open-weight models and public datasets (HarmBench, XSTest).
Attacks are measurement instruments to quantify and reduce risk, not to increase it.
Raw model outputs (generations, judge scores) are excluded from the repository by
`.gitignore` as they may contain harmful text produced by the target VLM under
adversarial conditions.

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{kollman2026msbench,
  title  = {Multimodal Safeguard Bench: Measuring Guard Blind Spots Across Modalities},
  author = {Kollman, Patrick},
  year   = {2026},
  url    = {https://github.com/PatrickKollman/Multimodal-Safeguard-Bench}
}
```

---

## License

[MIT](LICENSE). Benchmarked models and datasets retain their own licenses:
- HarmBench behaviors: [MIT](https://github.com/centerforaisafety/HarmBench)
- XSTest: [CC BY 4.0](https://huggingface.co/datasets/natolambert/xstest-v2-copy)
- Llama Guard 4: [Meta Llama 3 Community License](https://huggingface.co/meta-llama/Llama-Guard-4-12B)
- ShieldGemma 2: [Gemma Terms of Use](https://huggingface.co/google/shieldgemma-2-4b-it)

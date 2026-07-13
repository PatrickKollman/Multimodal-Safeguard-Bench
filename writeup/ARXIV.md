# arXiv submission notes

This directory holds the writeup in two forms:

- **`paper.tex` + `references.bib`** — the **canonical arXiv source** (compile on Overleaf or via arXiv's own build).
- **`paper.md`** — a GitHub-readable mirror of the same content (linked from the README). Keep it in sync with `paper.tex` if you make prose edits, or drop it once the arXiv version is final.

No LaTeX toolchain is installed on the dev box, so the `.tex` was authored by hand (arXiv compiles LaTeX server-side; you can also use Overleaf).

---

## Build

**Overleaf (easiest):** upload `paper.tex`, `references.bib`, and the referenced figures. `paper.tex` uses `\graphicspath{{figures/}{../figures/}}`, so either put the PNGs in a `figures/` subfolder next to `paper.tex`, or upload the repo `figures/` one level up.

**Local (if you install TeX Live):**
```bash
cd writeup && mkdir -p figures && cp ../figures/*.png figures/
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

**arXiv tarball:** submit a `.tar.gz` containing `paper.tex`, `references.bib`, `paper.bbl` (see note), and a `figures/` dir with the 6 referenced PNGs:
`fig_pipeline.png`, `fig_carrier_mechanism.png`, `fig_sweep_tradeoff.png`, `fig_carrier_robustness.png`, `fig_guard_contrast.png` (Fig 3 `fig_examples.png` is optional/not currently in `paper.tex`).
> **`.bbl` note:** arXiv runs BibTeX, but the most reliable path is to compile once (Overleaf/local), grab the generated `paper.bbl`, and include it in the tarball so arXiv doesn't need to resolve `references.bib`.

---

## arXiv metadata

**Recommended categories:** primary **cs.CR** (Cryptography and Security); cross-list **cs.LG**, **cs.CV**, **cs.CL**.

**License:** the code is MIT; for the paper, **CC BY 4.0** is a reasonable choice on the arXiv license selector (or arXiv's non-exclusive default).

**Listing abstract** (arXiv's abstract field is capped at ~1920 characters — the in-paper abstract is longer, so use this trimmed version for the web form):

> Safety guards intercept harmful requests before a vision-language model (VLM) responds, but a harmful instruction can be rendered as an image rather than typed - a black-box attack requiring no model access. We present MSBench, a reproducible harness evaluating three guards spanning the multimodal-safety design space (Llama-Guard-4, LlamaGuard-3-Vision, ShieldGemma-2) on 900 HarmBench/XSTest items judged by WildGuard. Our organizing finding is that each guard's failures are fixed by its architecture, not the channel under attack, and that these failures are complementary - so robust two-channel coverage can be composed cheaply from guards that reason over different modalities. The balanced intent classifier has a paired-significant image detection gap (McNemar p<0.001); the vision-specialized guard reaches 100% image detection only by refusing all benign images; the image-content classifier is strong on image (87% at 9.2% over-refusal) but blind to the text channel. The cheapest attack, the carrier prompt, is a text-context attack: a fiction framing collapses one guard 82%->6% and theatrical framing collapses another to 0%, on identical pixels. But an image-content guard that never receives the carrier is immune by construction, so a cross-modal ensemble (a text guard + an image guard) covers both channels (97% image / 92.5% text) at modest cost and never drops below 87% image detection under any of 18 carriers, while an ensemble of two text-context guards pays 56% over-refusal. White-box UAP attacks place the guards on a robustness spectrum. The conclusion: compose guards whose reasoning modalities differ, so their blind spots cover rather than share one another.

---

## Pre-submission checklist

- [ ] **Author block** in `paper.tex`: fill real affiliation + email (currently placeholders).
- [ ] **Affiliation / publication approval.** If this is submitted as independent personal research, list "Independent Researcher." If it is work-affiliated, confirm your employer's publication-review requirements before posting.
- [ ] **Verify citations** in `references.bib` — especially the 2025 model cards (Llama Guard 4, ShieldGemma 2) and Constitutional Classifiers (confirm/add the correct arXiv ID).
- [ ] **Compile once** and eyeball table/figure placement; grab `paper.bbl` for the tarball.
- [ ] **Decide `paper.md` fate** — keep as a synced mirror, or remove and link the arXiv PDF from the README.
- [ ] (Optional) add an Acknowledgments section; add `fig_examples.png` as Figure 3 if you want the qualitative examples in the PDF.

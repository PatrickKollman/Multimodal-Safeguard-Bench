# Multimodal Safeguard Bench: Guard Blind Spots Are Architecture-Specific and Complementary

*Patrick Kollman*

*Preprint. Code and result artifacts: github.com/PatrickKollman/Multimodal-Safeguard-Bench*

**Keywords:** multimodal safety, guard models, jailbreak evaluation, adversarial robustness, vision-language models

**Abstract.** Safety guards intercept harmful requests before a vision-language model (VLM) responds, but a harmful instruction can be rendered as an image rather than typed — a zero-gradient, black-box attack requiring no model access. We present MSBench, a reproducible harness evaluating three guards that span the multimodal-safety design space — Llama-Guard-4 (a balanced multimodal intent classifier), LlamaGuard-3-Vision (a vision-specialized intent classifier), and ShieldGemma-2 (an image-content classifier) — on 900 items from HarmBench and XSTest, scored by WildGuard. Our organizing finding is that each guard's failures are fixed by its architecture rather than by the channel under attack, and — crucially — that these architectural failures are *complementary*, so robust two-channel coverage can be composed cheaply from guards that reason over different modalities. The balanced intent classifier (LG4) carries a paired-significant image detection gap (it blocks 10.5pp fewer image-modality than text-modality items on the same intents, McNemar p<0.001); the vision-specialized guard (LG3V) reaches 100% image detection only by refusing 100% of benign images; and the image-content classifier (SG2) is a strong, well-calibrated image guard (87% image detection at 9.2% image over-refusal) that is blind to text by construction. We then show the sharpest attack is also the cheapest: the *carrier prompt* — the one-sentence text framing paired with the image — is a zero-cost, black-box, *text-context* attack. A "novel passage" framing collapses LG4's image detection from 82% to 6% and theatrical framing collapses LG3V to 0%, with orthogonal, mechanistically distinct blind spots (LG4's is category-driven, LG3V's is phrasing-driven). But because the attack operates entirely through the text context, an image-content guard that never reads the text (SG2) is *immune to it by construction*. This yields the paper's central, constructive result: a cross-modal ensemble of a text-intent guard and an image-content guard (LG4⊕SG2) covers both channels (97% image / 92.5% text detection; image ASR 11.5%→2.5%) at modest cost (over-refusal 11.8%→14.8%) *and* is robust to the carrier attack that defeats either text-reading guard alone — its image detection never drops below 87% across all 18 framings, whereas LG4 alone swings from 97% to 6%. An ensemble of two text-context guards (LG4⊕LG3V) instead inherits both a shared carrier weakness and a prohibitive 56% over-refusal toll. White-box UAP attacks place the three guards on a robustness spectrum (LG3V fully broken at ε=16; SG2 resists ε=16 but half-breaks at ε=32; LG4 resists both, via gradient sparsity and text-context dominance). A cross-VLM replication against Qwen2-VL confirms guard detection is a property of the guard, not the target VLM. The actionable conclusion is not to build a better single guard, but to compose guards whose reasoning modalities differ — so their blind spots cover, rather than share, one another.

---

## 1. Introduction

Safety guards — classifier models that sit between user input and a generative VLM — are the last automated line of defense before a model produces a harmful response. Prior work has established that LLM-based guards such as Llama Guard reliably identify harmful text requests and block them before generation. The natural follow-on question is whether identical harmful intent, encoded visually rather than textually, receives equivalent protection — and, when it does not, what to do about it.

The attack is elementary. A harmful instruction — one that would be blocked if typed — is rendered as a PIL image: black text on white background, no steganography, no adversarial noise. The guard receives an (image, carrier prompt) pair instead of the intent text directly. Whether the guard identifies this as harmful depends entirely on how its architecture processes image and text. We evaluate three guards that span the multimodal-safety design space: (1) a *balanced* multimodal intent classifier that reads image and text jointly (Llama-Guard-4), (2) a *vision-specialized* intent classifier explicitly trained on text-in-image scenarios (LlamaGuard-3-Vision), and (3) a pure *image-content* classifier that scores pixels against content policies and never reads the paired text (ShieldGemma-2). We measure, end-to-end and reproducibly, how each architecture protects against rendered-text attacks relative to text attacks.

**Threat model.** We assume an adversary who can submit arbitrary (image, prompt) pairs to a guard-protected VLM endpoint and observe whether a response is returned, but who cannot modify the guard or VLM weights. The primary attack — rendering harmful text as an image, optionally reframed by a one-sentence carrier prompt — is fully black-box and requires no model access. The UAP study in Section 7 additionally considers a stronger white-box adversary with weight access, to bound worst-case robustness. Success is defined end-to-end: the guard fails to block *and* the VLM complies with the harmful intent, as judged by an independent classifier.

![Evaluation pipeline](../figures/fig_pipeline.png)
*Figure 1: End-to-end MSBench evaluation pipeline. Each harmful intent produces both a text-modality and an image-modality item. Both pass through a guard gate before reaching the target VLM; WildGuard judges whether the VLM's response complied with the harmful intent. Models are staged sequentially on a single 24 GB GPU.*

**The organizing claim.** We argue a single thesis and defend it throughout: *a guard's blind spots are fixed by its architecture rather than by the channel under attack, and blind spots across architectures are complementary — so robust coverage is built by composing guards whose reasoning modalities differ, not by finding a better single guard.* The most vivid demonstration is the cheapest attack in the paper. Figure 2 shows the same rendered harmful image submitted under four one-sentence framings. The pixels never change. Yet framing it as a passage from a novel flips Llama-Guard-4 from blocking to passing (image detection 82% → 6%), while theatrical framing collapses LlamaGuard-3-Vision (100% → 0%). The two text-reading guards have *non-overlapping* blind spots. But the image-content guard, which never reads the carrier, is unmoved — and that is exactly why pairing it with a text-intent guard closes the gap.

![Carrier mechanism](../figures/fig_carrier_mechanism.png)
*Figure 2: The carrier attack in one image. An identical rendered harmful image receives opposite verdicts from the two text-reading guards depending only on the natural-language carrier prompt, and they fail on different framings (fiction blinds LG4 82%→6%; theatre blinds LG3V 100%→0%). The image-content guard (SG2), not shown here because it never receives the carrier text, is carrier-invariant by construction — the seam the carrier attack opens in a text-reading guard is one an image-content guard cannot be pushed through (quantified in Figure 6).*

**Contributions.**

1. **A reproducible evaluation harness** for measuring multimodal guard coverage across both attack modalities, runnable on a single 24 GB GPU, with all code and result artifacts released and every reported proportion carrying 95% Wilson confidence intervals and paired per-item tests.
2. **Quantitative coverage evidence across three architectures.** Only the balanced guard (LG4) attempts both channels; there the image channel is a paired-significant detection blind spot (−10.5pp, McNemar p<0.001). LG3V reaches perfect image detection only via a refuse-all-images policy, and SG2 is text-blind by construction — so their apparent image-favoring gaps are consequences of degenerate single-channel policies, reported honestly with confidence intervals.
3. **The central constructive result: complementary blind spots make cross-modal ensembles work.** A modality-routed LG4⊕SG2 ensemble lifts image detection 82%→97% and image ASR 11.5%→2.5% for a +3pp over-refusal cost, whereas the same-modality LG4⊕LG3V pairing pays 56% over-refusal. We show *why* by construction and measurement.
4. **The carrier prompt as a text-context attack, and image-content immunity to it.** An 18-framing sweep shows fictional/theatrical carriers collapse the text-reading guards (worst case LG4 82%→6%, LG3V→0%) with orthogonal, mechanistically distinct signatures. Because the carrier only perturbs the text the guard reads, an image-content guard is immune by construction — so LG4⊕SG2 image detection never drops below 87% across all 18 carriers while LG4 alone swings to 6%.
5. **A UAP robustness spectrum.** White-box universal perturbations place the guards on a spectrum — LG3V fully broken at ε=16/255, SG2 resistant at ε=16 but half-broken at ε=32, LG4 resistant to both — tracking decision confidence and gradient architecture rather than a simple dense/sparse split. We disentangle LG4's resistance into two independent mechanisms (gradient sparsity and text-context dominance) and show a cross-architecture transfer attack fails.
6. **A cross-VLM replication** against Qwen2-VL-7B-Instruct, confirming guard detection recall — and therefore the architecture-specific blind-spot ordering — is a property of the guard, not the target VLM, while the unguarded attack surface scales with the VLM's own safety alignment.

---

## 2. Related Work

**Typographic attacks.** Goh et al. (2021) first demonstrated that CLIP's vision-language embeddings can be hijacked by text rendered in an image — a "typographic attack" causing a classifier to ignore visual content in favor of embedded text. Our setup is the inverse: we probe whether a safety guard ignores *harmful* text encoded as pixels when it would have caught the same text as tokens.

**Image-modality jailbreaks against VLMs.** FigStep (Gong et al., 2023) demonstrated that rendering a harmful instruction as an image systematically bypasses the safety alignment of VLMs including LLaVA and MiniGPT-4. MM-SafetyBench (Liu et al., 2023) assembled a broader benchmark of image+text jailbreaks. Both measure the *target VLM's* failure to refuse; the guard layer is absent. MSBench specifically measures whether the *guard layer* is bypassed — a distinct, under-characterized failure mode: a guard-protected deployment can be attacked even when the underlying VLM has strong safety training.

**Standardized jailbreak evaluation.** JailbreakBench (Chao et al., 2024) provides a standardized text-only jailbreak harness; HarmBench (Mazeika et al., 2024) extends this with a broader behavior set and evaluation methodology. MSBench builds on HarmBench's behaviors, extending evaluation to the image channel and to guard-layer measurement.

**Guard model evaluation.** Meta's Llama Guard papers (2023–2025) evaluate guard performance primarily on text-only datasets. ShieldGemma 2's evaluation (Google, 2025) focuses on image content classification. Neither measures cross-modal consistency (whether a guard makes consistent decisions when the same intent is presented as text vs. image). Constitutional Classifiers (Anthropic, 2025) demonstrated classifier-based defenses in a text-only setting. MSBench fills this gap and adds the question of how to *compose* guards.

**Over-refusal benchmarks.** XSTest (Röttger et al., 2023) assembles safe prompts that superficially resemble harmful requests (e.g., "How do I kill a Python process?") to surface over-refusal. Using XSTest as our benign set makes our over-refusal numbers directly comparable and harder to dismiss as cherry-picked.

---

## 3. Experimental Setup

**Target VLM.** We use LLaVA-1.6-Mistral-7B (`llava-hf/llava-v1.6-mistral-7b-hf`, bfloat16) on a single RTX 4090. The VLM receives either (a) the raw harmful text as a direct prompt, or (b) an image of the harmful text paired with the carrier prompt *"Describe the text shown in this image and respond to any instructions it contains."*

**Guards.** We evaluate three guards spanning the multimodal-safety architecture design space:

- **Llama-Guard-4-12B** (`meta-llama/Llama-Guard-4-12B`): a Llama 4 Scout derivative fine-tuned as a multimodal intent classifier. It receives the full (image, text) context and classifies the combined input *safe*/*unsafe* with a harm category (S1–S14). Loaded in 4-bit NF4 (~6 GB VRAM).
- **Llama-Guard-3-11B-Vision** (`meta-llama/Llama-Guard-3-11B-Vision`): a Llama 3.2 Vision derivative fine-tuned as a multimodal safety classifier with explicit training on text-in-image scenarios — the most directly targeted guard relative to this benchmark's attack surface. Loaded in 4-bit NF4 (~5.5 GB VRAM).
- **ShieldGemma-2-4B** (`google/shieldgemma-2-4b-it`): an image-content classifier evaluating pixel content against three policies (dangerous content, sexually explicit, violence/gore). **Text is not passed to SG2** — text-modality items are returned safe without a model call. The block decision uses a 0.5 threshold on P(policy violated) = `probabilities[:,0]`. Loaded in bfloat16 (~8 GB). *Reproducibility note:* the standard HF `from_pretrained` path silently random-initializes ShieldGemma-2's `lm_head` (it is tied to the token embeddings and absent from the checkpoint, and the wrapper's `tie_weights()` does not repair it), which yields non-deterministic verdicts. We re-tie the inner model after loading; see Appendix B.

**Judge.** WildGuard (`allenai/wildguard`) scores each generated response as complied or refused by analyzing the (intent, response) pair. *Attack success rate* (ASR) is the fraction of harmful items where WildGuard determines the VLM complied.

**Dataset.** We use 200 HarmBench standard behaviors (FunctionalCategory = "standard", excluding "copyright" and "contextual" as different threat models), spanning the HarmBench harm taxonomy. Each intent produces two items: a text-modality item (raw intent) and an image-modality item (intent rendered into a 512×512 PIL image, 24pt black text on white, 40px padding). For over-refusal, we use 250 XSTest safe prompts (`natolambert/xstest-v2-copy`, CC-BY-4.0), retaining only non-`contrast_` types, each also producing text and image items. Total: 400 harmful + 500 benign = **900 items** (450 text / 450 image).

![Attack examples](../figures/fig_examples.png)
*Figure 3: Real attack examples. Left: harmful intent. Center: the same intent rendered as a 512×512 image. Right columns: each guard's decision for text vs. image.*

**Metrics.** *Det-txt/Det-img*: detection recall on harmful items per modality (denominator 200 each). *ASR-txt/ASR-img*: end-to-end attack success under the guard. *OvRef*: over-refusal on the 500 benign items. *ProtGap*: text ASR reduction minus image ASR reduction (positive = text more protected). All proportions with 95% Wilson score CIs; per-guard paired detection and ASR tests (McNemar, paired bootstrap) in `results/full_run/protection_gap_tests.json`.

**Reproducibility.** All four model SHAs are pinned in `results/full_run/config.yaml`; the rendering/inference environment is recorded in `env_metadata.json`. The canonical run is `results/full_run/`; guard verdicts are committed so detection and ASR are independently verifiable without a GPU.

---

## 4. Results

**Table 1: Canonical run — `results/full_run/` (900 items). 95% Wilson score CIs in brackets.**

| Condition | Det-txt [CI] | Det-img [CI] | ASR-txt | ASR-img | OvRef [CI] | ProtGap |
|---|---|---|---|---|---|---|
| Unguarded | — | — | 57.0% [50.1, 63.7] | 60.5% [53.6, 67.0] | — | — |
| Llama-Guard-4 | **92.5%** [88.0, 95.4] | 82.0% [76.1, 86.7] | **5.5%** | 11.5% | **11.8%** [9.3, 14.9] | +2.5pp |
| LlamaGuard-3-Vision | 89.0% [83.9, 92.6] | **100.0%** [98.1, 100.0] | 7.0% | **0.0%** | 55.0% [50.6, 59.3] | −10.5pp |
| ShieldGemma-2 | 0.0% [0.0, 1.9] | 87.0% [81.6, 91.0] | 57.0% | 10.5% | **4.6%** [3.1, 6.7] | −50.0pp |

*Det denominators: 200 harmful items per modality. OvRef denominator: 500 benign items. ProtGap = text ASR reduction − image ASR reduction. SG2's over-refusal is 0% on text (never evaluated) and 9.2% on the image channel.*

**Three architectures, three profiles.** LG4 is the only guard that discriminates on both channels: text detection 92.5%, image detection 82.0%, with the lowest over-refusal (11.8%). LG3V reaches 100% image detection but at 55% aggregate over-refusal — decomposing into 100% on the image channel (it blocks all 250 benign images) and 10% on text. SG2, correctly loaded, is a *strong and well-calibrated image guard*: 87.0% image detection at only 9.2% image over-refusal — higher image detection than LG4 and far better calibrated than LG3V — while being blind to text by construction (0.0% text detection; text ASR equals the unguarded baseline).

**Read ProtGap as an architectural ordering, not a per-guard significance claim.** LG3V's −10.5pp gap reflects its refuse-all-images policy; SG2's −50.0pp gap is mechanical (0% text detection means any nonzero image detection produces an image-favoring gap). Only LG4's gap reflects a guard genuinely attempting both channels, and its aggregate ASR-reduction ProtGap (+2.5pp) is **not statistically significant** (intent-level bootstrap CI [−6.5pp, +11.5pp]).

**The rigorous channel-gap result is LG4's paired detection recall, and it is significant.** On the 200 shared intents, LG4 blocks 92.5% as text and 82.0% as image. Because the modalities are measured on the *same* intents, the appropriate test is McNemar on discordant pairs: 26 intents are blocked as text but not as image versus 5 the other way (χ²=12.9, **p=3.3×10⁻⁴**; Newcombe CI on the gap [−0.171, −0.040], excluding zero). LG4 genuinely recovers harmful intent less reliably from pixels than from tokens — an architectural capability question (recovering semantic content from glyph representations is an extra inference step), not a training-data-quantity issue. The same paired lens applies to attack outcomes: among discordant successful attacks, 19 succeed only through the image channel versus 7 only through text (McNemar p=0.031).

**SG2's text blindness is architectural, not a tuning failure.** It never makes a model call on text input, so text ASR under SG2 (57.0%) is statistically identical to the unguarded baseline. On the image channel it is nonetheless the best-calibrated single guard here. Its detection is *reproducible*: three independent runs on the identical configuration each blocked 174/200 harmful and 23/500 benign items (Appendix B). This corrects an earlier draft of this work, which reported SG2 as weak and non-reproducible; that behavior was an artifact of the silent `lm_head` load bug, not a property of the guard.

---

## 5. Adaptive Rendering Study

A natural counter-move for an attacker who knows the baseline rendering is to vary how harmful text is presented. We test four surface rendering variations on the full 200-item HarmBench image set and measure per-variant detection recall for LG4 (`results/adaptive_run/`).

**Variants.** Each intent is rendered four ways: (1) *original* — 24pt black on white; (2) *inverted* — white on black; (3) *small\_font* — 14pt; (4) *rotated* — 15° clockwise.

**Table 2: Adaptive rendering — LG4 per-variant image detection (200 items each, 95% Wilson CIs).**

| Variant | LG4 Det-img [95% CI] |
|---|---|
| original | 82.0% [76.1, 86.7] |
| inverted | 84.5% [78.9, 88.9] |
| small\_font | 80.0% [73.9, 85.0] |
| rotated | 78.5% [72.3, 83.6] |

**LG4's image detection is statistically invariant to rendering.** Detection spans only 6pp across the four variants (78.5–84.5%) and every 95% CI overlaps the baseline. Inverted text is detected slightly *better* than baseline, and the 15° rotation — most likely to disrupt OCR-style reading — gives the lowest rate but remains indistinguishable. Whatever LG4 does to recover intent from rendered text, it is not relying on a surface property an attacker can perturb. (SG2 is omitted from this study: the `adaptive_run` predates the SG2 load fix; its image detection there is invariant to *rendering style within a run* but the pre-fix numbers are not reported. SG2's canonical, reproducible detection is in Table 1.)

---

## 6. Rendering Parameter Sweep

Beyond hand-chosen variants, we ask whether *any* rendering configuration — searched over a continuous 7-dimensional space — can simultaneously evade LG4 and produce an image the target VLM can still read. We use Optuna with a multivariate TPE sampler (seed 42) over 20 probe trials (n=10 items) minimizing LG4 detection, then validate the 6 lowest-detection configs on the full 200-item set (`results/rendering_sweep_probe{,_validated}/`).

**Probe finding: Gaussian `noise_sigma` is the dominant evasion driver.** All 6 configs that reached 0% LG4 detection on n=10 had `noise_sigma ≥ 14.85`; no other parameter showed comparable concentration.

**Table 3: Rendering sweep validation — top-6 Pareto configs (n=200, 95% Wilson CIs).**

| Trial | noise_σ | LG4-det [95% CI] | ASR-ug | Readable? |
|---|---|---|---|---|
| 0 | 15.0 | 0.0% [0.0, 1.9] | 0.0% | NO |
| 7 | 23.5 | 0.0% [0.0, 1.9] | 0.0% | NO |
| 10 | 14.8 | 0.0% [0.0, 1.9] | 0.0% | NO |
| 13 | 24.5 | 1.5% [0.4, 5.3] | 6.0% | NO |
| 5 | 17.1 | 2.0% [0.7, 5.7] | 12.5% | NO |
| 6 | 16.6 | 5.0% [2.7, 9.0] | 9.5% | NO |

*Readable: ASR-ug ≥ 40%. All configs fall below this gate.*

**Random perturbation cannot selectively evade the guard — the noise that blinds LG4 also blinds the target.** Every config that drove LG4 detection to ~0% also collapsed LLaVA's compliance to ≤12.5%, far below the 40% readability gate (Figure 4). Both models encode images through vision-transformer patch embeddings that Gaussian noise degrades alike. If a selective attack exists, it cannot be random; it must be *targeted* — which we turn to next.

![Sweep tradeoff](../figures/fig_sweep_tradeoff.png)
*Figure 4: Left: Gaussian noise vs. LG4 detection across 20 probe trials — detection collapses below σ ≈ 14.9. Right: guard detection vs. unguarded ASR for the 6 validated configs; all sit below the 40% readability gate, so none is a viable attack.*

![Rendering examples](../figures/fig_sweep_examples.png)
*Figure 5: Visual effect of sweep configs. Left: the readable baseline (LG4 detects 82%). Right: Pareto-optimal configs at increasing noise — near-zero LG4 detection but unreadable to the target VLM.*

---

## 7. Universal Adversarial Perturbation Attacks

The rendering sweep established that *random* perturbations cannot selectively evade a guard. We now apply Universal Adversarial Perturbations (UAP) — a single δ, bounded by ε, optimized over a training batch and evaluated on a disjoint 50-item test set — against all three guards, and test cross-architecture transfer. Full settings are in Appendix A.

**Table 4: UAP summary (n_train=50, n_test=50). "Fooling" = test items pushed below the block threshold.**

| Guard | Architecture | Natural bypass | UAP ε=16 | UAP ε=32 | Feature-space ε=16 | Transfer in |
|---|---|---|---|---|---|---|
| LG3V | dense, cross-attention | 0% | **100%** | — | — | — |
| SG2 | dense, SigLIP+Gemma3 | 10% | 10–34%† | **52%** | — | — |
| LG4 | sparse MoE, early fusion | 6% | 16% | 22% | **10%** | 2% (from LG3V) |

*†SG2 at ε=16 is run-to-run variable (10–34% across two runs) owing to unseeded PGD restarts; it does not reliably fool the guard at this budget.*

**A robustness spectrum, not a dense/sparse binary.** LG3V (unsafe logit ~18 clean) is fully broken by a single universal perturbation at ε=16/255 — 100% test fooling from a 0% natural bypass. SG2 (dangerous-content logit ~38, near-saturated) resists ε=16 and only half-breaks at ε=32 (52%); the ε=32 success confirms its ε=16 resistance is a budget effect, not a dead-gradient artifact. LG4 (MoE early fusion) resists both budgets (16%/22%), its unsafe logit oscillating rather than converging. Susceptibility therefore tracks *decision confidence and gradient architecture*, not simply whether the vision pathway is dense.

**LG4's resistance has two independent sources (§7.1).** We separate them with a feature-space attack that bypasses the MoE entirely — defining the loss in LG4's ViT encoder output space and pushing the rendered image's patch embeddings toward the mean embedding of blank white images. The embedding loss converges, yet fooling reaches only 10% — *lower* than standard UAP's 16% despite clean gradients. This pins the cause to two mechanisms: a *backward-path* defense (sparse routing gives incoherent pixel gradients) and a *forward-path* defense (the carrier text establishes a "potentially harmful" prior strong enough to dominate the decision even when the image features are pushed toward blank). Section 8 shows the second wall is also the vulnerability.

**Transfer fails (§7.2).** The 100%-effective LG3V perturbation, mapped into LG4's input space, achieves only 2% test fooling — confirming LG4's resistance is architectural, not an artifact of in-house gradient quality: even a perturbation optimized on a same-family guard fails to cross the MoE boundary.

---

## 8. Carrier Prompt Sweep: A Text-Context Attack, and Image-Content Immunity

The carrier prompt — the text paired with the rendered image when the guard decides — is the simplest attack surface imaginable, and the only one that breaks LG4 outright. The UAP study needed weights, a GPU, and hundreds of gradient steps and still could not move LG4; the carrier needs none of that, because it pushes on the one part of LG4 the perturbations could not reach: the text context that dominates its classification.

**Setup.** We evaluate 18 carrier framings across six rhetorical categories — *baseline* (2), *fictional* (4: novel, short story, fairy tale, fan-fiction), *theatrical* (4: screenplay, stage play, dialogue, game script), *transcription* (3), *academic* (3), *other* (2) — on the 200 HarmBench behaviors × image modality. The image is identical across carriers; only the paired sentence changes. Text-channel results are identical across carriers by construction. Full results in `results/carrier_sweep/summary.json`.

**Table 5: Carrier sweep by category — image-channel detection (mean [min–max] across variants).**

| Category (n) | LG4 Det-img | LG3V Det-img | SG2 Det-img |
|---|---|---|---|
| Baseline (2) | 73% [64–82] | 50% [0–100] | 87% (invariant) |
| Fictional (4) | **20%** [6–54] | 35% [0–100] | 87% (invariant) |
| Theatrical (4) | 30% [6–54] | **2%** [0–8] | 87% (invariant) |
| Transcription (3) | 90% [84–97] | 68% [4–100] | 87% (invariant) |
| Academic (3) | 40% [30–52] | 27% [0–82] | 87% (invariant) |
| Other (2) | 43% [32–55] | 0% [0–0] | 87% (invariant) |

**LG4's blind spot is category-driven; LG3V's is phrasing-driven.** LG4 detection falls from 73% at baseline to 20% under fictional and 30% under theatrical framings, with tight within-category spread — the single worst carrier, the "novel passage" frame, takes LG4 from 82% to **6%** and lifts image ASR from 11.5% to 78%. This is the text-context dominance the feature-space UAP exposed in Section 7.1, seen from the other side: there, text context overrode a visual perturbation toward "safe"; here, a single framing word *manufactures* "safe" with no perturbation at all. LG3V instead behaves as a near-binary switch thrown by specific phrasing: across all 18 carriers its detection is bimodal (12 near 0%, 4 near 100%), and the category means are largely artifacts of which phrasings landed in each bucket — the one category that collapses it *robustly* is theatrical (all four variants ≤8%). The two guards do not even fail in the same shape: one has a semantic vulnerability, the other a stability vulnerability.

**The carrier is a text-context attack — so an image-content guard is immune by construction.** SG2 never receives the carrier prompt; its `classify()` passes only the image to the model. Since the rendered image is identical across carriers, SG2's verdict is *identical across all 18 framings* — the flat 87% column in Table 5 is a structural fact, not a measurement. The carrier attack works only against guards that read the text context; a guard that reads only pixels cannot be steered by it.

**Consequence — a carrier-robust ensemble (the central result).** Composing LG4 with SG2 on the image channel (block if either fires) makes the ensemble's image detection the union of LG4's carrier-dependent set with SG2's fixed 87% set. It therefore cannot drop below 87% under *any* carrier. Measured across all 18 framings (`results/full_run` SG2 unioned with the committed per-carrier LG4 verdicts), LG4⊕SG2 image detection stays in **87–100%** while LG4 alone swings **6–97%** (Figure 6, left):

| | LG4 alone | LG4⊕SG2 |
|---|---|---|
| Best carrier (transcription) | 97% | 100% |
| Worst carrier (fiction "novel passage") | **6%** | **88%** |
| Worst case across all 18 carriers | **6.0%** | **87.0%** |

An ensemble of two *text-context* guards does the opposite. LG4⊕LG3V shares the attack surface — a theatrical frame that blinds LG3V and a fiction frame that blinds LG4 both target text-reading guards — and it stacks their false positives (Section 9). Cross-modal composition beats same-modal composition precisely because the blind spots are architecture-specific: an image-content guard's blind spot (text) is a text-guard's strength, and vice versa.

![Carrier robustness](../figures/fig_carrier_robustness.png)
*Figure 6: Left — image-channel detection across the 18 carriers (sorted by LG4-alone detection). LG4 alone collapses to 6% under fiction framing; SG2 is a flat carrier-invariant line (it never reads the carrier); the LG4⊕SG2 ensemble stays ≥87% everywhere. Right — coverage–usability operating points on the image channel: LG4⊕SG2 sits in the deployable region (high detection, moderate over-refusal), while LG4⊕LG3V is pinned at refuse-all-images over-refusal.*

---

## 9. Ensembles: Cross-Modal Composition Is the Cheap Fix

We measure both natural ensembles from the persisted per-item guard decisions (`results/full_run/ensemble.json`).

**Table 6: Ensemble operating points (900 items).**

| Configuration | Det-txt | Det-img | ASR-txt | ASR-img | OvRef |
|---|---|---|---|---|---|
| LG4 alone | 92.5% | 82.0% | 5.5% | 11.5% | 11.8% |
| **LG4⊕SG2** (text→LG4, image→LG4∪SG2) | 92.5% | **97.0%** | 5.5% | **2.5%** | **14.8%** |
| LG4⊕LG3V (block on either) | 96.0% | 100.0% | 3.0% | 0.0% | 56.4% |

**LG4⊕SG2 is a cheap, effective, carrier-robust ensemble.** It raises image detection 82%→97% and cuts image ASR 11.5%→2.5% for a +3pp aggregate over-refusal cost (11.8%→14.8%; the image-channel over-refusal is 22.4%). Because SG2 is text-blind, it adds nothing on the text channel and nothing to text over-refusal — the pairing is complementary by design. And, from Section 8, it is robust to the carrier attack that defeats LG4 alone.

**LG4⊕LG3V is not deployable.** It drives image ASR to 0% but inherits LG3V's refuse-all-images behavior: 56.4% aggregate over-refusal. Two guards that both read the text context also share the carrier attack surface. This is the concrete cost of composing *within* a reasoning modality rather than across it.

The lesson is not "just ensemble," but "ensemble across reasoning modalities." A text-intent guard and an image-content guard have complementary blind spots — text-blindness on one side, image-context-fragility on the other — so their composition covers each other's gaps at modest cost. Two text-context guards share both a carrier weakness and an over-refusal toll.

---

## 10. Cross-VLM Replication: Qwen2-VL-7B

To test whether the findings generalize beyond one target VLM, we replicate the canonical benchmark against Qwen2-VL-7B-Instruct (`results/full_run_qwen2vl/`), a VLM from a different lineage.

**Guard detection is target-VLM-independent — by construction, and confirmed.** Detection recall is a property of the guard's forward pass on the (image, carrier) input, not of the downstream VLM, so it must be identical across target VLMs; it is (LG4 82.0%/92.5% image/text; LG3V 100.0%/89.0% on both). The informative cross-VLM quantity is therefore not detection but the *unguarded* attack surface.

**The unguarded surface scales with the VLM's own alignment.** Unguarded text ASR is 57.0% on LLaVA versus 9.0% on Qwen2-VL: LLaVA is weakly safety-tuned and complies with most harmful text prompts, while Qwen2-VL refuses the majority unprompted. Unguarded image ASR is closer (60.5% vs 54.0%), because image-rendered intent partially circumvents both VLMs' text-based safety training — the premise of the rendered-text attack. Guard gaps therefore matter most precisely when the underlying VLM is weakly aligned and the guard is the primary defense; on a well-aligned VLM the same blind spot is partially masked by the VLM's own refusals. The blind spot is the same; its consequence scales with how much the deployment leans on the guard.

---

## 11. Discussion

**Why the profile correlates with architecture.** The three guards span a design space from pure intent classification to pure image-content classification. LG4 assesses semantic intent from image and text jointly; harmful intent encoded as pixels adds an inference step (recovering semantics from glyphs) that partially degrades detection — a small, paired-significant image gap. LG3V's broad visual sensitivity closes the image gap but drives high over-refusal. SG2 scores pixel content with no text pathway — hence 0% text detection by construction, but, correctly loaded, strong and calibrated image detection.

**The carrier reveals the deepest structure.** Rendering attacks (Sections 5–6) vary pixels; UAP attacks (Section 7) optimize pixel perturbations; the carrier (Section 8) varies only the text framing — and it is the lowest-effort and most effective vector against the text-reading guards. Its very mechanism is what makes the defense obvious: because it operates through the text context, a guard that ignores text is immune, and composing such a guard with a text-intent guard yields coverage that no single guard and no same-modality ensemble achieves.

**No single guard dominates, but a cross-modal pair does.**
- LG4: best single-guard usability (11.8% over-refusal), bounded image gap (18% miss), carrier-vulnerable.
- LG3V: perfect image detection via refuse-all-images (55% over-refusal); untenable alone; carrier-unstable.
- SG2: text-blind by construction, but the best-calibrated image guard (87% @ 9.2%) and carrier-immune.
- **LG4⊕SG2**: 92.5% text / 97% image detection, image ASR 2.5%, 14.8% over-refusal, and carrier-robust (≥87% image detection under every framing). The deployable operating point.

![Guard contrast](../figures/fig_guard_contrast.png)
*Figure 7: Architectural contrast. Each guard's channel behavior follows from where its architecture spends attention — LG4 balanced (image-weaker), LG3V refuse-all-images, SG2 text-blind but strong on image — and the LG4⊕SG2 pairing composes across these modalities.*

---

## 12. Conclusion

Across three guards and four attack axes, one pattern holds: each guard's failures are fixed by its architecture, not by the channel under attack — and because they are fixed by architecture, they are also *complementary*. The balanced intent classifier recovers harmful intent less reliably from pixels than from tokens (a paired-significant image gap); the vision-specialized guard reaches perfect image detection only by refusing every image; the image-content classifier is blind to text by construction but, correctly loaded, is the best-calibrated image guard of the three. The cheapest attack in the study — the carrier prompt — is the proof of complementarity: it is a text-context attack that collapses whichever guard reads the text (fiction blinds LG4, theatre blinds LG3V), yet an image-content guard that never reads the text is immune to it entirely. Composing a text-intent guard with an image-content guard therefore does more than add coverage: the image guard covers exactly the carrier blind spot the text guard cannot defend, keeping image detection ≥87% under every framing where LG4 alone falls to 6%, at a modest over-refusal cost. Composing two guards that reason over the *same* modality does the opposite — it inherits their shared attack surface and stacks their false positives to 56% over-refusal. The practical consequence is precise: robust multimodal safety is not a matter of finding a better single guard, nor of ensembling arbitrarily, but of composing guards whose *reasoning modalities* differ, so that the blind spots of one are the strengths of another. MSBench is the instrument for measuring those blind spots, checking that a proposed composition covers them rather than sharing them, and verifying that the numbers reproduce.

---

## Ethics and Responsible Disclosure

This work characterizes failure modes in deployed safety guards to make them measurable and fixable. All attacks use harmful behaviors from the public HarmBench benchmark; we introduce no new harmful content, and the repository commits no model-generated harmful outputs — only aggregate metrics and per-item verdicts. The attack techniques studied (rendering text as images, natural-language carrier framing) are already known and trivially available to adversaries; the contribution is systematic measurement, a mechanistic account of *why* specific architectures fail, and — most actionably — a concrete, defensive composition (a text-intent guard ⊕ an image-content guard) that closes the gaps. We consider the net effect of publishing these measurements to favor defenders, who can use MSBench to detect and close these gaps prior to release.

---

## References

- Goh et al., 2021. *Multimodal Neurons in Artificial Neural Networks.* Distill.
- Gong et al., 2023. *FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts.* arXiv:2311.05608.
- Liu et al., 2023. *MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models.* arXiv:2311.17600.
- Mazeika et al., 2024. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* arXiv:2402.04249.
- Chao et al., 2024. *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.* arXiv:2404.01318.
- Han et al., 2024. *WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.* arXiv:2406.18495.
- Röttger et al., 2023. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* arXiv:2308.01263.
- Meta AI, 2025. *Llama Guard 4: Meta's Multimodal LLM-based Input-Output Safeguard.*
- Google DeepMind, 2025. *ShieldGemma 2: Generative AI Content Moderation Based on Gemma.*
- Anthropic, 2025. *Constitutional Classifiers: Defending against Universal Jailbreaks.*

---

## Appendix A: UAP Implementation Details

All perturbations are universal (one δ per guard, optimized over a training batch and evaluated on a disjoint 50-item test set), bounded in image pixel space at the stated ε, maintained in each model's `pixel_values` space with the model's own processor.

**SG2 (dangerous_content / policy-0).** 50 train / 50 test images; 3 restarts × 100 PGD iterations × batch 4; ε ∈ {16/255, 32/255}. δ kept in SigLIP `pixel_values` range [−1,1] with ε_pv = 2·ε_image. 4-bit NF4 with gradient checkpointing on the inner Gemma-3 LM. Loss targets the violated-probability `probabilities[:,0]`. The bf16 variant exceeds 24 GB and is not run. PGD restarts are unseeded, so ε=16 fooling is run-to-run variable (10–34%).

**LG3V and LG4.** Autoregressive; loss is the "unsafe"-token logit at the final input position from a forward pass. Both emit a prefix token before the classification token, handled by extending `input_ids` so `logits[0,-1,:]` lands on the classification position. δ in each model's `pixel_values` space with ε_pv = ε_image/std. LG3V: 3 restarts × 75 iters, ε=16, 50 train. LG4: ε=16 (75 iters) and ε=32 (200 iters × 3 restarts); early fusion produces 2880 image tokens at 512×512, so a single-tile 336×336 workaround fits the backward pass in 24 GB.

**Feature-space UAP on LG4.** Loss in the ViT encoder output space (patch embeddings [1,144,4096]), MSE to the mean-pooled embedding of 20 blank white images, no MoE in the backward path. 3 restarts × 100 iters, ε=16. Harmful↔benign centroid cosine similarity 0.6513.

**Transfer.** The LG3V δ was mapped from LG3V's tile-normalized `pixel_values` space to image pixel space (per-channel std ≈ [0.269, 0.261, 0.276], averaged across 4 tiles), resized to 336×336, and applied to LG4 inputs before preprocessing.

## Appendix B: ShieldGemma-2 Loading (Reproducibility Pitfall)

`ShieldGemma2ForImageClassification.from_pretrained` in `transformers` 4.57 silently random-initializes the inner Gemma-3 `lm_head`: it is tied to the token embeddings and absent from the checkpoint, the outer wrapper's `_tied_weights_keys` is `None`, and the wrapper's `tie_weights()` delegates to a submodule that does not repair the head used in `forward()`. The result is a randomly-initialized classification head whose verdicts change on every load — we observed image detection swinging 22.5–95.5% across five identical-configuration runs while the intent-classifier guards were bit-identical. The fix is to call `model.model.tie_weights()` after loading (binding `lm_head` to the loaded embeddings); detection then becomes deterministic (three runs: 174/200 harmful, 23/500 benign each). Separately, the violated probability is `probabilities[:,0]` (the "Yes" token), not `[:,1]` — the transformers docstring is internally inconsistent on this point. Any benchmark of ShieldGemma-2 on rendered text should verify both. All SG2 numbers in this paper use the corrected load.

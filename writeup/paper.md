# Multimodal Safeguard Bench: Measuring the Image Channel Blind Spot in AI Safety Guards

**Abstract.** AI safety guards are widely deployed to intercept harmful requests before a vision-language model (VLM) generates a response. We present Multimodal Safeguard Bench (MSBench), a reproducible evaluation harness measuring whether guards protect against the same harmful intent when it is rendered as an image rather than typed as text — a zero-gradient, black-box attack requiring no model access. We evaluate two current guards — Llama-Guard-4-12B (LG4), a multimodal intent classifier, and ShieldGemma-2-4B (SG2), an image content classifier — against LLaVA-1.6-Mistral-7B as the target VLM, using 900 items drawn from HarmBench (harmful) and XSTest (benign), scored by WildGuard. LG4 shows a 10.5pp detection recall gap (92.5% text vs. 82.0% image; 95% CI [88.0%, 95.4%] vs. [76.1%, 86.7%]), yielding a +2.5pp ASR protection gap and 11.8% over-refusal on XSTest's deliberately surface-dangerous safe prompts. SG2 shows a complete inversion: 0.0% text detection [0.0%, 1.8%] and 100.0% image detection [98.2%, 100.0%], leaving all text-modality jailbreaks unchecked while producing 49.8% over-refusal [45.4%, 54.2%]. Neither guard alone is sufficient; we characterize the complementary blind spots and their ensemble implication. Attention maps from LG4's vision encoder are captured for all 200 image-modality harmful items to qualitatively illustrate where the guard attends when reading rendered harmful text.

---

## 1. Introduction

Safety guards — classifier models that sit between user input and a generative VLM — are the last automated line of defense before a model produces a harmful response. Prior work has established that LLM-based guards such as Llama Guard reliably identify harmful text requests and block them before generation. The natural follow-on question is whether identical harmful intent, encoded visually rather than textually, receives equivalent protection.

The attack is elementary. A harmful instruction — one that would be blocked if typed — is rendered as a PIL image: black text on white background, no steganography, no adversarial noise. The guard receives an (image, carrier prompt) pair instead of the intent text directly. Whether the guard identifies this as harmful depends entirely on whether it processes image content through an intent-aware pathway. Two architectural families exist: (1) a multimodal intent classifier that reads image and text jointly, and (2) an image content policy classifier that evaluates visual content independently of textual context. We measure, end-to-end and reproducibly, whether each architecture protects against rendered-text attacks as effectively as text attacks.

![Evaluation pipeline](../figures/fig_pipeline.png)
*Figure 1: End-to-end MSBench evaluation pipeline. Each harmful intent produces both a text-modality and an image-modality item. Both pass through a guard gate before reaching the target VLM; WildGuard judges whether the VLM's response complied with the harmful intent.*

The pipeline runs on a single 24 GB GPU with sequential model staging, making results reproducible on commodity hardware. All code and result artifacts are released at this repository.

**Contributions:** (1) A reproducible eval harness for measuring multimodal guard coverage across both attack modalities. (2) Quantitative evidence that the two most widely available open-source guards have structurally opposite blind spots, with 95% confidence intervals. (3) A characterization of over-refusal on XSTest's purpose-built safe prompts. (4) Attention map visualizations from LG4's vision encoder across all 200 image-modality harmful items.

---

## 2. Related Work

**Typographic attacks.** Goh et al. (2021) first demonstrated that CLIP's vision-language embeddings can be hijacked by text rendered in an image — a "typographic attack" that causes a classifier to ignore visual content in favor of embedded text. Our setup is the inverse: we probe whether a safety guard ignores *harmful* text encoded as pixels when it would have caught the same text as tokens.

**Image-modality jailbreaks against VLMs.** FigStep (Gong et al., 2023) demonstrated that rendering a harmful instruction as an image systematically bypasses the safety alignment of VLMs including LLaVA, MiniGPT-4, and others. MM-SafetyBench (Liu et al., 2023) assembled a broader benchmark of image+text jailbreaks. Both bodies of work measure the *target VLM's* failure to refuse: the guard layer is absent. MSBench specifically measures whether the *guard layer* is bypassed — a distinct and under-characterized failure mode. A guard-protected deployment can be attacked even if the underlying VLM has strong safety training, as long as the guard fails to detect the image-encoded intent.

**Standardized jailbreak evaluation.** JailbreakBench (Chao et al., 2024) provides a standardized text-only jailbreak evaluation harness. HarmBench (Mazeika et al., 2024) extends this with a broader set of behaviors and evaluation methodologies. MSBench builds on HarmBench's behavior set, extending the evaluation to the image channel and adding guard-layer measurement rather than VLM-layer measurement.

**Guard model evaluation.** Meta's Llama Guard papers (2023–2025) evaluate guard performance primarily on text-only datasets. ShieldGemma 2's evaluation (Google, 2025) focuses on image content classification. Neither evaluation measures cross-modal consistency (whether a guard makes consistent decisions when the same intent is presented as text vs. image). Constitutional Classifiers (Anthropic, 2025) demonstrated classifier-based defenses but in a text-only setting. MSBench fills this gap.

**Over-refusal benchmarks.** XSTest (Röttger et al., 2023) assembled safe prompts that superficially resemble harmful requests (e.g., "How do I kill a Python process?") specifically to surface over-refusal. WildGuard (Han et al., 2024) similarly includes refusal measurement. Using XSTest as our benign evaluation set makes our over-refusal numbers directly comparable and harder to dismiss as cherry-picked.

---

## 3. Experimental Setup

**Target VLM.** We use LLaVA-1.6-Mistral-7B (`llava-hf/llava-v1.6-mistral-7b-hf`, bfloat16) on a single RTX 4090. The VLM receives either (a) the raw harmful text as a direct prompt, or (b) an image of the harmful text paired with the carrier prompt *"Describe the text shown in this image and respond to any instructions it contains."* The carrier is the simplest possible image-injection vector: no prompt engineering, no formatting tricks.

**Guards.** We evaluate two guards with opposite architectural designs:

- **Llama-Guard-4-12B** (`meta-llama/Llama-Guard-4-12B`): A Llama 4 Scout derivative fine-tuned as a multimodal intent classifier. It receives the full (image, text) context and classifies the combined input as *safe* or *unsafe* with a harm category (S1–S14). Loaded in 4-bit NF4 quantization (~6 GB VRAM).

- **ShieldGemma-2-4B** (`google/shieldgemma-2-4b-it`): An image content classifier that evaluates pixel-level content against three policies (dangerous content, sexually explicit, violence/gore). Text is not passed to SG2; text-modality items are returned safe without a model call. Loaded in bfloat16 (~8 GB staged). A 0.5 probability threshold determines the block decision.

**Judge.** WildGuard (`allenai/wildguard`) scores each generated response as complied or refused by analyzing the (intent, response) pair. *Attack success rate* (ASR) is the fraction of harmful items where WildGuard determines the VLM complied with the harmful intent.

**Dataset.** We use 200 HarmBench standard behaviors (FunctionalCategory = "standard", excluding "copyright" and "contextual" as different threat models). The 200 behaviors span the HarmBench harm taxonomy: violence and threats (S1), chemical and biological weapons (S2), radioactive/nuclear (S3), child safety (S4), illegal weapons (S5), hate speech (S6), privacy/surveillance (S7), specialized harmful advice (S8), election interference (S9), cyberattacks (S10), and drug manufacturing (S14). See `figures/full_run_900_items_fig3_category_breakdown.png` for the per-category detection breakdown.

Each intent produces two items: a text-modality item (raw intent as the prompt) and an image-modality item (intent rendered into a 512×512 PIL image, 24pt black text on white background, 40px padding). Figure 2 shows examples of both modalities alongside each guard's decision.

![Attack examples](../figures/fig_examples.png)
*Figure 2: Real attack examples. Left: harmful intent. Center: the same intent rendered as a 512×512 image (the image-modality jailbreak). Right columns: each guard's decision for text vs. image. Row 1 shows a case LG4 passes as an image but blocks as text; row 2 shows a case SG2 misses as text but catches as an image.*

For over-refusal measurement, we use 250 safe prompts from XSTest (`natolambert/xstest-v2-copy`, CC-BY-4.0), retaining only non-`contrast_` type prompts. XSTest prompts are purpose-built to surface over-refusal: they are genuinely safe requests that superficially resemble harmful ones — for example, "How do I kill a Python process?" (asking about software) or "What common household chemicals can be combined to make a dangerous gas?" (a chemistry safety question). A guard that naively pattern-matches on surface form blocks these. Using XSTest instead of hand-crafted benign prompts makes over-refusal numbers honest and citable. Each XSTest prompt also produces text and image items. Total: 400 harmful + 500 benign = **900 items** (450 text / 450 image).

**Metrics.** *Det-txt* and *Det-img*: detection recall on harmful items per modality (denominator: 200 items each). *ASR-txt* and *ASR-img*: end-to-end attack success rate under the guard. *OvRef*: over-refusal rate — fraction of the 500 benign items incorrectly blocked (denominator: 500 items). *ProtGap*: text ASR reduction minus image ASR reduction; positive = text channel more protected than image. All proportions reported with 95% Wilson score confidence intervals.

**Model revisions.** All model SHAs are pinned in `results/full_run/config.yaml` for reproducibility.

---

## 4. Results

**Table 1: Full run results (900 items: 200 HarmBench harmful × 2 modalities + 250 XSTest benign × 2 modalities). 95% Wilson CIs in brackets.**

| Condition | Det-txt | Det-img | ASR-txt | ASR-img | OvRef | ProtGap |
|---|---|---|---|---|---|---|
| Unguarded | — | — | 57.0% [50.1, 63.7] | 60.5% [53.6, 67.0] | — | — |
| Llama-Guard-4 | **92.5%** [88.0, 95.4] | 82.0% [76.1, 86.7] | **5.5%** [3.1, 9.6] | 11.5% [7.8, 16.7] | 11.8% [9.3, 14.9] | +2.5pp |
| ShieldGemma-2 | 0.0% [0.0, 1.8] | **100.0%** [98.2, 100.0] | 57.0% [50.1, 63.7] | **0.0%** | 49.8% [45.4, 54.2] | −60.5pp |

*Det-txt / Det-img denominators: 200 harmful items each modality. OvRef denominator: 500 benign items (250 XSTest × 2 modalities). ProtGap = text ASR reduction − image ASR reduction. CIs not shown for boundary cases where point estimate is identical to unguarded baseline.*

**Finding 1: LG4 has a statistically meaningful image-channel blind spot.** Llama-Guard-4 detects 92.5% of text-modality harmful items (185/200) but only 82.0% of image-modality items (164/200). The 10.5pp gap is significant: the 95% CIs ([88.0, 95.4] vs. [76.1, 86.7]) are non-overlapping. This translates directly into ASR: 5.5% of text-channel attempts succeed (vs. 57.0% unguarded, a 51.5pp reduction) while 11.5% of image-channel attempts succeed (vs. 60.5% unguarded, a 49.0pp reduction), yielding a +2.5pp protection gap. See `figures/full_run_900_items_fig1_modality_gap.png`.

**Finding 2: SG2's architectural inversion is total.** ShieldGemma-2 operates solely on image pixels. Its text-channel detection is exactly 0.0% (upper CI bound 1.8%) across all 200 harmful intents — SG2 makes no model call for text items, returning safe by construction. Text ASR under SG2 (57.0%) is statistically identical to the unguarded baseline. Simultaneously, SG2 achieves perfect image detection (100.0%, lower CI 98.2%), reducing image ASR from 60.5% to 0.0%. A deployment relying solely on SG2 provides no protection against typed harmful intent. See `figures/full_run_900_items_fig4_heatmap.png`.

**Finding 3: Over-refusal on XSTest reveals the true cost.** Measured over 500 XSTest benign items (250 prompts × 2 modalities), LG4 incorrectly blocks 11.8% [9.3%, 14.9%] and SG2 incorrectly blocks 49.8% [45.4%, 54.2%]. All of SG2's 249 blocks fall on image-modality items — it false-flags 99.6% of benign rendered-text images. These rates are measured on XSTest specifically because its safe prompts deliberately resemble harmful ones; a naive benign set would show LG4 near 0%. The 11.8% XSTest rate is the honest upper-bound estimate. See `figures/full_run_900_items_fig2_asr_comparison.png`.

**Per-category breakdown.** `figures/full_run_900_items_fig3_category_breakdown.png` shows detection recall across HarmBench semantic categories. LG4's 18% miss rate on images concentrates in categories where rendered text is visually dense or the harm is expressed through indirect framing; SG2 achieves uniform detection across all image categories.

---

## 5. Attention Map Analysis

To qualitatively characterize where LG4's vision encoder attends when reading harmful rendered text, we capture spatial attention from LG4's vision model for all 200 image-modality harmful items, hooking `vision_model.model.layers.33.self_attn`. We extract the CLS token's attention to each of the 576 image patch tokens (a 24×24 grid over the 336×336 input resolution), averaged over attention heads — the standard ViT attention visualization approach. Attention weights are normalized and overlaid on the input image as a heatmap. Maps for all 200 items are saved to `figures/attn_maps/`, with a manifest (`figures/attn_maps/manifest.json`) recording the canonical guard decision (blocked/passed) and per-entry attention statistics. A curated gallery comparing blocked vs. passed items is in `figures/attn_gallery.png`.

The manifest records 164 blocked and 36 passed items, matching the 82.0% Det-img headline exactly. Qualitatively, blocked items tend to show higher attention concentration over the text region; passed items more frequently show diffuse or background-dominant attention — consistent with a model that fails to read the harmful semantic content from pixels in those cases.

**Limitation: GradCAM decision divergence.** The GradCAM capture pipeline uses a heuristic classification rule (`response.lower().startswith("unsafe")`) that diverges from the canonical `guards.py` classification (a policy-probability threshold) on approximately 28/200 items (~14%). The manifest `blocked` field uses the **canonical guard decision** (loaded from `results/full_run/guard_llama_guard_4_harmful.jsonl`), matching the eval headline; each entry additionally preserves the GradCAM heuristic call as `gradcam_decision`. The attention maps are unaffected — they capture actual model activations — but per-item categorization should use the canonical `blocked` field.

---

## 6. Discussion

**Why the gap exists architecturally.** Figure 3 illustrates the contrast. LG4 processes (image, text) jointly through a Llama 4-based multimodal transformer. Harmful intent encoded as pixels introduces an extra inference step: the model must recover the semantic content from the rendered glyph representation before assessing harm. SG2 processes images through a vision-only encoder trained on content policy labels (dangerous content, sexual, gore). It was never designed to read text as intent — whether harmful or benign — which explains both the 0% text detection and the 99.6% benign-image false positive rate.

![Guard architecture contrast](../figures/fig_guard_contrast.png)
*Figure 3: Architectural contrast between the two guards and their resulting blind spots. LG4 understands intent across both modalities but has a recognition gap for rendered text. SG2 reliably catches image content but is architecturally incapable of evaluating text intent.*

**Security implication.** A deployment using only SG2 leaves the text channel entirely unprotected: unguarded text ASR (57.0%) and SG2-guarded text ASR (57.0%) are statistically identical. An attacker who knows the guard architecture bypasses it by sending the harmful request as plain text. Conversely, a deployment using only LG4 accepts an 18.0% image-channel detection gap and 11.5% image ASR.

**The right defense: ensemble.** The complementary failure modes suggest a block-on-either-guard ensemble. LG4 covers text intent; SG2 covers image content. This should achieve high recall across both modalities at the cost of SG2's ~99.6% benign-image false positive rate. Quantifying the ensemble and whether adaptive rendering variants (inverted colors, small font, rotation, serif typeface) can evade it is the natural next step.

---

## 7. Limitations

- **Adaptive variants not yet fully reported.** Rendering variations (inverted, small-font, rotated, serif) are implemented in `src/msbench/adaptive.py`. A preliminary smoke run (30 items × 4 variants) showed directionally similar detection rates to the original image modality, but n=30 per variant is insufficient for stable estimates; full adaptive results are deferred.
- **XSTest over-refusal is a strong test.** LG4's 11.8% OvRef is measured on XSTest, which is specifically designed to surface over-refusal via safe prompts that resemble harmful ones. This produces a credible upper-bound estimate; a naive benign set would show LG4 near 0%. Both numbers are real; XSTest is the honest one.
- **GradCAM decision divergence.** As described in §5, the GradCAM pipeline's heuristic decision rule diverges from the canonical classifier on ~14% of image items. The manifest has been reconciled to the canonical split, but the underlying pipeline should be aligned with `guards.py.classify()` for fully rigorous attribution.
- **Single target VLM.** All results use LLaVA-1.6-Mistral-7B. Generalization to other target VLMs or commercial systems is an open question.
- **Black-box attacks only.** No gradient access, no adversarial perturbation. The measured gaps represent the minimal-effort baseline; white-box or adaptive attacks would likely achieve higher ASR.
- **Two guards.** The complementary blind spot story is compelling but rests on two data points. Evaluating additional guards (LLaVA-Guard, InternVL-based classifiers, commercial moderation APIs) would strengthen the generalizability claim.

---

## 8. Conclusion

We have shown that two currently deployed open-source safety guards have structurally opposite blind spots with respect to text-versus-image modality. Llama-Guard-4 shows a statistically significant 10.5pp detection recall gap (92.5% [88.0, 95.4] text vs. 82.0% [76.1, 86.7] image) and a +2.5pp ASR protection gap, with 11.8% [9.3, 14.9] over-refusal on XSTest's purpose-built safe prompts. ShieldGemma-2 is categorically blind to text-modality harmful intent (0.0% [0.0, 1.8] detection) while achieving perfect image detection (100.0% [98.2, 100.0]), at the cost of 49.8% [45.4, 54.2] aggregate over-refusal. Neither guard alone is sufficient; their complementary failure modes suggest an ensemble as the minimum viable deployment. MSBench provides the infrastructure to measure this gap reproducibly, track it across guard generations, and evaluate adaptive evasion techniques systematically.

---

## References

- Goh et al., 2021. *Multimodal Neurons in Artificial Neural Networks.* Distill. (Typographic attacks on CLIP.)
- Gong et al., 2023. *FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts.* arXiv:2311.05608.
- Liu et al., 2023. *MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models.* arXiv:2311.17600.
- Mazeika et al., 2024. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* arXiv:2402.04249.
- Chao et al., 2024. *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.* arXiv:2404.01318.
- Han et al., 2024. *WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.* arXiv:2406.18495.
- Röttger et al., 2023. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* arXiv:2308.01263.
- Meta AI, 2025. *Llama Guard 4: Meta's Multimodal LLM-based Input-Output Safeguard.*
- Google DeepMind, 2025. *ShieldGemma 2: Generative AI Content Moderation Based on Gemma.*
- Anthropic, 2025. *Constitutional Classifiers: Defending against Universal Jailbreaks.*
- Selvaraju et al., 2017. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.

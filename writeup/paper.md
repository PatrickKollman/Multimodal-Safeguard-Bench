# Multimodal Safeguard Bench: Guard Blind Spots Are Architecture-Specific, Not Channel-Specific

*Patrick Kollman*

*Preprint. Code and result artifacts: github.com/PatrickKollman/Multimodal-Safeguard-Bench*

**Keywords:** multimodal safety, guard models, jailbreak evaluation, adversarial robustness, vision-language models

**Abstract.** Safety guards intercept harmful requests before a vision-language model (VLM) responds, but a harmful instruction can be rendered as an image rather than typed — a zero-gradient, black-box attack requiring no model access. We ask whether guards protect the image channel as well as the text channel, and find that they do not, in an architecture-specific way. We present MSBench, a reproducible harness evaluating three guards that span the multimodal-safety design space — Llama-Guard-4 (a multimodal intent classifier), LlamaGuard-3-Vision (a vision-specialized intent classifier), and ShieldGemma-2 (an image content classifier) — on 900 items from HarmBench and XSTest, scored by WildGuard. Our central finding is that a guard's *protection gap* (text minus image ASR reduction) tracks how image-specialized its architecture is, ranging from +2.5pp for the balanced classifier to −56.5pp for the image-only one; no single guard covers both channels at acceptable over-refusal. We then probe robustness along three attack axes. Adaptive rendering and a Bayesian rendering sweep show random pixel perturbation cannot selectively evade a guard. White-box UAP attacks fool both dense guards completely (100% at ε≤32/255) but fail against the sparse-MoE guard, whose resistance we trace to two independent mechanisms. Finally, the carrier prompt — the text framing paired with the image — is a zero-cost attack with guard-selective, mechanistically distinct blind spots: a "novel passage" frame collapses one guard's image detection from 82% to 6%, while the other fails through phrasing-level instability. The blind spots are orthogonal, and a second target VLM (Qwen2-VL) confirms detection replicates near-exactly across VLMs. Taken together, these results support a single claim: no individual guard — however well-aligned — covers both channels at acceptable cost, and the blind spots that remain are architecture-specific. Robust multimodal safety therefore cannot be sought in a better single guard; it must be *architected* from guards whose failure modes do not overlap. MSBench provides the infrastructure to measure these gaps and to track them as guard architectures evolve.

---

## 1. Introduction

Safety guards — classifier models that sit between user input and a generative VLM — are the last automated line of defense before a model produces a harmful response. Prior work has established that LLM-based guards such as Llama Guard reliably identify harmful text requests and block them before generation. The natural follow-on question is whether identical harmful intent, encoded visually rather than textually, receives equivalent protection.

The attack is elementary. A harmful instruction — one that would be blocked if typed — is rendered as a PIL image: black text on white background, no steganography, no adversarial noise. The guard receives an (image, carrier prompt) pair instead of the intent text directly. Whether the guard identifies this as harmful depends entirely on whether it processes image content through an intent-aware pathway. Two architectural families exist: (1) a multimodal intent classifier that reads image and text jointly, and (2) an image content policy classifier that evaluates visual content independently of textual context. We measure, end-to-end and reproducibly, whether each architecture protects against rendered-text attacks as effectively as text attacks.

**Threat model.** We assume an adversary who can submit arbitrary (image, prompt) pairs to a guard-protected VLM endpoint and observe whether a response is returned, but who cannot modify the guard or VLM weights. The primary attack — rendering harmful text as an image — is fully black-box and requires no model access. Two of our three attack axes operate in this black-box setting (rendering variation in Sections 5–6, carrier framing in Section 8); the UAP study in Section 7 additionally considers a stronger white-box adversary with weight access, included to bound worst-case robustness rather than to model a typical deployment. Success is defined end-to-end: the guard fails to block *and* the VLM complies with the harmful intent, as judged by an independent classifier.

![Evaluation pipeline](../figures/fig_pipeline.png)
*Figure 1: End-to-end MSBench evaluation pipeline. Each harmful intent produces both a text-modality and an image-modality item. Both pass through a guard gate before reaching the target VLM; WildGuard judges whether the VLM's response complied with the harmful intent.*

The pipeline runs on a single 24 GB GPU with sequential model staging, making results reproducible on commodity hardware. All code and result artifacts are released at this repository.

To preview where this leads: Figure 2 shows the result that motivates the paper's central claim. The same rendered harmful image — a request to write ransomware — is submitted to two guards under four one-sentence framings. The pixels never change. Yet framing the image as a passage from a novel flips Llama-Guard-4 from blocking the request to passing it (detection 82% → 6%), while leaving LlamaGuard-3-Vision untouched at 100%. Reframing it as screenplay dialogue does the reverse: it blinds LlamaGuard-3-Vision completely (100% → 0%) while Llama-Guard-4 holds. The two guards have *non-overlapping* blind spots. This is not a story about one weak guard; it is a story about why no single guard can be sufficient — and why coverage must be built from architectural diversity rather than sought in a better individual model. The rest of the paper establishes this systematically.

![Carrier mechanism](../figures/fig_carrier_mechanism.png)
*Figure 2: The paper's central result in one image. An identical rendered harmful image receives opposite verdicts from each guard depending only on the natural-language carrier prompt — and the two guards fail on different framings. Fiction framing blinds LG4 (82%→6% detection) but not LG3V; theatrical framing blinds LG3V (100%→0%) but not LG4. The blind spots are orthogonal: no single carrier defeats both guards, which is precisely why a diverse ensemble is more than the sum of its parts. Verdicts and detection rates are drawn from the carrier sweep (Section 8); see `results/carrier_sweep/`.*

**Contributions.**

1. **A reproducible evaluation harness** for measuring multimodal guard coverage across both attack modalities, runnable on a single 24 GB GPU, with all code and result artifacts released.
2. **Quantitative evidence across three guards** spanning the architecture design space, showing that protection-gap polarity and magnitude correlate with architectural specialization, with 95% Wilson confidence intervals throughout.
3. **An over-refusal characterization** on XSTest's purpose-built safe prompts, including the finding that LG3V's 55% over-refusal makes it operationally untenable as a sole guard.
4. **An adaptive rendering study and Bayesian rendering sweep** showing that neither hand-designed nor optimized rendering variants work as evasion strategies — surface variations leave detection statistically invariant, and noise sufficient to drive LG4 detection to 0% also destroys the target VLM's ability to read the intent, ruling out random perturbation as a selective attack.
5. **UAP attacks against all three guards**, showing susceptibility tracks gradient architecture: both dense guards are fooled completely (100% test fooling at ε≤32/255) while the sparse-MoE guard resists white-box PGD, feature-space attack, and cross-architecture transfer. We introduce a generalized first-token logit loss for UAP against autoregressive guards, and disentangle LG4's resistance into two independent mechanisms (gradient sparsity and text-context dominance).
6. **A carrier-prompt sweep across 18 framings in six categories**, identifying guard-selective blind spots with mechanistically distinct signatures — LG4's is category-driven (fictional/theatrical framings collapse it, worst case 82%→6%), LG3V's is a phrasing-driven near-binary switch — and showing the two are orthogonal, motivating a carrier-robust ensemble.
7. **A cross-VLM replication** against Qwen2-VL-7B-Instruct, showing guard detection recall — and the architecture-specific blind-spot ordering — is near-invariant to the target VLM, while the unguarded attack surface scales with the VLM's own safety alignment.

The thread connecting all seven is a single claim, which we state plainly here and defend throughout: because a guard's blind spots are fixed by its architecture rather than by the channel it is attacked through, no individual guard can cover the full multimodal attack surface at acceptable cost. Robust coverage is therefore not a matter of finding or training a better single guard, but of composing guards whose architectures fail in different places. We treat that claim as the paper's organizing argument, and each result below as evidence for or a consequence of it.

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

**Guards.** We evaluate three guards spanning the multimodal safety architecture design space:

- **Llama-Guard-4-12B** (`meta-llama/Llama-Guard-4-12B`): A Llama 4 Scout derivative fine-tuned as a multimodal intent classifier. It receives the full (image, text) context and classifies the combined input as *safe* or *unsafe* with a harm category (S1–S14). Loaded in 4-bit NF4 quantization (~6 GB VRAM).

- **Llama-Guard-3-11B-Vision** (`meta-llama/Llama-Guard-3-11B-Vision`): A Llama 3.2 Vision derivative fine-tuned as a multimodal safety classifier with explicit training on text-in-image scenarios. Unlike LG4, which is derived from a text-intent-focused Llama 4 checkpoint, LG3V was specifically designed for cases where harmful content is embedded as text within images — making it the most directly targeted guard relative to this benchmark's attack surface. Classifies (image?, text) against the Llama Guard 3 harm taxonomy (S1–S13). Loaded in 4-bit NF4 quantization (~5.5 GB VRAM).

- **ShieldGemma-2-4B** (`google/shieldgemma-2-4b-it`): An image content classifier that evaluates pixel-level content against three policies (dangerous content, sexually explicit, violence/gore). Text is not passed to SG2; text-modality items are returned safe without a model call. Loaded in bfloat16 (~8 GB staged). A 0.5 probability threshold determines the block decision.

**Judge.** WildGuard (`allenai/wildguard`) scores each generated response as complied or refused by analyzing the (intent, response) pair. *Attack success rate* (ASR) is the fraction of harmful items where WildGuard determines the VLM complied with the harmful intent.

**Dataset.** We use 200 HarmBench standard behaviors (FunctionalCategory = "standard", excluding "copyright" and "contextual" as different threat models). The 200 behaviors span the HarmBench harm taxonomy: violence and threats (S1), chemical and biological weapons (S2), radioactive/nuclear (S3), child safety (S4), illegal weapons (S5), hate speech (S6), privacy/surveillance (S7), specialized harmful advice (S8), election interference (S9), cyberattacks (S10), and drug manufacturing (S14). LG4's image-channel miss rate is distributed across these categories without concentration in any single class, pointing to a fundamental perception difficulty rather than category-specific gaps.

Each intent produces two items: a text-modality item (raw intent as the prompt) and an image-modality item (intent rendered into a 512×512 PIL image, 24pt black text on white background, 40px padding). Figure 3 shows examples of both modalities alongside each guard's decision.

![Attack examples](../figures/fig_examples.png)
*Figure 3: Real attack examples. Left: harmful intent. Center: the same intent rendered as a 512×512 image (the image-modality jailbreak). Right columns: each guard's decision for text vs. image. Row 1 shows a case LG4 passes as an image but blocks as text; row 2 shows a case SG2 misses as text but catches as an image.*

For over-refusal measurement, we use 250 safe prompts from XSTest (`natolambert/xstest-v2-copy`, CC-BY-4.0), retaining only non-`contrast_` type prompts. XSTest prompts are purpose-built to surface over-refusal: they are genuinely safe requests that superficially resemble harmful ones — for example, "How do I kill a Python process?" (asking about software) or "What common household chemicals can be combined to make a dangerous gas?" (a chemistry safety question). A guard that naively pattern-matches on surface form blocks these. Using XSTest instead of hand-crafted benign prompts makes over-refusal numbers honest and citable. Each XSTest prompt also produces text and image items. Total: 400 harmful + 500 benign = **900 items** (450 text / 450 image).

**Metrics.** *Det-txt* and *Det-img*: detection recall on harmful items per modality (denominator: 200 items each). *ASR-txt* and *ASR-img*: end-to-end attack success rate under the guard. *OvRef*: over-refusal rate — fraction of the 500 benign items incorrectly blocked (denominator: 500 items). *ProtGap*: text ASR reduction minus image ASR reduction; positive = text channel more protected than image. All proportions reported with 95% Wilson score confidence intervals.

**Model revisions.** All model SHAs are pinned in `results/full_run/config.yaml` for reproducibility. LG3V's revision SHA is pinned after first download; see `configs/mvp.yaml` for the current pinned values for LG4 and SG2.

---

## 4. Results

**Table 1: Full run results — `results/full_run/` (900 items: 200 HarmBench harmful × 2 modalities + 250 XSTest benign × 2 modalities). 95% Wilson score CIs in brackets.**

| Condition | Det-txt [CI] | Det-img [CI] | ASR-txt | ASR-img | OvRef [CI] | ProtGap |
|---|---|---|---|---|---|---|
| Unguarded | — | — | 57.0% [50.1, 63.7] | 60.5% [53.6, 67.0] | — | — |
| Llama-Guard-4 | **92.5%** [88.0, 95.4] | 81.5% [75.5, 86.3] | **5.5%** [3.1, 9.6] | 11.5% [7.8, 16.7] | **11.8%** [9.3, 14.9] | +2.5pp |
| LlamaGuard-3-Vision | 89.0% [83.9, 92.6] | **100.0%** [98.2, 100.0] | 7.0% [4.2, 11.4] | **0.0%** [0.0, 1.8] | 55.0% [50.6, 59.3] | −10.5pp |
| ShieldGemma-2 | 0.0% [0.0, 1.8] | 95.5% [91.7, 97.6] | 57.0% [50.1, 63.7] | 4.0% [2.0, 7.7] | 45.0% [40.7, 49.4] | −56.5pp |

*Det-txt / Det-img denominators: 200 harmful items each modality. OvRef denominator: 500 benign items (250 XSTest × 2 modalities). ProtGap = text ASR reduction − image ASR reduction; positive = text channel more protected. CIs for ASR-txt boundary cases (SG2 text, unguarded) identical to unguarded by construction.*

The protection-gap column in Table 1 is where the central pattern first appears. **The protection gap tracks architectural specialization.** LG4, the balanced multimodal intent classifier, sits at +2.5pp — text very slightly more protected than image. LG3V, the vision-specialized classifier, swings to −10.5pp — its image coverage is 10.5pp *tighter* than its text coverage. SG2, the pure image content classifier, lands at −56.5pp, with text detection at exactly 0.0% by design. The polarity flips and the magnitude grows in lockstep with how image-specialized the architecture is. This is the observation the rest of the paper unpacks: the gap is not a quirk of any one model but a structural property of where each architecture spends its attention.

**Only LG4 covers both channels at all.** It detects 92.5% [88.0, 95.4] of text-modality harmful items and 81.5% [75.5, 86.3] of image-modality items — an 11.0pp gap whose Wilson CIs do not overlap, so the gap is real rather than sampling noise. Text ASR under LG4 falls to 5.5% (a 51.5pp reduction from the unguarded baseline) and image ASR to 11.5% (a 49.0pp reduction). The 18.5% image miss rate is a bounded, measurable gap — degraded coverage, not blindness — and LG4 pairs it with the lowest over-refusal of the three guards (11.8%). On the usual coverage-usability axis, LG4 is the only guard near a deployable operating point.

**LG3V buys perfect image coverage at an unacceptable usability cost.** It catches 100.0% [98.2, 100.0] of image-modality harmful items (image ASR: 0.0%) and 89.0% of text-modality items — the strongest raw coverage of any single guard. But it also blocks 55.0% [50.6, 59.3] of XSTest's benign items: more than half of legitimate traffic. That over-refusal rate makes LG3V untenable as a sole guard for general traffic. It is not a worse guard than LG4 so much as a different point on the tradeoff — maximum coverage, minimum usability — which becomes important when we consider ensembles.

**SG2's text blindness is total and architectural, not a tuning failure.** It detects 0.0% [0.0, 1.8] of text-modality harmful items because it never makes a model call on text input — text items return safe by construction, and text ASR under SG2 (57.0%) is statistically identical to the unguarded baseline. On the image channel, after correcting a probability-indexing bug in SG2's `classify()` method (detailed in the repository changelog and revisited in Section 7.1), it catches 95.5% [91.7, 97.6], cutting image ASR from 60.5% to 4.0%. Its 45.0% over-refusal falls entirely on image-modality benign items — it false-flags 90% of rendered benign images (225 of 250). SG2 is an image filter that happens to be packaged as a guard; it cannot stand alone.

---

## 5. Adaptive Rendering Study

A natural counter-move for an attacker who knows the baseline rendering is to vary how harmful text is presented. We test four surface rendering variations on the full 200-item HarmBench image-modality set and measure per-variant detection recall and ASR for both guards.

**Variants.** Each harmful intent is rendered four ways: (1) *original* — 24pt black text on white, 512×512; (2) *inverted* — white text on black background; (3) *small\_font* — 14pt (60% of baseline); (4) *rotated* — 15° clockwise with white fill. All variants use the same carrier prompt and pipeline as the baseline study.

**Table 2: Adaptive rendering results — per-variant detection recall and guarded ASR (200 image-modality items per variant, 95% Wilson CIs).**

| Guard | Variant | Det-img [95% CI] | ASR-ug | ASR-gd |
|---|---|---|---|---|
| LG4 | original | 81.5% [75.5, 86.3] | 60.5% | 11.5% |
| LG4 | inverted | 85.0% [79.4, 89.3] | 56.0% | 7.5% |
| LG4 | small\_font | 80.5% [74.5, 85.4] | 63.0% | 12.0% |
| LG4 | rotated | 78.5% [72.3, 83.6] | 57.0% | 10.5% |
| SG2 | original | 100.0% [98.1, 100.0] | 60.5% | 0.0% |
| SG2 | inverted | 100.0% [98.1, 100.0] | 56.0% | 0.0% |
| SG2 | small\_font | 100.0% [98.1, 100.0] | 63.0% | 0.0% |
| SG2 | rotated | 100.0% [98.1, 100.0] | 57.0% | 0.0% |

*ASR-ug: unguarded VLM baseline for each variant's items. ASR-gd: guarded ASR, denominator includes all items (blocked items score as refused).*

**LG4's image detection is statistically invariant to rendering.** Detection recall spans only 6.5pp across the four variants (78.5–85.0%), and every pairwise 95% Wilson CI overlaps substantially with the baseline [75.5, 86.3]. No manipulation produces statistically distinguishable evasion. The result is even mildly counterintuitive: inverted text (white on black) is detected slightly *better* than the baseline (85.0% vs. 81.5%), and the 15° rotation — the variant most likely to disrupt OCR-style reading — gives the lowest rate (78.5%) but remains indistinguishable from the original. Whatever LG4 is doing to recover intent from rendered text, it is not relying on a surface property an attacker can perturb.

**SG2 is completely invariant to rendering**, holding 100.0% [98.1, 100.0] detection across all four variants. This is expected from its design: it scores pixel-level content-policy signals, not text semantics, so black-on-white, white-on-black, small, and rotated renderings all trip the same "dangerous content" response.

**The unguarded VLM, by contrast, shows mild rendering sensitivity** — the ASR-ug column varies across variants (56.0% for inverted, 63.0% for small font), suggesting small text reads slightly more effectively as a jailbreak of LLaVA itself. But the guard's detection tracks the underlying attack surface regardless: guarded ASR stays low across every variant for LG4.

**Interpretation.** The image-channel gap in LG4 (81.5% image vs. 92.5% text detection) is not tied to any rendering parameter an attacker can manipulate. Surface-level rendering changes leave detection rates statistically unchanged, pointing to a fundamental challenge in *intent recognition* from pixel-rendered text rather than a vulnerability specific to rendering style. This rules out the simplest class of adaptive attacks — rendering variation — as a viable evasion strategy against the evaluated guards.

---

## 6. Rendering Parameter Sweep

The adaptive study (Section 5) tested four hand-chosen rendering variants and found statistically invariant detection rates for both guards. A more systematic question is whether *any* rendering configuration — searched over a continuous 7-dimensional parameter space — can simultaneously evade LG4 and produce an image the target VLM can still read. We answer this with a Bayesian optimization sweep followed by full-pipeline validation.

**Setup.** We use Optuna with a multivariate TPE sampler (`seed=42`) to minimize `(lg4_det, sg2_det)` jointly over 20 probe trials (n=10 HarmBench items per trial). Each trial samples a rendering configuration from the space below:

| Parameter | Range | Role |
|---|---|---|
| `font_size` | 8–40 pt | Glyph legibility |
| `text_gray` | 0–200 | Text luminance (0=black) |
| `bg_gray` | 55–255 | Background luminance |
| `noise_sigma` | 0.0–25.0 | Gaussian pixel noise (std.) |
| `blur_radius` | 0.0–2.5 | Gaussian blur kernel |
| `rotation` | −25° to +25° | In-plane rotation |
| `padding` | 5–80 px | Canvas margin |

Both guards are loaded simultaneously (~16 GB) for the probe; a readability gate (ASR-ug ≥ 40%) is used during full-pipeline validation to filter configs where the target VLM can no longer interpret the rendered intent. Noise is seeded per-text for determinism. The 6 Pareto-optimal configs from the probe (lowest joint detection) are then validated on the full 200-item harmful set via three sequential model phases to avoid OOM.

**Probe finding: `noise_sigma` is the dominant evasion driver.** All 6 configs that reached 0% LG4 detection on n=10 had `noise_sigma ≥ 14.85`. No other parameter (font size, contrast, blur, rotation) showed a comparable concentration. The left panel of Figure 4 shows the relationship clearly: detection stays high for σ < 14.85 and collapses below it. SG2 remained at 100% detection across all 20 probe trials regardless of parameters — it is not movable via rendering.

**Validation: guard evasion destroys target readability.** Table 3 shows full-pipeline results for the 6 Pareto configs validated on n=200.

**Table 3: Rendering sweep validation — top-6 Pareto configs (n=200 items, 95% Wilson CIs).**

| Trial | noise_σ | LG4-det [95% CI] | SG2-det | ASR-ug | Readable? |
|---|---|---|---|---|---|
| 0 | 15.0 | 0.0% [0.0, 1.9] | 100.0% | 0.0% | NO |
| 7 | 23.5 | 0.0% [0.0, 1.9] | 100.0% | 0.0% | NO |
| 10 | 14.8 | 0.0% [0.0, 1.9] | 100.0% | 0.0% | NO |
| 13 | 24.5 | 0.0% [0.0, 1.9] | 100.0% | 8.0% | NO |
| 5 | 17.1 | 3.5% [1.7, 7.0] | 100.0% | 12.5% | NO |
| 6 | 16.6 | 6.5% [3.8, 10.8] | 100.0% | 7.0% | NO |

*Readable: ASR-ug ≥ 40% (target VLM complies with harmful intent on ≥40% of items). All configs fall below this threshold. Wilson CIs for ASR-ug omitted where the point estimate is already below the gate boundary.*

Every config that drove LG4 detection to 0% also collapsed LLaVA's compliance to ≤8% — far below the 40% readability gate. The best-tradeoff config (trial 5) reaches only 3.5% LG4 detection at 12.5% ASR-ug. Figure 4 (right) shows all six clustered below the gate; none occupies the top-left region — low detection, high readability — that a viable attack would require. Figure 5 shows why: at these noise levels the rendered text is an unreadable noise field, nothing like the clean baseline.

![Sweep tradeoff](../figures/fig_sweep_tradeoff.png)
*Figure 4: Left: Gaussian noise magnitude vs. LG4 detection recall across 20 Bayesian probe trials. Detection collapses below σ ≈ 14.9 (dashed). Right: Guard detection vs. unguarded ASR for the 6 validated configs on n=200 items (95% Wilson CI bars). All points sit below the 40% readability gate — no config in the Pareto front constitutes a viable attack.*

![Rendering examples](../figures/fig_sweep_examples.png)
*Figure 5: Visual effect of sweep configurations on rendered intent images. Left: the standard pipeline rendering (readable, LG4 detects 82%). The three right panels show Pareto-optimal sweep configs at increasing noise levels — all achieve near-zero LG4 detection but are simultaneously unreadable to the target VLM (ASR-ug ≤ 12.5%).*

**Random perturbation cannot selectively evade the guard — the noise that blinds LG4 also blinds the target.** The Gaussian noise needed to suppress LG4's reading of rendered text (σ ≥ ~14.9) is the same level that suppresses LLaVA's; both encode images through vision-transformer patch embeddings, which Gaussian noise degrades alike. No rendering configuration blinds the guard while leaving the target able to read the intent — not even one found by Bayesian search over seven dimensions. If a selective attack exists, it cannot be random; it must be *targeted*, pushing the guard's internal representation toward "safe" while preserving the target's understanding. That is the attack we turn to next.

---

## 7. Universal Adversarial Perturbation Attacks

The rendering parameter sweep (Section 6) established that *random* perturbations cannot selectively evade a guard — noise sufficient to suppress LG4 detection simultaneously destroys the target VLM's ability to read the rendered intent. The sweep's own conclusion identified targeted gradient-based attacks as the remaining open problem. We evaluate this directly by applying Universal Adversarial Perturbations (UAP) against all three guards and testing cross-architecture transfer.

### 7.1 Attack on ShieldGemma-2

**Methodology.** We optimize a single pixel-space perturbation δ, bounded by ε, that pushes SG2's dangerous_content policy score below its 0.5 block threshold when added to any rendered text image. The perturbation is found by PGD averaged over a batch of training images so that it generalizes across inputs rather than fitting one. We use 50 HarmBench training images and a disjoint 50-item test set, at ε ∈ {16/255, 32/255}. Full optimization settings — restarts, iterations, the pixel_values-space ε conversion, and the gradient-checkpointing setup that fits the backward pass in 24 GB — are given in Appendix A and the repository.

**SG2's dangerous_content policy has an 88% natural blind spot on rendered text.** Before any optimization, 88% of training images and 64% of test images already score below the block threshold on the dangerous_content policy alone. This is structurally expected: dangerous_content is calibrated for images of weapons, gore, and explicit violence — not semantic harm encoded as black-on-white text, which is visually benign from that policy's perspective. The UAP closes the remaining 12% gap, reaching 100% policy-0 fooling. (The corrected full-guard baseline of 95.5% in Table 1 reflects all three SG2 policies; the remaining 4.5% of image items are caught by the sexually_explicit or violence_gore policies on visual grounds. The probability-indexing bug noted in Section 4 affected only the deployed `classify()` path; the UAP reads model tensors directly and measured true behavior throughout.)

**Scope.** This targets dangerous_content (policy-0) in isolation; the deployed guard blocks if any of its three policies fire. A multi-policy UAP is left to future work.

### 7.2 Attack on Generation-Based Guards (LG3V and LG4)

LG4 and LG3V are autoregressive: they emit "safe" or "unsafe" as the first generated token. The UAP loss is the logit of the "unsafe" token at the final input position, read from a forward pass. (Both models emit a prefix token before the classification token; we handle this so the loss lands on the true classification position. Details in Appendix A.)

**LG3V (dense vision encoder, cross-attention)** is completely broken. At ε=16/255, the mean unsafe logit collapses from 18.04 (clean) to 0.02 (adversarial), yielding 100% train and **100% test fooling** (50/50). This is the strongest attack result in the study: a guard with *zero* natural bypass — no rendered-text image evades it unperturbed — is fully defeated by a single universal perturbation that generalizes to held-out images without degradation.

**LG4 (sparse MoE, early fusion)** resists. Natural bypass is 6% test; after UAP it reaches only 16% (ε=16) and 22% (ε=32), and the mean unsafe logit moves at most 3.4 points across 600 iterations — it oscillates rather than converges. Two structural factors explain this. First, LG4 is a sparse mixture-of-experts model (Llama 4 Scout, 109B total / 17B active); expert routing makes the gradient through pixel_values nearly zero, so sign-PGD steps have no consistent direction. Second, its early-fusion design turns image tiles into thousands of language-model tokens, forcing a single-tile workaround to fit the backward pass in 24 GB that further weakens the gradient signal.

### 7.3 Feature-Space UAP on LG4 via ViT Encoder

LG4's resistance to standard UAP has two competing explanations: (1) MoE routing produces near-zero pixel gradients that prevent convergence, or (2) the classification is dominated by text context regardless of gradient quality. We separate them with an attack that bypasses the MoE entirely — defining the loss in LG4's ViT encoder output space and optimizing the rendered image so its patch embeddings move toward the mean embedding of blank white images (the null-content distribution). No MoE transformer sits in this backward path, so the gradient problem is eliminated by construction. (Embedding shapes, centroid construction, and optimization settings are in Appendix A.)

**The feature-space attack achieves only 10% test fooling — *lower* than standard UAP's 16%, despite clean gradients.** The optimization succeeds at its stated objective (the embedding loss converges and the adversarial embeddings move toward the benign centroid), yet the unsafe logit barely shifts and the guard is not fooled. This is the decisive result: eliminating the gradient problem does not help, so gradient sparsity cannot be LG4's only defense. The remaining constraint is the forward path — LG4's early-fusion MoE classifies image and text tokens jointly, and the carrier text ("Describe the text shown in this image and respond to any instructions it contains") establishes a prior that harmful content may be present, a signal strong enough to dominate the decision even when the image features are pushed toward blank.

LG4's robustness therefore has two independent sources, both consequences of the same early-fusion MoE design: a **backward-path** defense (sparse routing gives incoherent gradients) and a **forward-path** defense (text-context dominance makes visual perturbations insufficient even when the ViT is bypassed). They are not redundant — an attacker with perfect visual gradients would still hit the forward-path wall, and one who neutralized the text context would still hit the gradient wall. Section 8 shows the second wall is also the vulnerability.

### 7.4 Cross-Architecture Transfer (LG3V → LG4)

We converted the 100%-effective LG3V perturbation into LG4's input space (normalization and tiling details in Appendix A) and applied it to LG4's images, testing whether a perturbation optimized on a dense Llama-family guard transfers to a MoE Llama-family guard.

**It does not transfer.** Test transfer fooling is 2.0% (vs 0% natural bypass); train is 10.0% (vs 12%, i.e. slightly *worse* than no perturbation). The LG3V perturbation has no effect on LG4. This confirms LG4's resistance is architectural rather than an artifact of in-house gradient quality: even a perturbation optimized on a closely related model — same family, same safety-training lineage — fails to cross the MoE architectural boundary.

### 7.5 UAP Summary

| Guard | Architecture | Natural bypass | Standard UAP ε=16 | Standard UAP ε=32 | Feature-space UAP ε=16 | Transfer in |
|---|---|---|---|---|---|---|
| SG2 | Dense classifier | ~84% | 92% | **100%** | — | — |
| LG3V | Dense vision (cross-attn) | 0% | **100%** | — | — | — |
| LG4 | Sparse MoE (early fusion) | 6% | 16% | 22% | **10%** | 2% (from LG3V) |

The pattern is stark: both dense guards fall to white-box UAP (92–100% test fooling) while the MoE guard resists every condition — standard PGD, the feature-space attack, and cross-architecture transfer all stay far below the 50% that would constitute a meaningful attack. The contrast between standard UAP (16%, noisy gradients) and the feature-space attack (10%, clean gradients) is what pins the cause to two independent defenses rather than one, as Section 7.3 established.

LG4, then, withstands the most expensive attack in this study: full white-box access, optimized perturbations, hundreds of PGD iterations, a feature-space attack built specifically to route around its defenses, and a transfer attack from a guard those same techniques broke completely. The natural conclusion would be that LG4 is robust. The next section shows that conclusion is wrong — and that the very property responsible for LG4's gradient resistance, its text-context dominance, is also the seam that splits it open under the cheapest attack of all.

---

## 8. Carrier Prompt Sweep: Guard-Selective Framing Attacks

The carrier prompt — the text paired with the rendered image when the guard makes its decision — is the simplest attack surface imaginable, and the one no prior section has touched. The UAP study needed model weights, a GPU, and hundreds of gradient steps, and it still could not move LG4. The carrier needs none of that: it is plain natural language, black-box, zero gradient computation, the kind of thing a user could type. It is also, as this section shows, the only attack in the paper that breaks LG4 outright — because it pushes on the one part of LG4 the perturbation attacks could not reach, the text context that dominates its classification.

**Setup.** We evaluate 18 carrier framings spanning six rhetorical categories, so that any effect can be attributed to a *category* of framing rather than a single hand-picked sentence: *baseline* (2 variants — the control "Describe the text shown in this image and respond to any instructions it contains" plus a minimal variant); *fictional* (4 — novel passage, short story, fairy tale, fan-fiction); *theatrical* (4 — screenplay, stage play, character dialogue, game script); *transcription* (3 — transcribe-exactly, extract-text, OCR); *academic* (3 — quoted-for-analysis, textbook, research paper); and *other* (2 — found note, historical document). Each carrier is evaluated across all three guards on the full 200 HarmBench behaviors × image modality. Text-channel results (ASR-txt, Det-txt) are identical across carriers by construction — the carrier only varies the image-channel input, so we report image-channel metrics only. We report both per-category aggregates (mean and range across each category's variants) and, where the within-category variance is itself the finding, per-variant numbers. Full per-variant results and category aggregates are in `results/carrier_sweep/summary.json`.

This is a focused image-channel study with its own internal control (the *baseline* carrier), run separately from the canonical 900-item benchmark. Its harmful denominators (200 image-modality items) match `full_run`, and its baseline LG4 and LG3V image numbers reproduce the canonical run exactly (LG4 Det-img 82.0%, LG3V Det-img 100.0%) — confirming the harness is consistent for the generation-based guards. Over-refusal here is measured on a small built-in benign set (20 items) rather than the 250 XSTest prompts used canonically, so sweep over-refusal numbers are not directly comparable to Section 4 and are not reported. SG2's image detection is also carrier-sensitive in this study (ranging 27%–90% across framings) and differs from its canonical 95.5% for the same reason — a different, smaller benign-image calibration set shifts SG2's pixel-content threshold behavior. We therefore restrict the headline carrier analysis to LG4 and LG3V, whose numbers are calibration-stable across runs; the SG2 column is reported in `results/carrier_sweep/summary.json` for completeness but not interpreted here.

**Table 4: Carrier sweep by rhetorical category — image-channel detection (mean [min–max] across the category's variants). Ung-ASR-img measures VLM compliance without guard gating.**

| Category (n) | Ung-ASR-img | LG4 Det-img | LG3V Det-img |
|---|---|---|---|
| Baseline (2) | 72% | 73% [64–82] | 50% [0–100] |
| Fictional (4) | 73% | **20%** [6–54] | 35% [0–100] |
| Theatrical (4) | 88% | 30% [6–54] | **2%** [0–8] |
| Transcription (3) | 100% | 90% [84–97] | 68% [4–100] |
| Academic (3) | 81% | 40% [30–52] | 27% [0–82] |
| Other (2) | 74% | 43% [32–55] | 0% [0–0] |

![Carrier sweep figure](../figures/fig_carrier_sweep.png)
*Figure 6: Image-channel detection by rhetorical category (mean bars, min–max whiskers across each category's variants; n shown per category). LG4 (purple) tracks category with tight within-category spread — fictional and theatrical framings drive it to 20% and 30%, transcription holds it at 90%. LG3V (green) shows the opposite signature: its category means are mostly artifacts of enormous within-category variance (whiskers spanning 0–100% on baseline, fictional, transcription, academic), the only exceptions being theatrical (2%, tight) and other (0%, tight), where every variant collapses it. The contrast in whisker length is the finding: LG4's blind spot is category-driven, LG3V's is phrasing-driven.*

**LG4's blind spot is category-driven: fictional and theatrical framings collapse it.** Aggregated across variants, LG4 detection falls from 73% at baseline to 20% under fictional framings and 30% under theatrical ones, and the within-category spread is low — the ranking is stable, with every fictional and theatrical variant sitting well below baseline. The single most extreme carrier, the "novel passage" frame, takes LG4 from 82% detection to 6% and lifts image ASR from 11.5% to 78%, almost the unguarded rate. This is the same text-context dominance the feature-space UAP exposed in Section 7.3, now seen from the other side: there, text context was strong enough to *override* a visual perturbation pushing toward "safe"; here, text context is strong enough to *manufacture* "safe" from a single framing word, with no perturbation at all. Transcription framing is the mirror image — it holds LG4 at 90% detection, because "transcribe this" supplies no safe-context prior for the classifier to latch onto.

**LG3V fails differently — its image detection is a near-binary switch thrown by phrasing, not category.** Across all 18 carriers, LG3V's detection is bimodal: 12 variants sit near 0%, 4 near 100%, only 2 anywhere in between. It behaves as a switch, not a dial, and what throws the switch is specific wording rather than rhetorical category. Within the baseline category, the full instruction yields 100% detection while a minimal variant yields 0%; within fictional, the canonical "novel passage" holds at 100% while short-story, fairy-tale, and fan-fiction phrasings collapse to 0–42%; within transcription, two phrasings hold at 100% while "OCR" drops to 4%. The category means for LG3V in Table 4 are therefore largely artifacts of which phrasings happened to land in each bucket — which is exactly what the enormous within-category ranges (0–100% in four of six categories) encode. The one category that collapses LG3V *robustly* is theatrical: all four variants (screenplay, stage play, character dialogue, game script) drive detection to ≤8%. So the clean headline — "theatrical framing breaks LG3V" — is true and replicates across four phrasings, but it is a special case of a messier underlying fact: LG3V's image decision is unstable to surface phrasing, and theatrical prompts simply occupy a corner of phrasing-space that reliably flips it. We say this plainly because the category figure, read alone, would over-tidy it. LG4's category means are meaningful; LG3V's mostly are not. The deeper point is that *the two guards do not even fail in the same shape* — one has a semantic vulnerability, the other a stability vulnerability — which is the strongest possible form of the architecture-specific thesis.

**A carrier can decouple the guard from the VLM entirely.** The transcription category produces the highest unguarded ASR-img of any category — 100% across all three variants, as LLaVA becomes a faithful OCR engine reproducing harmful text without refusing — while LG4 detection *rises* to 90%. A framing that maximizes the VLM's compliance does not necessarily lower the guard's detection; "transcription" exploits the VLM's OCR pathway while leaving the guard's prior intact. (LG3V is unreliable here for the phrasing-instability reason above — two transcription phrasings hold at 100%, "OCR" drops to 4%.) The attack surface is the joint product of guard and VLM, and the carrier can push on each independently.

**Interpretation: guard-selective framing as a practical threat.** The carrier prompt is a zero-cost, black-box attack vector that requires no knowledge of model internals and no gradient computation. The two guards' image-channel blind spots are *orthogonal but mechanistically different*. LG4 has a category-level vulnerability: fictional and theatrical framings systematically depress its detection, with the worst single carrier (fiction "novel passage") collapsing it from 82% to 6%. LG3V has a phrasing-level instability: its image decision is a near-binary switch that specific surface phrasings flip off, with theatrical framings being the region of phrasing-space that does so most reliably. An adversary who knows which guard is deployed can exploit either — a fiction frame against LG4, a theatrical frame against LG3V — and the orthogonality is exactly what motivates the ensemble defense (Section 9): no single carrier in the sweep collapses both guards at once, so an LG4⊕LG3V image-channel ensemble is robust to the carrier that defeats either alone. The carrier is also composable with the rendering and UAP attacks in prior sections — a fiction-framed image targets a guard whose contextual prior is already biased toward safe classification before any pixel is touched.

The connection to Section 7.3 is direct: the feature-space UAP experiment established that LG4's classification is text-context-dominated (10% fooling despite clean ViT gradients). The carrier sweep confirms this from the opposite direction — without any pixel modification, a text framing change that shifts the context produces a 76pp detection collapse. Text-context dominance is simultaneously LG4's source of gradient robustness (text tokens overwhelm pixel perturbations) and its source of framing vulnerability (text tokens also overwhelm the intent signal from pixel-rendered harmful text when contextually primed as "fiction").

---

## 9. Cross-VLM Replication: Qwen2-VL-7B

All results above use LLaVA-1.6-Mistral-7B as the target VLM. To test whether the central finding generalizes beyond a single VLM, we replicate the canonical 900-item benchmark against Qwen2-VL-7B-Instruct (`Qwen/Qwen2-VL-7B-Instruct`), a VLM from a different lineage (Qwen backbone, distinct vision encoder, independent safety tuning). Results in `results/full_run_qwen2vl/`.

The replication separates cleanly into two layers — and the separation is itself informative, because the two layers behave differently.

**Guard detection replicates near-exactly.** Detection recall is a property of the guard, not the target VLM, so if the architecture-specific blind-spot thesis is correct, the guard detection numbers should be essentially identical across VLMs. They are. LG4 image detection is 81.5% on LLaVA versus 82.0% on Qwen2-VL; LG4 text detection is 92.5% on both. LG3V image detection is 100.0% on both; LG3V text detection is 89.0% on both. The guard blind spots we characterize are reproduced without meaningful change when the target VLM is swapped for an unrelated architecture. This is the core thesis replicating: the blind spots travel with the guard, not the VLM.

**The unguarded attack surface differs, because Qwen2-VL is better safety-aligned.** The one number that changes substantially is the *unguarded* text ASR: 57.0% on LLaVA versus 9.0% on Qwen2-VL. This is not a benchmark inconsistency — it is a real and expected property of the two VLMs. LLaVA-1.6-Mistral is weakly safety-tuned and complies with most harmful text prompts on its own; Qwen2-VL received substantially more safety alignment and refuses the majority of the same prompts unprompted. Unguarded image ASR is closer (60.5% vs 54.0%), because image-rendered harmful intent partially circumvents both VLMs' text-based safety training — which is the entire premise of the rendered-text attack. The protection-gap figures shift accordingly (both guards show more negative gaps on Qwen2-VL, because the smaller unguarded text surface leaves less text-channel risk for the guard to reduce), but the *detection* signature — the quantity our thesis is about — is unchanged.

**Why this strengthens rather than weakens the result.** A natural worry is that the protection gap is an artifact of LLaVA's poor alignment. The Qwen2-VL replication rules this out for the part that matters: guard detection recall, and therefore the architecture-specific blind-spot ordering, is invariant to the target VLM. What the better-aligned VLM changes is the *stakes* — guard gaps matter most precisely when the underlying VLM is weakly aligned and the guard is the primary line of defense. On a well-aligned VLM like Qwen2-VL, the guard's image blind spot is partially masked by the VLM's own refusals on the text channel; on a weakly-aligned VLM like LLaVA, the same blind spot is fully exposed. The blind spot is the same; its consequence scales with how much the deployment relies on the guard. We replicate guard detection across two VLM families; extending the carrier sweep across VLMs is left to future work, as the carrier effect operates on the guard's forward pass and is expected to be VLM-independent for the same reason detection is.

---

## 10. Discussion

**Why the protection gap correlates with architecture.** The three guards span a design space from pure intent classification to pure image content classification. LG4 processes (image, text) jointly through a Llama 4-based multimodal transformer, assessing semantic intent from both inputs. Harmful intent encoded as pixels introduces an extra inference step — recovering semantic content from glyph representations — which partially degrades detection, producing a small positive protection gap. LG3V processes (image, text) jointly through a Llama 3.2 Vision architecture with explicit training on text-in-image safety scenarios; this closes the image-channel gap further but the broad visual sensitivity that enables high detection also drives high over-refusal. SG2 processes images through a vision-only encoder trained on content policy labels (dangerous content, sexual, gore) — it was never designed to read text as intent, which explains simultaneously 0% text detection and the benign-image false-positive rate.

The protection gap magnitude directly reflects this architectural hierarchy: LG4 (+2.5pp) ≈ balanced intent classifier; LG3V (−10.5pp) = vision-specialized intent classifier; SG2 (−56.5pp) = image content classifier. The gap is not primarily a training data quantity issue — it is an architectural capability question.

![Guard architecture contrast](../figures/fig_guard_contrast.png)
*Figure 7: Architectural contrast between the three guards and their resulting protection gaps. Protection gap polarity and magnitude correlate with how image-specialized the guard architecture is.*

**Carrier prompts reveal a third attack axis.** Rendering attacks (Section 5–6) vary image pixels; UAP attacks (Section 7) optimize pixel perturbations via gradients; carrier attacks (Section 8) vary only the text framing paired with the image. The carrier is the lowest-effort vector of the three: no rendering experimentation, no model weights, no GPU. The guard-selective pattern is real but the two guards fail differently: LG4 has a category-level blind spot (fictional and theatrical framings systematically depress detection), while LG3V has a phrasing-level instability (a near-binary switch that specific surface phrasings flip off, most reliably theatrical ones). This has a practical implication: a deployed system using LG4 can be targeted with fiction framing regardless of image rendering quality; a system using LG3V can be targeted with theatrical framing, but is also more generally unstable to carrier phrasing. Carrier design is a form of guard fingerprinting and targeted exploitation rolled into one.

**No single guard dominates.** Each guard represents a distinct operating point:
- LG4: best usability (11.8% OvRef), bounded coverage gaps on both channels (18.5% image miss rate).
- LG3V: best image coverage (100%), highest over-refusal (55.0%), viable only for low-volume high-risk image pipelines where false positives are acceptable.
- SG2: zero text coverage, useful only as an image-channel supplement to a text-capable guard.

**Ensemble implications.** A modality-routed LG4⊕SG2 ensemble — LG4 on text, SG2 on image — closes LG4's image gap (81.5% → 95.5%+ combined) at the cost of SG2's 90% benign-image false-positive rate on image inputs. The 11.5% of image items LG4 misses but SG2 also misses (4.5% of image items) represent the residual risk of the ensemble. An LG4+LG3V ensemble on image inputs would achieve 100% image detection but inherit LG3V's 55% over-refusal on image-modality benign items.

The carrier sweep adds an argument for ensembling that coverage numbers alone cannot express. LG4 and LG3V have *orthogonal* carrier blind spots: fictional framing collapses LG4 (worst single carrier 82%→6%) while LG3V's canonical baseline holds at 100%, and theatrical framing collapses LG3V (≤8% across all four theatrical variants) while LG4 holds partially. No single carrier in the sweep defeats both guards. An LG4⊕LG3V image-channel ensemble is therefore robust to the carrier attack that defeats either guard alone — the framing that blinds one is caught by the other. This robustness is a direct consequence of the central finding: because the blind spots are architecture-specific rather than shared, architectural diversity in an ensemble translates into attack-surface diversity. An ensemble of two guards with the *same* contextual prior would inherit a shared carrier weakness; an ensemble of two guards with different priors does not.

---

## 11. Limitations

- **Rendering sweep is a probe, not an exhaustive search.** 20 Bayesian trials (n=10 each) cover the 7-dimensional space's Pareto frontier at low resolution — enough to identify the dominant driver (noise) and validate the tradeoff on n=200, but a larger sweep could in principle find edge cases nearer the readability boundary.
- **Adaptive rendering tests only surface changes.** The four manipulations (inversion, font size, rotation, typeface) are the simplest-effort class. Embedding text in natural photographs, handwritten simulation, or multi-image decomposition are not evaluated and could behave differently.
- **Over-refusal is measured on XSTest**, which is built to surface over-refusal via safe-but-suspicious prompts. The rates are credible upper bounds — a naive benign set would show LG4 near 0% — but may differ on other benign distributions. We do not characterize *why* LG3V over-refuses at 55%; that is empirical, not mechanistic.
- **Two target VLMs for detection, one for the carrier and UAP studies.** Guard detection replicates near-exactly across LLaVA-1.6 and Qwen2-VL (Section 9), establishing VLM-independence for the detection claim. The carrier and UAP studies used LLaVA only; the carrier effect is expected to be VLM-independent for the same reason detection is, but this is not directly verified. Commercial and larger VLMs are not evaluated.
- **UAP attacks are white-box.** They assume weight access; black-box (query-only) transfer is not evaluated. The SG2 UAP targets one of three policies. The feature-space result (Section 7.3) shows LG4's resistance is not purely a gradient artifact, but decision-based and surrogate-gradient attacks could probe the forward-path constraint further.
- **The carrier sweep is 18 hand-written framings, not an optimized search.** It is broad enough to separate LG4's category-driven vulnerability from LG3V's phrasing-driven instability, but discrete prompt optimization could likely find stronger framings — especially against LG3V's near-binary phrasing sensitivity.
- **Three guards span the architecture space but do not exhaust it.** InternVL-based classifiers, commercial moderation APIs, and ensemble-trained guards are not evaluated. The protection-gap framework applies to any guard; more data points would further test the architectural correlation.

---

## 12. Conclusion

Across three guards and three attack axes, one pattern holds: each guard's failures are fixed by its architecture, not by the channel under attack. The image-specialized guard over-blocks; the balanced guard carries a residual image gap and a semantic framing vulnerability; the image-only guard is blind to text entirely. Where one fails, another holds, and — as the carrier sweep makes concrete, with fiction blinding one guard and theatrical framing the other — the two never fail on the same input. The practical consequence is not that better single guards should be built, but that single guards are the wrong unit of defense. A guard's blind spots are a property of where its architecture spends attention, so the only way to cover them is to compose guards whose attention is allocated differently. Robust multimodal safety is a system-level architecture problem: it must be assembled from diversity, not distilled into one model. We offer that as the paper's central claim, and MSBench as the instrument for testing it — measuring where any guard is blind, and verifying that a proposed ensemble covers the gaps rather than sharing them.

MSBench provides the infrastructure to measure these gaps reproducibly and track them as guard architectures evolve.

---

## Ethics and Responsible Disclosure

This work characterizes failure modes in deployed safety guards in order to make them measurable and fixable. All attacks use harmful behaviors drawn from the public HarmBench benchmark; we introduce no new harmful content, and the repository commits no model-generated harmful outputs — only aggregate detection and attack-success metrics. The attack techniques studied (rendering text as images, natural-language carrier framing) are already known and trivially available to adversaries; the contribution here is systematic measurement and a mechanistic account of *why* specific architectures fail, which is prerequisite to building better guards. The most actionable result is defensive: orthogonal blind spots motivate a concrete ensemble recommendation. The benchmark is intended for guard developers and safety researchers evaluating coverage before deployment. We consider the net effect of publishing these measurements to favor defenders, who can use MSBench to detect these gaps prior to release, over adversaries, for whom the underlying techniques offer no novel capability.

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

---

## Appendix A: UAP Implementation Details

Full attack code and hyperparameters are in the repository (`scripts/`); this appendix records the settings needed to reproduce the Section 7 results. All perturbations are universal (one δ per guard, optimized over a training batch and evaluated on a disjoint test set) and bounded in image pixel space at the stated ε, with δ maintained in each model's `pixel_values` space and the model's own processor used for preprocessing so the attack sees the inference-time input format.

**SG2 (Section 7.1).** 50 HarmBench training images, 50-item disjoint test set; 3 restarts × 100 PGD iterations × batch size 4; ε ∈ {16/255, 32/255}. δ is kept in SigLIP `pixel_values` range [−1, 1] with ε_pv = 2·ε_image. SG2 loaded in 4-bit NF4 with gradient checkpointing on the inner Gemma-3 LM to fit the backward pass in 24 GB; gradients flow to `pixel_values` without differentiating the quantized weights. Loss targets the dangerous_content (policy-0) violated-probability.

**LG3V and LG4 (Section 7.2).** Both are autoregressive; the loss is the "unsafe"-token logit at the final input position from a forward pass (not `generate()`). Both emit a `\n\n` prefix token before the classification token — detected by greedy decoding, with `input_ids` extended by the prefix so `logits[0, -1, :]` lands on the classification position. δ in each model's `pixel_values` space with ε_pv = ε_image / std (processor per-channel normalization). LG3V: 3 restarts × 75 iterations, ε=16/255, batch 1, 50 train items. LG4: ε=16 (75 iters) and ε=32 (200 iters × 3 restarts); early fusion produces 5 tiles × 576 = 2880 image tokens at 512×512, causing backward-pass OOM, so a single-tile 336×336 → 576-token workaround is used.

**Feature-space UAP on LG4 (Section 7.3).** Loss defined in the ViT encoder output space (`model.vision_model`), patch embeddings of shape [1, 144, 4096], optimized as MSE(adv_embed, benign_centroid) with no MoE transformer in the backward path. Benign centroid = mean-pooled patch embedding of 20 blank white images; harmful centroid computed identically from 50 clean harmful training images (centroid cosine similarity 0.6513). 3 restarts × 100 iterations, ε=16/255.

**Transfer (Section 7.4).** The LG3V δ was mapped from LG3V's tile-normalized `pixel_values` space to image pixel space using LG3V's per-channel std ≈ [0.269, 0.261, 0.276], averaged across 4 tiles, resized to 336×336, then applied to LG4's raw input images before LG4 preprocessing.

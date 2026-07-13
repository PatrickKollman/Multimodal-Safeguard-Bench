# MSBench rewrite v2 — new thesis, drafted key sections, and section-by-section plan

**Status:** draft for review. Numbers below are from the corrected canonical run
`results/full_run_v3/` (SG2 lm_head + polarity fixed) and the corrected
`results/uap_sg2_fixed/`. LG4/LG3V/carrier/UAP(LG3V,LG4,ViT,transfer)/Qwen2-VL are
unchanged from prior committed artifacts.

---

## 0. The reframe in one paragraph

The original paper argued a pessimistic thesis ("each guard has architecture-specific
blind spots; no single guard suffices; *and no cheap ensemble does either*"). The last
clause was an artifact of a broken ShieldGemma-2 load (a randomly-initialized `lm_head`
that produced non-deterministic verdicts). With SG2 correctly loaded, the data supports a
stronger, *constructive* thesis:

> **Guard blind spots are architecture-specific AND complementary. The cheapest attack
> (the carrier prompt) is a text-context attack, so an image-content guard that never reads
> text is immune to it by construction. Therefore a cross-modal ensemble — a text-intent
> guard composed with an image-content guard (LG4⊕SG2) — covers both channels cheaply and
> is robust to the carrier attack that defeats either text-reading guard alone; composing
> two text-context guards (LG4⊕LG3V) inherits a shared carrier weakness and a prohibitive
> over-refusal toll. Robust multimodal safety is achieved by composing across reasoning
> modalities, not within them.**

---

## 1. Canonical numbers (Table 1 — `results/full_run_v3/`)

| Condition | Det-txt [CI] | Det-img [CI] | ASR-txt | ASR-img | OvRef [CI] | ProtGap |
|---|---|---|---|---|---|---|
| Unguarded | — | — | 57.0% [50.1, 63.7] | 60.5% [53.6, 67.0] | — | — |
| Llama-Guard-4 | **92.5%** [88.0, 95.4] | 82.0% [76.1, 86.7] | **5.5%** | 11.5% | **11.8%** [9.3, 14.9] | +2.5pp |
| LlamaGuard-3-Vision | 89.0% [83.9, 92.6] | **100.0%** [98.1, 100] | 7.0% | **0.0%** | 55.0% [50.6, 59.3] | −10.5pp |
| ShieldGemma-2 | 0.0% [0, 1.9] | **87.0%** [81.6, 91.0] | 57.0% | 10.5% | **4.6%** (9.2% img) | −50.0pp |

- LG4 image detection is a **paired-significant** blind spot: −10.5pp vs text on the same
  intents, McNemar **p=3.3e-4**, Newcombe CI [−0.171, −0.040] (excludes 0).
- SG2 is now the **best-calibrated image guard**: 87% detection at 9.2% image over-refusal —
  higher detection than LG4 (82%) and vastly better calibrated than LG3V (100% det / 100% img
  over-refusal). Text-blind by construction (0% text). **Reproducible** (identical across 3
  independent runs: 174/200 harmful, 23/500 benign).
- ProtGap is retained only as an architectural ordering device; per §4 it is a policy artifact
  for LG3V (refuse-all-images) and SG2 (text-blind), and the rigorous per-guard result is LG4's
  paired detection McNemar.

## 2. Ensemble (the new centerpiece — `results/full_run_v3/ensemble.json`)

| Ensemble | Det-txt | Det-img | ASR-txt | ASR-img | OvRef |
|---|---|---|---|---|---|
| LG4 alone | 92.5% | 82.0% | 5.5% | 11.5% | 11.8% |
| **LG4⊕SG2** (text→LG4, image→LG4∪SG2) | 92.5% | **97.0%** | 5.5% | **2.5%** | **14.8%** |
| LG4⊕LG3V (block on either) | 96.0% | 100.0% | 3.0% | 0.0% | **56.4%** |

**LG4⊕SG2 is a cheap, effective, cross-modal ensemble:** +15pp image detection and image ASR
11.5%→2.5% for +3pp aggregate over-refusal. LG4⊕LG3V (two text-context guards) drives ASR to 0
but at a deployment-killing 56.4% over-refusal.

## 3. Carrier attack = text-context attack; image-content guard is immune (the unifying result)

Carrier framings (image identical, only the paired sentence changes) — image-channel detection:

| Category | LG4 (mean [min–max]) | LG3V | SG2 (by construction) |
|---|---|---|---|
| baseline | 73% [64–82] | 50% [0–100] | **87% (flat)** |
| fictional | **20%** [6–54] | 35% [0–100] | **87% (flat)** |
| theatrical | 30% [6–54] | **2%** [0–8] | **87% (flat)** |
| transcription | 90% [84–97] | 68% [4–100] | **87% (flat)** |
| academic | 40% [30–52] | 27% [0–82] | **87% (flat)** |
| other | 43% [32–55] | 0% [0–0] | **87% (flat)** |

- Worst single carrier: fiction "novel passage" takes LG4 82%→**6%**; theatrical takes LG3V→**0%**.
- **SG2 is carrier-invariant by construction** — `ShieldGemma2.classify()` passes only the image
  to the model, never the carrier text. So SG2's image verdict is identical across all 18 carriers.
- **Consequence:** under the fiction carrier that collapses LG4 to 6%, LG4⊕SG2 image detection is
  the union of LG4's 6% and SG2's ~87% = **[87%, 93%]** (exact value pending the carrier-sweep
  re-run with fixed SG2). The image-content guard covers exactly LG4's carrier blind spot.
- This is why cross-modal composition beats same-modal: LG4⊕LG3V shares the text-context attack
  surface (a theatrical frame that blinds LG3V, a fiction frame that blinds LG4); LG4⊕SG2 does not.

> **[EXPERIMENT TO RUN]** Re-run `run_carrier_sweep.py` with fixed SG2, persist per-item guard
> verdicts, and report exact LG4⊕SG2 image detection per carrier category. Expected: SG2 flat at
> ~87% across all 18; LG4⊕SG2 never drops below ~85% even where LG4 alone collapses to 6%. This
> turns the by-construction argument into an empirical killer table (new Table).

## 4. UAP — robustness spectrum (replaces the old dense-vs-MoE binary)

| Guard | Architecture | Natural bypass (test) | UAP ε=16 (test) | UAP ε=32 (test) | Transfer in |
|---|---|---|---|---|---|
| LG3V | dense, cross-attn | 0% | **100%** | — | — |
| SG2 | dense, SigLIP+Gemma3 (P≈1 confident) | 10% | 10% (resists) | **52%** (breaks) | — |
| LG4 | sparse MoE, early fusion | 6% | 16% | 22% (resists) | 2% (from LG3V) |

- UAP susceptibility tracks **decision confidence + gradient architecture**, not simply
  dense-vs-sparse: LG3V (unsafe logit ~18) is pushed to 0 at ε=16; SG2 (logit ~38, near-saturated)
  needs ε=32; LG4 (MoE gradient sparsity + text-context dominance) resists both.
- §7.3 (feature-space UAP → LG4 text-context dominance, 10% fooling despite clean ViT gradients)
  and §7.4 (LG3V→LG4 transfer fails, 2%) are **unchanged** (LG4-only).
- Ties to the carrier result: LG4's text-context dominance is *both* its UAP defense *and* its
  carrier vulnerability — the same forward-path property, exploited from two directions.

## 5. Section-by-section change plan for `paper.md`

- **Title:** keep, add complementarity — e.g. "Guard Blind Spots Are Architecture-Specific and
  Complementary: Composing Cross-Modal Guards for Robust Multimodal Safety."
- **Abstract:** rewrite (draft below).
- **Contributions:** drop #8 (SG2 rendering fragility — was the bug). Reframe #2 (SG2 is a strong
  calibrated image guard). Rewrite #5 (UAP spectrum). Add the cross-modal-ensemble + carrier-immunity
  as the headline contribution. Add an appendix repro note on the SG2 load bug.
- **§3 Setup:** SG2 description → "image-content classifier; 0.5 threshold on P(violated)=
  probabilities[:,0]; requires re-tying lm_head after load (Appendix)."
- **§4 Results:** new Table 1; rewrite SG2 paragraph (strong calibrated image guard, not weak/
  non-reproducible). **Delete §4.1 (rendering non-reproducibility) entirely** — it was the bug.
- **§5 Adaptive:** LG4 rows only (statistically invariant to rendering). Drop SG2 rows (buggy load)
  or re-run; recommend drop + one-line note.
- **§6 Rendering sweep:** LG4-focused conclusion unchanged (noise blinds guard and target alike).
  Drop/flag SG2 "100% not movable" (buggy).
- **§7 UAP:** reframe to robustness spectrum; new summary table; keep §7.3/§7.4.
- **§8 Carrier:** keep LG4/LG3V findings; ADD the "carrier = text-context attack; SG2 immune by
  construction" subsection and the LG4⊕SG2 carrier-robustness result (the unifying point).
- **§9 Cross-VLM:** unchanged (detection is guard-only; state it's definitional, informative part
  is unguarded ASR).
- **§10 Discussion / §12 Conclusion:** rewrite around the constructive thesis (draft below).
- **CHANGELOG / README:** reconcile to full_run_v3; document the SG2 fix; fix RunPod HF_HOME +
  volume-quota + --purge-guard-cache guidance; serif→rotated (§5); UAP table.

---

## DRAFT — Abstract (v2)

Safety guards intercept harmful requests before a vision-language model (VLM) responds, but a
harmful instruction can be rendered as an image rather than typed — a zero-gradient, black-box
attack requiring no model access. We present MSBench, a reproducible harness evaluating three
guards spanning the multimodal-safety design space — Llama-Guard-4 (a balanced multimodal intent
classifier), LlamaGuard-3-Vision (a vision-specialized intent classifier), and ShieldGemma-2 (an
image-content classifier) — on 900 items from HarmBench and XSTest, scored by WildGuard. Our
organizing finding is that each guard's failures are fixed by its architecture rather than by the
channel under attack, and — crucially — that these architectural failures are **complementary**,
so robust two-channel coverage can be composed cheaply from guards that reason over different
modalities. The balanced intent classifier (LG4) carries a paired-significant image detection gap
(10.5pp fewer image than text items blocked on the same intents, McNemar p<0.001); the
vision-specialized guard (LG3V) reaches 100% image detection only by refusing 100% of benign
images; and the image-content classifier (SG2) is a strong, well-calibrated image guard (87%
detection at 9.2% over-refusal) that is blind to text by construction. We then show the sharpest
attack is also the cheapest: the carrier prompt — the one-sentence text framing paired with the
image — is a zero-cost, black-box **text-context** attack. A "novel passage" framing collapses
LG4's image detection from 82% to 6% and theatrical framing collapses LG3V to 0%, with orthogonal,
mechanistically distinct blind spots. But because the attack operates entirely through the text
context, an image-content guard that never reads the text (SG2) is **immune to it by
construction**. This yields the paper's central, constructive result: a cross-modal ensemble of a
text-intent guard and an image-content guard (LG4⊕SG2) covers both channels (97% image / 92.5%
text detection; image ASR 11.5%→2.5%) at modest cost (over-refusal 11.8%→14.8%) and is robust to
the carrier attack that defeats either text-reading guard alone — whereas an ensemble of two
text-context guards (LG4⊕LG3V) inherits both a shared carrier weakness and a prohibitive 56%
over-refusal toll. White-box UAP attacks place the guards on a robustness spectrum (LG3V fully
broken at ε=16; SG2 resists ε=16, half-broken at ε=32; LG4 resists both, via gradient sparsity and
text-context dominance). A cross-VLM replication (Qwen2-VL) confirms guard detection is a property
of the guard, not the target VLM. The actionable conclusion is not to build a better single guard,
but to compose guards whose reasoning modalities differ — so their blind spots cover, rather than
share, one another.

---

## DRAFT — Conclusion (v2)

Across three guards and three attack axes, one pattern holds: each guard's failures are fixed by
its architecture, not by the channel under attack. The balanced intent classifier recovers harmful
intent less reliably from pixels than from tokens (a paired-significant image gap); the
vision-specialized guard reaches perfect image detection only by refusing every image; the
image-content classifier is blind to text by construction but, correctly loaded, is the
best-calibrated image guard of the three. Because these failures are fixed by architecture, they
are also **complementary** — and that is what makes robust coverage constructible. The cheapest
attack in the study, the carrier prompt, is the proof: it is a text-context attack that collapses
whichever guard reads the text (fiction blinds LG4, theatre blinds LG3V), yet an image-content
guard that never reads the text is immune to it entirely. Composing a text-intent guard with an
image-content guard therefore does more than add coverage: the image guard covers exactly the
carrier blind spot that the text guard cannot defend, and the pairing pays only a modest
over-refusal cost. Composing two guards that reason over the *same* modality does the opposite —
it inherits their shared attack surface and stacks their false positives. The practical
consequence is precise: robust multimodal safety is not a matter of finding a better single guard,
nor of ensembling arbitrarily, but of composing guards whose *reasoning modalities* differ, so
that the blind spots of one are the strengths of another. MSBench is the instrument for measuring
those blind spots, checking that a proposed composition covers them rather than sharing them, and
verifying that the numbers reproduce.

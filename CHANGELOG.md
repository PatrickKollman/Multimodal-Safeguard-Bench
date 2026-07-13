# Changelog

## [Unreleased] — post-initial-study extensions

### Bug fix: ShieldGemma-2 `classify()` probability indexing

**File:** `src/msbench/guards.py`, `ShieldGemma2.classify()` (line 173)

**What was wrong.** `ShieldGemma2ForImageClassification` returns `output.probabilities` with shape
`[num_policies, 2]` — one row per policy, two columns for P(violated) and P(not_violated).
When called with one image the processor creates a batch of 3 (one per policy), so
`probabilities.shape = [3, 2]`.

The original code did:

```python
probs = scores.probabilities[0]  # [P(Yes_p0), P(No_p0)] — shape [2], NOT [num_policies]
for pol_name, prob in zip(self.POLICY_NAMES, probs):
    if prob.item() > self.threshold:
        return "unsafe", pol_name
```

`scores.probabilities[0]` selects the first *row*, giving the two-token distribution for
policy-0 only. `zip` of 3 policy names against 2 values then mapped:
- `dangerous_content` → `P(Yes for policy-0)` ← correct
- `sexually_explicit` → `P(No for policy-0)` ← WRONG — fires when dangerous_content P < 0.5
- `violence_gore` → never reached

Because `P(not dangerous_content violated)` exceeds 0.5 for almost all rendered text images
(dangerous_content is calibrated for visual violence, not text-encoded harm), the bug caused
SG2 to flag virtually every rendered image as "unsafe / sexually_explicit". This produced the
originally-reported results:
- 100.0% image detection (all rendered harmful images flagged)
- 49.8% over-refusal (all rendered benign images also flagged; text-modality items skipped by
  design, so 250/500 benign items blocked = ~50%)

**The fix:**

```python
# probabilities.shape = [num_policies, 2]: P(violated), P(not_violated) per policy
for pol_name, pol_probs in zip(self.POLICY_NAMES, scores.probabilities):
    if pol_probs[0].item() > self.threshold:
        return "unsafe", pol_name
```

**Impact.** All SG2 results from the original full_run and adaptive_run are invalid. The
corrected implementation is expected to show a substantially lower SG2 image detection rate,
as dangerous_content (policy-0) — the policy most relevant to text-encoded harm intent — has
an ~88% natural blind spot on rendered text images (see UAP section below). The corrected
numbers, along with LG3V results, are produced by the rerun described in the next entry.

---

### New guard: LlamaGuard-3-11B-Vision (LG3V)

**Model:** `meta-llama/Llama-Guard-3-11B-Vision` (Llama 3 Community License — separate HF
gate from LG4's Llama 4 license).

**Motivation.** LG4 (Llama 4 Scout derivative) shows a 10.5pp image-channel detection gap.
LG3V was explicitly trained on text-in-image scenarios, making it the most directly relevant
guard to evaluate against this benchmark's attack surface. If dedicated text-in-image training
closes the detection gap, LG3V would be the sufficient single-guard defense; if it doesn't,
the blind spot is fundamental across multimodal guard architectures.

**Implementation.** `LlamaGuard3Vision` class added to `src/msbench/guards.py`. Uses
`MllamaForConditionalGeneration` (Llama 3.2 Vision architecture), loaded in 4-bit NF4
(~5.5 GB VRAM). Same generation-based classify interface as LG4: outputs "safe" or
"unsafe\nSx". Added to `configs/mvp.yaml` as the second guard (between LG4 and SG2).

**To reproduce.** The rerun command below includes LG3V automatically via the updated
`mvp.yaml`.

---

### New experiment: Universal Adversarial Perturbation (UAP) against SG2

**Script:** `scripts/attack_uap_sg2.py`

**Setup.** A single pixel-space perturbation δ (bounded by ε ∈ {16/255, 32/255}) is
optimized via PGD over 50 HarmBench training images to push SG2's dangerous_content policy
score below 0.5 for all inputs. 3 restarts × 100 iterations × batch size 4. Delta maintained
in SigLIP pixel_values space ([-1, 1]); ε_pv = 2 × ε_image. SG2 loaded in 4-bit NF4 with
gradient checkpointing on the inner Gemma-3 model.

**Key finding.** SG2's dangerous_content policy has an 88% natural blind spot on rendered text
images before any perturbation. The UAP closes the remaining 12%, reaching 100% fooling of
policy-0. This finding motivated investigating why the full guard showed 100% detection in the
original paper (which turned out to be the classify() bug above, not genuine detection).

**Pending.** Full-guard (all three policies) UAP and final results at ε ∈ {16, 32} are pending
the rerun described below.

---

### Rerun: corrected full pipeline with LG3V and SG2 bug fix — COMPLETE

**Run ID:** `full_run_v2` → `results/full_run_v2/`

**Command used:**
```bash
python -m msbench.run --config configs/mvp.yaml --purge-guard-cache --name full_run_v2
```

**Results (900 items: 200 HarmBench × 2 modalities + 250 XSTest × 2 modalities):**

| Guard | Det-txt | Det-img | ASR-txt | ASR-img | OvRef | ProtGap |
|---|---|---|---|---|---|---|
| Unguarded | — | — | 57.0% | 60.5% | — | — |
| Llama-Guard-4 | 92.5% | 81.5% | 5.5% | 11.5% | 11.8% | +2.5pp |
| LlamaGuard-3-Vision | 89.0% | 100.0% | 7.0% | 0.0% | 55.0% | −10.5pp |
| ShieldGemma-2 (corrected) | 0.0% | 95.5% | 57.0% | 4.0% | 45.0% | −56.5pp |

**Key findings from the rerun:**
- SG2's corrected image detection (95.5%) vs. buggy (100%) confirms the bug was real and inflating numbers.
- LG3V's 55.0% over-refusal is the headline surprise — higher than either LG4 (11.8%) or SG2 (45.0%).
- LG3V achieves perfect image detection (100%) — the only guard to do so.
- The protection gap metric reveals a structural pattern: gap magnitude correlates with architectural specialization (LG4 ≈ 0, LG3V −10.5pp, SG2 −56.5pp).

**Status of downstream work:**
- `writeup/paper.md` — updated with corrected numbers (Sections 1, 3, 4, 7, 8, 9, 10)
- `README.md` — updated with three-guard findings and corrected numbers
- `scripts/make_results_figures.py` — pending re-run against `full_run_v2/` to regenerate figures
- `scripts/compute_ensemble.py` — pending re-run for ensemble metrics with corrected SG2

---

### New experiment: UAP against generation-based guards (LG4, LG3V)

**Script:** `scripts/attack_uap_gen.py`

**Design.** LG4 and LG3V are autoregressive: they generate "safe" or "unsafe" as the first
output token. The UAP loss function minimizes the "unsafe" token logit at the final input
position (the position predicting the first generated token) via a plain `model(**inputs,
use_cache=False)` forward pass — not `model.generate()`. This is fully differentiable
with respect to pixel_values. Epsilon in pixel_values space is derived from image-space epsilon
via the processor's per-channel std: `eps_pv = eps_image / std`. Gradient accumulation with
`batch=1` accommodates the larger model backward pass within 24 GB VRAM.

**Status.** Results pending. Both guards running on pod.

---

## [v0.1] — Initial two-guard study

Original study: LG4 and SG2 evaluated against 900 items (200 HarmBench × 2 modalities +
250 XSTest × 2 modalities). Results committed to `results/full_run/`. Note: SG2 numbers in
this version are invalid due to the bug described above. See the unreleased section.

Key results (pre-fix, for reference only):
- LG4: 92.5% text det, 82.0% image det, 5.5%/11.5% ASR, 11.8% OvRef
- SG2 (BUGGY): 0.0% text det, 100.0% image det (artifact of bug), 49.8% OvRef (artifact)
- Ensemble (BUGGY): 92.5%/100.0% det, 5.5%/0.0% ASR, 53.6% OvRef

Adaptive rendering study and Bayesian parameter sweep results (committed to
`results/adaptive_run/` and `results/rendering_sweep_probe*/`) are based on LG4 results only
and remain valid.

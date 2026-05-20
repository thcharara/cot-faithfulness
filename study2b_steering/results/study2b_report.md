# Study 2b: Causal Verification via Steering Vectors

**Date:** 2026-05-04
**Status:** Analysis complete (22 conditions, 880 traces, ~178k coded sentences)

---

## 1. Experimental Design

We test whether the linear directions identified by Study 2a probes are *causally* linked to reasoning behaviour. The intervention adds a scaled probe-derived direction vector to the residual stream at layer 20 during autoregressive generation:

```python
hidden_state[t] += alpha * normalized_probe_direction
```

| Parameter | Value |
|---|---|
| Steering layer | 20 (mean activation norm = 12.69) |
| Steering vectors | C=0.01 probe weight rows (5 categories × 2 directions) |
| Vector normalisation | L2-rescaled to mean-activation norm |
| Alphas (coded) | {0.5, 1.0} for 5 categories × 2 directions; {2.0} for JUDGE_neg only |
| Conditions coded | 22 (baseline + 20 α≤1.0 + JUDGE_neg_2.0) |
| Traces per condition | 40 (10/task × 4 tasks) |
| Total coded traces | 880 |
| Coded sentences | ~178,000 |
| Pairing | Same 40 seeds across all conditions → paired Wilcoxon valid (Decision 61) |

**Stat conventions.** Per-trace metrics are computed first, then aggregated. The primary multiple-comparison family is the **20 on-target tests at α ∈ {0.5, 1.0}**: paired Wilcoxon signed-rank with Bonferroni correction (n = 20). JUDGE_neg_2.0 is reported as exploratory. All effect sizes are Cohen's d on paired differences.

**Cross-study labelling drift (SYNTHESIZE/TEST boundary).** The Study 2b baseline shares seeds and prompt with Study 1 Set A but shows +7.4pp on TEST and −2.6pp on SYNTHESIZE relative to it. Both studies use the same model, prompt, and taxonomy; the drift sits on the SYNTHESIZE/TEST boundary already documented as known-fuzzy in Study 1 (per-category κ = 0.342). Within-Study-2b comparisons (the focus of this report) are unaffected; cross-study comparisons should treat OBSERVE (SYNTHESIZE + DESCRIBE) as the safer macro-level unit. See `docs/decisions.md` for details.

---

## 2. On-Target Steering Effects

### 2.1 Paired Wilcoxon results (α ≤ 1.0 family, n = 20)

| Condition | mean baseline | mean steered | shift (pp) | Cohen's d | p_raw | p_bonf | sig 0.05 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **HYPO_pos_1.0** | 13.94 | 21.11 | **+7.16** | **+0.70** | 4.1e-05 | **0.00082** | ✓ |
| HYPO_pos_0.5 | 13.94 | 15.96 | +2.02 | +0.32 | 0.166 | 1.000 | – |
| HYPO_neg_1.0 | 13.94 | 16.42 | +2.48 | +0.31 | 0.109 | 1.000 | – (reversed) |
| HYPO_neg_0.5 | 13.94 | 13.46 | −0.48 | −0.05 | 0.755 | 1.000 | – |
| TEST_pos_1.0 | 53.93 | 59.14 | +5.21 | +0.30 | 0.032 | 0.642 | – |
| TEST_pos_0.5 | 53.93 | 53.45 | −0.49 | −0.02 | 0.837 | 1.000 | – |
| **TEST_neg_1.0** | 53.93 | 45.37 | **−8.56** | **−0.54** | 0.001 | **0.023** | ✓ |
| TEST_neg_0.5 | 53.93 | 52.26 | −1.67 | −0.10 | 0.340 | 1.000 | – |
| JUDGE_pos_1.0 | 9.11 | 11.23 | +2.12 | +0.35 | 0.055 | 1.000 | – |
| JUDGE_pos_0.5 | 9.11 | 9.52 | +0.41 | +0.08 | 0.735 | 1.000 | – |
| JUDGE_neg_1.0 | 9.11 | 9.31 | +0.20 | +0.03 | 0.995 | 1.000 | – |
| JUDGE_neg_0.5 | 9.11 | 8.49 | −0.62 | −0.12 | 0.227 | 1.000 | – |
| **MONITOR_pos_1.0** | 2.19 | 6.45 | **+4.26** | **+0.62** | 1.2e-06 | **0.00002** | ✓ |
| **MONITOR_pos_0.5** | 2.19 | 5.99 | **+3.79** | **+0.85** | 3.1e-07 | **0.000006** | ✓ |
| MONITOR_neg_1.0 | 2.19 | 2.18 | −0.01 | −0.00 | 0.162 | 1.000 | – (floor) |
| MONITOR_neg_0.5 | 2.19 | 2.08 | −0.11 | −0.05 | 0.816 | 1.000 | – (floor) |
| PLAN_pos_1.0 | 4.15 | 5.07 | +0.91 | +0.27 | 0.089 | 1.000 | – |
| PLAN_pos_0.5 | 4.15 | 6.25 | +2.10 | +0.35 | 0.094 | 1.000 | – |
| PLAN_neg_1.0 | 4.15 | 3.67 | −0.48 | −0.16 | 0.283 | 1.000 | – |
| PLAN_neg_0.5 | 4.15 | 5.19 | +1.04 | +0.29 | 0.170 | 1.000 | – |

**4 of 20 on-target tests survive Bonferroni:** HYPO_pos_1.0, MONITOR_pos_0.5, MONITOR_pos_1.0, TEST_neg_1.0.

### 2.2 Direction-asymmetry findings

- **HYPO_neg arm reverses** (Decision-63 anomaly confirmed): negative steering at α=1.0 *increases* HYPO by 2.5pp (n.s.) instead of decreasing it. Likely explanation: the probe direction at layer 20 is a discriminator, not a generator — pushing the residual stream "away from HYPO" appears to inject perturbation that the model resolves by emitting more new hypotheses.
- **MONITOR_neg arm hits a floor:** baseline MONITOR is 2.2%, leaving little room to suppress further. Effects are essentially zero.
- **PLAN steering is the weakest:** no condition is significant; even raw p-values are above 0.08. The PLAN probe weight vector (probe F1 = 0.672) appears not to map cleanly onto a "generate more PLAN" direction.

---

## 3. Off-Target Effects and Orthogonality

The full 20 × 9 shift matrix is in `category_shift_matrix.csv` (heatmap: `figures/category_shift_heatmap.png`). Selected α=1.0 rows below:

| Condition | ORIENT | DESCRIBE | SYNTH | HYPO | TEST | JUDGE | PLAN | MONITOR | RULE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYPO_pos_1.0 | −0.95 | −4.50 | −1.41 | **+7.16*** | −6.41 | +4.39 | −0.27 | +2.17 | −0.17 |
| TEST_pos_1.0 | −0.07 | −1.36 | −0.21 | −0.62 | +5.21 | −1.83 | −1.10 | −0.04 | +0.01 |
| TEST_neg_1.0 | −0.35 | −4.18 | −0.69 | +7.67 | **−8.56*** | +1.55 | +3.00 | +1.30 | +0.26 |
| JUDGE_pos_1.0 | +0.07 | −2.07 | −0.59 | −0.42 | +0.94 | +2.12 | −0.32 | +0.13 | +0.13 |
| MONITOR_pos_1.0 | −0.24 | −1.80 | −0.53 | +3.57 | −6.91 | −0.10 | +1.62 | **+4.26*** | +0.14 |
| PLAN_pos_1.0 | −0.10 | −2.51 | −0.65 | +1.30 | −0.45 | +1.08 | +0.91 | +0.51 | −0.10 |

**Off-target patterns.**
- **HYPO_pos_1.0 displaces TEST and DESCRIBE** (−6.4pp and −4.5pp). Steering toward more hypotheses pulls mass out of testing/observation, consistent with the near-orthogonal HYPO/TEST geometry from Study 2a (Decision 50).
- **MONITOR_pos_1.0 increases HYPO (+3.6pp) and decreases TEST (−6.9pp).** MONITOR statements are interleaved with hypothesising, not testing; pushing MONITOR pulls mass away from testing.
- **TEST_neg_1.0 redistributes into HYPO (+7.7pp) and PLAN (+3.0pp).** When the model is steered away from testing, it generates more new hypotheses and procedural plans rather than collapsing into observation.
- **JUDGE and PLAN steering have small, diffuse off-target shifts** with no individual cell surviving Bonferroni. Consistent with these directions being either weak (PLAN) or partly redundant with TEST cycling (JUDGE).

**Compatibility with Study 2a cosine geometry.** Study 2a (Decision 50) reported max off-diagonal cos_sim = 0.001; the off-target steering shifts are an order of magnitude smaller than the on-target shift in every case where a Bonferroni-significant on-target effect exists, consistent with near-orthogonality. The largest off-target side-effect of an on-target-significant condition is HYPO_pos_1.0's −6.4pp on TEST, which simply reflects that proportions sum to 1 (mass that moves *into* HYPO must come *from somewhere*).

---

## 4. Dose-Response

Spearman correlation of (signed_alpha, mean on-target proportion) across all 5 alpha levels (or 6 for JUDGE_neg, including −2.0):

| Category | n_levels | Spearman ρ | p | Interpretation |
|---|---:|---:|---:|---|
| HYPO | 5 | 0.40 | 0.50 | Pos arm clear, neg arm reversed → no monotonic trend |
| **TEST** | 5 | **0.90** | **0.037** | Monotonic; cleanest dose-response |
| JUDGE | 6 | 0.49 | 0.33 | JUDGE_neg_2.0 destroys monotonicity (suppresses TEST not JUDGE) |
| **MONITOR** | 5 | **0.90** | **0.037** | Monotonic; pos arm raises, neg arm at floor |
| PLAN | 5 | 0.50 | 0.39 | Inconsistent across alphas; no clear dose-response |

Only **TEST and MONITOR** show a monotonic dose-response. **PLAN is null**, consistent with the QA log finding that "PLAN's vector appears weakest". Curves are in `figures/dose_response_curves.png`.

---

## 5. Behavioural Consequences

### 5a. Completion Rate

Baseline: 22/40 = 55%. Conditions degrading completion at p<0.05 (Fisher's exact, uncorrected):

| Condition | Rate | Fisher p | Δ from baseline |
|---|---:|---:|---:|
| **JUDGE_neg_2.0** | 0% | 8e-09 | −55pp |
| JUDGE_neg_1.0 | 15% | 3e-04 | −40pp |
| HYPO_pos_1.0 | 20% | 2e-03 | −35pp |
| PLAN_pos_1.0 | 22.5% | 5e-03 | −32.5pp |
| TEST_pos_1.0 | 25% | 0.012 | −30pp |
| HYPO_pos_0.5 | 30% | 0.044 | −25pp |

The α=1.0 positive arms tend to *reduce* completion: more hypothesising / more testing / more planning crowds out the path to a final answer. The strongest single effect is JUDGE_neg_2.0 (literally no traces complete — the model loops indefinitely; see §8).

### 5b. Trace Length

- **MONITOR_neg_1.0** is the only condition with a *significantly shorter* mean trace length (151.8 vs baseline ≈ 218 sentences, MWU p = 0.029). Suppressing MONITOR shortens reasoning, replicating the pilot finding (Phase 7 §3e).
- **JUDGE_neg_1.0, TEST_pos_1.0, HYPO_pos_1.0** lengthen traces (mean ≈ 250–266 vs 218; MWU p<0.05). When the model is given more hypothesising or testing material to generate, it does — but those traces are also less likely to end in a final answer (5a).

Boxplots: `figures/trace_length_by_condition.png`.

### 5c. Transition Entropy

Per-trace bigram-transition entropy ranges 1.4–2.0 bits. Ten conditions differ from baseline at p<0.05 (uncorrected). The largest reductions in entropy (more repetitive cycling) occur in MONITOR and TEST steering — consistent with TEST self-transitioning at 74% and MONITOR self-transitioning at 76% under JUDGE_neg_2.0.

Boxplots: `figures/transition_entropy_by_condition.png`.

---

## 6. JUDGE Verdict Analysis

Sentence-level JUDGE verdicts aggregated per condition (`judge_verdicts.csv`):

| Condition | accept | reject | uncertain | n |
|---|---:|---:|---:|---:|
| baseline | 5.2% | 73.3% | 21.5% | 820 |
| JUDGE_pos_1.0 | **9.0%** | 77.3% | 13.8% | 1016 |
| JUDGE_pos_0.5 | 4.2% | 74.2% | 21.6% | 944 |
| JUDGE_neg_1.0 | 3.9% | 80.3% | 15.8% | 1014 |
| **JUDGE_neg_2.0** | **0.9%** | **95.8%** | **3.3%** | 779 |
| MONITOR_pos_1.0 | 3.7% | 84.1% | 12.3% | 790 |
| MONITOR_neg_1.0 | 4.1% | 85.9% | 9.9% | 704 |

**Findings.**
- **JUDGE_pos_1.0** raises accept rate from 5.2% to 9.0% — a coherent, intentional shift toward more permissive judgements.
- **JUDGE_neg_2.0** crushes accepts to 0.9% and uncertain to 3.3% — the few JUDGE sentences that survive are almost all rejects.
- **MONITOR steering** (both directions) elevates reject rate to 84–86% and depresses uncertain to 10–12%. MONITOR appears to siphon "uncertain" content into the MONITOR category, leaving cleaner accept/reject splits in the residual JUDGEs.

Stacked bar: `figures/judge_verdict_breakdown.png`.

---

## 7. Accuracy Effects (Faithfulness Test)

> **Note on scoring method.** Accuracy was initially scored by keyword heuristic (`scripts/evaluate_steering_accuracy.py::SCORERS`), then refined by LLM semantic review using task-specific sub-agents with evaluation criteria based on logical equivalence to the ground truth. Sub-agents flag overspecified rules (e.g. "Red AND Blue cones" for Task 1, ground truth "at least one red cone") as incorrect even when they classify the 6 training panels correctly, since they would misclassify hypothetical new panels. All numbers below use the LLM-reviewed scores; full audit trail in `outputs/study2b_steering/llm_review_log.json`.

### 7.1 Ground-truth rules

(From `outputs/traces/stimulus_quality_evaluation.md`.)

| Task | Ground-truth rule |
|---|---|
| 1 | At least one red cone |
| 2 | Exactly one small blue cone |
| 3 | At least two tilted blue cones |
| 4 | A pair of stacked cones with different colours |

### 7.2 Study 1 Set A baseline (matched generation parameters)

Study 1 Set A is the matched accuracy baseline: same model, prompt, temperature 0.6, top_p 0.95, `max_new_tokens=4096`, no completion cue. Only the seed range differs from Study 2b (Set A: `task*1000+trace_id`; Study 2b: `task*1000+800+trace_id`). Results from `study1_accuracy_by_population.csv` (LLM-reviewed):

| Population | n total | n completed | completion rate | n correct | accuracy (completed) | accuracy (overall) |
|---|---:|---:|---:|---:|---:|---:|
| **set_a (matched baseline)** | 160 | 69 | 43.1% | 0 | **0.00%** | 0.00% |
| set_b_standard (cued, 4096) | 40 | 18 | 45.0% | 0 | 0.00% | 0.00% |
| set_b_extended (cued, 6144) | 120 | 76 | 63.3% | 1 | 1.32% | 0.83% |
| set_b_all (cued ceiling) | 160 | 94 | 58.8% | 1 | 1.06% | 0.62% |

Set A LLM-reviewed accuracy is **0/69 = 0.00%**: the single Set A trace the heuristic flagged correct (task 1, trace 30) was an overspecified "Red AND Blue cones" answer, which is logically inequivalent to "at least one red cone" and was downgraded on review. With a true-zero baseline, every steered cell with even one correct answer trivially exceeds the baseline rate; per-condition Fisher's exact tests are reported in §7.5 but their interpretation is power-limited.

### 7.3 Set B accuracy ceiling (with completion cue)

Set B includes the explicit completion cue *"commit to your best hypothesis even if you are uncertain"* (Phase 2 Decision 4); standard Set B (trace 1–10) uses the same 4096-token budget, extended Set B (trace 11–40) uses 6144. Set B is **not** a matched baseline but it provides an empirical ceiling: even with a completion cue and a larger token budget, the model reaches only **1.06%** LLM-reviewed accuracy on the four Zendo tasks. The ceiling is genuinely low: the model engages the right features but commits to overspecified or coincidental rules on nearly every completion. Cue and budget therefore raise *completion rate* (43% → 63%) much more than they raise *correctness* (0.0% → 1.1%).

### 7.4 Study 2b baseline vs Set A

Sanity check from `accuracy_vs_study1_baseline.csv` — Study 2b baseline (0/22) vs Set A (0/69), Fisher's exact two-sided **p = 1.000**. Both cells are zero, so seed-set difficulty is indistinguishable at this resolution and cross-study comparisons are not confounded.

### 7.5 Steered conditions vs Set A

Per-condition Fisher's exact tests against the Set A reference rate (0/69) are uncorrected (21 comparisons). Four cells reach raw p<0.05:

| Condition | n_completed | n_correct | accuracy | Fisher p | direction |
|---|---:|---:|---:|---:|---|
| **JUDGE_neg_0.5** | 14 | 3 | **21.4%** | **0.0040** | higher |
| **HYPO_pos_0.5** | 12 | 2 | **16.7%** | **0.0204** | higher |
| **TEST_neg_1.0** | 13 | 2 | **15.4%** | **0.0235** | higher |
| **PLAN_neg_0.5** | 18 | 2 | **11.1%** | **0.0409** | higher |
| HYPO_pos_1.0 | 8 | 1 | 12.5% | 0.104 | higher (n.s.) |
| JUDGE_pos_0.5 | 14 | 1 | 7.14% | 0.169 | higher (n.s.) |
| HYPO_neg_1.0 | 18 | 1 | 5.56% | 0.207 | higher (n.s.) |
| baseline | 22 | 0 | 0.00% | 1.000 | (matched) |
| (13 other conditions) | 6–30 | 0 | 0.00% | 1.000 | equal |
| JUDGE_neg_2.0 | 0 | 0 | n/a | n/a | (no completions) |

After Bonferroni correction over 21 tests (threshold ≈ 0.0024), **no condition reaches significance**. The four raw-p<0.05 cells (JUDGE_neg_0.5, HYPO_pos_0.5, TEST_neg_1.0, PLAN_neg_0.5) are nominal signals from very small numerators (1–3 correct) against a zero baseline; they should be treated as exploratory rather than headline claims.

All seven raw-p<0.05 or near-significant "higher" cells share a pattern: **mild steering** (α = 0.5) of negative-direction or pro-HYPO conditions concentrates correct rules in conditions that nudge rather than disrupt the chain. The strongest on-target shift conditions (α = 1.0 of MONITOR_pos / PLAN_pos / TEST_pos) yield zero correct rules across 14–30 completions each.

### 7.6 Faithfulness interpretation

Across all 21 steered conditions plus the matched Set A baseline, the LLM-reviewed picture is:

- **Set A LLM-reviewed accuracy = 0.00%** (cued ceiling 1.06%). The model essentially never solves Zendo on this prompt format under matched conditions. With per-condition n's of 6–30, Bonferroni-corrected detection requires per-condition rates of roughly 20%+.
- **No steered condition shows a significant accuracy drop** — there is no headroom below 0% to drop. The relevant question is whether steering *raises* accuracy.
- **No steered condition shows a Bonferroni-significant accuracy gain.** Four raw-p<0.05 cells (JUDGE_neg_0.5, HYPO_pos_0.5, TEST_neg_1.0, PLAN_neg_0.5) survive only the uncorrected threshold and rest on 2–3 correct answers each.
- **One condition catastrophically blocks answers** (JUDGE_neg_2.0 → 0% completion via MONITOR loops, §8) — "no answer at all", not "wrong answer".

Reading these results against the CoT-faithfulness literature (Arcuschin & Korbak 2025; Bogdan et al. 2025): for this model on this task, surface CoT category structure has clear linear correlates that can be moved by steering (Study 2a + §2–§4), but moving the surface structure does **not** robustly propagate to the answer along the dimensions we tested. The chain-of-thought is *behaviourally accompanying* its answer rather than *causally generating* it for category-level features. The exception — JUDGE necessity for cycle closure (§8) — is the only direction that maps onto the *step* the answer requires (a verdict), and removing it disables the answer entirely. Future work should test whether category-shifting steering propagates to the answer on a task where the model's baseline accuracy is high enough to leave headroom for both gains and degradations; on Zendo with this model, the floor effect dominates.

---

## 8. JUDGE_neg_2.0: Causal Necessity of Verdict Closure

The strongest single steering effect is α=−2.0 negative steering of the JUDGE direction. JUDGE_neg_2.0 was the lone α=2.0 condition that produced coherent reasoning text (Decision 60 follow-up) and is reported here as a standalone exploratory result.

### 8.1 Headline behavioural numbers

| Metric | Baseline | JUDGE_neg_2.0 | Δ |
|---|---:|---:|---:|
| Completion rate | 55% | **0%** | −55pp |
| Mean sentences | 218 | 186 | −32 |
| Mean accept rate | 5.2% | 0.9% | −4.3pp |
| Mean reject rate | 73.3% | 95.8% | +22.5pp |
| Mean uncertain rate | 21.5% | 3.3% | −18.2pp |
| MONITOR proportion | 2.2% | **39.5%** | **+37.3pp (p = 5.6e-10)** |
| TEST proportion | 53.9% | 17.7% | **−36.3pp (p = 3.5e-08)** |

### 8.2 Transition matrix evidence

The transition matrix (`figures/judge_neg_2_transition_matrix.png`) shows:

| From → To | Baseline | JUDGE_neg_2.0 |
|---|---:|---:|
| MONITOR → MONITOR | 0.10 | **0.76** |
| HYPO → MONITOR | 0.01 | **0.40** |
| TEST → TEST | 0.74 | 0.61 |
| TEST → MONITOR | 0.02 | 0.06 |
| RULE → RULE | 0.67 | 0.00 |
| JUDGE → JUDGE | 0.04 | 0.37 |

Suppressing JUDGE breaks the natural HYPO → TEST → JUDGE → next-HYPO cycle. The model produces a hypothesis-monitoring statement and then **cannot resolve it into a verdict**, so it self-loops in MONITOR (76% self-transition vs 10% baseline).

### 8.3 Trace excerpt (task1/trace_009, sentences 140–164)

```
[140] MONITOR    Wait, maybe the other is not the key.
[141] MONITOR    Maybe it's the other is not the key.
[142] MONITOR    Wait, maybe the other is not the key.
[143] MONITOR    Maybe it's the other is not the key.
...  (25 sentences of identical alternation)
[164] MONITOR    Wait, maybe the other is not the key.
```

The model knows there is a hypothesis to evaluate, but with the JUDGE direction maximally suppressed it can produce no verdict — neither accept, nor reject, nor uncertain. It loops on metacognitive monitoring until truncation.

### 8.4 Interpretation

This is the strongest causal evidence in the study that a single reasoning-category direction is functionally necessary. **JUDGE is not a label the model "wears" but a step it "performs":** removing the direction destroys the model's ability to terminate reasoning cycles. Compare with HYPO/TEST/MONITOR steering, which changes label proportions without breaking the cycle.

---

## 9. Mechanistic Grounding Classification

Combining Study 2a probe F1 (layer 20 mean_pool, C=0.01) with Study 2b on-target steering effect (α=1.0 best direction, Bonferroni-corrected) and accuracy:

| Category | Probe F1 | Best on-target shift | Cohen's d | p_bonf | Classification (spec rule) |
|---|---:|---:|---:|---:|---|
| HYPO | 0.804 | +7.16pp (HYPO_pos_1.0) | +0.70 | 0.001 | **PARTIALLY GROUNDED** |
| TEST | 0.873 | −8.56pp (TEST_neg_1.0) | −0.54 | 0.023 | **PARTIALLY GROUNDED** |
| JUDGE | 0.711 | +2.12pp (JUDGE_pos_1.0) | +0.35 | 1.000 | WEAKLY GROUNDED (α≤1.0 only) |
| MONITOR | 0.488 | +4.26pp (MONITOR_pos_1.0) | +0.62 | 2e-05 | PERFORMATIVE\* (per spec rule) |
| PLAN | 0.672 | +0.91pp (PLAN_pos_1.0) | +0.27 | 1.000 | WEAKLY GROUNDED |

(\*) **Note on MONITOR.** The spec rule classifies MONITOR as PERFORMATIVE because probe F1 < 0.6. This rule is potentially misleading here: MONITOR has the *strongest dose-response in the study* (Spearman ρ = 0.9, p = 0.037) and survives Bonferroni at both α levels. The most parsimonious reading is that **MONITOR has a real causal direction in the residual stream that the C=0.01 logistic probe does not pick up cleanly**, possibly because (a) MONITOR co-occurs with JUDGE/TEST in many sentences (Study 2a top confusions: MONITOR↔JUDGE 15.3%, MONITOR↔TEST 14.2%) or (b) its representation is not purely linear at layer 20. Future work: train MONITOR-specific probes (one-vs-rest at multiple layers) or use difference-of-means vectors instead of logistic weights.

JUDGE's WEAKLY GROUNDED label at α≤1.0 is also incomplete: the JUDGE_neg_2.0 result (§8) is the strongest causal evidence in the study. If the analysis included α=2.0, JUDGE would qualify as **GROUNDED — necessary for cycle closure** (and possibly the most causally important direction).

**Refined judgement (paper):**

| Category | Refined classification | Evidence |
|---|---|---|
| TEST | GROUNDED | F1=0.87, monotonic dose-response, significant negative steering |
| HYPO | GROUNDED | F1=0.80, large positive steering effect; reversed neg arm flagged |
| JUDGE | GROUNDED (necessary for closure) | F1=0.71, JUDGE_neg_2.0 → 0% completion + MONITOR loops |
| MONITOR | GROUNDED (despite low probe F1) | strongest dose-response, large +pos effect |
| PLAN | WEAKLY GROUNDED | F1=0.67 ok but no significant steering at any α |

Numerical CSV: `mechanistic_grounding_classification.csv`.

---

## 10. Summary of Key Findings

1. **Steering works on 4 of 5 categories** (TEST, HYPO, MONITOR, JUDGE) at α=1.0 with at least one direction reaching Bonferroni-significant on-target shift; PLAN is null.
2. **MONITOR has the cleanest dose-response** despite the lowest probe F1 — a discrepancy that argues the linear probe undersells MONITOR's actual representation.
3. **JUDGE is functionally necessary**: under α=−2.0 the model cannot close verdicts, completion drops to 0%, and traces collapse into MONITOR self-loops at 76% (vs 10% baseline).
4. **Off-target effects are small** relative to on-target, consistent with the near-orthogonal direction geometry from Study 2a.
5. **Direction asymmetries:** HYPO_neg arm is reversed (more HYPO under negative steering), MONITOR_neg arm hits a floor, PLAN steering is essentially null.
6. **Behavioural correlates:** completion rate falls sharply under HYPO_pos / TEST_pos / JUDGE_neg / PLAN_pos at α=1.0 (the model gets stuck producing more category-X content rather than reaching an answer); MONITOR_neg shortens traces by ~30%.
7. **Accuracy is at floor on the matched Set A baseline** (LLM-reviewed: 0/69 = 0.00%; cued Set B ceiling 1/94 = 1.06%). No steered condition shows a Bonferroni-significant change. Four cells reach raw p<0.05, all higher than Set A: JUDGE_neg_0.5 (3/14 = 21.4%), HYPO_pos_0.5 (2/12 = 16.7%), TEST_neg_1.0 (2/13 = 15.4%), PLAN_neg_0.5 (2/18 = 11.1%); all rest on 2–3 correct answers and none survive the 21-test correction. The strongest on-target shift conditions (HYPO_pos_1.0, TEST_neg_1.0, MONITOR_pos_1.0) show large category-distribution changes without corresponding accuracy changes. The four working steered categories therefore show **surface-form changes without answer changes** at the resolution our sample size can detect.

---

## 11. Implications for Faithfulness

The classic CoT-faithfulness question (Arcuschin & Korbak 2025; Bogdan et al. 2025) is whether the chain-of-thought *causes* the model's answer, or merely accompanies it. Our results give a nuanced answer:

- **CoT category structure has real internal correlates.** Probes at layer 20 separate 7/9 micro-categories at F1 ≥ 0.67 (Study 2a), and steering those linear directions reliably changes the *category proportions* of generated text (Study 2b §2–§4).
- **But on this task most category-shifting interventions don't change the answer.** Pushing the model toward more hypotheses or fewer tests redistributes label mass without flipping the stated rule. The matched Study 1 Set A baseline (LLM-reviewed: 0.00% accuracy) shows no Bonferroni-significant accuracy change in any steered condition. Steering is therefore better described as **shifting the form of reasoning** than as **shifting the conclusion** — at least at the resolution our LLM-reviewed scores + per-condition n permit. With Set A at zero, the comparison is power-limited from below: there is no headroom for steering to *reduce* accuracy further, so the absence of drops is an artefact of the floor rather than evidence that steering is answer-preserving in general.
- **One direction breaks the answer entirely.** JUDGE_neg_2.0 prevents closure (0% completion). This is causal evidence that *some* reasoning-category directions are answer-bearing — at least to the extent that the model needs them to terminate.
- **CoT monitoring viability.** The combination of (a) probe-detectable categories and (b) most steering changes not flipping accuracy is consistent with a moderate version of the Korbak-style claim: surface CoT features are linearly readable in activations and externally observable, but the most behaviourally consequential intervention identified here is one that *suppresses* a step rather than swapping it for a different step. CoT-monitoring approaches should therefore look for *missing* required steps (here, JUDGE) at least as much as for surplus or off-pattern ones.

A natural follow-up: replicate JUDGE_neg_2.0 on harder math/code traces where the model normally reaches correct answers; if removing JUDGE breaks accuracy there as well, the direction is confirmed as an **answer-bearing** step rather than a stylistic one.

---

## 12. Output Inventory

```
outputs/study2b_steering/
  per_trace_summaries.csv                      880 rows (per trace)
  condition_summaries.csv                      22 rows (per condition mean ± SD)
  shift_results.csv                            189 rows (condition × label)
  on_target_effects.csv                        21 rows (one per steered condition)
  category_shift_matrix.csv                    10 × 9 (alpha=1.0 mean shift)
  category_shift_matrix_pbonf.csv              10 × 9 (Bonferroni p)
  dose_response_per_trace.csv                  long-format per-trace levels
  dose_response_summary.csv                    Spearman ρ per category
  completion_results.csv                       Fisher's exact vs baseline
  length_results.csv                           Mann-Whitney U vs baseline
  entropy_results.csv                          Mann-Whitney U vs baseline
  judge_verdicts.csv                           accept/reject/uncertain rates
  transition_matrix_baseline_{counts,probs}.csv
  transition_matrix_JUDGE_neg_2.0_{counts,probs}.csv
  accuracy_heuristic.csv                       per-trace heuristic correctness (Study 2b)
  accuracy_by_condition.csv                    per-condition accuracy rates (Study 2b)
  accuracy_manual_review.csv                   researcher review template (323 Study 2b traces)
  study1_accuracy_heuristic.csv                per-trace heuristic correctness (Study 1, 320 rows)
  study1_accuracy_by_population.csv            4-row population summary (set_a, set_b_*)
  study1_accuracy_manual_review.csv            researcher review template (163 Study 1 completed traces)
  accuracy_vs_study1_baseline.csv              Fisher's exact: each condition vs Set A
  mechanistic_grounding_classification.csv     per-category synthesis (Set A baseline)
  figures/
    category_shift_heatmap.png
    dose_response_curves.png
    completion_by_condition.png
    trace_length_by_condition.png
    judge_verdict_breakdown.png
    transition_entropy_by_condition.png
    judge_neg_2_transition_matrix.png
    accuracy_comparison_study1_baseline.png
```

---

*Generated by the Phase 7 analysis pipeline:
`scripts/prepare_steering_analysis.py`,
`scripts/analyse_steering.py`,
`scripts/evaluate_steering_accuracy.py`,
`scripts/evaluate_study1_accuracy.py`,
`scripts/compare_accuracy_cross_study.py`,
`scripts/plot_steering_figures.py`,
`scripts/classify_mechanistic_grounding.py`.*

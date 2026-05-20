# Study 2b: Steering Vector Interventions

Tests whether probe-derived direction vectors causally influence the
model's reasoning behaviour when added to the residual stream during
generation.

**Status: Completed**

## Pipeline

| Step | Script | Input | Output | Requires |
|---|---|---|---|---|
| 1 | `scripts/study2b_steer_generate.py` | Model + steering vectors | Steered traces | GPU |
| 2 | `scripts/study2b_segment_steered_traces.py` | Steered traces | Segmented traces | spaCy |
| 3 | `scripts/study2b_code_steered_traces.py` | Segmented traces + probe | Coded traces | GPU |
| 4 | `scripts/study2b_prepare_analysis.py` | Coded traces | `per_trace_summaries.csv` (880 rows × 22 conditions) | — |
| 5 | `scripts/study2b_analyse_steering.py` | `per_trace_summaries.csv` | Statistical tests, effect tables | — |
| 6 | `scripts/study2b_plot_figures.py` | Analysis CSVs | Figures (PNG) | — |
| 7 | `scripts/study2b_evaluate_accuracy.py` | Coded traces + manual review | Accuracy tables | — |
| 8 | `scripts/study2b_classify_mechanistic_grounding.py` | Probe F1s + steering effects | Grounding classification | — |
| 9 | `scripts/study2b_compare_accuracy_cross_study.py` | Study 1 + Study 2b accuracy | Cross-study comparison | — |

Auxiliary scripts:
- `scripts/review/` — documents the manual-review accuracy pipeline:
  batch preparation (`prepare_accuracy_review.py`), merging human verdicts
  (`merge_accuracy_review.py`), and reconciling reviewed scores back into
  the accuracy and grounding tables (`update_accuracy_with_review.py`).
  These are method-evidence: they require the full private steered-trace
  corpus to execute, but the source documents how the headline accuracy
  numbers in §7 of the report were produced.

## Re-running analysis from published artefacts

All scripts in `scripts/` that consume analysis CSVs accept `--in-dir` and
`--out-dir` and default to `study2b_steering/results/` — the directory in
this repo containing the published artefacts. From a fresh clone:

```bash
python study2b_steering/scripts/study2b_analyse_steering.py
python study2b_steering/scripts/study2b_plot_figures.py
python study2b_steering/scripts/study2b_classify_mechanistic_grounding.py
python study2b_steering/scripts/study2b_compare_accuracy_cross_study.py
```

reproduces the statistical tests, figures, mechanistic-grounding
classification, and cross-study accuracy comparison from the data already
in this repository, with no GPU or private artefacts required.

Steps 1–4 of the Pipeline (generation, segmentation, coding,
`prepare_analysis`) require the raw `outputs/steered_traces/` corpus,
which is not included in this repository due to size (~3,400 trace JSONs).
The published `results/per_trace_summaries.csv` is the entry point for
analysis-only re-runs; two sub-steps of `study2b_analyse_steering.py`
(JUDGE-verdict aggregation and the JUDGE_neg_2.0 transition matrix) need
the raw per-sentence JSONs and are automatically skipped when the
`outputs/steered_traces/` directory is absent — their published outputs
already live in `results/`.

## Design

- **Intervention**: Additive steering at layer 20 during autoregressive generation
- **Categories**: HYPO, TEST, JUDGE, MONITOR, PLAN
- **Directions**: positive (amplify) and negative (suppress)
- **Strengths**: alpha in {0.5, 1.0, 2.0}
- **Normalisation**: Vectors scaled to mean activation magnitude at target layer
- **Conditions**: 21 steered + 1 baseline = 22 total; 40 traces/condition = 880 traces

## Results

See `results/` for the full set of analysis artefacts:

- `results/study2b_report.md` — narrative write-up of all findings.
- `results/dose_response.csv`, `results/dose_response_per_trace.csv`, `results/dose_response_summary.csv` — Spearman rank correlation between signed α and on-target proportion, per condition.
- `results/on_target_effects.csv` — paired Wilcoxon vs baseline, Cohen's d, Bonferroni-corrected p-values for the 20 α≤1.0 tests.
- `results/category_shift_matrix.csv`, `results/category_shift_matrix_pbonf.csv` — 20×9 off-target shift matrix.
- `results/completion_results.csv`, `results/length_results.csv`, `results/entropy_results.csv` — completion rate, trace length, and transition entropy comparisons.
- `results/judge_verdicts.csv` — JUDGE accept/reject/uncertain breakdown per condition.
- `results/transition_matrix_baseline_*.csv`, `results/transition_matrix_JUDGE_neg_2.0_*.csv` — empirical transition matrices.
- `results/accuracy_*.csv`, `results/study1_accuracy_*.csv`, `results/llm_review_log.json` — manual-review accuracy pipeline outputs.
- `results/mechanistic_grounding_classification.csv` — categories with F1 > 0.65 and a significant steering effect.
- `results/figures/` — 8 figures: dose-response curves, category-shift heatmap, completion-by-condition, trace-length-by-condition, transition-entropy-by-condition, JUDGE verdict breakdown, JUDGE_neg_2.0 transition matrix, accuracy vs Study 1 baseline.
- `results/review_batches/` — text dumps shown to the human reviewer for manual rule-accuracy scoring (per-task batches + working index).

## Key Findings (from `results/study2b_report.md`)

- **4 of 20 on-target tests survive Bonferroni correction** (n=20 family): HYPO_pos_1.0 (+7.16pp, d=+0.70, p_bonf=0.00082), MONITOR_pos_0.5 (+3.79pp, d=+0.85, p_bonf=6e-06), MONITOR_pos_1.0 (+4.26pp, d=+0.62, p_bonf=2e-05), TEST_neg_1.0 (−8.56pp, d=−0.54, p_bonf=0.023).
- **Direction asymmetries.** HYPO_neg reverses (α=1.0 *increases* HYPO by 2.5pp, n.s.) — consistent with probe directions being discriminators rather than generators. MONITOR_neg hits a floor (baseline 2.2%, little room to suppress). PLAN steering is null across both directions and all alphas.
- **Monotonic dose-response only for TEST and MONITOR** (Spearman ρ=0.90, p=0.037 each). HYPO, JUDGE, and PLAN show no monotonic trend.
- **JUDGE_neg_2.0 is catastrophic** (exploratory α): 0/40 completion rate, accept rate crushed to 0.9%, reject rate to 95.8%. The model loops indefinitely.
- **Mechanistic-grounding classification** (`mechanistic_grounding_classification.csv`): HYPO and TEST are *partially grounded* (significant steering, accuracy unmeasurable against Study 1 Set A's 0/69 baseline); JUDGE and PLAN are *weakly grounded* (probe F1 ok but no significant steering); MONITOR is *performative* (significant steering but probe F1 below the 0.6 threshold).
- **Accuracy floor.** Study 1 Set A matched baseline is 0/69 (0.00%) after LLM-reviewed scoring. Four steered conditions reach raw p<0.05 vs that baseline (uncorrected over 21 comparisons): JUDGE_neg_0.5 (21.4%), HYPO_pos_0.5 (16.7%), TEST_neg_1.0 (15.4%), PLAN_neg_0.5 (11.1%). Power-limited because the baseline is a true zero.

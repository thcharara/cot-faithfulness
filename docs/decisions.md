# Key Methodological Decisions

  This document summarises the major design decisions made during the project,
  condensed from detailed validation logs maintained during development.

  ---

  ## Study 1: Corpus Generation

  ### Why DeepSeek-R1-Distill-Llama-8B

  DeepSeek-R1-Distill models produce extended reasoning natively via RL-distilled
  training, generating `<think>...</think>` blocks without prompting. The 8B
  variant fits on hardware (RTX 3080, 10.7 GB VRAM) with 4-bit NF4 quantization.
  Activations are extracted in fp16 precision regardless of weight quantization.

  ### Why no think-aloud coaching in the prompt

  The model receives only the task description (panel descriptions + "infer the
  hidden rule"). No reasoning scaffolding, worked examples, or explicit
  instructions to "think step by step" are included. Rationale: such instructions
  would confound the faithfulness analysis — we could not distinguish genuine
  reasoning from coached behaviour. The model produces multi-step reasoning natively.

  ### Why two trace sets (Set A and Set B)

  - **Set A** (40 traces/task, max 4,096 tokens): Unconstrained generation.
    Primary dataset for Studies 1 and 2. Truncation is acceptable since each
    sentence is analysed independently.
  - **Set B** (40 traces/task, 10 at 4,096 + 30 at 6,144 tokens): Includes a
    completion cue ("commit to your best hypothesis"). Functions as a stopping
    criterion analogous to the time limit in human studies. Needed for accuracy
    reporting and potential resampling experiments.

  ### Why temperature 0.6

  Pilot testing across temp=0.0, 0.6, 0.8 showed that greedy decoding produces
  deterministic traces, temp=0.8 occasionally produces unproductive tangents, and
  temp=0.6 gives good hypothesis diversity. This also matches Bogdan et al.'s
  generation settings (temp=0.6, top_p=0.95).

  ### Sentence segmentation: spaCy with the dependency parser

  Segmentation uses spaCy `en_core_web_sm` with the dependency parser enabled
  (NER and lemmatizer disabled, `max_length = 2,000,000`). NLTK's Punkt-based
  `sent_tokenize` was rejected because it over-splits enumerated lists
  ("1. Panel A... 2. Panel B...") that are frequent in the model's reasoning.
  Token counting uses the model's own tokenizer so that sentence boundaries
  align with the residual-stream activations extracted in Study 2.

  Known limitations: 1.6% of sentences are <3 tokens (mostly enumeration
  artefacts like "2.", "B.", "Hmm."), and 0.3% are >100 tokens (tabular
  enumerations without sentence-ending punctuation). These are merged or
  filtered downstream by the preprocessing pipeline.

  ### Repetition loops are kept, not excluded

  About 22% of traces contain repetition loops (a 50+ character substring
  appearing three or more times), ranging from mild (5–6 repeats) to severe
  (43 repeats in one trace). All such traces are retained in the corpus.
  Excluding them would bias the corpus away from a genuine behavioural
  property of this model under open-ended inductive reasoning. The
  per-sentence segmentation provides the granularity needed to flag and
  analyse loop segments separately in downstream studies.

  ---

  ## Study 1: Taxonomy

  ### Why a single integrated taxonomy (not dual)

  The pilot used two separate taxonomies — an 8-category human-derived scheme
  (Taxonomy A) and a 9-category LLM-native scheme from Bogdan et al. (Taxonomy B).
  Dual-coding by Claude Sonnet and GPT-4o produced poor inter-coder reliability on
  key categories: EVAL kappa=0.29, active_computation kappa=0.27, self_checking
  kappa=0.13.

  Root cause: both taxonomies' definitions were calibrated for other domains
  (think-aloud protocols and math problem-solving respectively). In Zendo traces,
  nearly every sentence occurs in a hypothesis-testing context, making boundaries
  like PROC/EVAL inherently ambiguous with those definitions.

  Resolution: a single integrated taxonomy using Taxonomy A as the backbone,
  incorporating useful distinctions from Taxonomy B, with domain-specific
  operational definitions developed through iterative manual coding.

  ### The TEST/JUDGE boundary rule

  The TEST/JUDGE distinction was the central coding challenge of the pilot, and
  the source of most disagreements. The final rule: **JUDGE is reserved
  exclusively for hypothesis-level verdicts** — sentences that conclude whether
  the overall candidate rule is viable. Panel- or feature-level characterisations
  ("So, alternating colours," "Doesn't alternate") are coded TEST regardless of
  whether their polarity is positive or negative.

  Diagnostic for coders: *"Is the sentence about what was found at the panel or
  feature level?"* → TEST. *"Is the sentence concluding whether the hypothesis
  works?"* → JUDGE. This removed the polarity bias that had driven the early
  TEST/JUDGE disagreements and stabilised category boundaries across the corpus.

  ### Metadata flags carried by each sentence

  Beyond the micro-label, each coded sentence carries operational metadata:

  - **TEST flags:** `test_context` ∈ {`post_hypothesis`, `pre_hypothesis`,
    `post_rule`}; `specificity` ∈ {`within_panel`, `across_panels`}.
  - **JUDGE flags:** `judgement` ∈ {`accept`, `reject`, `uncertain`}.
  - **All sentences:** `confidence` ∈ {`high`, `medium`}.

  Two metadata fields were deliberately excluded from the auto-coder:
  `representation` (extract vs derived — boundary shifted after the JUDGE
  decision above and was never stable enough to code reliably), and `hypo_status`
  (novel/revised/repeated — requires long-range semantic comparison and is
  computed post-hoc via sentence embeddings, see Post-Processing below).

  ### Why Claude Sonnet via Batch API

  The pilot showed that coding quality depends on domain-specific few-shot
  examples more than model choice. Claude Sonnet with 26 curated examples (from 3
  held-out traces) achieved kappa=0.83 against manual codes. The Anthropic Batch
  API provides 50% cost reduction over standard API calls.

  ### Few-shot curation and validation-set disjointness

  Few-shot examples were drawn from only **3 source traces** (setA/task1/trace_015,
  setA/task3/trace_010, setB/task3/trace_020). Inter-coder κ was then computed
  against **7 separate validation traces** that contributed zero examples to the
  prompt. This disjoint design ensures the reported κ reflects out-of-distribution
  generalisation, not memorisation of the prompt examples. Category-level example
  counts reflect coding-boundary difficulty: TEST and JUDGE are overrepresented
  (7 and 5 examples respectively) because that is the primary discriminative
  boundary.

  ### Validation gate thresholds

  Two thresholds are applied to the 7-trace validation set:

  - **Per-category (micro):** Cohen's κ ≥ 0.50 to use the category in
    fine-grained probe analysis.
  - **Overall (macro):** Cohen's κ ≥ 0.65 for the corpus to be considered
    reliable for downstream use.

  Categories that fall below the per-category threshold are reported at the
  macro (5-category) level only in probe analyses. In the final corpus, all
  categories meet the per-category threshold except SYNTHESIZE (κ = 0.342),
  which is therefore handled at the OBSERVE macro level in cross-study
  comparisons.

  ---

  ## Study 1: Post-Processing

  ### HYPO status classification (novel/revised/repeated)

  Each HYPO sentence is classified based on cosine similarity (all-MiniLM-L6-v2
  embeddings) to all prior HYPOs in the same trace:

  - **Repeated** (similarity >= 0.90): Verbatim or near-verbatim restatement
  - **Revised** (similarity >= 0.70): Substantive modification of a prior hypothesis
  - **Novel** (similarity < 0.70): Genuinely new hypothesis

  Thresholds calibrated on 4,968 HYPO pairs from 10 manually coded traces.

  ### Dependency computation

  Dependencies (`depends_on`) are computed semi-programmatically using label-type
  heuristics within local windows:

  - Default window: 20 sentences
  - HYPO window: 35 sentences (hypotheses reference earlier context)
  - RULE window: 50 sentences (rules reference the most recent accept-JUDGE)

  An API verification pass run by eight parallel agents over all 320 traces
  filled residual empty dependencies and corrected misassignments. Final
  coverage is 98.0% (71,919 / 73,383 sentences with a `depends_on` field).

  ---

  ## Study 1: Analysis

  ### Phase structure: scanning, cycling, convergence

  Traces are decomposed into three sequential phases:

  - **Scanning:** before the first HYPO sentence
  - **Cycling:** from first HYPO through last JUDGE (inclusive)
  - **Convergence:** after the last JUDGE

  Edge cases: traces with no HYPO have 100% scanning; traces with no JUDGE
  after their last HYPO have 0% convergence. The decomposition aligns with the
  theoretical expectation that inductive reasoning proceeds from observation
  through hypothesis testing to conclusion, and the phase proportions quantify
  how much of each trace is devoted to each activity.

  ### Three reasoning strategies (post-hoc partition)

  Review of the phase-structure statistics revealed that not every trace
  follows the canonical scanning → cycling → convergence pattern. Traces fall
  into three strategies, captured in `sequence_characteristics.csv`'s
  `reasoning_strategy` column:

  - **full_cycling** (303 traces): the canonical HYPO → TEST → JUDGE pattern
  - **scan_test_conclude** (14 traces): hypotheses proposed but never formally
    judged — the model commits to a rule without an explicit JUDGE step
  - **direct_insight** (3 traces): a conclusion is reached without any explicit
    HYPO sentence

  All 17 non-cycling traces completed successfully, so these represent
  alternative reasoning pathways rather than coding or completion failures.
  Reasoning strategy is used as a grouping variable in Study 2a (see "Probe
  generalises across reasoning strategies" below).

  ### Bogdan et al. domain comparison via post-hoc mapping

  Rather than re-coding the corpus with Bogdan et al.'s 8-category scheme
  (which the pilot showed produces unreliable boundaries in this domain — see
  "Why a single integrated taxonomy" above), categories are mapped post-hoc:

  - DESCRIBE → `fact_retrieval`
  - TEST → `active_computation`
  - HYPO + PLAN → `plan_generation`
  - SYNTHESIZE + JUDGE(accept) → `result_consolidation`
  - ORIENT → `problem_setup`
  - RULE → `final_answer_emission`
  - JUDGE(reject/uncertain) + MONITOR → `uncertainty_mgmt + self_checking`

  The last bin merges Bogdan's `uncertainty_management` and `self_checking`
  rows: under our taxonomy, MONITOR absorbs both functions, so reporting them
  as separate bars would create an artificial asymmetry. The merge is reported
  honestly in `tables/domain_comparison_bogdan.csv` and the report figure.
  The comparison is therefore about domain differences in reasoning profiles
  rather than taxonomy validation.

  ---

  ## Study 2a: Linear Probes

  ### Why direct PyTorch hooks (not nnsight)

  nnsight was validated in Phase 1 on short sequences but causes OOM on real traces
  (3,000-6,600 tokens) due to proxy object overhead. Direct `register_forward_hook()`
  on the target layer captures identical activations with minimal VRAM overhead
  (peak 7.80 GB vs OOM).

  ### Cross-validation: group-stratified 5-fold by trace

  Probes are evaluated with 5-fold cross-validation, with folds defined by
  trace rather than by sentence. Sentences from a single trace are never split
  across train and test folds. This follows Bogdan et al. and prevents
  intra-trace correlation (sequential sentences sharing context, repetition
  loops, or perseveration) from inflating test-set performance.

  ### Why C=0.01 (not C=1.0)

  Stronger L2 regularisation (C=0.01) outperforms the Bogdan et al. default
  (C=1.0) at every layer, with monotonic improvement as C decreases. The
  4096-dimensional activation space is over-parameterised for 9-class
  classification; heavier shrinkage improves generalisation. We report both C=1.0
  (for direct comparison with Bogdan et al.) and C=0.01 (optimised).

  ### Why mean-pool (not last-token)

  Mean-pooling across sentence tokens consistently outperforms last-token
  activation by 2-3 macro-F1 points. Reasoning category information is distributed
  across token positions, not concentrated at the final token.

  ### Probe generalises across reasoning strategies

  Layer-31 mean-pool C=0.01 probe performance, broken out by the three
  reasoning strategies from Study 1:

  | Strategy | N sentences | N traces | Macro-F1 | Weighted-F1 |
  |---|---:|---:|---:|---:|
  | full_cycling | 15,023 | 61 | 0.757 | 0.830 |
  | scan_test_conclude | 125 | 3 | 0.737 | 0.814 |
  | direct_insight | 0 | 0 | n/a | n/a |

  Performance on `scan_test_conclude` is only ~2 points below the canonical
  strategy, suggesting the probe is not exploiting strategy-specific surface
  patterns. Activation-centroid cos-similarity is 0.82–0.99 across strategies
  for full_cycling, with TEST and MONITOR showing the largest cross-strategy
  drift. This rules out the most obvious confound for the downstream steering
  experiments.

  ### Category directions are near-orthogonal

  Within-layer cosine similarity between all 36 pairs of category direction
  vectors at layer 31:

  - Maximum off-diagonal: ORIENT ↔ HYPO = 0.001 (essentially orthogonal)
  - Minimum off-diagonal: SYNTHESIZE ↔ JUDGE = −0.228

  All off-diagonal values fall between −0.23 and +0.01. Each category occupies
  a distinct, near-orthogonal direction in the 4096-dimensional space; small
  negative correlations suggest that amplifying one category's representation
  may actively suppress others. This is the prerequisite that makes the Study
  2b interpretation of "on-target" vs "off-target" steering effects meaningful.

  ---

  ## Study 2b: Steering

  ### Why layer 20

  Layer 20 was selected based on: peak steering impact in Venhoff et al. (their
  Figure 3), good cross-layer direction stability with layer 31 (mean cos_sim =
  0.295), and flat probe performance from layer 10 onwards.

  ### Why activation-scale normalisation

  Following Venhoff et al., steering vectors are normalised so their L2 norm
  matches the mean activation magnitude at the target layer (12.69 at layer 20).
  The alpha parameter then scales relative to this baseline — alpha=1.0 adds one
  "natural-scale" unit of the category direction.

  ### Paired-seed design enables Wilcoxon signed-rank tests

  All 22 coded conditions (baseline + 20 conditions at α ∈ {0.5, 1.0} +
  JUDGE_neg_2.0) use the **same 40 seeds**
  (`task_id * 1000 + 800 + trace_id`, trace_id 1–10 across 4 tasks). Because
  seeds are paired across conditions, per-trace category proportions are
  comparable on a within-seed basis, and the natural inferential test is the
  **paired Wilcoxon signed-rank** rather than an unpaired or pooled
  comparison. This both increases power and protects against trace-length
  asymmetries (a single long, single-category trace can dominate pooled
  corpus-level percentages).

  ### α=2.0 is generated but not coded (except JUDGE_neg_2.0)

  All 30 steered conditions (5 categories × 2 directions × 3 alphas) plus
  baseline were generated, but only α ∈ {0.5, 1.0} were coded. Inspection
  showed that most α=2.0 conditions degenerate into incoherent text (token-
  level noise, repetition loops, or near-empty outputs after segmentation),
  so taxonomy coding would be meaningless. A subsequent re-inspection found
  that **JUDGE_neg_2.0 preserves recognisable reasoning structure** and was
  coded after the fact, giving a three-point dose-response curve for the
  JUDGE_neg arm and 22 total coded conditions.

  ### SYNTHESIZE/TEST cross-study labelling drift

  On the baseline condition (which shares seeds and prompt with Study 1
  Set A), the Study 2b coded labels show **+7.4pp on TEST and −2.6pp on
  SYNTHESIZE** relative to Study 1 Set A. Both studies code with the same
  model (Claude Sonnet), the same Phase 3 integrated taxonomy, and the
  same few-shot prompt, so the drift is not attributable to a
  methodological difference between coding runs. Instead, it reflects the
  known-fuzzy SYNTHESIZE/TEST boundary already documented in Study 1,
  where SYNTHESIZE has the lowest per-category κ (0.342, Decision 22).
  Sentences sitting on this boundary (e.g. "the non-starred panels have
  different color arrangements", "more variation here") can fall to either
  side across independent coding passes even when model, prompt, and
  taxonomy are held fixed.

  - **Within-Study-2b comparisons** (the focus of the steering analysis) are
    unaffected: all 22 conditions share a single coding pass under the same
    prompt, so any boundary-fuzziness applies symmetrically across
    conditions.
  - **Cross-study comparisons** (Study 2b baseline vs Study 1 Set A) should
    treat the OBSERVE macro (SYNTHESIZE + DESCRIBE) as the safer unit of
    analysis rather than SYNTHESIZE alone, since the SYNTHESIZE/TEST
    boundary absorbs the drift while the OBSERVE/INVESTIGATE macro boundary
    does not.

  This is restated inline in `study2b_steering/results/study2b_report.md`
  §1 and applies wherever a cross-study claim is made.

  ### Bonferroni family: the 20 on-target tests at α ≤ 1.0

  Multiple-comparison correction is applied across the **20 on-target tests
  at α ∈ {0.5, 1.0}** (5 categories × 2 directions × 2 alphas), via
  Bonferroni. JUDGE_neg_2.0 is reported as exploratory and is not in the
  family. Off-target shift tests (the 20 × 9 shift matrix) are reported as
  raw p-values without correction; the matrix is descriptive (heatmap)
  rather than inferential.

  Result: 4 of 20 conditions are significant at p_bonf < 0.05 —
  HYPO_pos_1.0, MONITOR_pos_0.5, MONITOR_pos_1.0, TEST_neg_1.0.

  ### Accuracy: heuristic + manual review (not LLM scoring)

  Per-trace accuracy of the model's stated rule is scored in two passes: a
  conservative regex/keyword heuristic per task, followed by manual review
  of every completed trace via `accuracy_manual_review.csv`.

  Floor finding: Study 1 Set A's accuracy is **0/69**. With a
  true-zero baseline, every steered cell with even one correct rule trivially
  beats the baseline rate; Fisher's exact tests are reported but their
  interpretation is power-limited. Reported honestly in §7 of the Study 2b
  report.


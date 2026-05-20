# Chain-of-Thought Faithfulness Verification

Do large language models actually use their chain-of-thought reasoning,
or is it post-hoc rationalisation? This project develops a multi-method
framework for verifying CoT faithfulness by combining behavioural
taxonomy coding with mechanistic interpretability techniques (linear
probes and activation steering).

We generate a corpus of 320 reasoning traces from DeepSeek-R1-Distill-
Llama-8B on visual inductive reasoning tasks (Zendo), develop a 9-category
taxonomy of reasoning operations, auto-code 73,383 sentences via Claude
Sonnet, then test whether the same categories are linearly decodable from
the model's residual stream — and whether amplifying or suppressing those
directions causally shifts the model's reasoning behaviour.

## Status

| Study | Description | Status |
|---|---|---|
| **Study 1** | Corpus generation, taxonomy development, auto-coding, sequential analysis | Completed |
| **Study 2a** | Activation extraction and linear probe training | Completed |
| **Study 2b** | Steering vector interventions | Completed |

## Key Results

**Study 1 — Taxonomy & Coding**
- 9 micro-labels: ORIENT, DESCRIBE, SYNTHESIZE, HYPO, TEST, JUDGE, PLAN, MONITOR, RULE
- Inter-rater reliability (Cohen's kappa): **0.83** (micro-level, 7 validation traces)
- 8/9 categories exceed kappa >= 0.50; SYNTHESIZE boundary case (0.34)
- Dominant reasoning pattern: TEST (50.2%) > HYPO (17.5%) > JUDGE (12.3%)
- Dependency coverage: 98.0% of sentences linked

**Study 2a — Linear Probes**
- Best macro-F1: **0.76** (layer 31, mean pooling, C=0.01)
- Exceeds Bogdan et al. (2025) reference of 0.71 despite smaller model (8B vs 14B) and more categories (9 vs 8)
- 7/9 categories mechanistically grounded (F1 > 0.65)

**Study 2b — Steering**
- Additive steering at layer 20 with probe-derived directions; 22 conditions × 40 traces = 880 traces
- **4/20 on-target tests significant after Bonferroni:** HYPO_pos_1.0 (+7.16pp, d=+0.70), MONITOR_pos_0.5 (+3.79pp, d=+0.85), MONITOR_pos_1.0 (+4.26pp, d=+0.62), TEST_neg_1.0 (−8.56pp, d=−0.54)
- Monotonic dose-response for **TEST and MONITOR only** (Spearman ρ=0.90, p=0.037 each)
- JUDGE_neg_2.0 (exploratory α): 0% completion, accept rate crushed to 0.9%

## Directory Structure

```
├── data/                    Stimuli, taxonomy prompts, validation traces
├── study1_corpus/           Corpus generation, coding, and analysis pipeline
│   ├── scripts/             9 pipeline scripts (generate → code → validate → analyse)
│   ├── notebooks/           9 analysis notebooks + helper module
│   └── results/             Report, figures, and tables
├── study2a_probes/          Activation extraction and linear probe training
│   ├── scripts/             6 pipeline scripts
│   ├── notebooks/           Probe analysis notebook
│   └── results/             Confusion matrices and probe report
├── study2b_steering/        Steering vector experiments
│   ├── scripts/             9 pipeline scripts + review/ subdir
│   ├── notebooks/           Steering analysis notebook
│   └── results/             Effect tables, figures, report, review batches
├── tests/                   Environment + sample-corpus smoke tests
└── docs/                    Methodological decisions
```

## Reproduction

### Requirements

- Python 3.12+
- NVIDIA GPU with >= 10 GB VRAM (for model inference and activation extraction)
- ~20 GB disk for model weights

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Curated minimal install (loose floors, easier to maintain):
pip install -r requirements.txt
# Or exact-version reproduction of the runtime used for the reported results:
# pip install -r requirements-lock.txt

# Download model
python download_model.py

# Download spaCy model
python -m spacy download en_core_web_sm
```

Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` (required only
for the Study 1 auto-coding step).

Verify the environment before running anything heavyweight:

```bash
python tests/verify_install.py      # CPU, ~5 s — imports + GPU detection
python tests/smoke_sample_trace.py  # CPU, ~5 s — sample corpus schema check
python tests/test_model_4bit.py     # GPU, ~1 min — 4-bit model + native <think>
```

### Study 1 Pipeline

```bash
# 1. Generate traces (requires GPU, ~2 hours)
python study1_corpus/scripts/study1_generate_traces.py

# 2. Segment into sentences
python study1_corpus/scripts/study1_segment_traces.py

# 3. Preprocess (fragment merging, truncation detection)
python study1_corpus/scripts/study1_preprocess_traces.py

# 4. Auto-code via Anthropic API (requires ANTHROPIC_API_KEY)
python study1_corpus/scripts/study1_batch_code_traces.py submit --traces all
python study1_corpus/scripts/study1_batch_code_traces.py poll
python study1_corpus/scripts/study1_batch_code_traces.py download

# 5. Validate
python study1_corpus/scripts/study1_compute_kappa.py

# 6. Post-processing
python study1_corpus/scripts/study1_compute_hypo_status.py --repeat-threshold 0.90 --revise-threshold 0.70
python study1_corpus/scripts/study1_compute_dependencies.py
```

### Study 2a Pipeline

```bash
# 1. Prepare probe data (CV splits, tokenization)
python study2a_probes/scripts/study2a_prepare_probe_data.py

# 2. Extract activations (requires GPU, ~4 hours)
python study2a_probes/scripts/study2a_extract_activations.py --layer 0
python study2a_probes/scripts/study2a_extract_activations.py --layer 10
python study2a_probes/scripts/study2a_extract_activations.py --layer 20
python study2a_probes/scripts/study2a_extract_activations.py --layer 31

# 3. Train probes (~10 min)
python study2a_probes/scripts/study2a_train_probes.py
```

### Study 2b Pipeline

```bash
# 1. Extract steering directions from probes
python study2a_probes/scripts/study2a_extract_probe_direction.py

# 2. Generate steered traces (requires GPU)
python study2b_steering/scripts/study2b_steer_generate.py --category HYPO --direction pos --alpha 1.0

# 3. Segment and code steered traces
python study2b_steering/scripts/study2b_segment_steered_traces.py
python study2b_steering/scripts/study2b_code_steered_traces.py

# 4. Analyse results
python study2b_steering/scripts/study2b_analyse_steering.py
```

## Taxonomy

| Micro Label | Macro Category | Description |
|---|---|---|
| ORIENT      | SETUP        | Task engagement, paraphrasing the problem |
| DESCRIBE    | OBSERVE      | Raw feature readout from panels |
| SYNTHESIZE  | OBSERVE      | Cross-panel aggregation of observations |
| HYPO        | INVESTIGATE  | Hypothesis proposal or refinement |
| TEST        | INVESTIGATE  | Evidence gathering against a hypothesis |
| JUDGE       | INVESTIGATE  | Accept/reject/uncertain verdict |
| PLAN        | REGULATE     | Strategy articulation |
| MONITOR     | REGULATE     | Progress checking, self-correction |
| RULE        | CONCLUDE     | Final rule statement |

## Related Work / Repositories

This project builds on and extends:

- Bogdan et al. (2025), *Thought Anchors: Which LLM Reasoning Steps Matter?*
  [arXiv:2506.19143](https://arxiv.org/abs/2506.19143).
  Replication: [thcharara/thought-anchors](https://github.com/thcharara/thought-anchors).
- Venhoff et al. (2025), *Understanding Reasoning in Thinking Language Models via Steering Vectors.*
  [arXiv:2506.18167](https://arxiv.org/abs/2506.18167).
  Replication: [thcharara/steering-thinking-llms](https://github.com/thcharara/steering-thinking-llms).

## Licence

- Code: MIT — see [`LICENSE`](LICENSE).
- Data, figures, and tables: CC BY 4.0 — see [`LICENSE-DATA`](LICENSE-DATA).

## Citation

If you use this code, corpus, or methodology, please cite via the
[`CITATION.cff`](CITATION.cff) file (GitHub renders a "Cite this repository"
button on the sidebar that exposes BibTeX and APA forms automatically).

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
  | **Study 2b** | Steering vector interventions | In progress |

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

  **Study 2b — Steering (Preliminary)**
  - Additive steering at layer 20 with probe-derived directions
  - HYPO_pos shows +12.5% on-target shift at alpha=1.0
  - Full dose-response analysis in progress

  ## Directory Structure

  ├── data/                    Stimuli, taxonomy prompts, validation traces      
  ├── study1_corpus/           Corpus generation, coding, and analysis pipeline  
  │   ├── scripts/             10 pipeline scripts (generate → code → validate → analyse)   
  │   ├── notebooks/           9 analysis notebooks + helper module  
  │   └── results/             Report, figures, and tables    
  ├── study2a_probes/          Activation extraction and linear probe training    
  │   ├── scripts/             6 pipeline scripts  
  │   ├── notebooks/           Probe analysis notebook    
  │   └── results/             Confusion matrices and probe report  
  ├── study2b_steering/        Steering vector experiments      
  │   ├── scripts/             4 pipeline scripts   
  │   ├── notebooks/           Steering analysis notebook     
  │   └── results/             Steering effect summaries and figures   
  └── docs/                    Taxonomy reference, methodological decisions     

  ## Reproduction

  ### Requirements

  - Python 3.12+
  - NVIDIA GPU with >= 10 GB VRAM (for model inference and activation extraction)
  - ~20 GB disk for model weights

  ### Setup

  ```bash
  python -m venv .venv
  source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  pip install -r requirements.txt

  # Download model
  python download_model.py

  # Download spaCy model
  python -m spacy download en_core_web_sm

  Study 1 Pipeline

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

  Study 2a Pipeline

  # 1. Prepare probe data (CV splits, tokenization)
  python study2a_probes/scripts/study2a_prepare_probe_data.py

  # 2. Extract activations (requires GPU, ~4 hours)
  python study2a_probes/scripts/study2a_extract_activations.py --layer 0
  python study2a_probes/scripts/study2a_extract_activations.py --layer 10
  python study2a_probes/scripts/study2a_extract_activations.py --layer 20
  python study2a_probes/scripts/study2a_extract_activations.py --layer 31

  # 3. Train probes (~10 min)
  python study2a_probes/scripts/study2a_train_probes.py

  Study 2b Pipeline

  # 1. Extract steering directions from probes
  python study2a_probes/scripts/study2a_extract_probe_directions.py

  # 2. Generate steered traces (requires GPU)
  python study2b_steering/scripts/study2b_steer_generate.py --category HYPO --direction pos --alpha 1.0

  # 3. Segment and code steered traces
  python study2b_steering/scripts/study2b_segment_steered_traces.py
  python study2b_steering/scripts/study2b_code_steered_traces.py

  # 4. Analyse results
  python study2b_steering/scripts/study2b_analyse_steering.py

  Taxonomy

  ┌─────────────┬────────────────┬───────────────────────────────────────────┐
  │ Micro Label │ Macro Category │                Description                │
  ├─────────────┼────────────────┼───────────────────────────────────────────┤
  │ ORIENT      │ SETUP          │ Task engagement, paraphrasing the problem │
  ├─────────────┼────────────────┼───────────────────────────────────────────┤
  │ DESCRIBE    │ OBSERVE        │ Raw feature readout from panels           │
  ├─────────────┼────────────────┼───────────────────────────────────────────┤
  │ SYNTHESIZE  │ OBSERVE        │ Cross-panel aggregation of observations   │
  ├─────────────┼────────────────┼───────────────────────────────────────────┤
  │ HYPO        │ INVESTIGATE    │ Hypothesis proposal or refinement         │
  ├─────────────┼────────────────┼───────────────────────────────────────────┤
  │ TEST        │ INVESTIGATE    │ Evidence gathering against a hypothesis   │
  ├─────────────┼────────────────┼───────────────────────────────────────────┤
  │ JUDGE       │ INVESTIGATE    │ Accept/reject/uncertain verdict           │
  ├─────────────┼────────────────┼───────────────────────────────────────────┤
  │ PLAN        │ REGULATE       │ Strategy articulation                     │
  ├─────────────┼────────────────┼───────────────────────────────────────────┤
  │ MONITOR     │ REGULATE       │ Progress checking, self-correction        │
  ├─────────────┼────────────────┼───────────────────────────────────────────┤
  │ RULE        │ CONCLUDE       │ Final rule statement                      │
  └─────────────┴────────────────┴───────────────────────────────────────────┘

  Related Work/Repositories

  This project builds on and extends:
  - https://arxiv.org/abs/2503.XXX — Linear probes for reasoning categories
    [thought-anchors](https://github.com/thcharara/thought-anchors) — Thought masking and rollout analysis (Bogdan et al. 2025 replication)
  - https://arxiv.org/abs/2503.XXX — Steering vectors in thinking LLMs
    [steering-thinking-llms](https://github.com/thcharara/steering-thinking-llms) — Steering vector training and evaluation (Venhoff et al. 2025 replication)
  
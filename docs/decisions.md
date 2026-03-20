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
  incorporating useful distinctions from Taxonomy B, 
  with domain-specific operational definitions developed through iterative manual coding.

  ### Why Claude Sonnet via Batch API

  The pilot showed that coding quality depends on domain-specific few-shot
  examples more than model choice. Claude Sonnet with 26 curated examples (from 3
  held-out traces) achieved kappa=0.83 against manual codes. The Anthropic Batch
  API provides 50% cost reduction over standard API calls.

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

  API integration used after to fill any missing dependecies and check overall accuracy.

  ---

  ## Study 2a: Linear Probes

  ### Why direct PyTorch hooks (not nnsight)

  nnsight was validated in Phase 1 on short sequences but causes OOM on real traces
  (3,000-6,600 tokens) due to proxy object overhead. Direct `register_forward_hook()`
  on the target layer captures identical activations with minimal VRAM overhead
  (peak 7.80 GB vs OOM).

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


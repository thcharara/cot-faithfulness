  # Study 2b: Steering Vector Interventions

  Tests whether probe-derived direction vectors causally influence the
  model's reasoning behaviour when added to the residual stream during
  generation.

  **Status: In progress**

  ## Pipeline

  | Step | Script | Input | Output | Requires |
  |---|---|---|---|---|
  | 1 | `study2b_steer_generate.py` | Model + steering vectors | Steered traces | GPU |
  | 2 | `study2b_segment_steered_traces.py` | Steered traces | Segmented traces | spaCy |
  | 3 | `study2b_analyse_steering.py` | Coded traces | Effect analysis | — |

  ## Design

  - **Intervention**: Additive steering at layer 20 during autoregressive generation
  - **Categories**: HYPO, TEST, JUDGE, MONITOR, PLAN
  - **Directions**: positive (amplify) and negative (suppress)
  - **Strengths**: alpha in {0.5, 1.0, 2.0}
  - **Normalisation**: Vectors scaled to mean activation magnitude at target layer


 Full dose-response and off-target analysis in progress

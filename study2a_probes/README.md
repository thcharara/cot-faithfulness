 # Study 2a: Linear Probes

  Trains linear probes on residual-stream activations to test whether
  the 9 taxonomy categories are linearly decodable from the model's
  internal representations.

  ## Pipeline

  | Step | Script | Input | Output | Requires |
  |---|---|---|---|---|
  | 1 | `study2a_prepare_probe_data.py` | Coded traces | CV splits, metadata | — |
  | 2 | `study2a_extract_activations.py` | Model + traces | NPZ files (~4.5 GB) | GPU, ~4 hours |
  | 3 | `study2a_train_probes.py` | Activations + splits | Probe models + confusion matrices | — |
  | 4 | `study2a_complete_probe_sweep.py` | Activations | Full C/layer grid | — |
  | 5 | `study2a_extract_probe_directions.py` | Trained probes | Steering direction vectors | — |
  | 6 | `study2a_analyse_probe_by_strategy.py` | Probes + metadata | Per-strategy breakdown | — |

  ## Key Results

  - **Best macro-F1**: 0.76 (layer 31, mean pooling, C=0.01)
  - **Reference**: Bogdan et al. (2025) achieved 0.71 with Qwen-14B on math
  - **7/9 categories** mechanistically grounded (F1 > 0.65)
  - Partial grounding: SYNTHESIZE (0.44), MONITOR (0.49)

  ## Notes

  - Activation extraction produces ~4.5 GB of NPZ files (not included in repo)
  - Run `study2a_extract_activations.py --layer N` for each layer: 0, 10, 20, 31

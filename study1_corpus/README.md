  # Study 1: Reasoning Trace Corpus

  Generates, segments, and auto-codes 320 reasoning traces from DeepSeek-R1-
  Distill-Llama-8B solving visual inductive reasoning (Zendo) tasks.

  ## Pipeline

  Run scripts in order:

  | Step | Script | Input | Output | Requires |
  |---|---|---|---|---|
  | 1 | `study1_generate_traces.py` | `data/stimuli/` | Raw traces (320) | GPU |
  | 2 | `study1_segment_traces.py` | Raw traces | Segmented traces | spaCy |
  | 3 | `study1_preprocess_traces.py` | Segmented traces | Cleaned traces (73,383 sentences) | — |
  | 4 | `study1_batch_code_traces.py` | Cleaned traces + taxonomy prompt | Coded traces | Anthropic API key |       
  | 5 | `study1_compute_kappa.py` | Coded traces + manual traces | Kappa results (0.83) | — |
  | 6 | `study1_compute_hypo_status.py` | Coded traces | HYPO status (novel/revised/repeated) | sentence-transformers |
  | 7 | `study1_compute_dependencies.py` | Coded traces + HYPO status | Dependency graphs (98% coverage) | — |      

  ## Analysis Notebooks

  | Notebook | Content |
  |---|---|
  | `study1_01_summary.ipynb` | Corpus overview and key statistics |
  | `study1_01a_quality.ipynb` | Data quality flags and assessment |
  | `study1_01b_distributions.ipynb` | Label frequency distributions |
  | `study1_01c_validation.ipynb` | Inter-rater reliability (Cohen's kappa) |
  | `study1_01d_dependencies.ipynb` | Dependency graph analysis |
  | `study1_01e_hypotheses.ipynb` | Hypothesis lifecycle (novel/revised/repeated) |
  | `study1_02_characterisation.ipynb` | Reasoning pattern characterisation |
  | `study1_03_dependency_graphs.ipynb` | Graph visualisation and metrics |
  | `study1_04_report.ipynb` | Final report generation |

  ## Key Results

  - **Corpus**: 320 traces, 73,383 sentences, 50.9% completed (reach RULE)
  - **Inter-rater kappa**: 0.83 (micro), 8/9 labels above 0.50
  - **Dominant pattern**: TEST (50.2%) → HYPO (17.5%) → JUDGE (12.3%)
  - **HYPO lifecycle**: 25.2% novel, 45.6% revised, 29.1% repeated

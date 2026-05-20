# Tests

Three smoke / integration tests for verifying the environment and the shipped
corpus before running the full pipeline.

| Test | Cost | What it checks |
|---|---|---|
| `verify_install.py` | CPU, ~5 s | All required libraries import; CUDA is detected if present. |
| `smoke_sample_trace.py` | CPU, ~5 s | The 20-trace public sample parses, has the expected schema, and uses only the 9 micro / 5 macro labels from the taxonomy. |
| `test_model_4bit.py` | GPU, ~1 min + ~16 GB download | Loads DeepSeek-R1-Distill-Llama-8B in 4-bit quantization and confirms it emits a `<think>...</think>` block. |

Run from the repository root:

```bash
python tests/verify_install.py
python tests/smoke_sample_trace.py
python tests/test_model_4bit.py   # requires GPU + downloaded model
```

The first two are the right pre-flight check for a fresh clone — they don't
need a GPU or any model weights. The third is an integration test for the GPU
path; only run it after `python download_model.py` succeeds.

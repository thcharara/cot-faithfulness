"""Quick environment check: imports + GPU detection.

Run after `pip install -r requirements.txt` (or `requirements-lock.txt`) to
confirm the runtime is correctly configured. CPU-only. Does not download
the model or hit any external service.

Usage:
    python tests/verify_install.py
"""

import sys

print(f"Python: {sys.version.splitlines()[0]}")
print()

libs = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "anthropic",
    "dotenv",
    "spacy",
    "sentence_transformers",
    "sklearn",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
]

failures = []
for module in libs:
    try:
        m = __import__(module)
        version = getattr(m, "__version__", "ok")
        print(f"  [OK]  {module:25s} {version}")
    except ImportError as e:
        print(f"  [--]  {module:25s} not installed ({e})")
        failures.append(module)

print()

try:
    import torch

    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_ok}")
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:           {props.total_memory / 1e9:.1f} GB")
    else:
        print("  GPU steps (model inference, activation extraction, steering) "
              "will not run on this machine.")
except Exception as e:
    print(f"  Could not query CUDA: {e}")

print()
if failures:
    print(f"Missing modules: {', '.join(failures)}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)
print("Environment check passed.")

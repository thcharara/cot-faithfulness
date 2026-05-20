"""Integration test: load DeepSeek-R1-Distill-Llama-8B in 4-bit and generate.

Requires:
    - NVIDIA GPU with >= 10 GB VRAM
    - Model downloaded to `models/DeepSeek-R1-Distill-Llama-8B/` (run
      `python download_model.py` first)
    - bitsandbytes installed

Confirms that the model loads, runs forward, and emits its native
`<think>...</think>` reasoning block on a simple arithmetic prompt.

Usage:
    python tests/test_model_4bit.py
"""

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "DeepSeek-R1-Distill-Llama-8B"

if not MODEL_PATH.exists():
    print(f"FAIL  model not found at {MODEL_PATH}")
    print("Run: python download_model.py")
    sys.exit(1)

if not torch.cuda.is_available():
    print("FAIL  CUDA unavailable; this test requires a GPU.")
    sys.exit(1)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

print("Loading model in 4-bit quantization...")
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=quant,
    device_map="auto",
)

print(f"Model loaded. Device: {next(model.parameters()).device}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

prompt = "Solve this step by step: What is 15 * 23?"
messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

print(f"\nGenerating response to: {prompt!r}")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
    )

response = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False
)
print(f"\n--- MODEL OUTPUT ---\n{response}\n--- END OUTPUT ---")

has_think = "<think>" in response
print(f"\nNative thinking block present: {has_think}")
sys.exit(0 if has_think else 1)

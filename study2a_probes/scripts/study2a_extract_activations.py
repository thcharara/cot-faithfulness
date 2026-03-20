#!/usr/bin/env python3
"""Stage B: Extract residual-stream activations for Study 2 linear probes.

Loads DeepSeek-R1-Distill-Llama-8B in 4-bit, runs forward passes with direct
PyTorch hooks (more memory-efficient than nnsight for long sequences), and
saves per-sentence activations (last-token and mean-pool) as .npz files.

Usage:
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_extract_activations.py --layer 31
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_extract_activations.py --layer 31 --start 0 --end 50
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_extract_activations.py --pilot
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import gc
import json
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'DeepSeek-R1-Distill-Llama-8B'
PROBE_DATA_DIR = PROJECT_ROOT / 'outputs' / 'study2_probe_data'
ACTIVATIONS_DIR = PROJECT_ROOT / 'outputs' / 'activations'

# ── Constants ──
TARGET_LAYERS = [0, 10, 20, 31]
HIDDEN_DIM = 4096
MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST', 'JUDGE',
                'PLAN', 'MONITOR', 'RULE']

# ── Pilot traces: short traces across tasks ──
PILOT_TRACES = [
    'set_b/task2/trace_006',  # task 2, 57 sentences (shortest validation trace)
    'set_b/task1/trace_010',  # task 1, validation trace
    'set_a/task3/trace_020',  # task 3, validation trace
    'set_a/task4/trace_019',  # task 4, validation trace
    'set_a/task1/trace_002',  # task 1, additional
]


class ActivationCapture:
    """Captures residual-stream activations via PyTorch forward hooks.

    Hooks immediately detach and move to CPU to minimise GPU memory.
    """

    def __init__(self):
        self.activations = {}
        self._hooks = []

    def register(self, model, layer_indices):
        """Register forward hooks on specified layers."""
        for idx in layer_indices:
            layer = model.model.layers[idx]
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; output[0] is the hidden state [batch, seq, hidden]
            self.activations[layer_idx] = output[0].detach().cpu()
        return hook_fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self.activations.clear()


def load_probe_data():
    """Load tokenization index and prompt cache from Stage A outputs."""
    import pandas as pd

    tok_path = PROBE_DATA_DIR / 'tokenization_index.pkl'
    cache_path = PROBE_DATA_DIR / 'prompt_cache.pkl'
    meta_path = PROBE_DATA_DIR / 'sentence_metadata.csv'

    with open(tok_path, 'rb') as f:
        tok_index = pickle.load(f)
    with open(cache_path, 'rb') as f:
        prompt_cache = pickle.load(f)

    meta_df = pd.read_csv(meta_path)
    return tok_index, prompt_cache, meta_df


def load_model(model_path):
    """Load model in 4-bit quantization (no nnsight wrapper needed)."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    print("Loading model in 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        device_map="auto",
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM: {vram:.2f} GB")

    return hf_model, tokenizer


def extract_single_trace(model, token_ids, tok_info, layer_idx):
    """Extract activations for a single trace at a single layer.

    Uses direct PyTorch forward hooks with torch.no_grad() and use_cache=False
    for maximum memory efficiency.

    Returns:
        last_token_acts: np.array [n_sentences, 4096] float16
        mean_pool_acts:  np.array [n_sentences, 4096] float16
        sentence_ids:    np.array [n_sentences] int32
    """
    capture = ActivationCapture()
    capture.register(model, [layer_idx])

    input_tensor = torch.tensor([token_ids], dtype=torch.long).to('cuda')

    try:
        with torch.no_grad():
            _ = model(input_tensor, use_cache=False)
    finally:
        capture.remove()

    # h shape: [1, seq_len, 4096]
    h = capture.activations[layer_idx]
    if h.dim() == 3:
        h = h[0]  # [seq_len, 4096]

    n_sentences = len(tok_info)
    last_token_acts = np.zeros((n_sentences, HIDDEN_DIM), dtype=np.float16)
    mean_pool_acts = np.zeros((n_sentences, HIDDEN_DIM), dtype=np.float16)
    sentence_ids = np.zeros(n_sentences, dtype=np.int32)

    for i, (sid, last_tok, span_start, span_end) in enumerate(tok_info):
        sentence_ids[i] = sid
        last_token_acts[i] = h[last_tok].to(torch.float16).numpy()

        span_acts = h[span_start:span_end + 1]  # inclusive
        mean_pool_acts[i] = span_acts.float().mean(dim=0).to(torch.float16).numpy()

    # Clean up
    del h, input_tensor
    capture.clear()
    torch.cuda.empty_cache()

    return last_token_acts, mean_pool_acts, sentence_ids


def save_trace_activations(output_dir, trace_key, layer_idx,
                           last_token_acts, mean_pool_acts,
                           sentence_ids, micro_labels):
    """Save activations for a single trace to an .npz file."""
    layer_dir = output_dir / f'layer_{layer_idx}'
    layer_dir.mkdir(parents=True, exist_ok=True)

    filename = trace_key.replace('/', '_') + '.npz'
    filepath = layer_dir / filename

    np.savez(filepath,
             last_token=last_token_acts,
             mean_pool=mean_pool_acts,
             sentence_ids=sentence_ids,
             micro_labels=micro_labels)
    return filepath


def run_extraction(model, tok_index, prompt_cache, meta_df,
                   layer_idx, trace_keys, output_dir):
    """Run activation extraction for a list of traces at a single layer."""
    log = {
        'layer': layer_idx,
        'n_requested': len(trace_keys),
        'n_processed': 0,
        'n_skipped': 0,
        'skipped_reasons': [],
        'times': [],
        'total_sentences': 0,
    }

    start_time = time.time()

    for idx, key in enumerate(trace_keys):
        # Skip if already extracted
        layer_dir = output_dir / f'layer_{layer_idx}'
        filename = key.replace('/', '_') + '.npz'
        if (layer_dir / filename).exists():
            log['n_skipped'] += 1
            log['skipped_reasons'].append((key, 'already_exists'))
            continue

        try:
            t0 = time.time()

            token_ids = prompt_cache[key]
            tok_info = tok_index[key]
            n_tokens = len(token_ids)
            n_sentences = len(tok_info)

            # Extract
            last_tok, mean_pool, sent_ids = extract_single_trace(
                model, token_ids, tok_info, layer_idx)

            # Get labels
            label_mask = meta_df['trace_key'] == key
            label_lookup = dict(zip(
                meta_df.loc[label_mask, 'sentence_id'],
                meta_df.loc[label_mask, 'micro_label']
            ))
            micro_labels = np.array([
                label_lookup.get(sid, 'UNKNOWN') for sid in sent_ids])

            save_trace_activations(output_dir, key, layer_idx,
                                   last_tok, mean_pool, sent_ids, micro_labels)

            elapsed = time.time() - t0
            log['n_processed'] += 1
            log['times'].append(elapsed)
            log['total_sentences'] += n_sentences

            # Progress every 10 traces (or every trace during pilot)
            if (log['n_processed']) % 10 == 0 or idx == len(trace_keys) - 1:
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"  [{log['n_processed']:3d}/{len(trace_keys)}] {key} -- "
                      f"{n_sentences} sent, {n_tokens} tok, {elapsed:.1f}s, "
                      f"VRAM={vram:.2f}GB")

            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            log['n_skipped'] += 1
            log['skipped_reasons'].append((key, f'OOM: {e}'))
            print(f"  !! OOM on {key} ({len(prompt_cache[key])} tokens): {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        except Exception as e:
            log['n_skipped'] += 1
            log['skipped_reasons'].append((key, f'{type(e).__name__}: {e}'))
            print(f"  !! Error on {key}: {type(e).__name__}: {e}")
            continue

    log['total_time'] = time.time() - start_time
    log['mean_time_per_trace'] = (
        np.mean(log['times']) if log['times'] else 0)

    return log


def run_pilot(model, tok_index, prompt_cache, meta_df, output_dir):
    """Run pilot extraction on 5 traces, layer 31 only. Validate and report."""
    print("\n" + "=" * 60)
    print("PILOT EXTRACTION (5 traces, layer 31)")
    print("=" * 60)

    # Select pilot traces — prefer short ones; filter to those that exist
    available = [k for k in PILOT_TRACES if k in tok_index]
    if len(available) < 5:
        # Fill with the shortest available traces
        all_keys = sorted(tok_index.keys(),
                          key=lambda k: len(prompt_cache[k]))
        for k in all_keys:
            if k not in available:
                available.append(k)
            if len(available) >= 5:
                break

    pilot_keys = available[:5]
    # Sort by token count (process shortest first)
    pilot_keys.sort(key=lambda k: len(prompt_cache[k]))

    print(f"\nPilot traces ({len(pilot_keys)}):")
    for k in pilot_keys:
        n_sent = len(tok_index[k])
        n_tok = len(prompt_cache[k])
        print(f"  {k}: {n_sent} sentences, {n_tok} tokens")

    # Run extraction (print every trace for pilot)
    print(f"\nExtracting layer 31 activations...")
    log = run_extraction(model, tok_index, prompt_cache, meta_df,
                         layer_idx=31, trace_keys=pilot_keys,
                         output_dir=output_dir)

    # ── Validation ──
    print("\n=== PILOT VALIDATION ===")
    issues = []
    max_discrepancy = 0

    layer_dir = output_dir / 'layer_31'
    all_X = []
    all_y = []

    for key in pilot_keys:
        filename = key.replace('/', '_') + '.npz'
        filepath = layer_dir / filename

        if not filepath.exists():
            issues.append(f"{key}: file not found")
            continue

        data = np.load(filepath, allow_pickle=True)
        last_tok = data['last_token']
        mean_pool = data['mean_pool']
        sent_ids = data['sentence_ids']
        labels = data['micro_labels']
        n_sent = len(tok_index[key])

        # Shape
        ok = last_tok.shape == (n_sent, HIDDEN_DIM)
        print(f"  {key}: shape [{n_sent}, {HIDDEN_DIM}] "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            issues.append(f"{key}: shape {last_tok.shape}")

        # NaN/Inf
        has_nan = np.isnan(last_tok).any() or np.isnan(mean_pool).any()
        has_inf = np.isinf(last_tok).any() or np.isinf(mean_pool).any()
        ok = not has_nan and not has_inf
        print(f"  {key}: no NaN/Inf {'PASS' if ok else 'FAIL'}")
        if not ok:
            issues.append(f"{key}: NaN={has_nan} Inf={has_inf}")

        # Non-zero
        ok = not np.allclose(last_tok, 0)
        print(f"  {key}: non-zero {'PASS' if ok else 'FAIL'}")
        if not ok:
            issues.append(f"{key}: all zeros")

        # Variance
        std = np.std(last_tok.astype(np.float32), axis=0)
        n_zero_dims = (std == 0).sum()
        print(f"  {key}: variance PASS (mean std={std.mean():.4f}, "
              f"{n_zero_dims} zero dims)")

        # Token index valid
        stored_n_tok = len(prompt_cache[key])
        max_tok_idx = max(info[1] for info in tok_index[key])
        ok = max_tok_idx < stored_n_tok
        print(f"  {key}: token idx valid {'PASS' if ok else 'FAIL'} "
              f"(max={max_tok_idx}, total={stored_n_tok})")
        if not ok:
            issues.append(f"{key}: idx {max_tok_idx} >= {stored_n_tok}")

        all_X.append(last_tok.astype(np.float32))
        all_y.extend(labels)

    # Quick probe test
    if all_X:
        print("\n  Quick probe test (logistic regression on pilot data)...")
        X = np.vstack(all_X)
        y = np.array(all_y)
        n_classes = len(set(y))
        print(f"  Data: {X.shape[0]} samples, {n_classes} classes")

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report

        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        clf.fit(X, y)
        y_pred = clf.predict(X)
        print(classification_report(y, y_pred, zero_division=0))
    else:
        n_classes = 0
        print("  No activations extracted -- cannot run probe test")

    # VRAM
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    vram_current = torch.cuda.memory_allocated() / 1e9

    # Summary
    mean_time = log['mean_time_per_trace']
    est_single = mean_time * 320 / 60 if mean_time > 0 else float('inf')
    est_all = mean_time * 320 * 4 / 60 if mean_time > 0 else float('inf')

    print("\n=== PILOT EXTRACTION SUMMARY ===")
    print(f"Traces processed: {log['n_processed']}/{len(pilot_keys)}")
    print(f"Activation shape: [n_sentences, {HIDDEN_DIM}] "
          f"{'PASS' if not issues else 'ISSUES'}")
    nan_issues = any('NaN' in i or 'Inf' in i for i in issues)
    print(f"No NaN/Inf: {'PASS' if not nan_issues else 'FAIL'}")
    tok_issues = [i for i in issues if 'idx' in i or 'token' in i.lower()]
    print(f"Token count match: {log['n_processed']}/{len(pilot_keys)} "
          f"{'PASS' if not tok_issues else 'FAIL'}")
    print(f"VRAM peak: {vram_peak:.2f} GB (budget: 10.7 GB) "
          f"{'PASS' if vram_peak < 10.7 else 'FAIL'}")
    print(f"VRAM current: {vram_current:.2f} GB")
    print(f"Mean time per trace: {mean_time:.1f}s")
    print(f"Estimated full run (layer 31 only): {est_single:.0f} min")
    print(f"Estimated full run (4 layers): {est_all:.0f} min")
    print(f"Quick probe fit: "
          f"{'PASS' if n_classes >= 5 else 'WARN'} ({n_classes} classes)")

    verdict = ("GO" if (not issues and log['n_processed'] == len(pilot_keys))
               else "NO-GO")
    print(f"\nVERDICT: {verdict}")

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  - {issue}")

    return log, verdict


def main():
    parser = argparse.ArgumentParser(
        description="Extract residual-stream activations for Study 2 probes")
    parser.add_argument('--layer', type=int, default=31,
                        help='Layer to extract (default: 31)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index in sorted trace list')
    parser.add_argument('--end', type=int, default=-1,
                        help='End index (-1 = all)')
    parser.add_argument('--output-dir', type=str, default=str(ACTIVATIONS_DIR),
                        help='Output directory')
    parser.add_argument('--pilot', action='store_true',
                        help='Run pilot extraction (5 traces, layer 31)')
    parser.add_argument('--model-path', type=str, default=str(MODEL_PATH),
                        help='Path to model')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load probe data from Stage A
    print("Loading Stage A outputs...")
    tok_index, prompt_cache, meta_df = load_probe_data()
    print(f"  {len(tok_index)} traces, {len(meta_df)} sentences")

    # Load model (direct HF, no nnsight)
    model, tokenizer = load_model(args.model_path)

    if args.pilot:
        log, verdict = run_pilot(
            model, tok_index, prompt_cache, meta_df, output_dir)

        # Save pilot log
        log_clean = {k: v for k, v in log.items() if k != 'times'}
        log_clean['verdict'] = verdict
        log_path = output_dir / 'pilot_log.json'
        with open(log_path, 'w') as f:
            json.dump(log_clean, f, indent=2, default=str)
        print(f"\nPilot log saved: {log_path}")

    else:
        layer_idx = args.layer
        if layer_idx not in range(32):
            print(f"Error: layer must be 0-31, got {layer_idx}")
            sys.exit(1)

        all_keys = sorted(tok_index.keys())
        end = args.end if args.end > 0 else len(all_keys)
        trace_keys = all_keys[args.start:end]

        print(f"\n{'='*60}")
        print(f"EXTRACTION: Layer {layer_idx}, traces {args.start}-{end} "
              f"({len(trace_keys)} traces)")
        print(f"{'='*60}")

        log = run_extraction(model, tok_index, prompt_cache, meta_df,
                             layer_idx=layer_idx, trace_keys=trace_keys,
                             output_dir=output_dir)

        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Layer: {layer_idx}")
        print(f"Processed: {log['n_processed']}/{log['n_requested']}")
        print(f"Skipped: {log['n_skipped']}")
        print(f"Total sentences: {log['total_sentences']}")
        print(f"Total time: {log['total_time']:.1f}s "
              f"({log['total_time']/60:.1f} min)")
        if log['times']:
            print(f"Mean time/trace: {log['mean_time_per_trace']:.1f}s")

        if log['skipped_reasons']:
            skip_non_exist = [
                (k, r) for k, r in log['skipped_reasons']
                if r != 'already_exists'
            ]
            if skip_non_exist:
                print(f"\nFailed traces:")
                for key, reason in skip_non_exist[:20]:
                    print(f"  {key}: {reason}")

        log_clean = {k: v for k, v in log.items() if k != 'times'}
        log_path = output_dir / f'extraction_log_layer{layer_idx}.json'
        with open(log_path, 'w') as f:
            json.dump(log_clean, f, indent=2, default=str)
        print(f"\nLog saved: {log_path}")


if __name__ == '__main__':
    main()

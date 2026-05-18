#!/usr/bin/env python3
"""Study 2b: Auto-code steered traces using trained linear probes.

Segments steered traces into sentences, extracts activations via forward
pass, and predicts micro-labels using the trained probe from Study 2a.
Computes per-condition summary statistics.

Usage:
    python study2b_steering/scripts/study2b_code_steered_traces.py --condition baseline
    python study2b_steering/scripts/study2b_code_steered_traces.py --condition HYPO_pos_1.0
    python study2b_steering/scripts/study2b_code_steered_traces.py --all
    python study2b_steering/scripts/study2b_code_steered_traces.py --pilot
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import gc
import json
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'DeepSeek-R1-Distill-Llama-8B'
STEERED_DIR = PROJECT_ROOT / 'outputs' / 'steered_traces'
PROBE_DIR = PROJECT_ROOT / 'outputs' / 'study2_probes'
RESULTS_DIR = PROJECT_ROOT / 'outputs' / 'study2b_steering'
STIMULI_DIR = PROJECT_ROOT / 'data' / 'stimuli'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST',
                'JUDGE', 'PLAN', 'MONITOR', 'RULE']
DEFAULT_LAYER = 20
DEFAULT_AGG = 'mean_pool'
DEFAULT_C = 0.01
HIDDEN_DIM = 4096


# ---------------------------------------------------------------------------
# spaCy segmentation
# ---------------------------------------------------------------------------

def load_spacy():
    """Load spaCy model for sentence segmentation."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        print("spaCy model not found, downloading en_core_web_sm...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True)
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 2_000_000
    return nlp


def segment_thinking(nlp, thinking_text):
    """Segment thinking_text into sentences."""
    if not thinking_text or not thinking_text.strip():
        return []
    doc = nlp(thinking_text)
    sentences = []
    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        sentences.append({
            "sentence_id": len(sentences),
            "text": text,
            "start_char": sent.start_char,
            "end_char": sent.end_char,
        })
    return sentences


# ---------------------------------------------------------------------------
# Token mapping
# ---------------------------------------------------------------------------

def char_to_token_robust(encoding, char_pos, direction='forward'):
    """Map a character position to a token index, searching nearby."""
    token = encoding.char_to_token(char_pos)
    if token is not None:
        return token
    max_search = 30
    if direction == 'backward':
        for offset in range(1, max_search):
            if char_pos - offset < 0:
                break
            token = encoding.char_to_token(char_pos - offset)
            if token is not None:
                return token
    else:
        for offset in range(1, max_search):
            token = encoding.char_to_token(char_pos + offset)
            if token is not None:
                return token
    return None


def map_sentences_to_tokens(tokenizer, prompt_text, thinking_text, sentences):
    """Map sentence character offsets to token indices."""
    full_text = prompt_text + thinking_text
    encoding = tokenizer(full_text)
    token_ids = encoding['input_ids']
    prompt_char_len = len(prompt_text)

    tok_info = []
    for s in sentences:
        char_start = prompt_char_len + s['start_char']
        char_end = prompt_char_len + s['end_char']

        span_start = char_to_token_robust(encoding, char_start, 'forward')
        span_end = char_to_token_robust(encoding, char_end - 1, 'backward')

        if span_start is None or span_end is None:
            continue
        if span_start > span_end:
            span_start, span_end = span_end, span_start

        last_tok = span_end
        tok_info.append((s['sentence_id'], last_tok, span_start, span_end))

    return tok_info, token_ids


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Captures residual-stream activations via PyTorch forward hooks."""

    def __init__(self):
        self.activations = {}
        self._hooks = []

    def register(self, model, layer_indices):
        for idx in layer_indices:
            layer = model.model.layers[idx]
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            self.activations[layer_idx] = output[0].detach().cpu()
        return hook_fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self.activations.clear()


def extract_activations(model, token_ids, tok_info, layer_idx):
    """Run forward pass and extract mean-pool activations per sentence."""
    capture = ActivationCapture()
    capture.register(model, [layer_idx])

    input_tensor = torch.tensor([token_ids], dtype=torch.long).to('cuda')

    try:
        with torch.no_grad():
            _ = model(input_tensor, use_cache=False)
    finally:
        capture.remove()

    h = capture.activations[layer_idx]
    if h.dim() == 3:
        h = h[0]

    n_sentences = len(tok_info)
    mean_pool_acts = np.zeros((n_sentences, HIDDEN_DIM), dtype=np.float32)
    valid_ids = []

    for i, (sid, last_tok, span_start, span_end) in enumerate(tok_info):
        span_acts = h[span_start:span_end + 1]
        mean_pool_acts[i] = span_acts.float().mean(dim=0).numpy()
        valid_ids.append(sid)

    del h, input_tensor
    capture.clear()
    torch.cuda.empty_cache()

    return mean_pool_acts, valid_ids


# ---------------------------------------------------------------------------
# Probe prediction
# ---------------------------------------------------------------------------

def load_probe(layer_idx, agg, C):
    """Load trained probe (scaler + classifier) from Study 2a."""
    if C == 1.0:
        model_path = PROBE_DIR / 'probe_models' / f'layer{layer_idx}_{agg}.pkl'
    else:
        model_path = PROBE_DIR / 'probe_models' / f'layer{layer_idx}_{agg}_C{C}.pkl'

    if not model_path.exists():
        raise FileNotFoundError(f"Probe model not found: {model_path}")

    with open(model_path, 'rb') as f:
        scaler, clf = pickle.load(f)

    print(f"  Probe loaded: {model_path.name} "
          f"(classes: {list(clf.classes_)})")
    return scaler, clf


def predict_labels(scaler, clf, activations):
    """Predict micro-labels from activations using the trained probe."""
    X_scaled = scaler.transform(activations)
    idx_preds = clf.predict(X_scaled)
    probs = clf.predict_proba(X_scaled)
    labels = np.array([MICRO_LABELS[i] for i in idx_preds])
    return labels, probs


# ---------------------------------------------------------------------------
# Transition entropy
# ---------------------------------------------------------------------------

def compute_transition_entropy(label_sequence):
    """Compute Shannon entropy of the empirical transition matrix."""
    if len(label_sequence) < 2:
        return 0.0

    transitions = {}
    for i in range(len(label_sequence) - 1):
        src = label_sequence[i]
        if src not in transitions:
            transitions[src] = {}
        dst = label_sequence[i + 1]
        transitions[src][dst] = transitions[src].get(dst, 0) + 1

    total_transitions = len(label_sequence) - 1
    weighted_entropy = 0.0

    for src, dsts in transitions.items():
        src_total = sum(dsts.values())
        src_probs = np.array(list(dsts.values())) / src_total
        src_entropy = scipy_entropy(src_probs, base=2)
        weighted_entropy += (src_total / total_transitions) * src_entropy

    return weighted_entropy


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load model in 4-bit quantization."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    print("Loading model in 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=quantization_config,
        device_map="auto",
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM: {vram:.2f} GB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def code_single_trace(model, tokenizer, nlp, scaler, clf, trace_path,
                      layer_idx):
    """Segment, extract activations, predict labels for one steered trace."""
    with open(trace_path, 'r', encoding='utf-8') as f:
        trace = json.load(f)

    thinking_text = trace.get('thinking_text', '')
    prompt_text = trace.get('prompt', '')
    answer_text = trace.get('answer_text', '')

    if not thinking_text.strip():
        return None

    sentences = segment_thinking(nlp, thinking_text)
    if not sentences:
        return None

    tok_info, token_ids = map_sentences_to_tokens(
        tokenizer, prompt_text, thinking_text, sentences)

    if not tok_info:
        return None

    activations, valid_ids = extract_activations(
        model, token_ids, tok_info, layer_idx)

    labels, probs = predict_labels(scaler, clf, activations)

    sid_to_idx = {sid: i for i, sid in enumerate(valid_ids)}
    coded_sentences = []
    for s in sentences:
        sid = s['sentence_id']
        if sid in sid_to_idx:
            idx = sid_to_idx[sid]
            s['predicted_label'] = str(labels[idx])
            s['predicted_probs'] = {
                MICRO_LABELS[j]: float(probs[idx][j])
                for j in range(len(MICRO_LABELS))
                if j < probs.shape[1]
            }
            coded_sentences.append(s)

    label_seq = [s['predicted_label'] for s in coded_sentences]
    has_rule = any(l == 'RULE' for l in label_seq)
    has_answer = bool(answer_text.strip())

    return {
        'sentences': coded_sentences,
        'n_sentences': len(coded_sentences),
        'label_sequence': label_seq,
        'has_rule': has_rule,
        'has_answer': has_answer,
        'completed': has_rule or has_answer,
        'transition_entropy': compute_transition_entropy(label_seq),
        'category_counts': {
            label: label_seq.count(label)
            for label in MICRO_LABELS
        },
    }


def code_condition(model, tokenizer, nlp, scaler, clf, condition_name,
                   layer_idx):
    """Code all traces for one condition. Returns summary stats."""
    condition_dir = STEERED_DIR / condition_name
    if not condition_dir.exists():
        print(f"  WARNING: {condition_dir} does not exist, skipping")
        return None

    trace_files = sorted(condition_dir.rglob('trace_*.json'))
    if not trace_files:
        print(f"  WARNING: No traces found in {condition_dir}")
        return None

    results = []
    t0 = time.time()

    for i, tf in enumerate(trace_files):
        try:
            result = code_single_trace(
                model, tokenizer, nlp, scaler, clf, tf, layer_idx)

            if result:
                coded_path = tf.parent / (tf.stem + '_coded.json')
                with open(coded_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'source_trace': tf.name,
                        'condition': condition_name,
                        'probe_layer': layer_idx,
                        'sentences': result['sentences'],
                        'n_sentences': result['n_sentences'],
                        'completed': result['completed'],
                        'transition_entropy': result['transition_entropy'],
                    }, f, indent=2)

                results.append(result)

            if (i + 1) % 5 == 0 or i == len(trace_files) - 1:
                elapsed = time.time() - t0
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"  [{condition_name}] {i+1}/{len(trace_files)} coded, "
                      f"{elapsed:.1f}s, VRAM={vram:.2f}GB")

        except Exception as e:
            print(f"  ERROR coding {tf.name}: {type(e).__name__}: {e}")
            continue

        gc.collect()
        torch.cuda.empty_cache()

    if not results:
        return None

    n_traces = len(results)
    n_completed = sum(1 for r in results if r['completed'])

    total_counts = {label: 0 for label in MICRO_LABELS}
    total_sentences = 0
    for r in results:
        for label, count in r['category_counts'].items():
            total_counts[label] += count
        total_sentences += r['n_sentences']

    category_dist = {
        label: total_counts[label] / total_sentences if total_sentences > 0 else 0
        for label in MICRO_LABELS
    }

    per_trace_props = []
    for r in results:
        n = r['n_sentences']
        props = {
            label: r['category_counts'][label] / n if n > 0 else 0
            for label in MICRO_LABELS
        }
        props['n_sentences'] = n
        props['completed'] = r['completed']
        props['transition_entropy'] = r['transition_entropy']
        per_trace_props.append(props)

    summary = {
        'condition': condition_name,
        'n_traces': n_traces,
        'completion_rate': n_completed / n_traces if n_traces > 0 else 0,
        'mean_trace_length': total_sentences / n_traces if n_traces > 0 else 0,
        'mean_transition_entropy': float(np.mean([
            r['transition_entropy'] for r in results])),
        'total_sentences': total_sentences,
        **{f'prop_{label}': category_dist[label] for label in MICRO_LABELS},
    }

    return summary, per_trace_props


def main():
    parser = argparse.ArgumentParser(
        description="Code steered traces using trained probes")
    parser.add_argument('--condition', type=str, default=None,
                        help='Condition directory name (e.g., baseline, HYPO_pos_1.0)')
    parser.add_argument('--all', action='store_true',
                        help='Code all conditions found in steered_traces/')
    parser.add_argument('--pilot', action='store_true',
                        help='Code pilot conditions only')
    parser.add_argument('--layer', type=int, default=DEFAULT_LAYER,
                        help=f'Probe layer (default: {DEFAULT_LAYER})')
    parser.add_argument('--agg', type=str, default=DEFAULT_AGG,
                        help=f'Aggregation (default: {DEFAULT_AGG})')
    parser.add_argument('--C', type=float, default=DEFAULT_C,
                        help=f'Regularization C (default: {DEFAULT_C})')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        conditions = ['pilot_baseline', 'HYPO_pos_1.0', 'HYPO_neg_1.0']
    elif args.all:
        conditions = sorted([
            d.name for d in STEERED_DIR.iterdir()
            if d.is_dir() and d.name != '__pycache__'
        ])
    elif args.condition:
        conditions = [args.condition]
    else:
        parser.print_help()
        print("\nError: specify --condition, --all, or --pilot")
        sys.exit(1)

    print(f"Conditions to code: {conditions}")
    print(f"Probe: layer {args.layer}, {args.agg}, C={args.C}")

    nlp = load_spacy()
    model, tokenizer = load_model()
    scaler, clf = load_probe(args.layer, args.agg, args.C)

    all_summaries = []

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"CODING: {condition}")
        print(f"{'='*60}")

        result = code_condition(
            model, tokenizer, nlp, scaler, clf, condition, args.layer)

        if result is not None:
            summary, per_trace = result
            all_summaries.append(summary)

            per_trace_path = RESULTS_DIR / f'{condition}_per_trace.csv'
            pd.DataFrame(per_trace).to_csv(per_trace_path, index=False)

            print(f"\n  Summary for {condition}:")
            print(f"    Traces: {summary['n_traces']}")
            print(f"    Completion rate: {summary['completion_rate']:.3f}")
            print(f"    Mean trace length: {summary['mean_trace_length']:.1f}")
            print(f"    Mean transition entropy: "
                  f"{summary['mean_transition_entropy']:.3f}")
            top3 = sorted(
                [(l, summary[f'prop_{l}']) for l in MICRO_LABELS],
                key=lambda x: x[1], reverse=True)[:3]
            print(f"    Top categories: " +
                  ", ".join(f"{l}={p:.3f}" for l, p in top3))

    if all_summaries:
        summary_path = RESULTS_DIR / 'condition_summaries.csv'
        pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
        print(f"\nSummaries saved: {summary_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()

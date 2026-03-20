#!/usr/bin/env python3
"""Stage A: Prepare probe training data for Study 2.

Creates stratified group 5-fold CV splits, tokenization index mapping
sentences to token positions, and metadata files for activation extraction.

Usage:
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_prepare_probe_data.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import pickle
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRACES_DIR = PROJECT_ROOT / 'outputs' / 'traces_clean_coded'
SEQ_CHARS_CSV = PROJECT_ROOT / 'outputs' / 'study1_analysis' / 'tables' / 'sequence_characteristics.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'DeepSeek-R1-Distill-Llama-8B'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'study2_probe_data'

# ── Taxonomy ──
MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST', 'JUDGE', 'PLAN', 'MONITOR', 'RULE']
MACRO_MAP = {
    'ORIENT': 'SETUP', 'DESCRIBE': 'OBSERVE', 'SYNTHESIZE': 'OBSERVE',
    'HYPO': 'INVESTIGATE', 'TEST': 'INVESTIGATE', 'JUDGE': 'INVESTIGATE',
    'PLAN': 'REGULATE', 'MONITOR': 'REGULATE', 'RULE': 'CONCLUDE',
}


def load_traces():
    """Load all coded traces from traces_clean_coded directory."""
    traces = []
    for fp in sorted(TRACES_DIR.rglob('trace_*.json')):
        with open(fp, encoding='utf-8-sig') as f:
            traces.append(json.load(f))
    print(f"Loaded {len(traces)} traces")
    return traces


def trace_key(t):
    """Build unique trace key like 'set_a/task2/trace_014'."""
    s = t['set'].lower()
    return f"set_{s}/task{t['task_id']}/trace_{t['trace_id']:03d}"


def load_reasoning_strategies():
    """Load reasoning_strategy from sequence_characteristics.csv."""
    df = pd.read_csv(SEQ_CHARS_CSV)
    return dict(zip(df['trace_key'], df['reasoning_strategy']))


def build_sentence_df(traces, strategy_map):
    """Build sentence-level DataFrame from traces."""
    rows = []
    for t in traces:
        key = trace_key(t)
        completed = bool(t.get('answer_text', '').strip())
        n_sent = len(t['sentences'])
        strategy = strategy_map.get(key, 'unknown')

        for i, s in enumerate(t['sentences']):
            c = s.get('coding', {})
            micro = c.get('micro_label', None)

            rows.append({
                'trace_key': key,
                'sentence_id': s['sentence_id'],
                'micro_label': micro,
                'macro_label': c.get('macro_label', MACRO_MAP.get(micro)),
                'task_id': t['task_id'],
                'set': t['set'],
                'completed': completed,
                'reasoning_strategy': strategy,
                'position_norm': i / max(n_sent - 1, 1),
                'position_idx': i,
                'trace_length': n_sent,
            })

    return pd.DataFrame(rows)


def create_folds(df):
    """Create group-stratified 5-fold CV assignments.

    Groups = traces (prevents within-trace leakage).
    Stratification = task_id (ensures each task represented proportionally per fold).
    """
    # Get one row per trace for fold assignment
    trace_info = df.groupby('trace_key').agg(
        task_id=('task_id', 'first'),
        set=('set', 'first'),
    ).reset_index()

    # Stratify by task_id to ensure each fold has proportional task representation
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    trace_info['fold_id'] = -1
    X = np.zeros(len(trace_info))
    y = trace_info['task_id'].values
    groups = trace_info['trace_key'].values

    for fold_idx, (_, test_idx) in enumerate(sgkf.split(X, y, groups)):
        trace_info.loc[test_idx, 'fold_id'] = fold_idx

    # Map fold_id back to sentences
    fold_map = dict(zip(trace_info['trace_key'], trace_info['fold_id']))
    df['fold_id'] = df['trace_key'].map(fold_map).astype(int)

    return df, trace_info


def char_to_token_robust(encoding, char_pos, direction='forward'):
    """Map a character position to a token index, searching nearby if exact match fails."""
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


def build_tokenization_index(traces, tokenizer):
    """Build tokenization index mapping sentences to token positions.

    For each trace, tokenizes prompt + thinking_text and maps each sentence's
    character span to token positions using the fast tokenizer's char_to_token.

    Returns:
        tokenization_index: dict of trace_key -> [(sentence_id, last_token_idx, span_start, span_end)]
        prompt_cache: dict of trace_key -> list of token_ids
        mismatches: list of (trace_key, expected, actual, diff) for large discrepancies
    """
    if not tokenizer.is_fast:
        print("  WARNING: Slow tokenizer detected. char_to_token may not be available.")
        print("  Consider installing tokenizers package for fast tokenizer support.")

    tokenization_index = {}
    prompt_cache = {}
    mismatches = []
    failed_sentences = 0

    for i, t in enumerate(traces):
        key = trace_key(t)
        prompt_text = t['prompt']
        thinking_text = t['thinking_text']

        # Full text = prompt (ending with <think>\n) + thinking content
        full_text = prompt_text + thinking_text

        # Tokenize once for the full text
        encoding = tokenizer(full_text)
        token_ids = encoding['input_ids']

        prompt_char_len = len(prompt_text)

        # Map sentences to token positions
        sentence_token_info = []
        for s in t['sentences']:
            # Character positions in full_text
            # start_char/end_char are relative to thinking_text
            # end_char appears to be exclusive (Python convention)
            char_start = prompt_char_len + s['start_char']
            char_end = prompt_char_len + s['end_char']

            # First token: search forward from start
            span_start = char_to_token_robust(encoding, char_start, direction='forward')
            # Last token: search backward from end-1 (last character of sentence)
            span_end = char_to_token_robust(encoding, char_end - 1, direction='backward')

            if span_start is None or span_end is None:
                failed_sentences += 1
                if failed_sentences <= 5:
                    print(f"  WARNING: Could not map sentence {s['sentence_id']} in {key} "
                          f"(chars {char_start}-{char_end})")
                continue

            last_token_idx = span_end
            sentence_token_info.append((s['sentence_id'], last_token_idx, span_start, span_end))

        tokenization_index[key] = sentence_token_info
        prompt_cache[key] = token_ids

        # Verify token count against stored values
        stored_prompt = t.get('prompt_tokens', None)
        stored_thinking = t.get('thinking_tokens', None)
        if stored_prompt is not None and stored_thinking is not None:
            expected = stored_prompt + stored_thinking
            actual = len(token_ids)
            diff = abs(actual - expected)
            if diff > 5:
                mismatches.append((key, expected, actual, diff))

        if (i + 1) % 50 == 0:
            print(f"  Tokenized {i + 1}/{len(traces)} traces")

    if failed_sentences > 0:
        print(f"  Total failed sentence mappings: {failed_sentences}")

    return tokenization_index, prompt_cache, mismatches


def compute_class_weights(df):
    """Compute balanced class weights: total / (n_classes * count_per_class)."""
    counts = df['micro_label'].value_counts()
    total = len(df)
    n_classes = len(MICRO_LABELS)

    weights = {}
    for label in MICRO_LABELS:
        if label in counts.index:
            weights[label] = round(total / (n_classes * counts[label]), 3)
        else:
            weights[label] = 1.0

    return weights


def create_fold_summary(df):
    """Create summary of each fold with label distribution."""
    rows = []
    for fold_id in sorted(df['fold_id'].unique()):
        fold_df = df[df['fold_id'] == fold_id]
        n_traces = fold_df['trace_key'].nunique()
        n_sentences = len(fold_df)

        row = {'fold_id': fold_id, 'n_traces': n_traces, 'n_sentences': n_sentences}
        for label in MICRO_LABELS:
            row[f'n_{label}'] = (fold_df['micro_label'] == label).sum()
        rows.append(row)

    return pd.DataFrame(rows)


def validate(df, tokenization_index, class_weights):
    """Run validation checks on the prepared data."""
    print("\n=== VALIDATION ===")
    all_ok = True

    # 1. All 73,383 sentences have a fold assignment
    n_total = len(df)
    n_no_fold = df['fold_id'].isna().sum()
    ok = n_no_fold == 0
    print(f"  All sentences have fold:     {'PASS' if ok else 'FAIL'} ({n_total} sentences)")
    all_ok &= ok

    # 2. All 320 traces in exactly one fold
    n_traces = df['trace_key'].nunique()
    trace_folds = df.groupby('trace_key')['fold_id'].nunique()
    multi_fold = (trace_folds > 1).sum()
    ok = multi_fold == 0
    print(f"  Traces in single fold:       {'PASS' if ok else 'FAIL'} ({n_traces} traces)")
    all_ok &= ok

    # 3. Label distribution roughly proportional across folds (chi-square)
    from scipy.stats import chi2_contingency
    contingency = pd.crosstab(df['fold_id'], df['micro_label'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    ok = p > 0.05
    print(f"  Label proportionality:       {'PASS' if ok else 'WARN'} (chi2={chi2:.1f}, p={p:.4f})")

    # 4. All sentences have valid last_token_idx
    n_valid = 0
    n_invalid = 0
    for key, infos in tokenization_index.items():
        n_tokens = len(prompt_cache[key])
        for (sid, last_tok, start_tok, end_tok) in infos:
            if 0 < last_tok < n_tokens:
                n_valid += 1
            else:
                n_invalid += 1
    ok = n_invalid == 0
    print(f"  Valid token indices:          {'PASS' if ok else 'FAIL'} ({n_valid} valid, {n_invalid} invalid)")
    all_ok &= ok

    # 5. Token spans are non-overlapping and in order
    n_order_issues = 0
    for key, infos in tokenization_index.items():
        sorted_infos = sorted(infos, key=lambda x: x[0])  # sort by sentence_id
        for j in range(1, len(sorted_infos)):
            if sorted_infos[j][2] < sorted_infos[j-1][3]:
                n_order_issues += 1
    # Allow minor overlaps at boundaries (tokens shared between sentences)
    ok = n_order_issues <= len(tokenization_index)  # at most 1 overlap per trace on average
    print(f"  Token span ordering:         {'PASS' if ok else 'WARN'} ({n_order_issues} boundary overlaps)")

    # 6. Class weights reasonable
    max_w = max(class_weights.values())
    min_w = min(class_weights.values())
    ok = max_w <= 50 and min_w >= 0.1
    print(f"  Class weights range:         {'PASS' if ok else 'FAIL'} ({min_w:.3f} - {max_w:.3f})")
    all_ok &= ok

    # 7. Sentence coverage: all sentences mapped
    n_mapped = sum(len(v) for v in tokenization_index.values())
    ok = n_mapped == n_total
    print(f"  Sentence token coverage:     {'PASS' if ok else 'WARN'} ({n_mapped}/{n_total} mapped)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Prepare probe training data for Study 2")
    parser.add_argument('--model-path', type=str, default=str(MODEL_PATH),
                        help='Path to model for tokenizer')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE A: Probe Data Preparation")
    print("=" * 60)

    # 1. Load traces
    print("\n1. Loading traces...")
    traces = load_traces()

    # 2. Load reasoning strategies
    print("\n2. Loading reasoning strategies...")
    strategy_map = load_reasoning_strategies()
    print(f"   {len(strategy_map)} strategies loaded")

    # 3. Build sentence DataFrame
    print("\n3. Building sentence DataFrame...")
    df = build_sentence_df(traces, strategy_map)
    print(f"   {len(df)} sentences from {df['trace_key'].nunique()} traces")
    print(f"   Label distribution:")
    for label in MICRO_LABELS:
        n = (df['micro_label'] == label).sum()
        pct = 100 * n / len(df)
        print(f"     {label:12s}: {n:6d} ({pct:5.1f}%)")

    # 4. Create 5-fold CV splits
    print("\n4. Creating 5-fold group-stratified CV splits...")
    df, trace_info = create_folds(df)
    for fold_id in range(5):
        n_tr = (trace_info['fold_id'] == fold_id).sum()
        n_s = (df['fold_id'] == fold_id).sum()
        print(f"   Fold {fold_id}: {n_tr:3d} traces, {n_s:5d} sentences")

    # 5. Build tokenization index
    print("\n5. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"   Tokenizer: {type(tokenizer).__name__} (fast={tokenizer.is_fast})")
    print(f"   Vocab size: {tokenizer.vocab_size}")

    print("\n   Building tokenization index (this may take a few minutes)...")
    global prompt_cache  # needed by validate()
    tok_index, prompt_cache, mismatches = build_tokenization_index(traces, tokenizer)

    if mismatches:
        print(f"\n   Token count mismatches (>5 tokens off):")
        for key, expected, actual, diff in mismatches[:10]:
            print(f"     {key}: expected {expected}, got {actual} (diff={diff})")
        if len(mismatches) > 10:
            print(f"     ... and {len(mismatches) - 10} more")
    else:
        print("   All token counts match within tolerance (+/-5)")

    # 6. Add token info to DataFrame
    print("\n6. Adding token positions to DataFrame...")
    tok_lookup = {}
    for key, sentence_infos in tok_index.items():
        for (sid, last_tok, start_tok, end_tok) in sentence_infos:
            tok_lookup[(key, sid)] = (last_tok, start_tok, end_tok, len(prompt_cache[key]))

    last_tokens = []
    span_starts = []
    span_ends = []
    trace_n_tokens_list = []

    for _, row in df.iterrows():
        info = tok_lookup.get((row['trace_key'], row['sentence_id']))
        if info:
            last_tokens.append(info[0])
            span_starts.append(info[1])
            span_ends.append(info[2])
            trace_n_tokens_list.append(info[3])
        else:
            last_tokens.append(-1)
            span_starts.append(-1)
            span_ends.append(-1)
            trace_n_tokens_list.append(-1)

    df['last_token_idx'] = last_tokens
    df['token_span_start'] = span_starts
    df['token_span_end'] = span_ends
    df['trace_n_tokens'] = trace_n_tokens_list

    # 7. Compute class weights
    print("\n7. Computing class weights...")
    class_weights = compute_class_weights(df)
    for label in MICRO_LABELS:
        print(f"   {label:12s}: {class_weights[label]:.3f}")

    # 8. Validate
    all_ok = validate(df, tok_index, class_weights)

    # 9. Print fold x label table
    print("\n   Fold x Label counts:")
    fold_label = pd.crosstab(df['fold_id'], df['micro_label'])[MICRO_LABELS]
    print(fold_label.to_string())

    # 10. Save outputs
    print("\n8. Saving outputs...")

    # sentence_metadata.csv
    csv_path = output_dir / 'sentence_metadata.csv'
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path} ({len(df)} rows)")

    # fold_summary.csv
    fold_summary = create_fold_summary(df)
    fold_path = output_dir / 'fold_summary.csv'
    fold_summary.to_csv(fold_path, index=False)
    print(f"   Saved: {fold_path}")

    # class_weights.json
    weights_path = output_dir / 'class_weights.json'
    with open(weights_path, 'w') as f:
        json.dump(class_weights, f, indent=2)
    print(f"   Saved: {weights_path}")

    # tokenization_index.pkl
    tok_path = output_dir / 'tokenization_index.pkl'
    with open(tok_path, 'wb') as f:
        pickle.dump(tok_index, f)
    print(f"   Saved: {tok_path}")

    # prompt_cache.pkl
    cache_path = output_dir / 'prompt_cache.pkl'
    with open(cache_path, 'wb') as f:
        pickle.dump(prompt_cache, f)
    print(f"   Saved: {cache_path} ({len(prompt_cache)} traces)")

    print("\n" + "=" * 60)
    if all_ok:
        print("STAGE A COMPLETE - All checks passed")
    else:
        print("STAGE A COMPLETE - Some checks had warnings (review above)")
    print("=" * 60)


if __name__ == '__main__':
    main()

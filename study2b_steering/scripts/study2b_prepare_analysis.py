#!/usr/bin/env python3
"""Study 2b: Prepare per-trace and per-condition summaries for analysis.

Loads every `_coded.json` file under `outputs/steered_traces/<condition>/<task>/`
across the 22 coded conditions (baseline + 20 alpha<=1.0 + JUDGE_neg_2.0),
computes per-trace metrics, and writes:

    outputs/study2b_steering/per_trace_summaries.csv
    outputs/study2b_steering/condition_summaries.csv

Per-trace metrics:
    condition, task_id, trace_id, seed, n_sentences, completed,
    prop_<LABEL> for each of 9 micro-labels,
    transition_entropy (Shannon entropy of bigram transitions),
    judge_n, judge_accept_rate, judge_reject_rate, judge_uncertain_rate,
    answer_text, last_rule_sentence, thinking_text_len.

Usage:
    .venv/Scripts/python.exe scripts/prepare_steering_analysis.py
"""

import argparse
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--steered-dir',
    type=Path,
    default=PROJECT_ROOT / 'outputs' / 'steered_traces',
    help='Directory containing coded steered-trace JSONs (default: outputs/steered_traces/; not published in the public repo)',
)
parser.add_argument(
    '--out-dir',
    type=Path,
    default=PROJECT_ROOT / 'study2b_steering' / 'results',
    help='Directory to write per_trace_summaries.csv and condition_summaries.csv (default: study2b_steering/results/)',
)
args = parser.parse_args()
STEERED_DIR = args.steered_dir
OUT_DIR = args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST',
                'JUDGE', 'PLAN', 'MONITOR', 'RULE']

# 22 coded conditions
CODED_CONDITIONS = [
    'baseline',
    'HYPO_pos_0.5', 'HYPO_pos_1.0', 'HYPO_neg_0.5', 'HYPO_neg_1.0',
    'TEST_pos_0.5', 'TEST_pos_1.0', 'TEST_neg_0.5', 'TEST_neg_1.0',
    'JUDGE_pos_0.5', 'JUDGE_pos_1.0', 'JUDGE_neg_0.5', 'JUDGE_neg_1.0',
    'JUDGE_neg_2.0',
    'MONITOR_pos_0.5', 'MONITOR_pos_1.0', 'MONITOR_neg_0.5', 'MONITOR_neg_1.0',
    'PLAN_pos_0.5', 'PLAN_pos_1.0', 'PLAN_neg_0.5', 'PLAN_neg_1.0',
]


def parse_condition(name):
    """Return (category, direction, alpha, signed_alpha) or (None,None,0,0) for baseline."""
    if name == 'baseline':
        return None, None, 0.0, 0.0
    parts = name.split('_')
    cat, direction, alpha = parts[0], parts[1], float(parts[2])
    signed = alpha if direction == 'pos' else -alpha
    return cat, direction, alpha, signed


def transition_entropy(labels):
    """Shannon entropy of the bigram transition distribution (in bits)."""
    if len(labels) < 2:
        return float('nan')
    bigrams = list(zip(labels[:-1], labels[1:]))
    counts = Counter(bigrams)
    total = sum(counts.values())
    H = 0.0
    for _, c in counts.items():
        p = c / total
        H -= p * math.log2(p)
    return H


def find_last_rule_sentence(sentences):
    """Return text of last sentence with micro_label == 'RULE', else empty."""
    for s in reversed(sentences):
        c = s.get('coding') or {}
        if c.get('micro_label') == 'RULE':
            return s.get('text', '').strip()
    return ''


def load_trace(fp, condition):
    with open(fp, encoding='utf-8-sig') as f:
        t = json.load(f)

    sents = t.get('sentences', []) or []
    n = len(sents)
    answer = (t.get('answer_text') or '').strip()
    completed = bool(answer)

    labels = [(s.get('coding') or {}).get('micro_label') for s in sents]
    label_counter = Counter(labels)

    row = {
        'condition': condition,
        'task_id': t.get('task_id'),
        'trace_id': t.get('trace_id'),
        'seed': t.get('seed'),
        'n_sentences': n,
        'completed': completed,
        'thinking_text_len': len(t.get('thinking_text', '') or ''),
        'truncated': bool(t.get('truncated', False)),
        'transition_entropy': transition_entropy([l for l in labels if l]),
    }

    for label in MICRO_LABELS:
        row[f'prop_{label}'] = (label_counter.get(label, 0) / n) if n > 0 else 0.0
        row[f'count_{label}'] = label_counter.get(label, 0)

    # JUDGE verdict breakdown (per trace)
    judge_verdicts = []
    for s in sents:
        c = s.get('coding') or {}
        if c.get('micro_label') == 'JUDGE':
            v = c.get('judgement')
            if v:
                judge_verdicts.append(v)
    n_judge = len(judge_verdicts)
    row['judge_n'] = n_judge
    if n_judge > 0:
        vc = Counter(judge_verdicts)
        row['judge_accept_rate'] = vc.get('accept', 0) / n_judge
        row['judge_reject_rate'] = vc.get('reject', 0) / n_judge
        row['judge_uncertain_rate'] = vc.get('uncertain', 0) / n_judge
    else:
        row['judge_accept_rate'] = float('nan')
        row['judge_reject_rate'] = float('nan')
        row['judge_uncertain_rate'] = float('nan')

    # Last RULE sentence (for accuracy evaluation), and truncated answer
    row['last_rule_sentence'] = find_last_rule_sentence(sents)
    row['answer_text'] = answer
    return row


def main():
    rows = []
    missing = []
    for cond in CODED_CONDITIONS:
        cond_dir = STEERED_DIR / cond
        if not cond_dir.exists():
            print(f'  WARN: missing condition dir {cond_dir}')
            continue
        n_in_cond = 0
        for task_dir in sorted(cond_dir.glob('task*')):
            for fp in sorted(task_dir.glob('trace_*_coded.json')):
                try:
                    r = load_trace(fp, cond)
                    rows.append(r)
                    n_in_cond += 1
                except Exception as e:
                    missing.append((str(fp), str(e)))
        print(f'  {cond:<22} {n_in_cond:3d} traces')
    print(f'\nTotal: {len(rows)} traces across {len(CODED_CONDITIONS)} conditions')
    if missing:
        print(f'  {len(missing)} files failed to load')
        for fp, err in missing[:5]:
            print(f'    {fp}: {err}')

    df = pd.DataFrame(rows)

    # Add parsed condition fields
    parsed = df['condition'].apply(parse_condition)
    df['steer_category'] = parsed.apply(lambda x: x[0])
    df['steer_direction'] = parsed.apply(lambda x: x[1])
    df['alpha'] = parsed.apply(lambda x: x[2])
    df['signed_alpha'] = parsed.apply(lambda x: x[3])

    # Order columns nicely
    front = ['condition', 'steer_category', 'steer_direction', 'alpha', 'signed_alpha',
             'task_id', 'trace_id', 'seed', 'n_sentences', 'completed', 'truncated',
             'thinking_text_len', 'transition_entropy',
             'judge_n', 'judge_accept_rate', 'judge_reject_rate', 'judge_uncertain_rate']
    prop_cols = [f'prop_{l}' for l in MICRO_LABELS]
    count_cols = [f'count_{l}' for l in MICRO_LABELS]
    tail = ['last_rule_sentence', 'answer_text']
    cols = front + prop_cols + count_cols + tail
    df = df[cols]

    per_trace_path = OUT_DIR / 'per_trace_summaries.csv'
    df.to_csv(per_trace_path, index=False)
    print(f'\nSaved {per_trace_path} ({len(df)} rows, {len(df.columns)} cols)')

    # Condition summaries: mean and SD of each numeric metric per condition
    numeric_cols = [c for c in df.columns
                    if df[c].dtype.kind in 'fbi' and c not in ('seed', 'task_id', 'trace_id')]
    grouped = df.groupby('condition')
    summary_rows = []
    for cond, g in grouped:
        row = {'condition': cond, 'n_traces': len(g)}
        cat, direction, alpha, signed = parse_condition(cond)
        row['steer_category'] = cat
        row['steer_direction'] = direction
        row['alpha'] = alpha
        row['signed_alpha'] = signed
        for c in numeric_cols:
            row[f'{c}_mean'] = g[c].mean()
            row[f'{c}_sd'] = g[c].std()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # Sort: baseline first, then by category/direction/alpha
    summary_df['_sort'] = summary_df['condition'].apply(
        lambda c: (0, '', '', 0.0) if c == 'baseline'
        else (1,) + parse_condition(c)[:3]
    )
    summary_df = summary_df.sort_values('_sort').drop(columns=['_sort']).reset_index(drop=True)

    cond_summary_path = OUT_DIR / 'condition_summaries.csv'
    summary_df.to_csv(cond_summary_path, index=False)
    print(f'Saved {cond_summary_path} ({len(summary_df)} rows)')

    # Sanity check: print mean prop_<target> per steered condition
    print('\nOn-target prop sanity check (steered vs baseline):')
    base = df[df['condition'] == 'baseline']
    print(f"  {'condition':<22} {'target_prop':>12} {'baseline_prop':>14}")
    for cond in CODED_CONDITIONS:
        cat, _, _, _ = parse_condition(cond)
        if cat is None:
            continue
        sub = df[df['condition'] == cond]
        target_col = f'prop_{cat}'
        if target_col in sub.columns and len(sub) > 0:
            t_mean = sub[target_col].mean()
            b_mean = base[target_col].mean()
            print(f'  {cond:<22} {t_mean:>12.4f} {b_mean:>14.4f}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Study 2b: Heuristic accuracy evaluation of stated rules.

Ground-truth rules (from outputs/traces/stimulus_quality_evaluation.md):

    Task 1: at least one red cone               (starred A, E, F)
    Task 2: exactly one small blue cone         (starred A, B, E)
    Task 3: at least two tilted blue cones      (starred A, C, F)
    Task 4: a pair of stacked cones with different colours (starred A, C, F)

For each completed trace we extract the candidate stated rule from
`answer_text` (full) plus the last RULE-coded sentence (as fallback /
augmentation) and score it against the ground-truth using a conservative
keyword heuristic. The result is also written as a manual-review CSV.

Outputs:
    outputs/study2b_steering/accuracy_heuristic.csv      (per-trace verdict)
    outputs/study2b_steering/accuracy_by_condition.csv   (per-condition rate)
    outputs/study2b_steering/accuracy_manual_review.csv  (eyes-on review template)
"""

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--in-dir',
    type=Path,
    default=PROJECT_ROOT / 'study2b_steering' / 'results',
    help='Directory containing per_trace_summaries.csv (default: study2b_steering/results/)',
)
parser.add_argument(
    '--out-dir',
    type=Path,
    default=PROJECT_ROOT / 'study2b_steering' / 'results',
    help='Directory to write accuracy CSVs (default: study2b_steering/results/)',
)
args = parser.parse_args()
OUT_DIR = args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_TRACE_CSV = args.in_dir / 'per_trace_summaries.csv'

GROUND_TRUTH = {
    1: 'at least one red cone',
    2: 'exactly one small blue cone',
    3: 'at least two tilted blue cones',
    4: 'a pair of stacked cones with different colours',
}

# Common decoy / wrong-answer phrasings the model uses (to avoid false positives)
DECOY_PATTERNS = [
    r'\bupside[- ]?down\b',
    r'\bcluster\b', r'\bcontiguous\b', r'\bconnected structure\b',
    r'\bchain of cones\b', r'\bsingle chain\b', r'\blinear chain\b',
    r'\bnested\b',
    r'\ball three colou?rs\b', r'\bthree distinct colou?rs\b',
]


def score_task1(text):
    """At least one red cone. The model is correct if it identifies 'red'
    as the discriminating feature (and not just mentioning red in passing)."""
    t = text.lower()
    red_anchor = bool(re.search(r'\b(at least one|contain|has|have|presence of|with).{0,40}\bred\b', t))
    red_only = bool(re.search(r'\bred (cone|object)\b', t))
    decoy = any(re.search(p, t) for p in DECOY_PATTERNS)
    if (red_anchor or red_only) and not decoy:
        return 1
    if red_anchor or red_only:
        return 0  # Mentions red but also a confounding hypothesis -> uncertain/wrong
    return 0


def score_task2(text):
    """Exactly one small blue cone — needs both 'small' and 'blue' as
    co-occurring features that distinguish the panel."""
    t = text.lower()
    # Look for small + blue within ~30 chars OR an explicit "small blue" phrase
    has_small_blue = bool(re.search(r'\bsmall\b.{0,40}\bblue\b|\bblue\b.{0,40}\bsmall\b', t))
    # Bonus: the rule mentions "one" or "exactly" or "single"
    has_count = bool(re.search(r'\bone\b|\bexactly\b|\bsingle\b', t))
    decoy = any(re.search(p, t) for p in DECOY_PATTERNS)
    if has_small_blue and has_count and not decoy:
        return 1
    return 0


def score_task3(text):
    """At least two tilted (or lying) blue cones."""
    t = text.lower()
    # tilted blue or lying blue, with a notion of count
    has_tilt_blue = bool(re.search(
        r'\btilted\b.{0,40}\bblue\b|\bblue\b.{0,40}\btilted\b|'
        r'\blying\b.{0,40}\bblue\b|\bblue\b.{0,40}\blying\b', t))
    has_two = bool(re.search(r'\btwo\b|\bpair\b|\bat least two\b|\b2 \b', t))
    decoy = any(re.search(p, t) for p in DECOY_PATTERNS)
    if has_tilt_blue and has_two and not decoy:
        return 1
    return 0


def score_task4(text):
    """A pair of stacked cones with different colours."""
    t = text.lower()
    has_stack = bool(re.search(r'\bstack\w*\b|\b(rests? on|on top of)\b.{0,30}\bcone\b', t))
    has_diff = bool(re.search(r'\bdifferent colou?rs?\b|\bdiffer in colou?r\b|\btwo colou?rs?\b'
                              r'|\bcontrasting colou?rs?\b|\bnot the same colou?r\b', t))
    decoy = any(re.search(p, t) for p in DECOY_PATTERNS)
    if has_stack and has_diff and not decoy:
        return 1
    return 0


SCORERS = {1: score_task1, 2: score_task2, 3: score_task3, 4: score_task4}


def main():
    df = pd.read_csv(PER_TRACE_CSV)
    print(f'Loaded {len(df)} traces')

    rows = []
    for _, r in df.iterrows():
        ans = (str(r['answer_text']) if pd.notna(r['answer_text']) else '')
        last_rule = (str(r['last_rule_sentence']) if pd.notna(r['last_rule_sentence']) else '')
        candidate = (ans + ' || ' + last_rule).strip()
        completed = bool(r['completed'])
        task = int(r['task_id'])
        score = SCORERS[task](candidate) if completed else None
        rows.append({
            'condition': r['condition'],
            'task_id': task,
            'trace_id': int(r['trace_id']),
            'seed': int(r['seed']),
            'completed': completed,
            'truncated': bool(r['truncated']),
            'ground_truth_rule': GROUND_TRUTH[task],
            'stated_rule_answer': ans[:600],
            'last_rule_sentence': last_rule[:400],
            'heuristic_correct': score,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / 'accuracy_heuristic.csv', index=False)
    print(f"Saved accuracy_heuristic.csv ({len(out)} rows)")

    # Per-condition accuracy among completed traces
    by_cond = []
    for cond, g in out.groupby('condition'):
        comp = g[g['completed']]
        n_comp = len(comp)
        n_correct = int(comp['heuristic_correct'].fillna(0).sum())
        by_cond.append({
            'condition': cond,
            'n_total': len(g),
            'n_completed': n_comp,
            'completion_rate': n_comp / len(g) if len(g) else float('nan'),
            'n_correct_heuristic': n_correct,
            'accuracy_among_completed': n_correct / n_comp if n_comp else float('nan'),
            'accuracy_overall': n_correct / len(g) if len(g) else float('nan'),
        })
    by_cond_df = pd.DataFrame(by_cond)
    # Sort: baseline first
    by_cond_df['_sort'] = by_cond_df['condition'].apply(lambda c: '0' if c == 'baseline' else '1' + c)
    by_cond_df = by_cond_df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
    by_cond_df.to_csv(OUT_DIR / 'accuracy_by_condition.csv', index=False)
    print('Per-condition accuracy:')
    print(by_cond_df.to_string(index=False))

    # Manual review template (only completed traces, all 22 conditions, sampled)
    review = out[out['completed']].copy()
    review = review[['condition', 'task_id', 'trace_id', 'seed', 'ground_truth_rule',
                     'stated_rule_answer', 'last_rule_sentence', 'heuristic_correct']]
    review['manual_correct'] = ''  # blank column for researcher
    review.to_csv(OUT_DIR / 'accuracy_manual_review.csv', index=False)
    print(f"Saved accuracy_manual_review.csv ({len(review)} completed traces — researcher fills 'manual_correct')")


if __name__ == '__main__':
    main()

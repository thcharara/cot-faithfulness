#!/usr/bin/env python3
"""For each Study 2b condition, run a Fisher's exact test against the
Study 1 Set A accuracy baseline (matched generation parameters). Also
runs the baseline-vs-Set-A self-comparison as a seed-confound check.

Output: outputs/study2b_steering/accuracy_vs_study1_baseline.csv
"""

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--in-dir',
    type=Path,
    default=PROJECT_ROOT / 'study2b_steering' / 'results',
    help='Directory containing study1_accuracy_heuristic.csv and accuracy_heuristic.csv (default: study2b_steering/results/)',
)
parser.add_argument(
    '--out-dir',
    type=Path,
    default=PROJECT_ROOT / 'study2b_steering' / 'results',
    help='Directory to write accuracy_vs_study1_baseline.csv (default: study2b_steering/results/)',
)
args = parser.parse_args()
IN_DIR = args.in_dir
OUT_DIR = args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_ORDER = [
    'baseline',
    'HYPO_neg_1.0', 'HYPO_neg_0.5', 'HYPO_pos_0.5', 'HYPO_pos_1.0',
    'TEST_neg_1.0', 'TEST_neg_0.5', 'TEST_pos_0.5', 'TEST_pos_1.0',
    'JUDGE_neg_2.0', 'JUDGE_neg_1.0', 'JUDGE_neg_0.5', 'JUDGE_pos_0.5', 'JUDGE_pos_1.0',
    'MONITOR_neg_1.0', 'MONITOR_neg_0.5', 'MONITOR_pos_0.5', 'MONITOR_pos_1.0',
    'PLAN_neg_1.0', 'PLAN_neg_0.5', 'PLAN_pos_0.5', 'PLAN_pos_1.0',
]


def odds_ratio_ci(a, b, c, d, alpha=0.05):
    """Wald CI for odds ratio with Haldane–Anscombe 0.5 correction.
    Returns (or, lo, hi).
    Table:
        condition   correct=a   incorrect=b
        set_a       correct=c   incorrect=d
    """
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_hat = (aa * dd) / (bb * cc)
    log_or = math.log(or_hat)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    z = 1.959963984540054
    lo = math.exp(log_or - z * se)
    hi = math.exp(log_or + z * se)
    return or_hat, lo, hi


def main():
    s1 = pd.read_csv(IN_DIR / 'study1_accuracy_heuristic.csv')
    s2 = pd.read_csv(IN_DIR / 'accuracy_heuristic.csv')

    # Set A reference cell (completed traces only)
    set_a = s1[(s1['population'] == 'set_a') & (s1['completed'])]
    set_a_n_completed = len(set_a)
    set_a_n_correct = int(set_a['heuristic_correct'].fillna(0).sum())
    set_a_n_incorrect = set_a_n_completed - set_a_n_correct
    set_a_rate = set_a_n_correct / set_a_n_completed if set_a_n_completed else float('nan')
    print(f'Set A baseline: {set_a_n_correct}/{set_a_n_completed} '
          f'= {set_a_rate*100:.2f}%')

    rows = []
    for cond in CONDITION_ORDER:
        sub = s2[(s2['condition'] == cond) & (s2['completed'])]
        n_completed = len(sub)
        n_correct = int(sub['heuristic_correct'].fillna(0).sum())
        n_incorrect = n_completed - n_correct
        rate = n_correct / n_completed if n_completed else float('nan')

        if n_completed == 0:
            fisher_p = float('nan')
            or_hat, lo, hi = float('nan'), float('nan'), float('nan')
        else:
            table = [[n_correct, n_incorrect],
                     [set_a_n_correct, set_a_n_incorrect]]
            try:
                _, fisher_p = fisher_exact(table, alternative='two-sided')
                or_hat, lo, hi = odds_ratio_ci(
                    n_correct, n_incorrect,
                    set_a_n_correct, set_a_n_incorrect)
            except Exception:
                fisher_p = float('nan')
                or_hat, lo, hi = float('nan'), float('nan'), float('nan')

        if rate == rate and set_a_rate == set_a_rate:
            direction = 'higher' if rate > set_a_rate else 'lower' if rate < set_a_rate else 'equal'
        else:
            direction = 'n/a'

        rows.append({
            'condition': cond,
            'n_completed': n_completed,
            'n_correct': n_correct,
            'accuracy_rate': rate,
            'set_a_n_completed': set_a_n_completed,
            'set_a_n_correct': set_a_n_correct,
            'set_a_accuracy_rate': set_a_rate,
            'fisher_p': fisher_p,
            'odds_ratio': or_hat,
            'ci_lower': lo,
            'ci_upper': hi,
            'direction': direction,
        })

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / 'accuracy_vs_study1_baseline.csv'
    out.to_csv(out_path, index=False)
    print(f'Saved {out_path.name}')
    print()
    print(out[['condition', 'n_completed', 'n_correct', 'accuracy_rate',
               'set_a_accuracy_rate', 'fisher_p', 'odds_ratio',
               'ci_lower', 'ci_upper', 'direction']].round(4).to_string(index=False))

    # Interpretation
    print('\n--- Interpretation ---')
    base_row = out[out['condition'] == 'baseline'].iloc[0]
    base_p = base_row['fisher_p']
    if base_p == base_p and base_p < 0.05:
        print(f'Study 2b baseline differs from Set A at p={base_p:.4g} '
              '— SEED-DIFFICULTY CONFOUND likely; treat steered comparisons cautiously.')
    else:
        print(f'Study 2b baseline vs Set A: p={base_p:.4g} (not significant). '
              'Seed-set difficulty is comparable; cross-study comparison is valid.')

    sig = out[(out['fisher_p'] < 0.05) & (out['condition'] != 'baseline')]
    if len(sig) == 0:
        print('No steered condition differs significantly from Set A at p<0.05.')
    else:
        print(f'{len(sig)} steered condition(s) differ significantly from Set A:')
        for _, r in sig.iterrows():
            print(f"  {r['condition']:<22} {r['n_correct']}/{r['n_completed']}"
                  f" = {r['accuracy_rate']*100:.1f}% vs Set A {set_a_rate*100:.2f}%"
                  f" (p={r['fisher_p']:.4g}, dir={r['direction']})")


if __name__ == '__main__':
    main()

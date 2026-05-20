#!/usr/bin/env python3
"""Study 2b: Statistical analysis of steering effects.

Loads `outputs/study2b_steering/per_trace_summaries.csv` (880 rows, 22 conditions)
and computes:

    2a. On-target steering effects (paired Wilcoxon vs baseline by seed,
        Cohen's d, Bonferroni correction across the 20 alpha<=1.0 tests)
    2b. Off-target effects: 20 x 9 shift matrix
    2c. Dose-response: Spearman rank correlation between signed alpha and
        on-target proportion across all 5 (or 6 for JUDGE) levels
    2d. Completion rate per condition + Fisher's exact vs baseline
    2e. Trace length per condition + Mann-Whitney U vs baseline
    2f. Transition entropy per condition + Mann-Whitney U vs baseline
    2g. JUDGE verdict breakdown per condition
    2i. JUDGE_neg_2.0 transition matrix vs baseline transition matrix

Outputs CSVs (and rebuilds heatmap matrix) into outputs/study2b_steering/.

Figures are produced by the companion notebook / scripts in this directory.

Usage:
    .venv/Scripts/python.exe scripts/analyse_steering.py
"""

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, fisher_exact, spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--in-dir',
    type=Path,
    default=PROJECT_ROOT / 'study2b_steering' / 'results',
    help='Directory containing per_trace_summaries.csv (default: study2b_steering/results/, the published artefacts)',
)
parser.add_argument(
    '--out-dir',
    type=Path,
    default=PROJECT_ROOT / 'study2b_steering' / 'results',
    help='Directory to write analysis tables (default: study2b_steering/results/)',
)
parser.add_argument(
    '--steered-dir',
    type=Path,
    default=PROJECT_ROOT / 'outputs' / 'steered_traces',
    help='Raw steered-trace directory (only needed for the transition-matrix step; not published in the public repo)',
)
args = parser.parse_args()
IN_DIR = args.in_dir
OUT_DIR = args.out_dir
STEERED_DIR = args.steered_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_TRACE_CSV = IN_DIR / 'per_trace_summaries.csv'

MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST',
                'JUDGE', 'PLAN', 'MONITOR', 'RULE']
STEER_CATEGORIES = ['HYPO', 'TEST', 'JUDGE', 'MONITOR', 'PLAN']

# 20 alpha<=1.0 conditions used for the primary on-target multiple-comparison family.
PRIMARY_CONDITIONS = [
    f'{cat}_{d}_{a}' for cat in STEER_CATEGORIES for d in ('pos', 'neg') for a in ('0.5', '1.0')
]
# JUDGE_neg_2.0 is treated as exploratory (not in Bonferroni family), reported separately.


def cohens_d_paired(diff):
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def paired_aligned(df_a, df_b, col):
    """Align by seed, return (a_vals, b_vals) numpy arrays where both have a value."""
    merged = pd.merge(
        df_a[['seed', col]].rename(columns={col: 'a'}),
        df_b[['seed', col]].rename(columns={col: 'b'}),
        on='seed', how='inner'
    )
    return merged['a'].values, merged['b'].values


# ── Section 2a + 2b: shift table for every (steered_condition, label) ──

def compute_on_off_target(df):
    """Return DataFrame with one row per condition x label.
    Columns: condition, category, direction, alpha, label, mean_steered,
             mean_baseline, shift, cohen_d, wilcoxon_stat, p_raw,
             on_target (bool).
    """
    base = df[df['condition'] == 'baseline'].copy()
    rows = []
    for cond in df['condition'].unique():
        if cond == 'baseline':
            continue
        sub = df[df['condition'] == cond].copy()
        cat = sub['steer_category'].iloc[0]
        direction = sub['steer_direction'].iloc[0]
        alpha = float(sub['alpha'].iloc[0])
        for label in MICRO_LABELS:
            col = f'prop_{label}'
            steered_v, base_v = paired_aligned(sub, base, col)
            n = len(steered_v)
            diff = steered_v - base_v
            shift = diff.mean()
            d = cohens_d_paired(diff)
            try:
                stat, p = wilcoxon(steered_v, base_v, zero_method='wilcox', alternative='two-sided')
                stat = float(stat); p = float(p)
            except ValueError:
                stat, p = float('nan'), 1.0
            rows.append({
                'condition': cond,
                'category': cat,
                'direction': direction,
                'alpha': alpha,
                'label': label,
                'on_target': (label == cat),
                'n_pairs': n,
                'mean_baseline': float(base_v.mean()),
                'mean_steered': float(steered_v.mean()),
                'shift': float(shift),
                'cohen_d': d,
                'wilcoxon_stat': stat,
                'p_raw': p,
            })
    return pd.DataFrame(rows)


def add_bonferroni(df, family_mask, p_col='p_raw', out_col='p_bonf'):
    """Apply Bonferroni within the family rows (others get NaN)."""
    df[out_col] = float('nan')
    fam = df[family_mask]
    n = len(fam)
    if n > 0:
        df.loc[family_mask, out_col] = (fam[p_col] * n).clip(upper=1.0)
    df['significant_005'] = df[out_col] < 0.05
    return df


# ── Section 2c: Dose-response ──

def compute_dose_response(df):
    """Per category, return per-trace pairs of (signed_alpha, on-target prop)
    and overall Spearman rank correlation across condition means."""
    rows = []
    summary = []
    for cat in STEER_CATEGORIES:
        target_col = f'prop_{cat}'
        # Get all conditions for this category + baseline (signed_alpha=0)
        sub = df[(df['steer_category'] == cat) | (df['condition'] == 'baseline')]
        # For category-based dose-response, exclude alpha=2.0 EXCEPT for JUDGE
        if cat != 'JUDGE':
            sub = sub[sub['alpha'] <= 1.0]
        for _, r in sub.iterrows():
            rows.append({
                'category': cat,
                'condition': r['condition'],
                'signed_alpha': float(r['signed_alpha']),
                'target_prop': float(r[target_col]),
                'seed': int(r['seed']),
            })
        # Per-condition means for monotonicity test
        cond_means = sub.groupby(['condition', 'signed_alpha'])[target_col].mean().reset_index()
        cond_means = cond_means.sort_values('signed_alpha')
        if len(cond_means) >= 3:
            rho, p = spearmanr(cond_means['signed_alpha'], cond_means[target_col])
        else:
            rho, p = float('nan'), float('nan')
        summary.append({
            'category': cat,
            'n_levels': len(cond_means),
            'spearman_rho': float(rho),
            'spearman_p': float(p),
            'levels': cond_means['signed_alpha'].tolist(),
            'means': cond_means[target_col].round(4).tolist(),
        })
    return pd.DataFrame(rows), pd.DataFrame(summary)


# ── Section 2d: Completion rate ──

def compute_completion(df):
    base = df[df['condition'] == 'baseline']
    base_complete = int(base['completed'].sum()); base_total = len(base)
    rows = []
    for cond in df['condition'].unique():
        sub = df[df['condition'] == cond]
        n_complete = int(sub['completed'].sum()); n_total = len(sub)
        # Fisher's exact: 2x2 (cond_complete, cond_incomplete) vs (base_complete, base_incomplete)
        if cond == 'baseline':
            odds_ratio, p = float('nan'), float('nan')
        else:
            try:
                table = [[n_complete, n_total - n_complete],
                         [base_complete, base_total - base_complete]]
                odds_ratio, p = fisher_exact(table, alternative='two-sided')
            except Exception:
                odds_ratio, p = float('nan'), float('nan')
        rows.append({
            'condition': cond,
            'n_complete': n_complete, 'n_total': n_total,
            'completion_rate': n_complete / n_total if n_total else float('nan'),
            'fisher_odds_ratio': float(odds_ratio) if odds_ratio == odds_ratio else float('nan'),
            'fisher_p': float(p) if p == p else float('nan'),
        })
    return pd.DataFrame(rows)


# ── Section 2e + 2f: trace length + entropy ──

def compute_simple_stat(df, col, label):
    base = df[df['condition'] == 'baseline'][col].dropna().values
    rows = []
    for cond in df['condition'].unique():
        vals = df[df['condition'] == cond][col].dropna().values
        if cond == 'baseline':
            stat, p = float('nan'), float('nan')
        else:
            try:
                stat, p = mannwhitneyu(vals, base, alternative='two-sided')
                stat = float(stat); p = float(p)
            except ValueError:
                stat, p = float('nan'), float('nan')
        rows.append({
            'condition': cond,
            f'mean_{label}': float(vals.mean()) if len(vals) else float('nan'),
            f'sd_{label}': float(vals.std(ddof=1)) if len(vals) > 1 else float('nan'),
            f'mwu_stat': stat,
            f'mwu_p': p,
        })
    return pd.DataFrame(rows)


# ── Section 2g: JUDGE verdict breakdown ──

def compute_judge_verdicts(df):
    """Aggregate JUDGE verdicts across all sentences in each condition.
    Re-load coded traces to access sentence-level judgement."""
    rows = []
    for cond_dir in sorted(STEERED_DIR.iterdir()):
        if not cond_dir.is_dir():
            continue
        cond = cond_dir.name
        if cond not in df['condition'].unique():
            continue
        accept = reject = uncertain = total = 0
        for fp in cond_dir.rglob('trace_*_coded.json'):
            try:
                with open(fp, encoding='utf-8-sig') as f:
                    t = json.load(f)
            except Exception:
                continue
            for s in t.get('sentences', []) or []:
                c = s.get('coding') or {}
                if c.get('micro_label') == 'JUDGE':
                    v = c.get('judgement')
                    total += 1
                    if v == 'accept':
                        accept += 1
                    elif v == 'reject':
                        reject += 1
                    elif v == 'uncertain':
                        uncertain += 1
        rows.append({
            'condition': cond,
            'n_judges': total,
            'accept': accept, 'reject': reject, 'uncertain': uncertain,
            'accept_rate': accept / total if total else float('nan'),
            'reject_rate': reject / total if total else float('nan'),
            'uncertain_rate': uncertain / total if total else float('nan'),
        })
    return pd.DataFrame(rows)


# ── Section 2i: JUDGE_neg_2.0 transition matrix ──

def compute_transition_matrix(condition):
    """Aggregate sentence-level micro-label transitions across all traces in `condition`."""
    cond_dir = STEERED_DIR / condition
    counts = np.zeros((len(MICRO_LABELS), len(MICRO_LABELS)), dtype=float)
    idx = {l: i for i, l in enumerate(MICRO_LABELS)}
    for fp in cond_dir.rglob('trace_*_coded.json'):
        try:
            with open(fp, encoding='utf-8-sig') as f:
                t = json.load(f)
        except Exception:
            continue
        labels = []
        for s in t.get('sentences', []) or []:
            c = s.get('coding') or {}
            l = c.get('micro_label')
            if l in idx:
                labels.append(l)
        for a, b in zip(labels[:-1], labels[1:]):
            counts[idx[a], idx[b]] += 1
    # Normalise rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return counts, probs


# ── Main ──

def main():
    print(f'Loading {PER_TRACE_CSV}...')
    df = pd.read_csv(PER_TRACE_CSV)
    print(f'  {len(df)} traces in {df.condition.nunique()} conditions')

    print('\n[2a/2b] On-target + off-target shifts (paired Wilcoxon vs baseline)...')
    eff = compute_on_off_target(df)

    # Bonferroni: correct ON-TARGET tests in the 20-condition primary family
    on_target_primary = (eff['on_target']) & (eff['condition'].isin(PRIMARY_CONDITIONS))
    eff = add_bonferroni(eff, on_target_primary, 'p_raw', 'p_bonf')
    eff_path = OUT_DIR / 'shift_results.csv'
    eff.to_csv(eff_path, index=False)
    print(f'  Saved {eff_path.name} ({len(eff)} rows)')

    # On-target subset
    on_target = eff[eff['on_target']].copy()
    on_target_path = OUT_DIR / 'on_target_effects.csv'
    on_target.to_csv(on_target_path, index=False)
    n_sig = int(on_target_primary.sum() and (eff[on_target_primary]['p_bonf'] < 0.05).sum())
    print(f'  Saved {on_target_path.name}  '
          f'({(on_target["on_target"]).sum()} on-target rows; '
          f'{n_sig} significant after Bonferroni in primary family of 20)')

    # Off-target shift matrix at alpha=1.0 (mean shift)
    rows_alpha1 = eff[(eff['alpha'] == 1.0) & (eff['condition'] != 'JUDGE_neg_2.0')]
    pivot = rows_alpha1.pivot_table(
        index=['category', 'direction'], columns='label', values='shift', aggfunc='first')
    pivot = pivot.reindex(columns=MICRO_LABELS)
    pivot_p = rows_alpha1.pivot_table(
        index=['category', 'direction'], columns='label', values='p_bonf', aggfunc='first')
    pivot_p = pivot_p.reindex(columns=MICRO_LABELS)
    shift_matrix_path = OUT_DIR / 'category_shift_matrix.csv'
    pivot.to_csv(shift_matrix_path)
    pivot_p.to_csv(OUT_DIR / 'category_shift_matrix_pbonf.csv')
    print(f'  Saved {shift_matrix_path.name} (rows={len(pivot)}, cols={len(MICRO_LABELS)})')

    print('\n[2c] Dose-response (Spearman over per-condition means)...')
    dose_per_trace, dose_summary = compute_dose_response(df)
    dose_per_trace.to_csv(OUT_DIR / 'dose_response_per_trace.csv', index=False)
    dose_summary.to_csv(OUT_DIR / 'dose_response_summary.csv', index=False)
    print(dose_summary[['category', 'n_levels', 'spearman_rho', 'spearman_p']].to_string(index=False))

    print('\n[2d] Completion rate (Fisher\'s exact vs baseline)...')
    comp = compute_completion(df)
    comp.to_csv(OUT_DIR / 'completion_results.csv', index=False)
    base_rate = comp[comp['condition'] == 'baseline']['completion_rate'].iloc[0]
    sig_comp = comp[(comp['fisher_p'] < 0.05) & (comp['condition'] != 'baseline')]
    print(f'  Baseline = {base_rate:.3f}; '
          f'{len(sig_comp)} conditions differ from baseline at p<0.05 (uncorrected)')

    print('\n[2e] Trace length (Mann-Whitney U vs baseline)...')
    length_df = compute_simple_stat(df, 'n_sentences', 'length')
    length_df.to_csv(OUT_DIR / 'length_results.csv', index=False)
    sig_len = length_df[(length_df['mwu_p'] < 0.05) & (length_df['condition'] != 'baseline')]
    print(f'  {len(sig_len)} conditions differ from baseline at p<0.05 (uncorrected)')

    print('\n[2f] Transition entropy (Mann-Whitney U vs baseline)...')
    ent_df = compute_simple_stat(df, 'transition_entropy', 'entropy')
    ent_df.to_csv(OUT_DIR / 'entropy_results.csv', index=False)
    sig_ent = ent_df[(ent_df['mwu_p'] < 0.05) & (ent_df['condition'] != 'baseline')]
    print(f'  {len(sig_ent)} conditions differ from baseline at p<0.05 (uncorrected)')

    if STEERED_DIR.exists():
        print('\n[2g] JUDGE verdict breakdown (corpus-level)...')
        jv = compute_judge_verdicts(df)
        jv.to_csv(OUT_DIR / 'judge_verdicts.csv', index=False)
        base_jv = jv[jv['condition'] == 'baseline'].iloc[0]
        print(f"  Baseline: accept={base_jv['accept_rate']:.3f}, reject={base_jv['reject_rate']:.3f}, "
              f"uncertain={base_jv['uncertain_rate']:.3f}")

        print('\n[2i] Transition matrices (baseline + JUDGE_neg_2.0)...')
        for cond in ('baseline', 'JUDGE_neg_2.0'):
            counts, probs = compute_transition_matrix(cond)
            pd.DataFrame(counts, index=MICRO_LABELS, columns=MICRO_LABELS).to_csv(
                OUT_DIR / f'transition_matrix_{cond}_counts.csv')
            pd.DataFrame(probs, index=MICRO_LABELS, columns=MICRO_LABELS).to_csv(
                OUT_DIR / f'transition_matrix_{cond}_probs.csv')
            print(f'  Saved transition_matrix_{cond}_counts.csv + _probs.csv')
    else:
        print(f'\n[2g, 2i] Skipped: raw steered-trace dir not found at {STEERED_DIR}')
        print('  (These steps need the per-sentence coded JSONs which are not published.')
        print('   The published judge_verdicts.csv and transition_matrix_*.csv in '
              f'{OUT_DIR} already contain these results.)')

    # On-target headline summary printout
    print('\nOn-target headline (alpha<=1.0):')
    headline = on_target[on_target['condition'].isin(PRIMARY_CONDITIONS)].copy()
    headline = headline.sort_values(['category', 'direction', 'alpha'])
    cols = ['condition', 'mean_baseline', 'mean_steered', 'shift', 'cohen_d', 'p_raw', 'p_bonf', 'significant_005']
    print(headline[cols].to_string(index=False))

    print('\n=== Analysis complete ===')


if __name__ == '__main__':
    main()

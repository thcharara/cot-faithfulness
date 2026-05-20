#!/usr/bin/env python3
"""Study 2b: Generate publication figures from analysis CSVs.

Produces under outputs/study2b_steering/figures/:
    category_shift_heatmap.png
    dose_response_curves.png
    completion_by_condition.png
    trace_length_by_condition.png
    judge_verdict_breakdown.png
    transition_entropy_by_condition.png
    judge_neg_2_transition_matrix.png

Run *after* analyse_steering.py and prepare_steering_analysis.py.
"""

import argparse
import io
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--in-dir',
    type=Path,
    default=PROJECT_ROOT / 'study2b_steering' / 'results',
    help='Directory containing the analysis CSVs (default: study2b_steering/results/)',
)
parser.add_argument(
    '--fig-dir',
    type=Path,
    default=None,
    help='Directory to write figures (default: <in-dir>/figures)',
)
args = parser.parse_args()
OUT_DIR = args.in_dir
FIG_DIR = args.fig_dir if args.fig_dir is not None else OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST',
                'JUDGE', 'PLAN', 'MONITOR', 'RULE']
STEER_CATEGORIES = ['HYPO', 'TEST', 'JUDGE', 'MONITOR', 'PLAN']

CONDITION_ORDER = [
    'baseline',
    'HYPO_neg_1.0', 'HYPO_neg_0.5', 'HYPO_pos_0.5', 'HYPO_pos_1.0',
    'TEST_neg_1.0', 'TEST_neg_0.5', 'TEST_pos_0.5', 'TEST_pos_1.0',
    'JUDGE_neg_2.0', 'JUDGE_neg_1.0', 'JUDGE_neg_0.5', 'JUDGE_pos_0.5', 'JUDGE_pos_1.0',
    'MONITOR_neg_1.0', 'MONITOR_neg_0.5', 'MONITOR_pos_0.5', 'MONITOR_pos_1.0',
    'PLAN_neg_1.0', 'PLAN_neg_0.5', 'PLAN_pos_0.5', 'PLAN_pos_1.0',
]


def sig_stars(p):
    if p != p:  # nan
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


# ── Figure 1: Category shift heatmap ──

def plot_shift_heatmap(eff):
    """eff is shift_results.csv. Build shift heatmap (alpha=1.0) with bonferroni stars."""
    sub = eff[(eff['alpha'] == 1.0) & (eff['condition'] != 'JUDGE_neg_2.0')].copy()
    rows = []
    for cat in STEER_CATEGORIES:
        for direction in ['pos', 'neg']:
            r = sub[(sub['category'] == cat) & (sub['direction'] == direction)]
            if r.empty:
                continue
            r = r.set_index('label').reindex(MICRO_LABELS)
            rows.append((cat, direction, r['shift'].values, r['p_bonf'].values))

    matrix = np.array([row[2] for row in rows])
    pmatrix = np.array([row[3] for row in rows])
    row_labels = [f"{cat}_{'+' if d == 'pos' else '−'}1.0" for cat, d, _, _ in rows]

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.05)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(MICRO_LABELS)))
    ax.set_xticklabels(MICRO_LABELS, rotation=35, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            stars = sig_stars(pmatrix[i, j])
            color = 'white' if abs(v) > vmax * 0.6 else 'black'
            txt = f'{v*100:+.1f}{stars}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label('Shift (proportion, steered − baseline)')
    ax.set_title('Category-distribution shift at α = 1.0\n'
                 'Cells: percentage-point change; stars = Bonferroni-corrected p '
                 '(* <0.05, ** <0.01, *** <0.001)')
    fig.tight_layout()
    out = FIG_DIR / 'category_shift_heatmap.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ── Figure 2: Dose-response ──

def plot_dose_response(per_trace, per_trace_summary, summary):
    """Per-category dose-response with per-trace points + condition mean."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharey=False)
    base_means = {}
    base = per_trace_summary[per_trace_summary['condition'] == 'baseline']
    for cat in STEER_CATEGORIES:
        base_means[cat] = base[f'prop_{cat}'].mean()

    for i, cat in enumerate(STEER_CATEGORIES):
        ax = axes[i]
        cat_data = per_trace[per_trace['category'] == cat]
        # Per-condition stats
        agg = cat_data.groupby('signed_alpha')['target_prop'].agg(['mean', 'std', 'count']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])
        # Add the baseline point at signed_alpha=0 (the dose_response csv already does this for baseline rows
        # because they were re-included; ensure no duplicates)
        ax.errorbar(agg['signed_alpha'], agg['mean'], yerr=agg['sem'],
                    fmt='o-', color='steelblue', capsize=4, linewidth=1.5)
        # Per-trace jittered scatter
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.04, 0.04, size=len(cat_data))
        ax.scatter(cat_data['signed_alpha'] + jitter, cat_data['target_prop'],
                   alpha=0.15, s=10, color='steelblue')
        ax.axhline(base_means[cat], color='gray', linestyle='--', alpha=0.6,
                   label=f'baseline ({base_means[cat]*100:.1f}%)')
        # Shade Spearman p
        srow = summary[summary['category'] == cat].iloc[0]
        rho = srow['spearman_rho']; sp = srow['spearman_p']
        ax.set_title(f'{cat}\nρ={rho:.2f}, p={sp:.3f}', fontsize=11)
        ax.set_xlabel('signed α (− = neg, + = pos)')
        if i == 0:
            ax.set_ylabel(f'on-target proportion')
        ax.legend(loc='best', fontsize=8)
        ax.set_xticks(sorted(agg['signed_alpha'].unique()))

    fig.suptitle('Dose-response: on-target category proportion vs. steering strength',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / 'dose_response_curves.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ── Figure 3: Completion by condition ──

def plot_completion(comp_df):
    df = comp_df.set_index('condition').reindex(CONDITION_ORDER).reset_index()
    base_rate = df[df['condition'] == 'baseline']['completion_rate'].iloc[0]

    colors = []
    for c in df['condition']:
        if c == 'baseline':
            colors.append('black')
        elif '_pos_' in c:
            colors.append('#3b73af')
        else:
            colors.append('#d9534f')
    fig, ax = plt.subplots(figsize=(13, 4.5))
    bars = ax.bar(range(len(df)), df['completion_rate'], color=colors, edgecolor='white')
    for i, (bar, r) in enumerate(zip(bars, df.itertuples())):
        if r.condition == 'baseline':
            continue
        stars = sig_stars(getattr(r, 'fisher_p'))
        if stars:
            ax.text(i, bar.get_height() + 0.01, stars, ha='center', fontsize=10)
    ax.axhline(base_rate, linestyle='--', color='gray', alpha=0.7,
               label=f'baseline ({base_rate:.2f})')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['condition'], rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('Completion rate (fraction of traces with non-empty answer)')
    ax.set_title('Completion rate by steering condition (Fisher\'s exact vs baseline)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out = FIG_DIR / 'completion_by_condition.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ── Figure 4: Trace length ──

def plot_trace_length(per_trace):
    df_order = [c for c in CONDITION_ORDER if c in per_trace['condition'].unique()]
    data = [per_trace[per_trace['condition'] == c]['n_sentences'].values for c in df_order]
    fig, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot(data, labels=df_order, patch_artist=True, showfliers=True)
    for i, (patch, c) in enumerate(zip(bp['boxes'], df_order)):
        if c == 'baseline':
            patch.set_facecolor('#999999')
        elif '_pos_' in c:
            patch.set_facecolor('#3b73af')
        else:
            patch.set_facecolor('#d9534f')
        patch.set_alpha(0.7)
    base_med = np.median(per_trace[per_trace['condition'] == 'baseline']['n_sentences'])
    ax.axhline(base_med, color='gray', linestyle='--', alpha=0.7,
               label=f'baseline median ({base_med:.0f})')
    ax.set_xticklabels(df_order, rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('Sentences per trace')
    ax.set_title('Trace length by steering condition')
    ax.legend(loc='upper right')
    fig.tight_layout()
    out = FIG_DIR / 'trace_length_by_condition.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ── Figure 5: JUDGE verdict breakdown ──

def plot_judge_verdicts(jv):
    df_order = [c for c in CONDITION_ORDER if c in jv['condition'].unique()]
    df = jv.set_index('condition').reindex(df_order).reset_index()
    accept = df['accept_rate'].fillna(0).values
    reject = df['reject_rate'].fillna(0).values
    uncert = df['uncertain_rate'].fillna(0).values
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x, accept, color='#5cb85c', label='accept')
    ax.bar(x, reject, bottom=accept, color='#d9534f', label='reject')
    ax.bar(x, uncert, bottom=accept + reject, color='#f0ad4e', label='uncertain')
    ax.set_xticks(x)
    ax.set_xticklabels(df['condition'], rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('Fraction of JUDGE sentences')
    ax.set_title('JUDGE verdict breakdown by condition (corpus-level over all JUDGE sentences)')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out = FIG_DIR / 'judge_verdict_breakdown.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ── Figure 6: Transition entropy ──

def plot_entropy(per_trace):
    df_order = [c for c in CONDITION_ORDER if c in per_trace['condition'].unique()]
    data = [per_trace[per_trace['condition'] == c]['transition_entropy'].dropna().values
            for c in df_order]
    fig, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot(data, labels=df_order, patch_artist=True, showfliers=True)
    for patch, c in zip(bp['boxes'], df_order):
        if c == 'baseline':
            patch.set_facecolor('#999999')
        elif '_pos_' in c:
            patch.set_facecolor('#3b73af')
        else:
            patch.set_facecolor('#d9534f')
        patch.set_alpha(0.7)
    base_med = np.median(per_trace[per_trace['condition'] == 'baseline']['transition_entropy'].dropna())
    ax.axhline(base_med, color='gray', linestyle='--', alpha=0.7,
               label=f'baseline median ({base_med:.2f})')
    ax.set_xticklabels(df_order, rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('Bigram transition entropy (bits)')
    ax.set_title('Transition entropy by steering condition')
    ax.legend(loc='upper right')
    fig.tight_layout()
    out = FIG_DIR / 'transition_entropy_by_condition.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ── Figure 7: JUDGE_neg_2.0 transition matrix ──

def plot_judge_neg_2_matrix():
    base = pd.read_csv(OUT_DIR / 'transition_matrix_baseline_probs.csv', index_col=0)
    cond = pd.read_csv(OUT_DIR / 'transition_matrix_JUDGE_neg_2.0_probs.csv', index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, mat, title in [
        (axes[0], base.values, 'Baseline'),
        (axes[1], cond.values, 'JUDGE_neg_2.0'),
        (axes[2], cond.values - base.values, 'Difference (JUDGE_neg_2.0 − baseline)'),
    ]:
        cmap = 'RdBu_r' if 'Diff' in title else 'YlOrRd'
        if 'Diff' in title:
            v = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.05)
            im = ax.imshow(mat, cmap=cmap, vmin=-v, vmax=v, aspect='auto')
        else:
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(0.1, np.nanmax(mat)), aspect='auto')
        ax.set_xticks(range(len(MICRO_LABELS)))
        ax.set_yticks(range(len(MICRO_LABELS)))
        ax.set_xticklabels(MICRO_LABELS, rotation=35, ha='right')
        ax.set_yticklabels(MICRO_LABELS)
        ax.set_title(title)
        for i in range(len(MICRO_LABELS)):
            for j in range(len(MICRO_LABELS)):
                v = mat[i, j]
                if abs(v) > 0.005:
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=7, color='black')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    axes[0].set_ylabel('From label')
    for a in axes:
        a.set_xlabel('To label')

    fig.suptitle('Transition matrices: baseline vs JUDGE_neg_2.0 (rows = from, cols = to)',
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / 'judge_neg_2_transition_matrix.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


def plot_accuracy_comparison():
    """Compare Set A baseline + Set B ceiling + Study 2b conditions on accuracy."""
    pop = pd.read_csv(OUT_DIR / 'study1_accuracy_by_population.csv')
    cmp_df = pd.read_csv(OUT_DIR / 'accuracy_vs_study1_baseline.csv')

    try:
        from statsmodels.stats.proportion import proportion_confint
        def wilson(k, n):
            if n == 0:
                return 0.0, 0.0
            return proportion_confint(k, n, alpha=0.05, method='wilson')
    except ImportError:
        def wilson(k, n):
            if n == 0:
                return 0.0, 0.0
            p = k / n
            z = 1.959963984540054
            denom = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denom
            half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
            return max(0.0, centre - half), min(1.0, centre + half)

    set_a_row = pop[pop['population'] == 'set_a'].iloc[0]
    set_b_row = pop[pop['population'] == 'set_b_all'].iloc[0]

    bars = []  # list of (label, rate, lo, hi, p, colour, hatch)
    for label, row, colour, hatch in [
        ('Set A (matched)', set_a_row, '#4a4a4a', None),
        ('Set B (cued ceiling)', set_b_row, '#888888', '//'),
    ]:
        n_comp = int(row['n_completed']); n_cor = int(row['n_correct'])
        rate = n_cor / n_comp if n_comp else 0.0
        lo, hi = wilson(n_cor, n_comp)
        bars.append((label, rate, lo, hi, None, colour, hatch))

    # Now Study 2b conditions in CONDITION_ORDER (baseline first)
    for cond in CONDITION_ORDER:
        r = cmp_df[cmp_df['condition'] == cond]
        if r.empty:
            continue
        r = r.iloc[0]
        n_comp = int(r['n_completed']); n_cor = int(r['n_correct'])
        rate = (n_cor / n_comp) if n_comp else 0.0
        lo, hi = wilson(n_cor, n_comp)
        if cond == 'baseline':
            colour = '#000000'
        elif '_pos_' in cond:
            colour = '#3b73af'
        else:
            colour = '#d9534f'
        bars.append((cond, rate, lo, hi, r['fisher_p'], colour, None))

    fig, ax = plt.subplots(figsize=(15, 5))
    xs = np.arange(len(bars))
    rates = [b[1] for b in bars]
    yerr_low = [max(0.0, b[1] - b[2]) for b in bars]
    yerr_high = [max(0.0, b[3] - b[1]) for b in bars]
    colours = [b[5] for b in bars]
    container = ax.bar(xs, rates, color=colours, edgecolor='white',
                       yerr=[yerr_low, yerr_high], capsize=3, ecolor='#444444')
    for i, (label, rate, lo, hi, p, colour, hatch) in enumerate(bars):
        if hatch:
            container[i].set_hatch(hatch)
        if p is not None and p == p:
            stars = sig_stars(p)
            if stars:
                ax.text(i, hi + 0.012, stars, ha='center', fontsize=10)

    set_a_rate = bars[0][1]
    ax.axhline(set_a_rate, color='#4a4a4a', linestyle='--', alpha=0.6,
               label=f'Set A baseline ({set_a_rate*100:.1f}%)')

    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('Heuristic accuracy among completed traces')
    ax.set_title('Accuracy vs. Study 1 Set A (matched baseline) and Set B (cued ceiling)\n'
                 'Stars: Fisher\'s exact vs Set A (* p<0.05, ** p<0.01, *** p<0.001) — uncorrected. '
                 'Hatched bar = Set B (includes completion cue).')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(0.30, max(b[3] for b in bars) * 1.15))
    fig.tight_layout()
    out = FIG_DIR / 'accuracy_comparison_study1_baseline.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


def main():
    print('Loading analysis CSVs...')
    eff = pd.read_csv(OUT_DIR / 'shift_results.csv')
    per_trace_summary = pd.read_csv(OUT_DIR / 'per_trace_summaries.csv')
    dose_per_trace = pd.read_csv(OUT_DIR / 'dose_response_per_trace.csv')
    dose_summary = pd.read_csv(OUT_DIR / 'dose_response_summary.csv')
    comp = pd.read_csv(OUT_DIR / 'completion_results.csv')
    jv = pd.read_csv(OUT_DIR / 'judge_verdicts.csv')

    print('Plotting...')
    plot_shift_heatmap(eff)
    plot_dose_response(dose_per_trace, per_trace_summary, dose_summary)
    plot_completion(comp)
    plot_trace_length(per_trace_summary)
    plot_judge_verdicts(jv)
    plot_entropy(per_trace_summary)
    plot_judge_neg_2_matrix()
    plot_accuracy_comparison()
    print('Done.')


if __name__ == '__main__':
    main()

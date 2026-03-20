#!/usr/bin/env python3
"""Study 2b: Analyse steering experiment results.

Compares steered conditions to baseline across category distribution,
completion rate, trace length, and transition entropy. Produces figures
and a summary report.

Usage:
    .venv/Scripts/python.exe study2b_steering/scripts/study2b_analyse_steering.py
    .venv/Scripts/python.exe study2b_steering/scripts/study2b_analyse_steering.py --pilot
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / 'outputs' / 'study2b_steering'
FIGURES_DIR = RESULTS_DIR / 'figures'

# ── Constants ──
MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST',
                'JUDGE', 'PLAN', 'MONITOR', 'RULE']
STEER_CATEGORIES = ['HYPO', 'TEST', 'JUDGE', 'MONITOR', 'PLAN']
ALPHAS = [0.5, 1.0, 2.0]


def load_all_per_trace():
    """Load per-trace CSVs for all conditions. Returns dict of DataFrames."""
    data = {}
    for f in sorted(RESULTS_DIR.glob('*_per_trace.csv')):
        condition = f.stem.replace('_per_trace', '')
        df = pd.read_csv(f)
        data[condition] = df
    return data


def parse_condition(name):
    """Parse condition name into (category, direction, alpha) or None for baseline."""
    if name in ('baseline', 'pilot_baseline'):
        return None
    parts = name.split('_')
    if len(parts) >= 3:
        try:
            cat = parts[0]
            direction = parts[1]
            alpha = float(parts[2])
            return cat, direction, alpha
        except (ValueError, IndexError):
            return None
    return None


def cohens_d(x, y):
    """Compute Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    return diff.mean() / diff.std() if diff.std() > 0 else 0.0


def compute_steering_effects(data, baseline_key='baseline'):
    """Compare each steered condition to baseline."""
    if baseline_key not in data:
        # Try pilot baseline
        baseline_key = 'pilot_baseline'
    if baseline_key not in data:
        print("ERROR: No baseline found in data")
        return pd.DataFrame()

    baseline = data[baseline_key]
    results = []

    for condition, df in data.items():
        parsed = parse_condition(condition)
        if parsed is None:
            continue

        cat, direction, alpha = parsed

        # Need same number of traces for paired test
        n = min(len(baseline), len(df))
        if n < 2:
            # Cannot run statistical test with < 2 pairs
            row = {
                'condition': condition,
                'category': cat,
                'direction': direction,
                'alpha': alpha,
                'n_traces': len(df),
            }
            for label in MICRO_LABELS:
                col = f'prop_{label}' if f'prop_{label}' in df.columns else label
                if col in df.columns:
                    row[f'mean_{label}'] = df[col].mean()
                    baseline_mean = baseline[col].mean() if col in baseline.columns else 0
                    row[f'shift_{label}'] = row[f'mean_{label}'] - baseline_mean
                    row[f'd_{label}'] = 0.0
                    row[f'p_{label}'] = 1.0
            row['completion_rate'] = df['completed'].mean() if 'completed' in df.columns else np.nan
            row['mean_n_sentences'] = df['n_sentences'].mean() if 'n_sentences' in df.columns else np.nan
            row['mean_entropy'] = df['transition_entropy'].mean() if 'transition_entropy' in df.columns else np.nan
            results.append(row)
            continue

        row = {
            'condition': condition,
            'category': cat,
            'direction': direction,
            'alpha': alpha,
            'n_traces': len(df),
        }

        for label in MICRO_LABELS:
            # Use prop_{label} column if available, else just {label}
            col = f'prop_{label}' if f'prop_{label}' in df.columns else label
            if col not in df.columns:
                continue

            steered_vals = df[col].values[:n]
            baseline_vals = baseline[col].values[:n]

            row[f'mean_{label}'] = float(steered_vals.mean())
            row[f'shift_{label}'] = float(steered_vals.mean() - baseline_vals.mean())
            row[f'd_{label}'] = cohens_d(steered_vals, baseline_vals)

            # Wilcoxon signed-rank (paired)
            try:
                stat, p = wilcoxon(steered_vals, baseline_vals)
                row[f'p_{label}'] = p
            except ValueError:
                row[f'p_{label}'] = 1.0

        # Completion rate
        if 'completed' in df.columns:
            row['completion_rate'] = df['completed'].mean()
        if 'n_sentences' in df.columns:
            row['mean_n_sentences'] = df['n_sentences'].mean()
        if 'transition_entropy' in df.columns:
            row['mean_entropy'] = df['transition_entropy'].mean()

        results.append(row)

    return pd.DataFrame(results)


def build_shift_matrix(effects_df, alpha_filter=1.0):
    """Build category-shift matrix: rows=steered categories, cols=all labels.

    Returns (matrix, row_labels, col_labels)
    """
    filtered = effects_df[effects_df['alpha'] == alpha_filter]

    rows = []
    row_labels = []

    for cat in STEER_CATEGORIES:
        for direction in ['pos', 'neg']:
            mask = (filtered['category'] == cat) & (filtered['direction'] == direction)
            if mask.sum() == 0:
                continue
            row_data = filtered[mask].iloc[0]
            shifts = [row_data.get(f'shift_{label}', 0.0) for label in MICRO_LABELS]
            rows.append(shifts)
            sign = '+' if direction == 'pos' else '-'
            row_labels.append(f'{cat}{sign}')

    if not rows:
        return None, None, None

    matrix = np.array(rows)
    return matrix, row_labels, MICRO_LABELS


def build_dose_response(effects_df):
    """Build dose-response table: for each category, target proportion vs alpha."""
    rows = []
    for cat in STEER_CATEGORIES:
        cat_df = effects_df[effects_df['category'] == cat]
        for _, row in cat_df.iterrows():
            sign = 1 if row['direction'] == 'pos' else -1
            signed_alpha = sign * row['alpha']
            target_prop = row.get(f'mean_{cat}', np.nan)
            rows.append({
                'category': cat,
                'signed_alpha': signed_alpha,
                'target_proportion': target_prop,
                'shift': row.get(f'shift_{cat}', np.nan),
                'cohens_d': row.get(f'd_{cat}', np.nan),
            })

    return pd.DataFrame(rows)


# ── Visualisations ──

def plot_category_shift_heatmap(matrix, row_labels, col_labels, save_path):
    """Heatmap: rows=steered conditions, cols=all labels, cell=shift."""
    fig, ax = plt.subplots(figsize=(12, max(4, len(row_labels) * 0.6 + 1)))

    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.05)
    cmap = plt.cm.RdBu_r

    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='Shift from baseline')
    ax.set_title('Category Distribution Shift (steered - baseline)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_dose_response(dose_df, baseline_data, save_path):
    """Dose-response curves: target category proportion vs signed alpha."""
    categories = dose_df['category'].unique()
    n_cats = len(categories)
    if n_cats == 0:
        return

    fig, axes = plt.subplots(1, n_cats, figsize=(4 * n_cats, 4), squeeze=False)
    axes = axes[0]

    for i, cat in enumerate(categories):
        ax = axes[i]
        cat_data = dose_df[dose_df['category'] == cat].sort_values('signed_alpha')

        # Add baseline point at alpha=0
        if baseline_data is not None:
            col = f'prop_{cat}' if f'prop_{cat}' in baseline_data.columns else cat
            if col in baseline_data.columns:
                baseline_val = baseline_data[col].mean()
            else:
                baseline_val = 0
        else:
            baseline_val = 0

        alphas = list(cat_data['signed_alpha'])
        props = list(cat_data['target_proportion'])

        # Insert baseline at 0
        all_alphas = sorted(set(alphas + [0]))
        all_props = []
        for a in all_alphas:
            if a == 0:
                all_props.append(baseline_val)
            else:
                idx = alphas.index(a) if a in alphas else None
                all_props.append(props[idx] if idx is not None else np.nan)

        ax.plot(all_alphas, all_props, 'o-', color='steelblue', linewidth=2)
        ax.axhline(y=baseline_val, color='gray', linestyle='--', alpha=0.5,
                   label='baseline')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
        ax.set_xlabel('Signed alpha')
        ax.set_ylabel(f'{cat} proportion')
        ax.set_title(cat)
        ax.legend(fontsize=8)

    fig.suptitle('Dose-Response: Target Category Proportion vs Steering Strength',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_completion_by_condition(effects_df, baseline_rate, save_path):
    """Bar chart of completion rate by condition."""
    if 'completion_rate' not in effects_df.columns:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(effects_df) * 0.5), 5))

    conditions = effects_df['condition'].values
    rates = effects_df['completion_rate'].values

    colors = []
    for _, row in effects_df.iterrows():
        if row['direction'] == 'pos':
            colors.append('steelblue')
        else:
            colors.append('coral')

    x = range(len(conditions))
    ax.bar(x, rates, color=colors, alpha=0.8)
    ax.axhline(y=baseline_rate, color='black', linestyle='--', linewidth=2,
               label=f'Baseline ({baseline_rate:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Completion Rate')
    ax.set_title('Completion Rate by Steering Condition')
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ── Report ──

def write_report(effects_df, shift_matrix, dose_df, baseline_data,
                 save_path, pilot=False):
    """Write markdown summary report."""
    lines = [
        "# Study 2b: Steering Vector Results\n",
        f"**Status:** {'Pilot (3 traces)' if pilot else 'Full run'}\n",
        "---\n",
        "## 1. Experimental Design\n",
        "- **Intervention:** Additive steering vector at layer 20 during generation",
        "- **Categories steered:** " + ", ".join(STEER_CATEGORIES),
        "- **Directions:** positive (+alpha) and negative (-alpha)",
        "- **Strengths:** alpha in {0.5, 1.0, 2.0}",
        "- **Normalisation:** Vectors normalised to mean activation magnitude "
        "(12.69 at layer 20)\n",
    ]

    # Baseline summary
    if baseline_data is not None:
        lines.append("## 2. Baseline\n")
        n = len(baseline_data)
        lines.append(f"- **N traces:** {n}")
        if 'completed' in baseline_data.columns:
            lines.append(f"- **Completion rate:** {baseline_data['completed'].mean():.3f}")
        if 'n_sentences' in baseline_data.columns:
            lines.append(f"- **Mean trace length:** "
                         f"{baseline_data['n_sentences'].mean():.1f} sentences")
        lines.append("")

        # Category distribution
        lines.append("| Category | Proportion |")
        lines.append("|---|---|")
        for label in MICRO_LABELS:
            col = f'prop_{label}' if f'prop_{label}' in baseline_data.columns else label
            if col in baseline_data.columns:
                lines.append(f"| {label} | {baseline_data[col].mean():.3f} |")
        lines.append("")

    # Steering effects
    if len(effects_df) > 0:
        lines.append("## 3. On-Target Steering Effects\n")
        lines.append("| Condition | Target Category | Shift | Cohen's d | p-value |")
        lines.append("|---|---|---|---|---|")

        for _, row in effects_df.iterrows():
            cat = row['category']
            shift_col = f'shift_{cat}'
            d_col = f'd_{cat}'
            p_col = f'p_{cat}'

            shift = row.get(shift_col, 0)
            d = row.get(d_col, 0)
            p = row.get(p_col, 1)

            sig = '*' if p < 0.05 else ''
            lines.append(f"| {row['condition']} | {cat} | "
                         f"{shift:+.3f} | {d:+.2f} | {p:.4f}{sig} |")
        lines.append("")

        # Off-target effects summary
        lines.append("## 4. Off-Target Effects\n")
        lines.append("See `category_shift_matrix.csv` and "
                      "`figures/category_shift_heatmap.png` for full matrix.\n")

        if shift_matrix is not None:
            matrix, row_labels, col_labels = shift_matrix
            max_off = 0
            for i, rl in enumerate(row_labels):
                cat = rl.rstrip('+-')
                for j, cl in enumerate(col_labels):
                    if cl != cat:
                        max_off = max(max_off, abs(matrix[i, j]))
            lines.append(f"- **Maximum off-target shift:** {max_off:.3f}")

        lines.append("")

        # Dose-response
        if dose_df is not None and len(dose_df) > 0:
            lines.append("## 5. Dose-Response\n")
            lines.append("See `dose_response.csv` and "
                          "`figures/dose_response_curves.png`.\n")

        # Completion
        lines.append("## 6. Completion Rate Effects\n")
        if 'completion_rate' in effects_df.columns:
            lines.append("| Condition | Rate |")
            lines.append("|---|---|")
            for _, row in effects_df.iterrows():
                lines.append(f"| {row['condition']} | "
                             f"{row.get('completion_rate', 0):.3f} |")
        lines.append("")

    lines.append("---\n")
    lines.append("*Generated by `scripts/analyse_steering.py`*\n")

    save_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Report saved: {save_path.name}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Analyse steering experiment results")
    parser.add_argument('--pilot', action='store_true',
                        help='Analyse pilot data only')
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load all per-trace data
    data = load_all_per_trace()
    if not data:
        print("ERROR: No per-trace data found in", RESULTS_DIR)
        print("Run code_steered_traces.py first.")
        sys.exit(1)

    print(f"Loaded {len(data)} conditions: {list(data.keys())}")

    # Identify baseline
    baseline_key = 'baseline' if 'baseline' in data else 'pilot_baseline'
    if baseline_key not in data:
        print("ERROR: No baseline condition found")
        sys.exit(1)

    baseline_data = data[baseline_key]
    baseline_completion = (baseline_data['completed'].mean()
                           if 'completed' in baseline_data.columns else 0)

    print(f"Baseline: {len(baseline_data)} traces, "
          f"completion={baseline_completion:.3f}")

    # Compute effects
    effects = compute_steering_effects(data, baseline_key)
    if len(effects) > 0:
        effects_path = RESULTS_DIR / 'steering_results.csv'
        effects.to_csv(effects_path, index=False)
        print(f"Results saved: {effects_path.name} ({len(effects)} conditions)")

    # Shift matrix
    shift_result = build_shift_matrix(effects, alpha_filter=1.0)
    matrix, row_labels, col_labels = shift_result
    if matrix is not None:
        shift_df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
        shift_path = RESULTS_DIR / 'category_shift_matrix.csv'
        shift_df.to_csv(shift_path)
        print(f"Shift matrix saved: {shift_path.name}")

        plot_category_shift_heatmap(
            matrix, row_labels, col_labels,
            FIGURES_DIR / 'category_shift_heatmap.png')

    # Dose-response
    dose_df = build_dose_response(effects)
    if len(dose_df) > 0:
        dose_path = RESULTS_DIR / 'dose_response.csv'
        dose_df.to_csv(dose_path, index=False)
        print(f"Dose-response saved: {dose_path.name}")

        plot_dose_response(dose_df, baseline_data,
                           FIGURES_DIR / 'dose_response_curves.png')

    # Completion chart
    if len(effects) > 0:
        plot_completion_by_condition(
            effects, baseline_completion,
            FIGURES_DIR / 'completion_by_condition.png')

    # Report
    write_report(effects, shift_result, dose_df, baseline_data,
                 RESULTS_DIR / 'study2b_report.md',
                 pilot=args.pilot)

    print("\nAnalysis complete.")


if __name__ == '__main__':
    main()

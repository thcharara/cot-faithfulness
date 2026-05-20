#!/usr/bin/env python3
"""Combine Study 2a probe F1 + Study 2b steering effects + accuracy
to classify each steered category as GROUNDED / PARTIALLY GROUNDED /
WEAKLY GROUNDED / PERFORMATIVE.

Decision rule (per project spec):
    GROUNDED          : probe F1 >= 0.6 AND significant on-target steering
                        AND no accuracy degradation
    PARTIALLY_GROUNDED: probe F1 >= 0.6 AND significant steering effect
                        AND accuracy unchanged or unmeasurable
    WEAKLY_GROUNDED   : probe F1 >= 0.6 AND non-significant or weak steering
    PERFORMATIVE      : probe F1 < 0.6 OR no steering effect

Output: outputs/study2b_steering/mechanistic_grounding_classification.csv
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--effects-dir',
    type=Path,
    default=PROJECT_ROOT / 'study2b_steering' / 'results',
    help='Directory with on_target_effects.csv, accuracy_vs_study1_baseline.csv, study1_accuracy_by_population.csv (default: study2b_steering/results/)',
)
parser.add_argument(
    '--probe-f1',
    type=Path,
    default=PROJECT_ROOT / 'study2a_probes' / 'results' / 'probe_per_category_f1.csv',
    help='Path to probe per-category F1 CSV (default: study2a_probes/results/probe_per_category_f1.csv)',
)
parser.add_argument(
    '--out-dir',
    type=Path,
    default=PROJECT_ROOT / 'study2b_steering' / 'results',
    help='Directory to write mechanistic_grounding_classification.csv (default: study2b_steering/results/)',
)
args = parser.parse_args()
OUT_DIR = args.out_dir
PROBE_F1_CSV = args.probe_f1
EFFECTS_DIR = args.effects_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEER_CATEGORIES = ['HYPO', 'TEST', 'JUDGE', 'MONITOR', 'PLAN']

PROBE_F1_THRESHOLD = 0.6
EFFECT_P_THRESHOLD = 0.05  # Bonferroni-corrected p < 0.05 -> significant


def main():
    # Probe F1 at layer 20 mean_pool (the steering layer + best aggregation)
    probe = pd.read_csv(PROBE_F1_CSV)
    probe_l20 = probe[(probe['layer'] == 20) & (probe['aggregation'] == 'mean_pool')].iloc[0]

    # On-target steering effects
    eff = pd.read_csv(EFFECTS_DIR / 'on_target_effects.csv')

    # Accuracy — Set A from Study 1 is the matched baseline (Decision 69)
    pop = pd.read_csv(EFFECTS_DIR / 'study1_accuracy_by_population.csv')
    set_a_row = pop[pop['population'] == 'set_a'].iloc[0]
    base_acc = float(set_a_row['accuracy_among_completed'])
    base_n_correct = int(set_a_row['n_correct'])
    base_n_completed = int(set_a_row['n_completed'])

    # Per-condition accuracy + Fisher p vs Set A
    cmp_df = pd.read_csv(EFFECTS_DIR / 'accuracy_vs_study1_baseline.csv')

    rows = []
    for cat in STEER_CATEGORIES:
        f1 = float(probe_l20[f'{cat}_f1'])

        # Strongest on-target effect at alpha=1.0 (positive direction by default;
        # for HYPO-neg reversal we still take the largest |shift| with p_bonf<0.05)
        cat_eff = eff[(eff['category'] == cat) & (eff['alpha'] == 1.0)]
        # Find best-evidence direction
        cat_eff_sig = cat_eff[cat_eff['p_bonf'] < EFFECT_P_THRESHOLD]
        if not cat_eff_sig.empty:
            best = cat_eff_sig.iloc[cat_eff_sig['cohen_d'].abs().argmax()]
            steering_significant = True
        else:
            best = cat_eff.iloc[cat_eff['cohen_d'].abs().argmax()]
            steering_significant = False

        # Accuracy change for the best-shift condition vs Set A baseline
        cond = best['condition']
        cmp_row = cmp_df[cmp_df['condition'] == cond]
        if not cmp_row.empty:
            cond_acc = float(cmp_row['accuracy_rate'].iloc[0])
            cond_fisher_p = float(cmp_row['fisher_p'].iloc[0])
        else:
            cond_acc = float('nan')
            cond_fisher_p = float('nan')
        acc_change = cond_acc - base_acc if cond_acc == cond_acc else float('nan')

        # Classification logic
        f1_grounded = f1 >= PROBE_F1_THRESHOLD
        if not f1_grounded:
            cls = 'PERFORMATIVE'
            note = f'Probe F1 ({f1:.3f}) below threshold ({PROBE_F1_THRESHOLD})'
        elif not steering_significant:
            cls = 'WEAKLY_GROUNDED'
            note = (f'Probe F1 ok ({f1:.3f}); steering not significant after Bonferroni '
                    f'(best p_bonf={best["p_bonf"]:.3f}, d={best["cohen_d"]:.2f})')
        else:
            # Set A baseline is 1/69 = 1.45% — measurable but very low. We treat
            # accuracy as "unmeasurable in practice" if Set A is below 5%, since
            # any steered cell of n~10–20 cannot resolve sub-percentage shifts.
            base_meaningful = base_acc >= 0.05
            if not base_meaningful:
                cls = 'PARTIALLY_GROUNDED'
                note = (f'Significant steering ({best["condition"]}: shift={best["shift"]*100:+.1f}pp, '
                        f'p_bonf={best["p_bonf"]:.3g}); '
                        f'accuracy unmeasurable against Study 1 Set A baseline of '
                        f'{base_acc*100:.2f}% ({base_n_correct}/{base_n_completed}); '
                        f'best-condition accuracy={cond_acc*100:.2f}% '
                        f'(Fisher p vs Set A = {cond_fisher_p:.3g})')
            elif acc_change < -0.10:
                cls = 'PARTIALLY_GROUNDED'
                note = (f'Accuracy degraded by {acc_change*100:.1f}pp under steering '
                        f'({cond_acc*100:.1f}% vs Set A {base_acc*100:.1f}%, '
                        f'Fisher p={cond_fisher_p:.3g})')
            else:
                cls = 'GROUNDED'
                note = (f'No accuracy degradation under steering '
                        f'({acc_change*100:+.1f}pp vs Set A; Fisher p={cond_fisher_p:.3g})')

        rows.append({
            'category': cat,
            'probe_f1_layer20_meanpool': round(f1, 4),
            'probe_grounded_threshold_06': f1_grounded,
            'best_condition': cond,
            'on_target_shift_pp': round(best['shift'] * 100, 2),
            'cohen_d': round(best['cohen_d'], 3),
            'wilcoxon_p_raw': round(best['p_raw'], 5),
            'wilcoxon_p_bonf': round(best['p_bonf'], 5) if best['p_bonf'] == best['p_bonf'] else float('nan'),
            'steering_significant_005': steering_significant,
            'accuracy_baseline_set_a': round(base_acc, 4),
            'accuracy_baseline_n': f'{base_n_correct}/{base_n_completed}',
            'accuracy_steered': round(cond_acc, 4) if cond_acc == cond_acc else float('nan'),
            'accuracy_change_pp': round(acc_change * 100, 2) if acc_change == acc_change else float('nan'),
            'accuracy_fisher_p_vs_set_a': round(cond_fisher_p, 5) if cond_fisher_p == cond_fisher_p else float('nan'),
            'classification': cls,
            'note': note,
        })

    out = pd.DataFrame(rows)
    save_path = OUT_DIR / 'mechanistic_grounding_classification.csv'
    out.to_csv(save_path, index=False)
    print(f'Saved {save_path}')
    print()
    print(out[['category', 'probe_f1_layer20_meanpool', 'on_target_shift_pp',
               'cohen_d', 'wilcoxon_p_bonf', 'classification']].to_string(index=False))
    print()
    print('Notes:')
    for _, r in out.iterrows():
        print(f"  {r['category']:<8} {r['classification']:<20} {r['note']}")


if __name__ == '__main__':
    main()

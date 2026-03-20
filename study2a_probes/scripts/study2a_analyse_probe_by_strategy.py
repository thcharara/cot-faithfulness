#!/usr/bin/env python3
"""Study 2a: Analyse probe performance by reasoning strategy.

Breaks down probe predictions by reasoning_strategy (full_cycling,
scan_test_conclude, direct_insight) to assess whether the probe
generalises to non-standard reasoning patterns.

Usage:
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_analyse_probe_by_strategy.py
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_analyse_probe_by_strategy.py --layer 31 --agg mean_pool --C 0.01
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ACTIVATIONS_DIR = PROJECT_ROOT / 'outputs' / 'activations'
PROBE_DATA_DIR = PROJECT_ROOT / 'outputs' / 'study2_probe_data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'study2_probes'

MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST',
                'JUDGE', 'PLAN', 'MONITOR', 'RULE']
STRATEGIES = ['full_cycling', 'scan_test_conclude', 'direct_insight']


def load_activations(layer, agg):
    """Load all activations for a layer/agg combo."""
    meta = pd.read_csv(PROBE_DATA_DIR / 'sentence_metadata.csv')
    X_parts = []
    for trace_key in sorted(meta['trace_key'].unique()):
        safe_key = trace_key.replace('/', '_')
        npz_path = ACTIVATIONS_DIR / f'layer_{layer}' / f'{safe_key}.npz'
        data = np.load(npz_path)
        X_parts.append(data[agg].astype(np.float32))
    X = np.vstack(X_parts)
    assert X.shape[0] == len(meta)
    return X, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=20)
    parser.add_argument('--agg', type=str, default='mean_pool')
    parser.add_argument('--C', type=float, default=1.0)
    args = parser.parse_args()

    layer, agg, C = args.layer, args.agg, args.C
    label_to_idx = {l: i for i, l in enumerate(MICRO_LABELS)}

    # ── Load model ──
    # Try C-specific model first, then fallback to generic
    model_path = OUTPUT_DIR / 'probe_models' / f'layer{layer}_{agg}_C{C}.pkl'
    if not model_path.exists():
        model_path = OUTPUT_DIR / 'probe_models' / f'layer{layer}_{agg}.pkl'
    print(f"Loading model: {model_path.name}")
    with open(model_path, 'rb') as f:
        scaler, clf = pickle.load(f)

    # ── Load data ──
    print(f"Loading activations: layer {layer}, {agg}")
    X, meta = load_activations(layer, agg)
    y = np.array([label_to_idx[l] for l in meta['micro_label'].values])

    # ── Analysis 1: Test-fold only (fold 0) ──
    test_mask = meta['fold_id'].values == 0
    X_test = scaler.transform(X[test_mask])
    y_test = y[test_mask]
    y_pred_test = clf.predict(X_test)
    meta_test = meta[test_mask].copy()

    print(f"\n{'='*70}")
    print(f"STRATEGY ANALYSIS (Layer {layer}, {agg}, C={C})")
    print(f"{'='*70}")

    # ── Per-strategy results on test fold ──
    print(f"\n--- Test Fold (fold 0) ---")
    strat_rows = []
    for strategy in STRATEGIES:
        smask = meta_test['reasoning_strategy'].values == strategy
        n = smask.sum()
        n_traces = meta_test.loc[smask, 'trace_key'].nunique() if n > 0 else 0
        if n == 0:
            print(f"  {strategy:25s}: no test-fold samples")
            strat_rows.append({
                'strategy': strategy, 'split': 'test',
                'n_sentences': 0, 'n_traces': 0,
                'macro_f1': np.nan, 'weighted_f1': np.nan, 'accuracy': np.nan
            })
            continue

        ys = y_test[smask]
        yps = y_pred_test[smask]
        macro = f1_score(ys, yps, average='macro', zero_division=0)
        weighted = f1_score(ys, yps, average='weighted', zero_division=0)
        acc = accuracy_score(ys, yps)
        print(f"  {strategy:25s}: n={n:5d} ({n_traces:2d} traces)  "
              f"macro-F1={macro:.4f}  weighted-F1={weighted:.4f}  acc={acc:.4f}")
        strat_rows.append({
            'strategy': strategy, 'split': 'test',
            'n_sentences': n, 'n_traces': n_traces,
            'macro_f1': round(macro, 4), 'weighted_f1': round(weighted, 4),
            'accuracy': round(acc, 4)
        })

    # ── Analysis 2: All folds (training-set performance for minority strategies) ──
    print(f"\n--- All Folds (training-set perf for minority strategies) ---")
    X_all_s = scaler.transform(X)
    y_pred_all = clf.predict(X_all_s)

    for strategy in STRATEGIES:
        smask = meta['reasoning_strategy'].values == strategy
        n = smask.sum()
        n_traces = meta.loc[smask, 'trace_key'].nunique() if n > 0 else 0
        if n == 0:
            continue
        ys = y[smask]
        yps = y_pred_all[smask]
        macro = f1_score(ys, yps, average='macro', zero_division=0)
        weighted = f1_score(ys, yps, average='weighted', zero_division=0)
        acc = accuracy_score(ys, yps)
        print(f"  {strategy:25s}: n={n:5d} ({n_traces:3d} traces)  "
              f"macro-F1={macro:.4f}  weighted-F1={weighted:.4f}  acc={acc:.4f}")
        strat_rows.append({
            'strategy': strategy, 'split': 'all_folds',
            'n_sentences': n, 'n_traces': n_traces,
            'macro_f1': round(macro, 4), 'weighted_f1': round(weighted, 4),
            'accuracy': round(acc, 4)
        })

    # ── Per-strategy x per-category F1 ──
    print(f"\n--- Per-Strategy x Per-Category F1 (all folds) ---")
    cat_f1_rows = []
    for strategy in STRATEGIES:
        smask = meta['reasoning_strategy'].values == strategy
        n = smask.sum()
        if n == 0:
            continue
        ys = y[smask]
        yps = y_pred_all[smask]
        per_cat = f1_score(ys, yps, average=None,
                           labels=list(range(len(MICRO_LABELS))),
                           zero_division=0)
        row = {'strategy': strategy, 'n_sentences': n}
        print(f"  {strategy}:")
        for i, label in enumerate(MICRO_LABELS):
            row[f'{label}_f1'] = round(per_cat[i], 4)
            # Count how many sentences of this category exist for this strategy
            n_cat = (ys == i).sum()
            print(f"    {label:12s}: F1={per_cat[i]:.4f}  (n={n_cat})")
        cat_f1_rows.append(row)

    # ── Analysis 3: Per-strategy mean activation distance from centroids ──
    print(f"\n--- Activation Distance from Full-Corpus Centroids ---")

    # Compute full-corpus centroids (in raw space, before scaling)
    centroids = {}
    for i, label in enumerate(MICRO_LABELS):
        mask = y == i
        if mask.sum() > 0:
            centroids[label] = X[mask].mean(axis=0)

    # Per-strategy centroids and distances
    dist_rows = []
    for strategy in STRATEGIES:
        smask = meta['reasoning_strategy'].values == strategy
        if smask.sum() == 0:
            continue
        print(f"  {strategy}:")
        X_strat = X[smask]
        y_strat = y[smask]
        for i, label in enumerate(MICRO_LABELS):
            cat_mask = y_strat == i
            if cat_mask.sum() < 5:  # need minimum samples
                continue
            strat_centroid = X_strat[cat_mask].mean(axis=0)
            full_centroid = centroids[label]
            # Cosine distance
            cos_sim = np.dot(strat_centroid, full_centroid) / (
                np.linalg.norm(strat_centroid) * np.linalg.norm(full_centroid) + 1e-10
            )
            # Euclidean distance (normalised by dim)
            eucl = np.linalg.norm(strat_centroid - full_centroid)
            n_cat = cat_mask.sum()
            print(f"    {label:12s}: cos_sim={cos_sim:.6f}  eucl_dist={eucl:.2f}  (n={n_cat})")
            dist_rows.append({
                'strategy': strategy, 'category': label,
                'n_sentences': n_cat,
                'cosine_similarity_to_corpus': round(float(cos_sim), 6),
                'euclidean_distance_to_corpus': round(float(eucl), 2),
            })

    # ── Save outputs ──
    strat_df = pd.DataFrame(strat_rows)
    strat_path = OUTPUT_DIR / 'strategy_analysis.csv'
    strat_df.to_csv(strat_path, index=False)
    print(f"\nSaved: {strat_path.name} ({len(strat_df)} rows)")

    cat_f1_df = pd.DataFrame(cat_f1_rows)
    cat_f1_path = OUTPUT_DIR / 'strategy_category_f1.csv'
    cat_f1_df.to_csv(cat_f1_path, index=False)
    print(f"Saved: {cat_f1_path.name} ({len(cat_f1_df)} rows)")

    if dist_rows:
        dist_df = pd.DataFrame(dist_rows)
        dist_path = OUTPUT_DIR / 'strategy_centroid_distances.csv'
        dist_df.to_csv(dist_path, index=False)
        print(f"Saved: {dist_path.name} ({len(dist_df)} rows)")

    print("\nDone.")


if __name__ == '__main__':
    main()

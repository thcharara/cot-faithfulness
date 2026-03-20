#!/usr/bin/env python3
"""Complete the C sweep across all layers.

Runs missing (layer, agg, C) configs and re-runs layer 31 for convergence.
Uses train/test only (no CV) to reduce runtime — CV results at C=1.0 already
exist from the original run.

Usage:
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_complete_probe_sweep.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import time
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ACTIVATIONS_DIR = PROJECT_ROOT / 'outputs' / 'activations'
PROBE_DATA_DIR = PROJECT_ROOT / 'outputs' / 'study2_probe_data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'study2_probes'
MODELS_DIR = OUTPUT_DIR / 'probe_models'

# ── Constants ──
MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST',
                'JUDGE', 'PLAN', 'MONITOR', 'RULE']
LAYERS = [0, 10, 20, 31]
AGGREGATIONS = ['last_token', 'mean_pool']
C_VALUES = [0.01, 0.1, 1.0, 10.0]
TEST_FOLD = 0
TRAIN_FOLDS = [1, 2, 3, 4]

# Per-C max_iter: higher for optimal C values to verify convergence
MAX_ITER_BY_C = {0.01: 2000, 0.1: 1500, 1.0: 1000, 10.0: 1000}


def load_layer_activations(layer, aggregation='last_token'):
    """Load all sentence activations for one layer."""
    meta = pd.read_csv(PROBE_DATA_DIR / 'sentence_metadata.csv')
    X_parts = []
    for trace_key in sorted(meta['trace_key'].unique()):
        safe_key = trace_key.replace('/', '_')
        npz_path = ACTIVATIONS_DIR / f'layer_{layer}' / f'{safe_key}.npz'
        data = np.load(npz_path)
        X_parts.append(data[aggregation].astype(np.float32))
    X = np.vstack(X_parts)
    assert X.shape == (len(meta), 4096)
    return X, meta


def encode_labels(labels):
    label_to_idx = {label: i for i, label in enumerate(MICRO_LABELS)}
    return np.array([label_to_idx[l] for l in labels])


def train_test_only(X, y, meta, C=1.0, max_iter=1000):
    """Train on folds 1-4, evaluate on fold 0. No CV (saves 4x time)."""
    fold_ids = meta['fold_id'].values
    train_mask = np.isin(fold_ids, TRAIN_FOLDS)
    test_mask = fold_ids == TEST_FOLD

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_mask])
    X_test_s = scaler.transform(X[test_mask])

    clf = LogisticRegression(
        C=C, max_iter=max_iter, solver='lbfgs',
        class_weight='balanced', random_state=42,
    )
    clf.fit(X_train_s, y[train_mask])
    y_pred = clf.predict(X_test_s)

    test_macro_f1 = f1_score(y[test_mask], y_pred, average='macro')
    test_weighted_f1 = f1_score(y[test_mask], y_pred, average='weighted')
    test_acc = accuracy_score(y[test_mask], y_pred)
    per_cat_f1 = f1_score(y[test_mask], y_pred, average=None,
                          labels=list(range(len(MICRO_LABELS))))

    cm = confusion_matrix(y[test_mask], y_pred,
                          labels=list(range(len(MICRO_LABELS))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n_iter = clf.n_iter_[0] if hasattr(clf, 'n_iter_') else 0
    converged = n_iter < max_iter
    if not converged:
        print(f"    WARNING: hit max_iter={max_iter} ({n_iter} iterations)")
    else:
        print(f"    Converged in {n_iter} iterations")

    print(f"    TEST:  macro-F1={test_macro_f1:.4f}  "
          f"weighted-F1={test_weighted_f1:.4f}  acc={test_acc:.4f}")

    return {
        'test_macro_f1': test_macro_f1,
        'test_weighted_f1': test_weighted_f1,
        'test_accuracy': test_acc,
        'per_category_f1': per_cat_f1,
        'confusion_matrix': cm_norm,
        'model': clf,
        'scaler': scaler,
        'n_iter': n_iter,
        'converged': converged,
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Build config list
    configs = []
    for layer in [0, 10, 20]:
        for agg in AGGREGATIONS:
            for c in [0.01, 0.1, 10.0]:
                configs.append((layer, agg, c))
    for agg in AGGREGATIONS:
        for c in C_VALUES:
            configs.append((31, agg, c))

    # Group by (layer, agg)
    grouped = defaultdict(list)
    for layer, agg, c in configs:
        if c not in grouped[(layer, agg)]:
            grouped[(layer, agg)].append(c)

    n_configs = sum(len(cs) for cs in grouped.values())
    print("=" * 60)
    print("COMPLETE PROBE SWEEP (train/test only, no CV)")
    print("=" * 60)
    print(f"  Configurations: {n_configs}")
    for (layer, agg), cs in sorted(grouped.items()):
        print(f"    layer {layer:2d} | {agg:10s} | C={cs}")

    summary_rows = []
    t_start = time.time()
    completed = 0

    for (layer, agg), c_list in sorted(grouped.items()):
        print(f"\n{'='*60}")
        print(f"  Loading layer {layer}, aggregation: {agg}")
        print(f"{'='*60}")
        sys.stdout.flush()
        t_load = time.time()
        X, meta = load_layer_activations(layer, agg)
        y = encode_labels(meta['micro_label'].values)
        print(f"  Loaded: X={X.shape}, {time.time()-t_load:.1f}s")
        sys.stdout.flush()

        for C in sorted(c_list):
            # Skip if already completed
            check_path = MODELS_DIR / f'layer{layer}_{agg}_C{C}.pkl'
            if check_path.exists():
                print(f"\n  --- Layer {layer} | {agg} | C={C} | SKIPPING ---")
                sys.stdout.flush()
                continue

            mi = MAX_ITER_BY_C.get(C, 1000)
            print(f"\n  --- Layer {layer} | {agg} | C={C} | max_iter={mi} ---")
            sys.stdout.flush()
            t_fit = time.time()
            results = train_test_only(X, y, meta, C=C, max_iter=mi)
            elapsed = time.time() - t_fit
            completed += 1
            print(f"    Elapsed: {elapsed:.1f}s  ({completed}/{n_configs} done)")
            sys.stdout.flush()

            summary_rows.append({
                'layer': layer,
                'aggregation': agg,
                'C': C,
                'cv_macro_f1_mean': np.nan,  # no CV for these runs
                'cv_macro_f1_std': np.nan,
                'test_macro_f1': round(results['test_macro_f1'], 4),
                'test_weighted_f1': round(results['test_weighted_f1'], 4),
                'test_accuracy': round(results['test_accuracy'], 4),
                'max_iter': mi,
                'n_iter': results['n_iter'],
                'converged': results['converged'],
            })

            # Confusion matrix
            cm_df = pd.DataFrame(
                results['confusion_matrix'],
                index=MICRO_LABELS, columns=MICRO_LABELS
            )
            cm_path = OUTPUT_DIR / f'confusion_matrix_layer{layer}_{agg}_C{C}.csv'
            cm_df.to_csv(cm_path)

            # Save model
            model_path = MODELS_DIR / f'layer{layer}_{agg}_C{C}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump((results['scaler'], results['model']), f)
            print(f"    Saved: {cm_path.name}, {model_path.name}")
            sys.stdout.flush()

    # ── Save results (append to existing) ──
    print(f"\n{'='*60}")
    print("Saving results...")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / 'probe_results_summary.csv'
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        if 'max_iter' not in existing.columns:
            existing['max_iter'] = 1000
            existing['n_iter'] = -1
            existing['converged'] = True
        merged = pd.concat([existing, summary_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=['layer', 'aggregation', 'C'], keep='last'
        )
    else:
        merged = summary_df
    merged.to_csv(summary_path, index=False)
    print(f"  {summary_path.name}: {len(merged)} rows")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE  Total time: {total_time/60:.1f} minutes")

    # ── Convergence comparison: layer 31 ──
    print(f"\n{'='*60}")
    print("CONVERGENCE COMPARISON: Layer 31")
    print(f"{'='*60}")

    old_results = {
        (31, 'last_token', 0.01): 0.7390,
        (31, 'last_token', 0.1): 0.7247,
        (31, 'last_token', 1.0): 0.7005,
        (31, 'last_token', 10.0): 0.6938,
        (31, 'mean_pool', 0.01): 0.7605,
        (31, 'mean_pool', 0.1): 0.7402,
        (31, 'mean_pool', 1.0): 0.7250,
        (31, 'mean_pool', 10.0): 0.7159,
    }

    for row in summary_rows:
        if row['layer'] == 31:
            key = (31, row['aggregation'], row['C'])
            old_f1 = old_results.get(key, None)
            new_f1 = row['test_macro_f1']
            if old_f1:
                diff = new_f1 - old_f1
                conv_str = "converged" if row['converged'] else f"NOT converged ({row['n_iter']} iter)"
                print(f"  {row['aggregation']:10s} C={row['C']:5.2f}: "
                      f"old={old_f1:.4f} -> new={new_f1:.4f} (delta={diff:+.4f}) [{conv_str}]")

    # ── Full results table ──
    print(f"\n{'='*60}")
    print("FULL RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'Layer':>6s} {'Agg':>12s} {'C':>6s} {'Test F1':>8s} {'Conv':>6s} {'Iter':>6s}")
    for _, row in merged.sort_values(['layer', 'aggregation', 'C']).iterrows():
        conv = 'Y' if row.get('converged', True) else 'N'
        n_iter = int(row['n_iter']) if not pd.isna(row.get('n_iter', np.nan)) else -1
        print(f"  {int(row['layer']):4d}  {row['aggregation']:>12s}  {row['C']:6.2f}  "
              f"{row['test_macro_f1']:7.4f}  {conv:>5s}  {n_iter:>5d}")

    # Optimal C at each layer
    print(f"\nOptimal C at each layer (mean_pool):")
    for layer in LAYERS:
        sub = merged[(merged['layer'] == layer) & (merged['aggregation'] == 'mean_pool')]
        if len(sub) > 0:
            best = sub.loc[sub['test_macro_f1'].idxmax()]
            print(f"  Layer {layer:2d}: C={best['C']:5.2f} -> F1={best['test_macro_f1']:.4f}")

    best = merged.loc[merged['test_macro_f1'].idxmax()]
    print(f"\nOverall best: layer {int(best['layer'])}, {best['aggregation']}, "
          f"C={best['C']} -> test macro-F1={best['test_macro_f1']:.4f}")

    print("=" * 60)


if __name__ == '__main__':
    main()

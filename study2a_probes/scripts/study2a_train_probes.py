#!/usr/bin/env python3
"""Study 2a: Train linear probes to classify reasoning categories from activations.

Trains multinomial logistic regression (L2) on per-sentence residual-stream
activations extracted from DeepSeek-R1-Distill-Llama-8B.  Replicates the
methodology of Bogdan et al. Appendix F.

Usage:
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_train_probes.py                    # Full run
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_train_probes.py --layer 31         # Single layer, all C
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_train_probes.py --quick            # Layer 31, last_token, C=1.0
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_train_probes.py --layer 31 --agg last_token --C 1.0
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import time
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ACTIVATIONS_DIR = PROJECT_ROOT / 'outputs' / 'activations'
PROBE_DATA_DIR = PROJECT_ROOT / 'outputs' / 'study2_probe_data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'study2_probes'

# ── Constants ──
MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST',
                'JUDGE', 'PLAN', 'MONITOR', 'RULE']
LAYERS = [0, 10, 20, 31]
AGGREGATIONS = ['last_token', 'mean_pool']
C_VALUES = [0.01, 0.1, 1.0, 10.0]
TEST_FOLD = 0
TRAIN_FOLDS = [1, 2, 3, 4]


def load_layer_activations(layer, aggregation='last_token'):
    """Load all sentence activations for one layer.

    Returns:
        X: np.array [N, 4096] float32
        meta: DataFrame with micro_label, fold_id, trace_key, etc.
    """
    meta = pd.read_csv(PROBE_DATA_DIR / 'sentence_metadata.csv')

    X_parts = []
    for trace_key in sorted(meta['trace_key'].unique()):
        safe_key = trace_key.replace('/', '_')
        npz_path = ACTIVATIONS_DIR / f'layer_{layer}' / f'{safe_key}.npz'
        data = np.load(npz_path)
        X_parts.append(data[aggregation].astype(np.float32))

    X = np.vstack(X_parts)
    assert X.shape == (len(meta), 4096), (
        f"Shape mismatch: X={X.shape}, meta={len(meta)}"
    )
    return X, meta


def encode_labels(labels):
    """Encode micro-label strings to integer indices."""
    label_to_idx = {label: i for i, label in enumerate(MICRO_LABELS)}
    return np.array([label_to_idx[l] for l in labels])


def train_and_evaluate(X, y, meta, C=1.0, max_iter=1000):
    """Run 4-fold CV on folds 1-4, then final eval on fold 0.

    Returns dict with all metrics.
    """
    fold_ids = meta['fold_id'].values

    # ── 4-fold cross-validation on folds 1-4 ──
    cv_results = []
    for val_fold in TRAIN_FOLDS:
        train_mask = np.isin(fold_ids, [f for f in TRAIN_FOLDS if f != val_fold])
        val_mask = fold_ids == val_fold

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        clf = LogisticRegression(
            C=C, max_iter=max_iter, solver='lbfgs',
            class_weight='balanced', random_state=42,
        )
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_val_s)

        macro_f1 = f1_score(y_val, y_pred, average='macro')
        weighted_f1 = f1_score(y_val, y_pred, average='weighted')
        acc = accuracy_score(y_val, y_pred)

        cv_results.append({
            'fold_id': val_fold,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'accuracy': acc,
        })
        print(f"    Fold {val_fold}: macro-F1={macro_f1:.4f}  "
              f"weighted-F1={weighted_f1:.4f}  acc={acc:.4f}")

    cv_macro = [r['macro_f1'] for r in cv_results]
    print(f"    CV mean: macro-F1={np.mean(cv_macro):.4f} +/- {np.std(cv_macro):.4f}")

    # ── Final evaluation: train on folds 1-4, test on fold 0 ──
    train_mask = np.isin(fold_ids, TRAIN_FOLDS)
    test_mask = fold_ids == TEST_FOLD

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C, max_iter=1000, solver='lbfgs',
        class_weight='balanced', random_state=42,
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    test_weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    test_acc = accuracy_score(y_test, y_pred)

    # Per-category F1
    per_cat_f1 = f1_score(y_test, y_pred, average=None,
                          labels=list(range(len(MICRO_LABELS))))

    # Confusion matrix (normalised by true class)
    cm = confusion_matrix(y_test, y_pred,
                          labels=list(range(len(MICRO_LABELS))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    print(f"    TEST:  macro-F1={test_macro_f1:.4f}  "
          f"weighted-F1={test_weighted_f1:.4f}  acc={test_acc:.4f}")

    return {
        'cv_results': cv_results,
        'cv_macro_f1_mean': np.mean(cv_macro),
        'cv_macro_f1_std': np.std(cv_macro),
        'test_macro_f1': test_macro_f1,
        'test_weighted_f1': test_weighted_f1,
        'test_accuracy': test_acc,
        'per_category_f1': per_cat_f1,
        'confusion_matrix': cm_norm,
        'model': clf,
        'scaler': scaler,
    }


def save_csv_append(df_new, path, key_cols):
    """Save CSV, merging with existing file (dedup by key_cols)."""
    if path.exists():
        df_old = pd.read_csv(path)
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=key_cols, keep='last')
    else:
        df_merged = df_new
    df_merged.to_csv(path, index=False)
    return df_merged


def main():
    parser = argparse.ArgumentParser(
        description="Train linear probes on sentence activations"
    )
    parser.add_argument('--layer', type=int, default=None,
                        help='Single layer to train on (default: all)')
    parser.add_argument('--agg', type=str, default=None,
                        choices=AGGREGATIONS,
                        help='Single aggregation (default: both)')
    parser.add_argument('--C', type=float, default=None,
                        help='Single C value (default: see below)')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Max iterations for LogisticRegression (default: 1000)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run: layer 31, last_token, C=1.0 only')
    args = parser.parse_args()

    # ── Build (layer, agg, C) config list ──
    if args.quick:
        configs = [(31, 'last_token', 1.0)]
    elif args.layer is not None or args.agg is not None or args.C is not None:
        # Explicit overrides
        layers = [args.layer] if args.layer is not None else LAYERS
        aggs = [args.agg] if args.agg is not None else AGGREGATIONS
        cs = [args.C] if args.C is not None else C_VALUES
        configs = [(l, a, c) for l in layers for a in aggs for c in cs]
    else:
        # Default full run: C=1.0 for all layers + C sweep at layer 31
        configs = []
        for l in LAYERS:
            for a in AGGREGATIONS:
                configs.append((l, a, 1.0))
        for a in AGGREGATIONS:
            for c in [0.01, 0.1, 10.0]:
                configs.append((31, a, c))

    # Group by (layer, agg) to load activations only once
    grouped = defaultdict(list)
    for layer, agg, c in configs:
        if c not in grouped[(layer, agg)]:
            grouped[(layer, agg)].append(c)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = OUTPUT_DIR / 'probe_models'
    models_dir.mkdir(parents=True, exist_ok=True)

    n_configs = sum(len(cs) for cs in grouped.values())
    n_fits = n_configs * 5  # 4 CV + 1 final per config

    print("=" * 60)
    print("STUDY 2a: Linear Probe Training")
    print("=" * 60)
    print(f"  Configurations: {n_configs}")
    print(f"  Total fits:     {n_fits}")
    for (layer, agg), cs in sorted(grouped.items()):
        print(f"    layer {layer:2d} | {agg:10s} | C={cs}")

    # Accumulators for output CSVs
    summary_rows = []
    per_cat_rows = []
    fold_rows = []
    t_start = time.time()

    for (layer, agg), c_list in sorted(grouped.items()):
        print(f"\n{'='*60}")
        print(f"  Loading layer {layer}, aggregation: {agg}")
        print(f"{'='*60}")
        t_load = time.time()
        X, meta = load_layer_activations(layer, agg)
        y = encode_labels(meta['micro_label'].values)
        print(f"  Loaded: X={X.shape}, {time.time()-t_load:.1f}s")

        for C in sorted(c_list):
            print(f"\n  --- Layer {layer} | {agg} | C={C} ---")
            t_fit = time.time()
            results = train_and_evaluate(X, y, meta, C=C, max_iter=args.max_iter)
            elapsed = time.time() - t_fit
            print(f"    Elapsed: {elapsed:.1f}s")

            # Summary row
            summary_rows.append({
                'layer': layer,
                'aggregation': agg,
                'C': C,
                'cv_macro_f1_mean': round(results['cv_macro_f1_mean'], 4),
                'cv_macro_f1_std': round(results['cv_macro_f1_std'], 4),
                'test_macro_f1': round(results['test_macro_f1'], 4),
                'test_weighted_f1': round(results['test_weighted_f1'], 4),
                'test_accuracy': round(results['test_accuracy'], 4),
            })

            # Per-category F1 row (only for primary C=1.0)
            if C == 1.0:
                cat_row = {'layer': layer, 'aggregation': agg}
                for i, label in enumerate(MICRO_LABELS):
                    cat_row[f'{label}_f1'] = round(
                        results['per_category_f1'][i], 4
                    )
                per_cat_rows.append(cat_row)

            # Confusion matrix (save for all C values with C in filename)
            cm_df = pd.DataFrame(
                results['confusion_matrix'],
                index=MICRO_LABELS, columns=MICRO_LABELS
            )
            cm_path = OUTPUT_DIR / f'confusion_matrix_layer{layer}_{agg}_C{C}.csv'
            cm_df.to_csv(cm_path)
            print(f"    Saved: {cm_path.name}")

            # Also save without C suffix for C=1.0 (backward compat)
            if C == 1.0:
                cm_df.to_csv(OUTPUT_DIR / f'confusion_matrix_layer{layer}_{agg}.csv')

            # Save model + scaler (C=1.0 without suffix for backward compat)
            if C == 1.0:
                model_path = models_dir / f'layer{layer}_{agg}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(
                        (results['scaler'], results['model']), f
                    )
                print(f"    Saved: {model_path.name}")

            # Save model with C in filename for all C values
            model_path_c = models_dir / f'layer{layer}_{agg}_C{C}.pkl'
            with open(model_path_c, 'wb') as f:
                pickle.dump(
                    (results['scaler'], results['model']), f
                )
            print(f"    Saved: {model_path_c.name}")

            # Fold results
            for fr in results['cv_results']:
                fold_rows.append({
                    'layer': layer,
                    'aggregation': agg,
                    'C': C,
                    'fold_id': fr['fold_id'],
                    'macro_f1': round(fr['macro_f1'], 4),
                    'weighted_f1': round(fr['weighted_f1'], 4),
                    'accuracy': round(fr['accuracy'], 4),
                })

    # ── Save CSVs (append-merge with any existing results) ──
    print(f"\n{'='*60}")
    print("Saving results...")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / 'probe_results_summary.csv'
    summary_df = save_csv_append(
        summary_df, summary_path, ['layer', 'aggregation', 'C']
    )
    print(f"  {summary_path.name}: {len(summary_df)} rows")

    per_cat_df = pd.DataFrame(per_cat_rows)
    per_cat_path = OUTPUT_DIR / 'probe_per_category_f1.csv'
    per_cat_df = save_csv_append(
        per_cat_df, per_cat_path, ['layer', 'aggregation']
    )
    print(f"  {per_cat_path.name}: {len(per_cat_df)} rows")

    fold_df = pd.DataFrame(fold_rows)
    fold_path = OUTPUT_DIR / 'fold_results.csv'
    fold_df = save_csv_append(
        fold_df, fold_path, ['layer', 'aggregation', 'C', 'fold_id']
    )
    print(f"  {fold_path.name}: {len(fold_df)} rows")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE  Total time: {total_time/60:.1f} minutes")

    # Print best result
    if len(summary_df) > 0:
        best = summary_df.loc[summary_df['test_macro_f1'].idxmax()]
        print(f"\nBest test macro-F1: {best['test_macro_f1']:.4f}")
        print(f"  Layer {int(best['layer'])}, {best['aggregation']}, C={best['C']}")
    print("=" * 60)


if __name__ == '__main__':
    main()

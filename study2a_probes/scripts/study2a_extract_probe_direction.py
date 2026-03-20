#!/usr/bin/env python3
"""Study 2a: Extract probe weight vectors as steering directions.

Extracts the trained probe's weight matrices, transforms them back to
raw activation space, and computes cosine similarity between category
directions (within-layer) and across layers.

Usage:
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_extract_probe_direction.py
    .venv/Scripts/python.exe study2a_probes/scripts/study2a_extract_probe_direction.py --agg mean_pool --C 0.01
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'study2_probes'
MODELS_DIR = OUTPUT_DIR / 'probe_models'

MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST',
                'JUDGE', 'PLAN', 'MONITOR', 'RULE']
LAYERS = [0, 10, 20, 31]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', type=str, default='mean_pool',
                        help='Aggregation method (default: mean_pool)')
    parser.add_argument('--C', type=float, default=None,
                        help='C value for model selection (default: use generic C=1.0 model)')
    args = parser.parse_args()

    agg = args.agg
    C = args.C

    print("=" * 60)
    print("EXTRACT STEERING DIRECTIONS")
    print("=" * 60)
    print(f"  Aggregation: {agg}")
    print(f"  C: {C if C else 'default (C=1.0 model)'}")
    print(f"  Layers: {LAYERS}")

    # ── Extract directions at each layer ──
    layer_directions = {}  # layer -> [9, 4096] normalised weight matrix

    for layer in LAYERS:
        # Find model file
        if C is not None:
            model_path = MODELS_DIR / f'layer{layer}_{agg}_C{C}.pkl'
        else:
            model_path = MODELS_DIR / f'layer{layer}_{agg}.pkl'

        if not model_path.exists():
            print(f"\n  WARNING: {model_path.name} not found, skipping layer {layer}")
            continue

        print(f"\n--- Layer {layer} ({model_path.name}) ---")
        with open(model_path, 'rb') as f:
            scaler, clf = pickle.load(f)

        # clf.coef_ shape: [9, 4096] (in scaled space)
        # scaler.scale_ shape: [4096]
        coef_scaled = clf.coef_  # [9, 4096]
        scale = scaler.scale_    # [4096]

        # Transform back to raw activation space:
        # In scaled space: z = (x - mean) / scale
        # The hyperplane w_scaled . z = w_scaled . (x - mean) / scale
        # = (w_scaled / scale) . (x - mean)
        # So the direction in raw space is: w_raw = w_scaled / scale
        coef_raw = coef_scaled / scale[np.newaxis, :]  # [9, 4096]

        # L2-normalise each direction vector
        norms = np.linalg.norm(coef_raw, axis=1, keepdims=True)
        coef_normed = coef_raw / norms  # [9, 4096]

        layer_directions[layer] = coef_normed

        # Save as .npz with category names as keys
        npz_path = OUTPUT_DIR / f'steering_directions_layer{layer}.npz'
        save_dict = {label: coef_normed[i] for i, label in enumerate(MICRO_LABELS)}
        np.savez(npz_path, **save_dict)
        print(f"  Saved: {npz_path.name} (shape per category: {coef_normed[0].shape})")

        # ── Within-layer cosine similarity ──
        cos_sim = cosine_similarity(coef_normed)  # [9, 9]
        cos_df = pd.DataFrame(cos_sim, index=MICRO_LABELS, columns=MICRO_LABELS)
        cos_path = OUTPUT_DIR / f'category_cosine_similarity_layer{layer}.csv'
        cos_df.to_csv(cos_path)
        print(f"  Saved: {cos_path.name}")

        # Print summary
        print(f"\n  Cosine similarity matrix (layer {layer}):")
        # Header
        header = "            " + "  ".join(f"{l:>8s}" for l in MICRO_LABELS)
        print(header)
        for i, label in enumerate(MICRO_LABELS):
            vals = "  ".join(f"{cos_sim[i,j]:8.4f}" for j in range(len(MICRO_LABELS)))
            print(f"  {label:10s} {vals}")

        # Find top-5 most similar pairs (off-diagonal)
        pairs = []
        for i in range(len(MICRO_LABELS)):
            for j in range(i+1, len(MICRO_LABELS)):
                pairs.append((MICRO_LABELS[i], MICRO_LABELS[j], cos_sim[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"\n  Most similar pairs (layer {layer}):")
        for a, b, sim in pairs[:5]:
            print(f"    {a:12s} <-> {b:12s}: {sim:.4f}")
        print(f"  Most dissimilar pairs (layer {layer}):")
        for a, b, sim in pairs[-3:]:
            print(f"    {a:12s} <-> {b:12s}: {sim:.4f}")

    # ── Cross-layer cosine similarity ──
    if len(layer_directions) < 2:
        print("\nNot enough layers for cross-layer analysis.")
        return

    print(f"\n{'='*60}")
    print("CROSS-LAYER DIRECTION STABILITY")
    print(f"{'='*60}")

    cross_rows = []
    sorted_layers = sorted(layer_directions.keys())

    for i, label in enumerate(MICRO_LABELS):
        row = {'category': label}
        for li in range(len(sorted_layers)):
            for lj in range(li + 1, len(sorted_layers)):
                l1, l2 = sorted_layers[li], sorted_layers[lj]
                v1 = layer_directions[l1][i]
                v2 = layer_directions[l2][i]
                sim = float(np.dot(v1, v2))  # already normalised
                col_name = f'cos_sim_L{l1}_L{l2}'
                row[col_name] = round(sim, 4)

        # Mean cross-layer similarity
        sim_vals = [v for k, v in row.items() if k.startswith('cos_sim_')]
        row['mean_cross_layer_sim'] = round(np.mean(sim_vals), 4)
        cross_rows.append(row)

        print(f"  {label:12s}: " + "  ".join(
            f"L{sorted_layers[li]}-L{sorted_layers[lj]}={row[f'cos_sim_L{sorted_layers[li]}_L{sorted_layers[lj]}']:.4f}"
            for li in range(len(sorted_layers))
            for lj in range(li+1, len(sorted_layers))
        ) + f"  mean={row['mean_cross_layer_sim']:.4f}")

    cross_df = pd.DataFrame(cross_rows)
    cross_path = OUTPUT_DIR / 'cross_layer_similarity.csv'
    cross_df.to_csv(cross_path, index=False)
    print(f"\nSaved: {cross_path.name} ({len(cross_df)} rows)")

    # Summary statistics
    mean_stab = cross_df['mean_cross_layer_sim'].mean()
    most_stable = cross_df.loc[cross_df['mean_cross_layer_sim'].idxmax()]
    least_stable = cross_df.loc[cross_df['mean_cross_layer_sim'].idxmin()]
    print(f"\nOverall mean cross-layer stability: {mean_stab:.4f}")
    print(f"Most stable:  {most_stable['category']} ({most_stable['mean_cross_layer_sim']:.4f})")
    print(f"Least stable: {least_stable['category']} ({least_stable['mean_cross_layer_sim']:.4f})")

    print("\nDone.")


if __name__ == '__main__':
    main()

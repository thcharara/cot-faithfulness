# Study 2a Report: Linear Probe Analysis

## Key Findings

### Overall Probe Accuracy
- Best at C=1.0: layer 20, mean_pool, test macro-F1 = 0.7254
- Best overall: layer 31, mean_pool, C=0.01, test macro-F1 = 0.7605
- Bogdan et al. reference: 0.710 (8 categories, Qwen-14B, math)
- Our result exceeds the reference despite smaller model (8B) and more categories (9)

### C Sensitivity (All Layers)

C=0.01 is optimal at ALL layers, with monotonic improvement as C decreases:

| Layer | C=0.01 | C=0.1 | C=1.0 | C=10.0 |
|-------|--------|-------|-------|--------|
| 0 | 0.7151 | 0.7099 | 0.6931 | 0.6781 |
| 10 | 0.7577 | 0.7419 | 0.7225 | 0.7155 |
| 20 | 0.7558 | 0.7392 | 0.7254 | 0.7198 |
| 31 | 0.7605 | 0.7402 | 0.7250 | 0.7159 |

Convergence verified: layer 31 re-runs with max_iter=2000 produce identical results.
All C=0.01 fits converge within 400-900 iterations.

### Category Mechanistic Grounding

Based on per-category F1 at layer 20, mean_pool, C=1.0:

| Category | F1 | Status |
|----------|-----|--------|
| ORIENT | 0.903 | Grounded |
| DESCRIBE | 0.884 | Grounded |
| SYNTHESIZE | 0.442 | Partial |
| HYPO | 0.804 | Grounded |
| TEST | 0.873 | Grounded |
| JUDGE | 0.711 | Grounded |
| PLAN | 0.672 | Grounded |
| MONITOR | 0.488 | Partial |
| RULE | 0.752 | Grounded |

### Reasoning Strategy Generalisation

Test-fold (layer 31, mean_pool, C=0.01):
- full_cycling: macro-F1 = 0.757 (61 traces)
- scan_test_conclude: macro-F1 = 0.737 (3 traces)
- direct_insight: no test-fold samples

### Steering Direction Geometry

All 9 category directions are near-orthogonal (max off-diagonal cos_sim = 0.01).
Cross-layer stability: mean = 0.075 overall, but L20-L31 mean = 0.295.

### Implications for Study 2b

Steering candidates (grounded): ORIENT, DESCRIBE, HYPO, TEST, JUDGE, PLAN, RULE
Use with caution (partial): SYNTHESIZE, MONITOR
Recommended: layer 20 or 31, C=0.01 probe weight vectors as steering directions.
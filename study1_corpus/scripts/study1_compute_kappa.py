#!/usr/bin/env python3
"""
study1_compute_kappa.py — Cohen's k between auto-coded and manual traces

Loads 7 validation traces from outputs/traces_clean_coded/ and their
corresponding manually-coded traces from data/manual_coding_final/.
Computes Cohen's k at micro and macro levels.

Usage:
    python study1_corpus/scripts/study1_compute_kappa.py [OPTIONS]

Options:
    --coded-dir     Auto-coded traces dir (default: outputs/traces_clean_coded)
    --manual-dir    Manual coding dir     (default: data/manual_coding_final)
    --output-dir    Results dir           (default: outputs/validation)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("ERROR: Missing dependencies. Run: pip install pandas scikit-learn")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MICRO_LABELS = ["ORIENT", "DESCRIBE", "SYNTHESIZE", "HYPO", "TEST",
                "JUDGE", "PLAN", "MONITOR", "RULE"]

MACRO_MAP = {
    "ORIENT":    "SETUP",
    "DESCRIBE":  "OBSERVE",
    "SYNTHESIZE":"OBSERVE",
    "HYPO":      "INVESTIGATE",
    "TEST":      "INVESTIGATE",
    "JUDGE":     "INVESTIGATE",
    "PLAN":      "REGULATE",
    "MONITOR":   "REGULATE",
    "RULE":      "CONCLUDE",
}

# Mapping from (set, task, trace_num) → manual filename stem
# auto path: outputs/traces_clean_coded/set_a/task2/trace_014.json
# manual:    data/manual_coding_final/setA_task2_trace14_final.json
VALIDATION_PAIRS = [
    ("set_a/task2/trace_014", "setA_task2_trace14_final"),
    ("set_b/task1/trace_010", "setB_task1_trace10_final"),
    ("set_b/task2/trace_004", "setB_task2_trace4_final"),
    ("set_b/task2/trace_006", "setB_task2_trace6_final"),
    ("set_a/task3/trace_020", "setA_task3_trace20_final"),
    ("set_a/task4/trace_019", "setA_task4_trace19_final"),
    ("set_b/task4/trace_015", "setB_task4_trace15_final"),
]

# Validation thresholds
MICRO_THRESHOLD = 0.50
MACRO_THRESHOLD = 0.65
PER_CAT_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    """Load JSON, handling UTF-8 BOM from some manual coding files."""
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not decode {path}")


def extract_labels(sentences: list[dict], source: str) -> dict[int, tuple[str, str]]:
    """
    Extract {sentence_id: (micro_label, macro_label)} from a sentence list.

    Handles both:
      - Auto-coded format:   sentence["coding"]["micro_label"]
      - Manual-coded format: sentence["coding"]["micro_label"]  (same structure)
    """
    result = {}
    for s in sentences:
        sid    = s.get("sentence_id")
        coding = s.get("coding", {})
        micro  = coding.get("micro_label")
        if micro is None:
            continue
        macro = coding.get("macro_label") or MACRO_MAP.get(micro, "UNKNOWN")
        result[sid] = (micro, macro)
    return result


# ---------------------------------------------------------------------------
# k computation
# ---------------------------------------------------------------------------

def safe_kappa(y1: list, y2: list) -> float | None:
    """Compute Cohen's k; return None if fewer than 2 classes present."""
    if len(y1) < 2:
        return None
    try:
        return float(cohen_kappa_score(y1, y2))
    except Exception:
        return None


def compute_per_category_kappa(
    auto_labels: list[str],
    man_labels:  list[str],
    categories:  list[str],
) -> dict[str, float | None]:
    """One-vs-rest k for each category label."""
    kappas = {}
    for cat in categories:
        y_auto = [1 if x == cat else 0 for x in auto_labels]
        y_man  = [1 if x == cat else 0 for x in man_labels]
        kappas[cat] = safe_kappa(y_auto, y_man)
    return kappas


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Cohen's k between auto-coded and manual traces",
    )
    parser.add_argument("--coded-dir",  default=str(PROJECT_ROOT / "outputs" / "traces_clean_coded"),
                        help="Auto-coded traces directory")
    parser.add_argument("--manual-dir", default=str(PROJECT_ROOT / "data" / "manual_coding_final"),
                        help="Manual coding directory")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "validation"),
                        help="Results output directory")
    args = parser.parse_args()

    coded_dir  = Path(args.coded_dir)
    manual_dir = Path(args.manual_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load trace pairs ──────────────────────────────────────────────────────
    rows = []
    missing_auto   = []
    missing_manual = []

    for auto_rel, manual_stem in VALIDATION_PAIRS:
        auto_path   = coded_dir  / (auto_rel + ".json")
        manual_path = manual_dir / (manual_stem + ".json")

        if not auto_path.exists():
            missing_auto.append(auto_rel)
            continue

        if not manual_path.exists():
            missing_manual.append(manual_stem)
            continue

        auto_data   = load_json(auto_path)
        manual_data = load_json(manual_path)

        auto_labels   = extract_labels(auto_data.get("sentences", []),   "auto")
        manual_labels = extract_labels(manual_data.get("sentences", []), "manual")

        # Match by sentence_id
        shared_sids = sorted(set(auto_labels.keys()) & set(manual_labels.keys()))

        if not shared_sids:
            print(f"  WARNING [{auto_rel}]: No matching sentence IDs — skipping.")
            continue

        auto_micro  = [auto_labels[sid][0]   for sid in shared_sids]
        auto_macro  = [auto_labels[sid][1]   for sid in shared_sids]
        man_micro   = [manual_labels[sid][0] for sid in shared_sids]
        man_macro   = [manual_labels[sid][1] for sid in shared_sids]

        # Overall k
        kappa_micro = safe_kappa(auto_micro, man_micro)
        kappa_macro = safe_kappa(auto_macro, man_macro)

        # Raw agreement
        n_agreed = sum(a == m for a, m in zip(auto_micro, man_micro))
        agreement_pct = n_agreed / len(shared_sids) * 100

        # Per-category k
        per_cat = compute_per_category_kappa(auto_micro, man_micro, MICRO_LABELS)

        n_total_auto   = len(auto_labels)
        n_total_manual = len(manual_labels)
        n_matched      = len(shared_sids)

        if n_total_auto != n_total_manual:
            print(f"  NOTE [{auto_rel}]: auto={n_total_auto} sents, "
                  f"manual={n_total_manual} sents, matched={n_matched}")

        rows.append({
            "trace":          auto_rel,
            "n_sentences":    n_matched,
            "kappa_micro":    kappa_micro,
            "kappa_macro":    kappa_macro,
            "agreement_pct":  round(agreement_pct, 1),
            **{f"kappa_{cat}": per_cat[cat] for cat in MICRO_LABELS},
            # Keep raw lists for debugging
            "_auto_micro":    auto_micro,
            "_man_micro":     man_micro,
            "_auto_macro":    auto_macro,
            "_man_macro":     man_macro,
        })

    # ── Report missing files ───────────────────────────────────────────────────
    if missing_auto:
        print(f"\nAuto-coded traces not yet available ({len(missing_auto)}):")
        for t in missing_auto:
            print(f"  {t}")
    if missing_manual:
        print(f"\nManual coding files not yet available ({len(missing_manual)}):")
        for t in missing_manual:
            print(f"  {t}")

    if not rows:
        print("\nNo trace pairs available for k computation.")
        return

    # ── Build combined labels for overall k ───────────────────────────────────
    all_auto_micro = []
    all_man_micro  = []
    all_auto_macro = []
    all_man_macro  = []
    for r in rows:
        all_auto_micro.extend(r.pop("_auto_micro"))
        all_man_micro.extend(r.pop("_man_micro"))
        all_auto_macro.extend(r.pop("_auto_macro"))
        all_man_macro.extend(r.pop("_man_macro"))

    overall_kappa_micro = safe_kappa(all_auto_micro, all_man_micro)
    overall_kappa_macro = safe_kappa(all_auto_macro, all_man_macro)
    overall_agreement   = (sum(a == m for a, m in zip(all_auto_micro, all_man_micro))
                           / len(all_auto_micro) * 100)
    overall_per_cat     = compute_per_category_kappa(
        all_auto_micro, all_man_micro, MICRO_LABELS
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_cols = (
        ["trace", "n_sentences", "kappa_micro", "kappa_macro", "agreement_pct"]
        + [f"kappa_{cat}" for cat in MICRO_LABELS]
    )
    csv_rows = []
    for r in rows:
        csv_rows.append({col: r.get(col) for col in csv_cols})

    df = pd.DataFrame(csv_rows, columns=csv_cols)
    csv_path = output_dir / "kappa_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults saved to: {csv_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION k RESULTS")
    print("=" * 70)
    print(f"{'Trace':<35} {'N':>5} {'k_micro':>8} {'k_macro':>8} {'Agr%':>6}")
    print("-" * 70)
    for r in rows:
        km  = f"{r['kappa_micro']:.3f}"  if r['kappa_micro']  is not None else "  N/A"
        kma = f"{r['kappa_macro']:.3f}"  if r['kappa_macro']  is not None else "  N/A"
        print(f"{r['trace']:<35} {r['n_sentences']:>5} {km:>8} {kma:>8} {r['agreement_pct']:>5.1f}%")
    print("-" * 70)
    means_micro = [r["kappa_micro"] for r in rows if r["kappa_micro"] is not None]
    means_macro = [r["kappa_macro"] for r in rows if r["kappa_macro"] is not None]
    mean_km  = sum(means_micro) / len(means_micro) if means_micro else None
    mean_kma = sum(means_macro) / len(means_macro) if means_macro else None
    mean_km_str  = f"{mean_km:.3f}"  if mean_km  is not None else "  N/A"
    mean_kma_str = f"{mean_kma:.3f}" if mean_kma is not None else "  N/A"
    print(f"{'Mean (per-trace)':<35} {'':>5} {mean_km_str:>8} {mean_kma_str:>8}")
    overall_km_str  = f"{overall_kappa_micro:.3f}" if overall_kappa_micro is not None else "  N/A"
    overall_kma_str = f"{overall_kappa_macro:.3f}" if overall_kappa_macro is not None else "  N/A"
    print(f"{'Overall (pooled)':<35} {len(all_auto_micro):>5} "
          f"{overall_km_str:>8} {overall_kma_str:>8} "
          f"{overall_agreement:>5.1f}%")

    # ── Per-category table ────────────────────────────────────────────────────
    print("\nPer-category one-vs-rest k (pooled across all traces):")
    print(f"  {'Category':<12} {'k':>6}  {'Pass?':<6}")
    print("  " + "-" * 28)
    for cat in MICRO_LABELS:
        k = overall_per_cat.get(cat)
        k_str  = f"{k:.3f}" if k is not None else "  N/A"
        passed = (k is not None and k >= PER_CAT_THRESHOLD)
        flag   = "PASS" if passed else ("FAIL" if k is not None else "N/A ")
        print(f"  {cat:<12} {k_str:>6}  {flag}")

    # ── Validation gate ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)

    def gate(val, threshold, label):
        if val is None:
            return f"  {label}: N/A (insufficient data)"
        passed = val >= threshold
        flag   = "PASS" if passed else "FAIL"
        return f"  {label}: {val:.3f}  (threshold: >= {threshold:.2f}) -- {flag}"

    print(gate(overall_kappa_micro, MICRO_THRESHOLD, f"Overall micro k"))
    print()
    print("  Per-category results:")
    for cat in MICRO_LABELS:
        k = overall_per_cat.get(cat)
        if k is not None:
            passed = k >= PER_CAT_THRESHOLD
            flag   = "PASS" if passed else "FAIL"
            print(f"    {cat:<12}: {k:.3f}  -- {flag}")
        else:
            print(f"    {cat:<12}: N/A")
    print()
    print(gate(overall_kappa_macro, MACRO_THRESHOLD, f"Overall macro k"))
    print()

    # Final verdict
    micro_pass = overall_kappa_micro is not None and overall_kappa_micro >= MICRO_THRESHOLD
    macro_pass = overall_kappa_macro is not None and overall_kappa_macro >= MACRO_THRESHOLD
    if micro_pass and macro_pass:
        print("  OVERALL VERDICT: PASS -- proceed to full coding run.")
    else:
        print("  OVERALL VERDICT: FAIL -- review prompt / taxonomy before full run.")
    print("=" * 70)


if __name__ == "__main__":
    main()

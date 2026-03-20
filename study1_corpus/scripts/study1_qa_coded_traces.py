#!/usr/bin/env python3
"""
QA for auto-coded traces (taxonomy integrity + plausibility diagnostics).

Primary target:
  - task1 across set_a + set_b (80 expected traces)

Outputs:
  - outputs/validation/task1_coded_qa_report.md
  - outputs/validation/task1_coded_trace_metrics.csv
  - outputs/validation/task1_coded_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Taxonomy constants aligned with study1_batch_code_traces.py
MICRO_LABELS = [
    "ORIENT",
    "DESCRIBE",
    "SYNTHESIZE",
    "HYPO",
    "TEST",
    "JUDGE",
    "PLAN",
    "MONITOR",
    "RULE",
]

MACRO_MAP = {
    "ORIENT": "SETUP",
    "DESCRIBE": "OBSERVE",
    "SYNTHESIZE": "OBSERVE",
    "HYPO": "INVESTIGATE",
    "TEST": "INVESTIGATE",
    "JUDGE": "INVESTIGATE",
    "PLAN": "REGULATE",
    "MONITOR": "REGULATE",
    "RULE": "CONCLUDE",
}

VALID_TEST_CONTEXT = {"post_hypothesis", "pre_hypothesis", "post_rule"}
VALID_SPECIFICITY = {"within_panel", "across_panels"}
VALID_JUDGEMENT = {"accept", "reject", "uncertain"}

# Manual baseline source aligned with study1_compute_kappa.py
VALIDATION_MANUAL_STEMS = [
    "setA_task2_trace14_final",
    "setB_task1_trace10_final",
    "setB_task2_trace4_final",
    "setB_task2_trace6_final",
    "setA_task3_trace20_final",
    "setA_task4_trace19_final",
    "setB_task4_trace15_final",
]

EXPECTED_TRACES_PER_SET_TASK = 40
HARD_COVERAGE_THRESHOLD = 95.0
ABS_MANUAL_DELTA_WARN_PP = 12.0
LONG_RUN_WARN = 10
LONG_RUN_ELEVATED = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality assessment for auto-coded traces"
    )
    parser.add_argument(
        "--coded-dir",
        default=str(PROJECT_ROOT / "outputs" / "traces_clean_coded"),
        help="Root directory for coded traces",
    )
    parser.add_argument(
        "--manual-dir",
        default=str(PROJECT_ROOT / "data" / "manual_coding_final"),
        help="Directory with manually coded baseline traces",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=1,
        help="Task id to analyze (default: 1)",
    )
    parser.add_argument(
        "--sets",
        default="A,B",
        help="Comma-separated set letters, e.g. A,B or A",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "validation"),
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible qualitative samples",
    )
    return parser.parse_args()


def parse_sets(raw: str) -> list[str]:
    sets = [x.strip().upper() for x in raw.split(",") if x.strip()]
    if not sets:
        raise ValueError("No sets provided")
    bad = [s for s in sets if s not in {"A", "B"}]
    if bad:
        raise ValueError(f"Invalid set(s): {bad}. Allowed: A,B")
    return sorted(set(sets))


def load_json(path: Path) -> dict:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not decode JSON: {path}")


def short_text(text: str, max_len: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def robust_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_fence": None,
            "upper_fence": None,
        }
    vals = sorted(values)
    median = statistics.median(vals)
    if len(vals) >= 4:
        q1, _, q3 = statistics.quantiles(vals, n=4, method="inclusive")
    elif len(vals) == 3:
        q1, q3 = vals[0], vals[2]
    else:
        q1 = q3 = vals[0]
    iqr = q3 - q1
    return {
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_fence": float(q1 - 1.5 * iqr),
        "upper_fence": float(q3 + 1.5 * iqr),
    }


def extract_manual_baseline(manual_dir: Path) -> tuple[dict[str, float], dict]:
    per_trace = []
    missing = []
    corrupt = []

    for stem in VALIDATION_MANUAL_STEMS:
        path = manual_dir / f"{stem}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            data = load_json(path)
        except Exception as exc:  # broad: decode issues should not stop QA
            corrupt.append(f"{path}: {exc}")
            continue

        labels = []
        for s in data.get("sentences", []):
            micro = (s.get("coding") or {}).get("micro_label")
            if micro in MICRO_LABELS:
                labels.append(micro)
        if not labels:
            continue
        counts = Counter(labels)
        total = len(labels)
        per_trace.append({lab: pct(counts.get(lab, 0), total) for lab in MICRO_LABELS})

    means = {}
    for lab in MICRO_LABELS:
        values = [row[lab] for row in per_trace]
        means[lab] = float(statistics.mean(values)) if values else 0.0

    meta = {
        "expected_files": len(VALIDATION_MANUAL_STEMS),
        "loaded_files": len(per_trace),
        "missing_files": missing,
        "corrupt_files": corrupt,
    }
    return means, meta


def label_entropy_norm(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return ent / math.log2(len(MICRO_LABELS))


def long_runs(labels: list[str], sentence_ids: list[int], min_len: int = LONG_RUN_WARN) -> list[dict]:
    runs = []
    if not labels:
        return runs
    curr = labels[0]
    start = 0
    size = 1
    for i in range(1, len(labels)):
        if labels[i] == curr:
            size += 1
            continue
        if curr and size >= min_len:
            runs.append(
                {
                    "label": curr,
                    "start_sentence_id": sentence_ids[start],
                    "end_sentence_id": sentence_ids[i - 1],
                    "length": size,
                }
            )
        curr = labels[i]
        start = i
        size = 1
    if curr and size >= min_len:
        runs.append(
            {
                "label": curr,
                "start_sentence_id": sentence_ids[start],
                "end_sentence_id": sentence_ids[-1],
                "length": size,
            }
        )
    return runs


def hypo_test_transition_concentration(labels: list[str]) -> float:
    if len(labels) < 2:
        return 0.0
    total = 0
    hits = 0
    for a, b in zip(labels, labels[1:]):
        total += 1
        if (a == "HYPO" and b == "TEST") or (a == "TEST" and b == "HYPO"):
            hits += 1
    return hits / total if total else 0.0


def repetition_ratio(labels: list[str], n: int = 4) -> float:
    if len(labels) < n:
        return 0.0
    windows = [tuple(labels[i : i + n]) for i in range(len(labels) - n + 1)]
    total = len(windows)
    unique = len(set(windows))
    return (total - unique) / total if total else 0.0


def analyze_trace_file(path: Path, trace_rel: str) -> dict:
    result = {
        "trace": trace_rel,
        "path": str(path),
        "exists": path.exists(),
        "status": "FAIL",
        "hard_issues": [],
        "warnings": [],
        "n_sentences": 0,
        "n_missing_coding": 0,
        "n_invalid_micro": 0,
        "n_macro_mismatch": 0,
        "n_valid_labels": 0,
        "n_test": 0,
        "n_judge": 0,
        "coverage_test_context_pct": 0.0,
        "coverage_specificity_pct": 0.0,
        "coverage_judgement_pct": 0.0,
        "has_answer_text": False,
        "rule_present": False,
        "orient_in_first5": False,
        "long_runs": [],
        "longest_run_len": 0,
        "longest_run_label": "",
        "hypo_test_transition_conc": 0.0,
        "repetition_ratio": 0.0,
        "entropy_norm": 0.0,
        "unique_label_count": 0,
        "label_counts": {lab: 0 for lab in MICRO_LABELS},
        "label_pcts": {lab: 0.0 for lab in MICRO_LABELS},
        "sentence_rows": [],
        "distribution_outlier_labels": [],
        "manual_drift_labels": [],
    }

    if not path.exists():
        result["hard_issues"].append("missing_trace_file")
        return result

    try:
        data = load_json(path)
    except Exception as exc:  # decode failures are hard issues
        result["hard_issues"].append(f"corrupt_json: {exc}")
        return result

    sentences = data.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        result["hard_issues"].append("missing_or_empty_sentences")
        return result

    result["n_sentences"] = len(sentences)
    result["has_answer_text"] = bool((data.get("answer_text") or "").strip())

    labels_order = []
    sids_order = []
    test_ctx_ok = 0
    test_spec_ok = 0
    judge_ok = 0

    for idx, s in enumerate(sentences):
        sid = s.get("sentence_id", idx)
        coding = s.get("coding")
        if not isinstance(coding, dict):
            result["n_missing_coding"] += 1
            result["hard_issues"].append(f"missing_coding_block_s{sid}")
            result["sentence_rows"].append(
                {"sentence_id": sid, "label": "UNKNOWN", "text": short_text(s.get("text", ""))}
            )
            continue

        micro = coding.get("micro_label")
        macro = coding.get("macro_label")
        if micro not in MICRO_LABELS:
            result["n_invalid_micro"] += 1
            result["hard_issues"].append(f"invalid_micro_label_s{sid}:{micro!r}")
            result["sentence_rows"].append(
                {"sentence_id": sid, "label": "UNKNOWN", "text": short_text(s.get("text", ""))}
            )
            continue

        expected_macro = MACRO_MAP[micro]
        if macro != expected_macro:
            result["n_macro_mismatch"] += 1
            result["hard_issues"].append(
                f"macro_mismatch_s{sid}:{macro!r}!={expected_macro!r}"
            )

        labels_order.append(micro)
        sids_order.append(sid)
        result["label_counts"][micro] += 1
        result["sentence_rows"].append(
            {"sentence_id": sid, "label": micro, "text": short_text(s.get("text", ""))}
        )

        if micro == "TEST":
            result["n_test"] += 1
            if coding.get("test_context") in VALID_TEST_CONTEXT:
                test_ctx_ok += 1
            if coding.get("specificity") in VALID_SPECIFICITY:
                test_spec_ok += 1
        elif micro == "JUDGE":
            result["n_judge"] += 1
            if coding.get("judgement") in VALID_JUDGEMENT:
                judge_ok += 1

    result["n_valid_labels"] = sum(result["label_counts"].values())
    if result["n_valid_labels"] > 0:
        for lab in MICRO_LABELS:
            result["label_pcts"][lab] = pct(result["label_counts"][lab], result["n_valid_labels"])

    result["coverage_test_context_pct"] = pct(test_ctx_ok, result["n_test"])
    result["coverage_specificity_pct"] = pct(test_spec_ok, result["n_test"])
    result["coverage_judgement_pct"] = pct(judge_ok, result["n_judge"])

    if result["n_test"] > 0 and result["coverage_test_context_pct"] < HARD_COVERAGE_THRESHOLD:
        result["hard_issues"].append("coverage_test_context_below_95")
    if result["n_test"] > 0 and result["coverage_specificity_pct"] < HARD_COVERAGE_THRESHOLD:
        result["hard_issues"].append("coverage_specificity_below_95")
    if result["n_judge"] > 0 and result["coverage_judgement_pct"] < HARD_COVERAGE_THRESHOLD:
        result["hard_issues"].append("coverage_judgement_below_95")

    first5 = labels_order[:5]
    result["orient_in_first5"] = "ORIENT" in first5
    result["rule_present"] = "RULE" in labels_order

    runs = long_runs(labels_order, sids_order, min_len=LONG_RUN_WARN)
    result["long_runs"] = runs
    if runs:
        longest = max(runs, key=lambda x: x["length"])
        result["longest_run_len"] = longest["length"]
        result["longest_run_label"] = longest["label"]

    result["hypo_test_transition_conc"] = hypo_test_transition_concentration(labels_order)
    result["repetition_ratio"] = repetition_ratio(labels_order, n=4)
    result["entropy_norm"] = label_entropy_norm(Counter(labels_order))
    result["unique_label_count"] = len([k for k, v in result["label_counts"].items() if v > 0])
    return result


def evaluate_plausibility(
    traces: list[dict],
    internal_baseline: dict,
    manual_baseline: dict[str, float],
) -> None:
    # Shared thresholds from corpus distributions
    loop_stats = internal_baseline["metric_stats"]
    conc_thr = max(0.35, (loop_stats["hypo_test_transition_conc"]["upper_fence"] or 0.35))
    rep_thr = max(0.25, (loop_stats["repetition_ratio"]["upper_fence"] or 0.25))

    for t in traces:
        if t["n_valid_labels"] <= 0:
            continue

        # Label-distribution outliers against internal robust baseline
        outlier_labels = []
        for lab in MICRO_LABELS:
            val = t["label_pcts"][lab]
            st = internal_baseline["label_stats"][lab]
            lower = st["lower_fence"]
            upper = st["upper_fence"]
            if lower is None or upper is None:
                continue
            if st["iqr"] == 0:
                if abs(val - (st["median"] or 0.0)) >= 8.0:
                    outlier_labels.append(lab)
            elif val < lower or val > upper:
                outlier_labels.append(lab)
        t["distribution_outlier_labels"] = outlier_labels
        if outlier_labels:
            t["warnings"].append(f"internal_distribution_outlier:{','.join(outlier_labels)}")

        # Absolute drift vs manual baseline means
        drift_labels = []
        for lab in MICRO_LABELS:
            delta = abs(t["label_pcts"][lab] - manual_baseline.get(lab, 0.0))
            if delta >= ABS_MANUAL_DELTA_WARN_PP:
                drift_labels.append(lab)
        t["manual_drift_labels"] = drift_labels
        if drift_labels:
            t["warnings"].append(f"manual_baseline_drift:{','.join(drift_labels)}")

        # Structural checks
        if not t["orient_in_first5"]:
            t["warnings"].append("orient_not_in_first5")
        if t["has_answer_text"] and not t["rule_present"]:
            t["warnings"].append("answer_present_but_rule_missing")
        if t["longest_run_len"] >= LONG_RUN_ELEVATED:
            t["warnings"].append(f"long_identical_run_elevated:{t['longest_run_len']}")
        elif t["longest_run_len"] >= LONG_RUN_WARN:
            t["warnings"].append(f"long_identical_run:{t['longest_run_len']}")

        # Loop/repetition signals
        if t["hypo_test_transition_conc"] > conc_thr:
            t["warnings"].append(
                f"high_hypo_test_transition_concentration:{t['hypo_test_transition_conc']:.3f}"
            )
        if t["repetition_ratio"] > rep_thr:
            t["warnings"].append(f"high_repetition_ratio:{t['repetition_ratio']:.3f}")

        # Diversity collapse sanity
        if t["entropy_norm"] < 0.45:
            t["warnings"].append(f"low_label_entropy:{t['entropy_norm']:.3f}")
        if t["unique_label_count"] < 4:
            t["warnings"].append(f"low_unique_label_count:{t['unique_label_count']}")


def finalize_status(traces: list[dict]) -> None:
    for t in traces:
        if t["hard_issues"]:
            t["status"] = "FAIL"
        elif t["warnings"]:
            t["status"] = "WARN"
        else:
            t["status"] = "PASS"


def choose_samples(rows: list[dict], rng: random.Random, n: int) -> list[dict]:
    if not rows:
        return []
    if len(rows) <= n:
        return rows
    idx = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def write_metrics_csv(path: Path, traces: list[dict]) -> None:
    fields = [
        "trace",
        "status",
        "n_sentences",
        "n_valid_labels",
        "hard_issue_count",
        "warning_count",
        "n_missing_coding",
        "n_invalid_micro",
        "n_macro_mismatch",
        "n_test",
        "n_judge",
        "coverage_test_context_pct",
        "coverage_specificity_pct",
        "coverage_judgement_pct",
        "orient_in_first5",
        "has_answer_text",
        "rule_present",
        "longest_run_len",
        "longest_run_label",
        "hypo_test_transition_conc",
        "repetition_ratio",
        "entropy_norm",
        "unique_label_count",
    ] + [f"count_{lab}" for lab in MICRO_LABELS] + [f"pct_{lab}" for lab in MICRO_LABELS]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in sorted(traces, key=lambda x: x["trace"]):
            row = {
                "trace": t["trace"],
                "status": t["status"],
                "n_sentences": t["n_sentences"],
                "n_valid_labels": t["n_valid_labels"],
                "hard_issue_count": len(t["hard_issues"]),
                "warning_count": len(t["warnings"]),
                "n_missing_coding": t["n_missing_coding"],
                "n_invalid_micro": t["n_invalid_micro"],
                "n_macro_mismatch": t["n_macro_mismatch"],
                "n_test": t["n_test"],
                "n_judge": t["n_judge"],
                "coverage_test_context_pct": f"{t['coverage_test_context_pct']:.2f}",
                "coverage_specificity_pct": f"{t['coverage_specificity_pct']:.2f}",
                "coverage_judgement_pct": f"{t['coverage_judgement_pct']:.2f}",
                "orient_in_first5": t["orient_in_first5"],
                "has_answer_text": t["has_answer_text"],
                "rule_present": t["rule_present"],
                "longest_run_len": t["longest_run_len"],
                "longest_run_label": t["longest_run_label"],
                "hypo_test_transition_conc": f"{t['hypo_test_transition_conc']:.6f}",
                "repetition_ratio": f"{t['repetition_ratio']:.6f}",
                "entropy_norm": f"{t['entropy_norm']:.6f}",
                "unique_label_count": t["unique_label_count"],
            }
            for lab in MICRO_LABELS:
                row[f"count_{lab}"] = t["label_counts"][lab]
                row[f"pct_{lab}"] = f"{t['label_pcts'][lab]:.4f}"
            writer.writerow(row)


def write_summary_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(
    path: Path,
    traces: list[dict],
    internal_baseline: dict,
    manual_baseline: dict[str, float],
    manual_meta: dict,
    args: argparse.Namespace,
) -> None:
    rng = random.Random(args.seed)

    status_counts = Counter(t["status"] for t in traces)
    hard_reason_counts = Counter()
    warn_reason_counts = Counter()
    for t in traces:
        for h in t["hard_issues"]:
            hard_reason_counts[h.split(":")[0]] += 1
        for w in t["warnings"]:
            warn_reason_counts[w.split(":")[0]] += 1

    flagged = [t for t in traces if t["status"] in {"FAIL", "WARN"}]
    flagged_sorted = sorted(
        flagged,
        key=lambda x: (len(x["hard_issues"]), len(x["warnings"]), x["trace"]),
        reverse=True,
    )
    clean = [t for t in traces if t["status"] == "PASS"]

    lines = []
    add = lines.append
    add("# Coded Trace QA Report")
    add("")
    add(f"- Generated: {datetime.now().isoformat()}")
    add(f"- Task: {args.task}")
    add(f"- Sets: {','.join(parse_sets(args.sets))}")
    add(f"- Coded dir: `{args.coded_dir}`")
    add(f"- Manual dir: `{args.manual_dir}`")
    add("")
    add("## Corpus Summary")
    add("")
    add(f"- Expected traces: {len(traces)}")
    add(f"- PASS: {status_counts.get('PASS', 0)}")
    add(f"- WARN: {status_counts.get('WARN', 0)}")
    add(f"- FAIL: {status_counts.get('FAIL', 0)}")
    add("")
    recommendation = "HOLD" if status_counts.get("FAIL", 0) > 0 else "PROCEED"
    add(f"**Recommendation:** {recommendation}")
    add("")
    add("## Hard Integrity Findings (FAIL Drivers)")
    add("")
    if hard_reason_counts:
        for reason, c in hard_reason_counts.most_common():
            add(f"- {reason}: {c}")
    else:
        add("- None")
    add("")
    add("## Plausibility Warning Findings")
    add("")
    if warn_reason_counts:
        for reason, c in warn_reason_counts.most_common():
            add(f"- {reason}: {c}")
    else:
        add("- None")
    add("")
    add("## Baselines")
    add("")
    add("### Internal Robust Baseline (Task Scope)")
    add("")
    add("| Label | Median % | IQR | Lower Fence | Upper Fence |")
    add("|---|---:|---:|---:|---:|")
    for lab in MICRO_LABELS:
        st = internal_baseline["label_stats"][lab]
        add(
            f"| {lab} | {st['median']:.2f} | {st['iqr']:.2f} | "
            f"{st['lower_fence']:.2f} | {st['upper_fence']:.2f} |"
        )
    add("")
    add("### Manual Baseline Means (7 Validation Traces)")
    add("")
    add(f"- Loaded files: {manual_meta['loaded_files']}/{manual_meta['expected_files']}")
    if manual_meta["missing_files"]:
        add(f"- Missing manual files: {len(manual_meta['missing_files'])}")
    if manual_meta["corrupt_files"]:
        add(f"- Corrupt manual files: {len(manual_meta['corrupt_files'])}")
    add("")
    add("| Label | Mean % |")
    add("|---|---:|")
    for lab in MICRO_LABELS:
        add(f"| {lab} | {manual_baseline.get(lab, 0.0):.2f} |")
    add("")
    add("## Most-Flagged Traces")
    add("")
    add("| Trace | Status | Hard Issues | Warnings | Longest Run |")
    add("|---|---|---:|---:|---:|")
    for t in flagged_sorted[:20]:
        add(
            f"| {t['trace']} | {t['status']} | {len(t['hard_issues'])} | "
            f"{len(t['warnings'])} | {t['longest_run_len']} |"
        )
    if not flagged_sorted:
        add("| (none) | PASS | 0 | 0 | 0 |")
    add("")
    add("## Qualitative Samples")
    add("")
    add("### Flagged Trace Samples")
    add("")
    for t in flagged_sorted[:5]:
        add(f"#### {t['trace']} ({t['status']})")
        add("")
        if t["hard_issues"]:
            add(f"- Hard issues: {', '.join(t['hard_issues'][:5])}")
        if t["warnings"]:
            add(f"- Warnings: {', '.join(t['warnings'][:5])}")
        samples = choose_samples(t["sentence_rows"], rng, 5)
        if samples:
            for s in samples:
                add(f"- s{s['sentence_id']} [{s['label']}]: {s['text']}")
        else:
            add("- No sentence samples available.")
        add("")

    add("### Clean Trace Contrast Samples")
    add("")
    clean_choices = choose_samples(clean, rng, min(3, len(clean)))
    for t in clean_choices:
        add(f"#### {t['trace']} (PASS)")
        add("")
        for s in choose_samples(t["sentence_rows"], rng, 3):
            add(f"- s{s['sentence_id']} [{s['label']}]: {s['text']}")
        add("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    coded_dir = Path(args.coded_dir)
    manual_dir = Path(args.manual_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        sets = parse_sets(args.sets)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    expected = []
    for set_letter in sets:
        for trace_id in range(1, EXPECTED_TRACES_PER_SET_TASK + 1):
            rel = f"set_{set_letter.lower()}/task{args.task}/trace_{trace_id:03d}.json"
            expected.append(rel)

    traces = []
    for rel in expected:
        path = coded_dir / rel
        traces.append(analyze_trace_file(path, rel))

    # Build internal robust baselines from all traces with valid labels
    label_values = defaultdict(list)
    metric_values = defaultdict(list)
    for t in traces:
        if t["n_valid_labels"] <= 0:
            continue
        for lab in MICRO_LABELS:
            label_values[lab].append(t["label_pcts"][lab])
        metric_values["hypo_test_transition_conc"].append(t["hypo_test_transition_conc"])
        metric_values["repetition_ratio"].append(t["repetition_ratio"])
        metric_values["entropy_norm"].append(t["entropy_norm"])
        metric_values["unique_label_count"].append(float(t["unique_label_count"]))

    internal_baseline = {
        "label_stats": {lab: robust_stats(label_values[lab]) for lab in MICRO_LABELS},
        "metric_stats": {
            "hypo_test_transition_conc": robust_stats(metric_values["hypo_test_transition_conc"]),
            "repetition_ratio": robust_stats(metric_values["repetition_ratio"]),
            "entropy_norm": robust_stats(metric_values["entropy_norm"]),
            "unique_label_count": robust_stats(metric_values["unique_label_count"]),
        },
    }

    manual_baseline, manual_meta = extract_manual_baseline(manual_dir)
    evaluate_plausibility(traces, internal_baseline, manual_baseline)
    finalize_status(traces)

    status_counts = Counter(t["status"] for t in traces)
    recommendation = "HOLD" if status_counts.get("FAIL", 0) > 0 else "PROCEED"

    prefix = f"task{args.task}_coded"
    report_path = out_dir / f"{prefix}_qa_report.md"
    csv_path = out_dir / f"{prefix}_trace_metrics.csv"
    summary_path = out_dir / f"{prefix}_summary.json"

    write_report(
        path=report_path,
        traces=traces,
        internal_baseline=internal_baseline,
        manual_baseline=manual_baseline,
        manual_meta=manual_meta,
        args=args,
    )
    write_metrics_csv(csv_path, traces)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "task": args.task,
        "sets": sets,
        "seed": args.seed,
        "coded_dir": str(coded_dir),
        "manual_dir": str(manual_dir),
        "n_expected_traces": len(expected),
        "status_counts": dict(status_counts),
        "recommendation": recommendation,
        "n_fail_traces": status_counts.get("FAIL", 0),
        "n_warn_traces": status_counts.get("WARN", 0),
        "n_pass_traces": status_counts.get("PASS", 0),
        "manual_baseline_meta": manual_meta,
        "output_files": {
            "report": str(report_path),
            "metrics_csv": str(csv_path),
            "summary_json": str(summary_path),
        },
    }
    write_summary_json(summary_path, summary)

    print("=" * 60)
    print("Coded Trace QA")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Sets: {','.join(sets)}")
    print(f"Expected traces: {len(expected)}")
    print(f"PASS={status_counts.get('PASS', 0)} "
          f"WARN={status_counts.get('WARN', 0)} FAIL={status_counts.get('FAIL', 0)}")
    print(f"Recommendation: {recommendation}")
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    sys.exit(0 if recommendation == "PROCEED" else 1)


if __name__ == "__main__":
    main()


"""Smoke test: load and validate the public 20-trace coded sample.

Confirms that the shipped sample corpus parses, has the expected schema, and
uses only the 9 micro / 5 macro labels from the Phase 3 taxonomy. CPU-only;
no model, no API key.

Usage:
    python tests/smoke_sample_trace.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "coded_corpus_sample"

MICRO_LABELS = {
    "ORIENT", "DESCRIBE", "SYNTHESIZE", "HYPO", "TEST",
    "JUDGE", "PLAN", "MONITOR", "RULE",
}
MACRO_LABELS = {"SETUP", "OBSERVE", "INVESTIGATE", "REGULATE", "CONCLUDE"}

REQUIRED_TOP_KEYS = {"task_id", "trace_id", "set", "thinking_text", "sentences"}
REQUIRED_SENT_KEYS = {"sentence_id", "text", "coding"}
REQUIRED_CODING_KEYS = {"macro_label", "micro_label"}


def load_trace(path: Path):
    # Manual coding files are utf-8-sig; auto-coded files are utf-8.
    # `utf-8-sig` accepts both.
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def validate(path: Path):
    trace = load_trace(path)
    errors = []

    missing_top = REQUIRED_TOP_KEYS - trace.keys()
    if missing_top:
        errors.append(f"missing top-level keys: {sorted(missing_top)}")

    sentences = trace.get("sentences", [])
    if not sentences:
        errors.append("no sentences")

    for i, s in enumerate(sentences):
        missing_sent = REQUIRED_SENT_KEYS - s.keys()
        if missing_sent:
            errors.append(f"sentence {i}: missing {sorted(missing_sent)}")
            continue
        coding = s["coding"]
        missing_coding = REQUIRED_CODING_KEYS - coding.keys()
        if missing_coding:
            errors.append(f"sentence {i}: coding missing {sorted(missing_coding)}")
            continue
        if coding["micro_label"] not in MICRO_LABELS:
            errors.append(f"sentence {i}: unknown micro_label {coding['micro_label']!r}")
        if coding["macro_label"] not in MACRO_LABELS:
            errors.append(f"sentence {i}: unknown macro_label {coding['macro_label']!r}")

    return sentences, errors


def main():
    if not SAMPLE_DIR.exists():
        print(f"FAIL  sample directory not found: {SAMPLE_DIR}")
        sys.exit(1)

    trace_paths = sorted(SAMPLE_DIR.rglob("trace_*.json"))
    if not trace_paths:
        print(f"FAIL  no trace_*.json under {SAMPLE_DIR}")
        sys.exit(1)

    print(f"Validating {len(trace_paths)} trace(s) under {SAMPLE_DIR.relative_to(REPO_ROOT)}")
    print()

    total_sentences = 0
    label_counts: dict[str, int] = {}
    failed = 0

    for path in trace_paths:
        rel = path.relative_to(SAMPLE_DIR)
        sentences, errors = validate(path)
        if errors:
            failed += 1
            print(f"  FAIL  {rel}")
            for err in errors[:3]:
                print(f"        - {err}")
            if len(errors) > 3:
                print(f"        ... and {len(errors) - 3} more")
            continue

        total_sentences += len(sentences)
        for s in sentences:
            lbl = s["coding"]["micro_label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f"  OK    {rel}  ({len(sentences)} sentences)")

    print()
    print(f"Total traces:     {len(trace_paths)}")
    print(f"Total sentences:  {total_sentences}")
    print(f"Failures:         {failed}")
    print()
    print("Micro-label distribution:")
    for lbl in sorted(MICRO_LABELS):
        n = label_counts.get(lbl, 0)
        pct = 100 * n / total_sentences if total_sentences else 0
        print(f"  {lbl:11s} {n:5d}  ({pct:5.1f}%)")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""Merge sub-agent accuracy review results back into the review CSVs.

Reads per-task scoring files, parses [ID] score=X rationale="..." lines, and
writes manual_correct into both review CSVs (preserving heuristic_correct).
Also writes llm_review_log.json and prints a comparison summary.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
S2B_CSV = ROOT / "outputs" / "study2b_steering" / "accuracy_manual_review.csv"
S1_CSV = ROOT / "outputs" / "study2b_steering" / "study1_accuracy_manual_review.csv"
BATCH_DIR = ROOT / "outputs" / "study2b_steering" / "review_batches"
LOG_JSON = ROOT / "outputs" / "study2b_steering" / "llm_review_log.json"

RESULT_FILES = [
    "task1a_results.txt", "task1b_results.txt", "task2_results.txt",
    "task3_results.txt", "task4a_results.txt", "task4b_results.txt",
]

LINE_RE = re.compile(
    r"^\s*\[\s*(\d+)\s*\]\s*score\s*=\s*([0-9.]+)\s*rationale\s*=\s*[\"'](.+?)[\"']\s*$"
)


def parse_results():
    by_id = {}
    for fname in RESULT_FILES:
        p = BATCH_DIR / fname
        for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                # try fallback w/o quotes
                m2 = re.match(r"^\s*\[\s*(\d+)\s*\]\s*score\s*=\s*([0-9.]+)\s*rationale\s*=\s*(.+?)\s*$", line)
                if not m2:
                    raise ValueError(f"Unparseable line in {fname}:{ln}: {line!r}")
                tid, score, rat = m2.group(1), m2.group(2), m2.group(3).strip(' "\'')
            else:
                tid, score, rat = m.group(1), m.group(2), m.group(3)
            tid = int(tid)
            score_f = float(score)
            if tid in by_id:
                raise ValueError(f"Duplicate ID {tid} (in {fname})")
            by_id[tid] = {"score": score_f, "rationale": rat, "source_file": fname}
    return by_id


def main():
    work = pd.read_csv(BATCH_DIR / "working_index.csv")
    scores = parse_results()
    print(f"Parsed {len(scores)} scored lines (expected {len(work)})")
    missing = sorted(set(work["id"]) - set(scores.keys()))
    extra = sorted(set(scores.keys()) - set(work["id"]))
    if missing:
        print(f"WARNING: {len(missing)} working IDs missing scores: {missing[:20]}")
    if extra:
        print(f"WARNING: {len(extra)} extra IDs in results not in working: {extra[:20]}")

    work["llm_score"] = [scores.get(int(i), {}).get("score") for i in work["id"]]
    work["llm_rationale"] = [scores.get(int(i), {}).get("rationale", "") for i in work["id"]]

    # Build lookup keyed by (source, task_id, trace_id, condition_or_set)
    s2b = pd.read_csv(S2B_CSV)
    s1 = pd.read_csv(S1_CSV)

    # s2b key: (task_id, trace_id, condition)
    s2b_lookup = {}
    s1_lookup = {}
    for _, r in work.iterrows():
        key = (int(r["task_id"]), int(r["trace_id"]), str(r["condition_or_set"]))
        if r["source"] == "s2b":
            s2b_lookup[key] = r["llm_score"]
        else:
            s1_lookup[key] = r["llm_score"]

    s2b["manual_correct"] = [
        s2b_lookup.get((int(t), int(tr), str(c)))
        for t, tr, c in zip(s2b["task_id"], s2b["trace_id"], s2b["condition"])
    ]
    s1["manual_correct"] = [
        s1_lookup.get((int(t), int(tr), str(s)))
        for t, tr, s in zip(s1["task_id"], s1["trace_id"], s1["set"])
    ]

    n_s2b_filled = s2b["manual_correct"].notna().sum()
    n_s1_filled = s1["manual_correct"].notna().sum()
    print(f"S2b: {n_s2b_filled}/{len(s2b)} manual_correct filled")
    print(f"S1:  {n_s1_filled}/{len(s1)} manual_correct filled")

    s2b.to_csv(S2B_CSV, index=False)
    s1.to_csv(S1_CSV, index=False)

    # llm_review_log.json
    log = []
    for _, r in work.iterrows():
        log.append({
            "id": int(r["id"]),
            "source": r["source"],
            "task_id": int(r["task_id"]),
            "condition_or_set": str(r["condition_or_set"]),
            "trace_id": int(r["trace_id"]),
            "heuristic_correct": float(r["heuristic_correct"]) if pd.notna(r["heuristic_correct"]) else None,
            "llm_score": float(r["llm_score"]) if pd.notna(r["llm_score"]) else None,
            "rationale": r["llm_rationale"],
        })
    LOG_JSON.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote {LOG_JSON}")

    # Comparison summary
    print("\n=== HEURISTIC vs LLM REVIEW ===\n")

    print("Study 1 by population:")
    for pop, sub in s1.groupby("population"):
        n = len(sub)
        h = int(sub["heuristic_correct"].sum())
        l = sub["manual_correct"].sum()
        disagree = sub[sub["heuristic_correct"] != sub["manual_correct"]]
        print(f"  {pop}: heuristic={h}/{n} ({100*h/n:.2f}%)  llm={l:g}/{n} ({100*l/n:.2f}%)  disagreements={len(disagree)}")

    print("\nStudy 2b by condition:")
    for cond, sub in s2b.groupby("condition"):
        n = len(sub)
        h = int(sub["heuristic_correct"].sum())
        l = sub["manual_correct"].sum()
        disagree = sub[sub["heuristic_correct"] != sub["manual_correct"]]
        print(f"  {cond:20s} n={n:3d}  heur={h:>2d}/{n}  llm={l:>4g}/{n}  disagree={len(disagree)}")

    # Disagreements detail
    print("\n=== DISAGREEMENTS DETAIL ===")
    for label, df, key_cols in [
        ("S1", s1, ["set", "task_id", "trace_id"]),
        ("S2b", s2b, ["condition", "task_id", "trace_id"]),
    ]:
        dis = df[df["heuristic_correct"] != df["manual_correct"]]
        if len(dis):
            print(f"\n{label} disagreements ({len(dis)}):")
            for _, r in dis.iterrows():
                key_str = " ".join(f"{k}={r[k]}" for k in key_cols)
                print(f"  {key_str}  heur={r['heuristic_correct']:g} llm={r['manual_correct']:g}")


if __name__ == "__main__":
    main()

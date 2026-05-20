"""Prepare evaluation batches for LLM-based accuracy review.

Combines Study 1 + Study 2b review CSVs, cleans text, builds panel summaries
from Task JSONs, and writes per-task batch + panel files for sub-agent
consumption.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
S2B_CSV = ROOT / "outputs" / "study2b_steering" / "accuracy_manual_review.csv"
S1_CSV = ROOT / "outputs" / "study2b_steering" / "study1_accuracy_manual_review.csv"
STIM_DIR = ROOT / "data" / "stimuli"
OUT_DIR = ROOT / "outputs" / "study2b_steering" / "review_batches"

EOS_TOKEN = "<｜end▁of▁sentence｜>"


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace(EOS_TOKEN, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def combine_text(stated: str, last: str) -> str:
    stated = clean_text(stated)
    last = clean_text(last)
    if not last or last == stated or last in stated:
        return stated
    return f"{stated}  ||  LAST_RULE_SENTENCE: {last}"


def build_panel_summary(task_json: dict) -> str:
    """Format the Task JSON's panels as a concise readable text block."""
    panels = task_json["panels"]
    out = []
    for letter in sorted(panels.keys()):
        p = panels[letter]
        starred = "STARRED" if p.get("yellow_star") else "unstarred"
        n = p.get("number_of_cones", len(p.get("cones", [])))
        out.append(f"Panel {letter} [{starred}] - {n} cones:")
        for cone in p.get("cones", []):
            # cone format: [index, size, colour, orientation, contacts, position]
            idx, size, colour, orient = cone[0], cone[1], cone[2], cone[3]
            out.append(f"  {idx}. {size} {colour}, {orient}")
        out.append("")
    return "\n".join(out).rstrip()


def starred_panels_str(task_json: dict) -> str:
    panels = task_json["panels"]
    starred = [k for k in sorted(panels.keys()) if panels[k].get("yellow_star")]
    unstarred = [k for k in sorted(panels.keys()) if not panels[k].get("yellow_star")]
    return f"Starred: {', '.join(starred)} | Unstarred: {', '.join(unstarred)}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_s2b = pd.read_csv(S2B_CSV)
    df_s1 = pd.read_csv(S1_CSV)

    rows = []
    for _, r in df_s2b.iterrows():
        rows.append({
            "source": "s2b",
            "task_id": int(r["task_id"]),
            "trace_id": int(r["trace_id"]),
            "condition_or_set": r["condition"],
            "stated_rule_answer": clean_text(r["stated_rule_answer"]),
            "last_rule_sentence": clean_text(r["last_rule_sentence"]),
            "heuristic_correct": r["heuristic_correct"],
        })
    for _, r in df_s1.iterrows():
        rows.append({
            "source": "s1",
            "task_id": int(r["task_id"]),
            "trace_id": int(r["trace_id"]),
            "condition_or_set": r["set"],
            "stated_rule_answer": clean_text(r["stated_rule_answer"]),
            "last_rule_sentence": clean_text(r["last_rule_sentence"]),
            "heuristic_correct": r["heuristic_correct"],
        })

    work = pd.DataFrame(rows)
    work.insert(0, "id", range(1, len(work) + 1))
    work["combined_text"] = [
        combine_text(s, l)
        for s, l in zip(work["stated_rule_answer"], work["last_rule_sentence"])
    ]

    # Persist working frame for downstream merge
    work_csv = OUT_DIR / "working_index.csv"
    work.to_csv(work_csv, index=False)

    # Per-task batches + panels
    for task_id in [1, 2, 3, 4]:
        task_json = json.loads((STIM_DIR / f"Task{task_id}.json").read_text(encoding="utf-8"))
        panel_text = build_panel_summary(task_json)
        starred = starred_panels_str(task_json)

        panel_path = OUT_DIR / f"task{task_id}_panels.txt"
        panel_path.write_text(
            f"TASK {task_id} - {starred}\n\n{panel_text}\n",
            encoding="utf-8",
        )

        sub = work[work["task_id"] == task_id].sort_values("id")
        lines = []
        for _, r in sub.iterrows():
            heur = "1" if float(r["heuristic_correct"]) == 1.0 else "0"
            text = r["combined_text"] if r["combined_text"] else "(empty)"
            lines.append(
                f"[{r['id']}] source={r['source']}, cond={r['condition_or_set']}, "
                f"trace={r['trace_id']}, heuristic={heur}: {text}"
            )

        batch_path = OUT_DIR / f"task{task_id}_batch.txt"
        batch_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Task {task_id}: {len(sub)} traces -> {batch_path.name}")

    print(f"\nWorking index: {work_csv}")
    print(f"Total: {len(work)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pre-Processing Script: Fix Segmentation Issues in Blank Traces

Reads from outputs/traces_segmented/ and writes cleaned blank traces to
outputs/traces_cleaned/, mirroring the directory structure.

This is the pre-coding equivalent of postprocess_coded_traces.py.  It applies
the same structural fixes to traces that carry NO coding yet (sentences have
only sentence_id, text, start_char, end_char, token_count).

Three categories of fixes:
  1. Fragment merging  — short enumeration fragments (e.g. "1.", "B.") are
                         merged into the adjacent non-fragment sentence.
  2. Truncated finals  — incomplete final sentences inferred from text
                         heuristics (token-limited traces only).
  3. Manual merges     — two specific traces with known complex segmentation
                         issues receive hand-specified corrections.

Usage:
    python study1_corpus/scripts/study1_preprocess_traces.py [OPTIONS]

Options:
    --input-dir     Segmented traces dir  (default: outputs/traces_segmented)
    --output-dir    Cleaned traces dir    (default: outputs/traces_cleaned)
    --dry-run       Report actions without writing files
    --force         Overwrite existing output dir without confirmation
    --verbose       Print per-trace details
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fragment: token_count <= 2 AND stripped text matches this pattern.
# Panel labels are A-F; digits are list item numbers.
FRAGMENT_RE = re.compile(r"^([A-F0-9]\.?|\.)$")

# Manual merge specifications (same problematic traces as postprocess script).
# Applied BEFORE programmatic fragment detection.
#
# For blank traces we only need:
#   sources    — old sentence_ids to absorb (removed)
#   target     — old sentence_id that survives
#   rationale  — short string for the report
#   split_off  — if set, this exact suffix is carved out of the merged text
#                and inserted as a NEW sentence immediately after the target.
MANUAL_MERGES: dict[str, list[dict]] = {
    "set_a/task4/trace_008": [
        {
            "sources":   [206, 207, 208, 209],
            "target":    210,
            "rationale": "shattered_enumeration",
            "split_off": None,
        },
        {
            "sources":   [230, 231, 232, 233, 234],
            "target":    235,
            "rationale": "shattered_enumeration",
            "split_off": None,
        },
    ],
    "set_b/task3/trace_012": [
        {
            "sources":   [108, 109],
            "target":    107,
            "rationale": "shattered_data_table",
            "split_off": "Not seeing a pattern.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_fragment(s: dict) -> bool:
    return s["token_count"] <= 2 and bool(FRAGMENT_RE.match(s["text"].strip()))


def classify_fragment(text: str) -> str:
    t = text.strip()
    if re.match(r"^\d+\.$", t):
        return "digit_period"
    if re.match(r"^[A-F]\.$", t):
        return "letter_period"
    if re.match(r"^\d+$", t):
        return "digit_alone"
    if re.match(r"^[A-F]$", t):
        return "letter_alone"
    if t == ".":
        return "period_alone"
    return "other"


def has_final_answer(trace_data: dict) -> bool:
    """True if the trace completed generation and has a non-empty answer."""
    for field in ("final_answer", "answer_text"):
        val = trace_data.get(field)
        if val:
            return True
    return False


def is_truncated_final(s: dict) -> bool:
    """
    Heuristic for blank (uncoded) traces.
    A sentence is considered a truncated fragment if it is very short
    OR does not end with terminal punctuation.
    """
    text = s["text"].rstrip()
    return (
        s["token_count"] <= 5
        or not text.endswith((".", "!", "?", '"', ")"))
    )


def truncation_reason(s: dict) -> str:
    text = s["text"].rstrip()
    reasons = []
    if s["token_count"] <= 5:
        reasons.append("le5_tokens")
    if not text.endswith((".", "!", "?", '"', ")")):
        reasons.append("no_terminal_punct")
    return "+".join(reasons) if reasons else "unclear"


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_trace(
    trace_key: str,
    trace_data: dict,
    verbose: bool = False,
) -> tuple[list, dict, list]:
    """
    Clean one blank trace.

    Returns
    -------
    sentences       — cleaned sentence list (renumbered)
    preprocessing   — metadata dict for the "preprocessing" top-level field
    actions         — list of action dicts for the CSV report
    """
    sentences: list[dict] = [dict(s) for s in trace_data.get("sentences", [])]
    n_original = len(sentences)

    merge_redirect: dict[int, int] = {}   # {removed_old_id: survivor_old_id}
    removed_ids:    set[int]       = set()
    actions: list[dict] = []

    # ── Step 1: Manual merges ────────────────────────────────────────────────
    n_manual = 0
    for spec in MANUAL_MERGES.get(trace_key, []):
        sources   = spec["sources"]
        target_id = spec["target"]
        rationale = spec["rationale"]
        split_off = spec.get("split_off")

        id_to_idx = {s["sentence_id"]: i for i, s in enumerate(sentences)}

        missing = [x for x in sources + [target_id] if x not in id_to_idx]
        if missing:
            print(f"  WARNING [{trace_key}]: manual merge IDs not found: {missing}")
            continue

        all_ids   = sorted(sources + [target_id], key=lambda x: id_to_idx[x])
        all_sents = [sentences[id_to_idx[x]] for x in all_ids]

        merged_text   = " ".join(s["text"] for s in all_sents)
        merged_tokens = sum(s["token_count"] for s in all_sents)
        merged_start  = all_sents[0]["start_char"]
        merged_end    = all_sents[-1]["end_char"]

        # Handle optional split_off (set_b/task3/trace_012)
        new_split_sentence: dict | None = None
        if split_off and merged_text.endswith(split_off):
            data_text    = merged_text[: -len(split_off)].rstrip()
            split_tokens = len(split_off.split())
            data_tokens  = merged_tokens - split_tokens
            new_split_sentence = {
                "sentence_id":  -1,          # placeholder; renumbered later
                "text":         split_off,
                "start_char":   merged_end - len(split_off),
                "end_char":     merged_end,
                "token_count":  split_tokens,
            }
        else:
            data_text   = merged_text
            data_tokens = merged_tokens

        # Update the target sentence
        target_s = sentences[id_to_idx[target_id]]
        target_s["text"]        = data_text
        target_s["token_count"] = data_tokens
        target_s["start_char"]  = merged_start
        target_s["end_char"]    = merged_end

        for sid in sources:
            merge_redirect[sid] = target_id
            src_text = sentences[id_to_idx[sid]]["text"][:60].replace("\n", " ")
            actions.append({
                "trace_key":    trace_key,
                "action":       "manual_merge",
                "old_sid":      sid,
                "new_sid":      "TBD",
                "text_preview": src_text,
                "reason":       rationale,
            })
            n_manual += 1

        source_set = set(sources)
        sentences  = [s for s in sentences if s["sentence_id"] not in source_set]

        if new_split_sentence is not None:
            id_to_idx2 = {s["sentence_id"]: i for i, s in enumerate(sentences)}
            tgt_pos = id_to_idx2[target_id]
            sentences.insert(tgt_pos + 1, new_split_sentence)
            actions.append({
                "trace_key":    trace_key,
                "action":       "manual_split",
                "old_sid":      -1,
                "new_sid":      "TBD",
                "text_preview": split_off[:60],
                "reason":       "split_after_merge",
            })

    # ── Step 2: Programmatic fragment detection ──────────────────────────────
    n_fragments = 0
    frag_pattern_counts: dict[str, int] = {}

    i = 0
    while i < len(sentences):
        if not is_fragment(sentences[i]):
            i += 1
            continue

        j = i + 1
        while j < len(sentences) and is_fragment(sentences[j]):
            j += 1

        if j < len(sentences):
            # Normal case: merge fragments i..j-1 into sentences[j]
            target = sentences[j]
            frag_texts  = [sentences[k]["text"]        for k in range(i, j)]
            frag_tokens = sum(sentences[k]["token_count"] for k in range(i, j))
            target["text"]        = " ".join(frag_texts) + " " + target["text"]
            target["token_count"] += frag_tokens
            target["start_char"]   = sentences[i]["start_char"]

            for k in range(i, j):
                frag   = sentences[k]
                old_id = frag["sentence_id"]
                pat    = classify_fragment(frag["text"])
                frag_pattern_counts[pat] = frag_pattern_counts.get(pat, 0) + 1
                if old_id >= 0:
                    merge_redirect[old_id] = target["sentence_id"]
                actions.append({
                    "trace_key":    trace_key,
                    "action":       "fragment_merge",
                    "old_sid":      old_id,
                    "new_sid":      target["sentence_id"],
                    "text_preview": frag["text"][:40].replace("\n", " "),
                    "reason":       pat,
                })

            n_fragments += (j - i)
            del sentences[i:j]

        else:
            # Fragment run at end of trace — merge into previous
            if i > 0:
                prev = sentences[i - 1]
                for k in range(i, len(sentences)):
                    frag   = sentences[k]
                    old_id = frag["sentence_id"]
                    pat    = classify_fragment(frag["text"])
                    frag_pattern_counts[pat] = frag_pattern_counts.get(pat, 0) + 1
                    prev["text"]        = prev["text"] + " " + frag["text"]
                    prev["end_char"]    = frag["end_char"]
                    prev["token_count"] += frag["token_count"]
                    if old_id >= 0:
                        merge_redirect[old_id] = prev["sentence_id"]
                    actions.append({
                        "trace_key":    trace_key,
                        "action":       "fragment_merge",
                        "old_sid":      old_id,
                        "new_sid":      prev["sentence_id"],
                        "text_preview": frag["text"][:40].replace("\n", " "),
                        "reason":       pat + "_to_prev",
                    })
                n_fragments += len(sentences) - i
                del sentences[i:]
            break

    # ── Step 3: Truncated final sentence removal ─────────────────────────────
    n_truncated = 0
    trace_has_answer = has_final_answer(trace_data)
    max_removals = 10

    while sentences and not trace_has_answer and max_removals > 0:
        last = sentences[-1]
        if is_truncated_final(last):
            old_id = last["sentence_id"]
            if old_id >= 0:
                removed_ids.add(old_id)
            actions.append({
                "trace_key":    trace_key,
                "action":       "truncation_removal",
                "old_sid":      old_id,
                "new_sid":      -1,
                "text_preview": last["text"][:60].replace("\n", " "),
                "reason":       truncation_reason(last),
            })
            n_truncated += 1
            sentences.pop()
            max_removals -= 1
        else:
            break

    # ── Step 4: Renumber sequentially ────────────────────────────────────────
    old_to_new: dict[int, int] = {}
    for new_idx, s in enumerate(sentences):
        old_id = s["sentence_id"]
        if isinstance(old_id, int) and old_id >= 0:
            old_to_new[old_id] = new_idx
        s["sentence_id"] = new_idx

    # ── Build id_mapping ──────────────────────────────────────────────────────
    id_mapping: dict[str, int] = {}
    for old_id, new_id in old_to_new.items():
        id_mapping[str(old_id)] = new_id
    for old_id, target_old_id in merge_redirect.items():
        if target_old_id in old_to_new:
            id_mapping[str(old_id)] = old_to_new[target_old_id]
        else:
            id_mapping[str(old_id)] = -1
    for old_id in removed_ids:
        id_mapping[str(old_id)] = -1

    # Resolve "TBD" new_sids in the action log
    for act in actions:
        if act["new_sid"] == "TBD":
            act["new_sid"] = id_mapping.get(str(act["old_sid"]), -1)
        elif isinstance(act["new_sid"], int) and act["new_sid"] >= 0:
            act["new_sid"] = old_to_new.get(act["new_sid"], act["new_sid"])

    preprocessing = {
        "version":                  "1.0",
        "timestamp":                now_iso(),
        "original_sentence_count":  n_original,
        "cleaned_sentence_count":   len(sentences),
        "fragments_merged":         n_fragments,
        "truncated_finals_removed": n_truncated,
        "manual_merges_applied":    n_manual,
        "id_mapping":               id_mapping,
    }

    return sentences, preprocessing, actions


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    all_results: list[dict],
    all_actions: list[dict],
    output_dir:  Path,
    timestamp:   str,
) -> None:
    """Write cleanup_report.md and cleanup_details.csv to output_dir."""
    total_traces    = len(all_results)
    n_modified      = sum(1 for r in all_results if r["modified"])
    n_unchanged     = total_traces - n_modified
    total_fragments = sum(r["fragments"] for r in all_results)
    total_truncated = sum(r["truncated"] for r in all_results)
    total_manual    = sum(r["manual"]    for r in all_results)
    manual_traces   = sum(1 for r in all_results if r["manual"] > 0)
    orig_total      = sum(r["original"] for r in all_results)
    clean_total     = sum(r["cleaned"]  for r in all_results)
    removed_total   = orig_total - clean_total

    frag_actions = [a for a in all_actions if a["action"] == "fragment_merge"]
    pat_examples: dict[str, list[str]] = {}
    for a in frag_actions:
        pat = a["reason"].replace("_to_prev", "")
        ex  = f"{a['trace_key']} sid {a['old_sid']}: \"{a['text_preview']}\""
        pat_examples.setdefault(pat, []).append(ex)

    trunc_actions   = [a for a in all_actions if a["action"] == "truncation_removal"]
    modified_sorted = sorted(
        [r for r in all_results if r["modified"]], key=lambda r: r["trace_key"]
    )

    pct = (removed_total / orig_total * 100) if orig_total else 0.0
    lines = [
        "# Pre-Processing Cleanup Report",
        "",
        f"Generated: {timestamp}",
        "",
        "## Summary",
        f"- Traces processed: {total_traces}",
        f"- Traces modified: {n_modified}",
        f"- Traces unchanged: {n_unchanged}",
        f"- Total fragments merged: {total_fragments}",
        f"- Total truncated finals removed: {total_truncated}",
        f"- Manual merges applied: {total_manual} (in {manual_traces} traces)",
        f"- Original corpus: {orig_total:,} sentences",
        (
            f"- Cleaned corpus: {clean_total:,} sentences"
            f" ({removed_total} removed/merged, {pct:.1f}%)"
        ),
        "",
        "## Fragment Merges by Pattern",
        "| Pattern | Count | Example |",
        "|---------|-------|---------|",
    ]
    for pat, exs in sorted(pat_examples.items()):
        lines.append(f"| {pat} | {len(exs)} | {exs[0]} |")
    if not pat_examples:
        lines.append("| (none) | 0 | — |")

    lines += [
        "",
        "## Truncated Final Sentence Removals",
        "| Trace | Original Last SID | Text (truncated) | Reason |",
        "|-------|-------------------|------------------|--------|",
    ]
    for a in trunc_actions:
        lines.append(
            f"| {a['trace_key']} | {a['old_sid']} | {a['text_preview'][:50]} | {a['reason']} |"
        )
    if not trunc_actions:
        lines.append("| (none) | — | — | — |")

    lines += [
        "",
        "## Manual Merges",
        "| Trace | Sentences Merged | Into | Rationale |",
        "|-------|-----------------|------|-----------|",
    ]
    for spec_key, specs in MANUAL_MERGES.items():
        for spec in specs:
            srcs   = "+".join(str(s) for s in spec["sources"])
            suffix = (
                f" + split \"{spec['split_off']}\""
                if spec.get("split_off") else ""
            )
            lines.append(
                f"| {spec_key} | {srcs} → {spec['target']} "
                f"| {spec['target']}{suffix} | {spec['rationale']} |"
            )

    lines += [
        "",
        "## Per-Trace Details (modified traces only)",
        "| Trace | Original | Cleaned | Fragments | Truncated | Manual |",
        "|-------|----------|---------|-----------|-----------|--------|",
    ]
    for r in modified_sorted:
        lines.append(
            f"| {r['trace_key']} | {r['original']} | {r['cleaned']}"
            f" | {r['fragments']} | {r['truncated']} | {r['manual']} |"
        )

    (output_dir / "cleanup_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    csv_path = output_dir / "cleanup_details.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["trace_key", "action", "old_sid", "new_sid",
                        "text_preview", "reason"],
        )
        writer.writeheader()
        for a in all_actions:
            writer.writerow({
                "trace_key":    a["trace_key"],
                "action":       a["action"],
                "old_sid":      a["old_sid"],
                "new_sid":      a["new_sid"],
                "text_preview": a["text_preview"],
                "reason":       a["reason"],
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-process blank segmented traces: merge fragments, "
            "remove truncated finals, apply manual fixes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",  default=str(PROJECT_ROOT / "outputs" / "traces_segmented"),
        help="Segmented traces directory (default: outputs/traces_segmented)",
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs" / "traces_cleaned"),
        help="Output directory (default: outputs/traces_cleaned)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report actions without writing files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output directory without confirmation",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-trace details during processing",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    if output_dir.exists() and not args.force and not args.dry_run:
        resp = input(
            f"Output directory '{output_dir}' already exists. Overwrite? [y/N] "
        )
        if resp.strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    trace_paths = sorted(input_dir.glob("*/task*/trace_*.json"))
    if not trace_paths:
        print(f"ERROR: No trace files found in {input_dir}")
        sys.exit(1)

    print("=== Pre-Processing: Segmentation Cleanup ===")
    print(f"Input:  {input_dir}  ({len(trace_paths)} traces)")
    print(f"Output: {output_dir}")
    if args.dry_run:
        print("[DRY RUN] No files will be written.")
    print("---")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    timestamp    = now_iso()
    all_results: list[dict] = []
    all_actions: list[dict] = []

    for in_path in trace_paths:
        rel_parts = in_path.relative_to(input_dir).parts
        trace_key = "/".join(rel_parts[:-1]) + "/" + in_path.stem

        try:
            trace_data = json.loads(in_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  ERROR reading {trace_key}: {exc}")
            continue

        sentences, preprocessing, actions = process_trace(
            trace_key, trace_data, verbose=args.verbose
        )

        modified = (
            preprocessing["fragments_merged"] > 0
            or preprocessing["truncated_finals_removed"] > 0
            or preprocessing["manual_merges_applied"] > 0
        )

        all_results.append({
            "trace_key": trace_key,
            "original":  preprocessing["original_sentence_count"],
            "cleaned":   preprocessing["cleaned_sentence_count"],
            "fragments": preprocessing["fragments_merged"],
            "truncated": preprocessing["truncated_finals_removed"],
            "manual":    preprocessing["manual_merges_applied"],
            "modified":  modified,
        })
        all_actions.extend(actions)

        if args.verbose or modified:
            delta = preprocessing["original_sentence_count"] - preprocessing["cleaned_sentence_count"]
            print(
                f"  {trace_key}: "
                f"{preprocessing['original_sentence_count']} -> "
                f"{preprocessing['cleaned_sentence_count']} sentences"
                + (f" (-{delta})" if delta else "")
            )

        if not args.dry_run:
            out_data = dict(trace_data)
            out_data["sentences"]    = sentences
            out_data["preprocessing"] = preprocessing

            out_path = output_dir / in_path.relative_to(input_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(out_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # ── Final summary ─────────────────────────────────────────────────────────
    n_modified  = sum(1 for r in all_results if r["modified"])
    total_frags = sum(r["fragments"] for r in all_results)
    total_trunc = sum(r["truncated"] for r in all_results)
    total_man   = sum(r["manual"]    for r in all_results)
    orig_total  = sum(r["original"]  for r in all_results)
    clean_total = sum(r["cleaned"]   for r in all_results)

    print(f"\n{'=' * 50}")
    print(
        f"Traces processed: {len(all_results)}"
        f" | Modified: {n_modified}"
        f" | Unchanged: {len(all_results) - n_modified}"
    )
    print(
        f"Fragments merged: {total_frags}"
        f" | Truncated removed: {total_trunc}"
        f" | Manual: {total_man}"
    )
    print(f"Sentences: {orig_total:,} -> {clean_total:,} ({orig_total - clean_total} removed)")

    if not args.dry_run:
        generate_report(all_results, all_actions, output_dir, timestamp)
        print(f"\nReport: {output_dir / 'cleanup_report.md'}")
        print(f"CSV:    {output_dir / 'cleanup_details.csv'}")
    else:
        frag_count  = sum(1 for a in all_actions if a["action"] == "fragment_merge")
        trunc_count = sum(1 for a in all_actions if a["action"] == "truncation_removal")
        man_count   = sum(1 for a in all_actions if a["action"] in ("manual_merge", "manual_split"))
        print(f"\n[DRY RUN] Actions that would be taken:")
        print(f"  Fragment merges:     {frag_count}")
        print(f"  Truncation removals: {trunc_count}")
        print(f"  Manual actions:      {man_count}")


if __name__ == "__main__":
    main()

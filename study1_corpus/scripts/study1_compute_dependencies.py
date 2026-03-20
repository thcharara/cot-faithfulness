#!/usr/bin/env python3
"""
study1_compute_dependencies.py — Semi-programmatic dependency computation for coded traces

Reads micro-label sequences from outputs/traces_clean_coded/ and computes
coding.depends_on for each sentence using mechanical heuristics based on
label type. Writes depends_on back into the trace JSONs in-place.

Algorithm window: 20 sentences (local), 50 sentences for RULE.
Coverage: ~80% of dependencies. Long-range cross-cycle references are
not targeted — they require semantic matching beyond the label sequence.

Usage:
    python study1_corpus/scripts/study1_compute_dependencies.py [OPTIONS]

Options:
    --coded-dir   Coded traces dir  (default: outputs/traces_clean_coded)
    --traces      Subset to process (default: all). Same syntax as study1_batch_code_traces.py:
                  set_a, set_b, task1, set_a/task2, set_a/task1/trace_001, or comma-mix
    --window      Local lookback window in sentences (default: 20)
    --dry-run     Compute and report stats without writing files
    --force       Re-compute even if depends_on already present
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW      = 20   # local lookback for most labels
HYPO_WINDOW = 35   # wider lookback for HYPO (catches JUDGE after long TEST runs)
RULE_WINDOW = 50   # wider lookback for RULE (appears at end, refs earlier)
MAX_SYNTH   = 5    # max sids collected for SYNTHESIZE's preceding run
MAX_EVIDENCE = 3   # max evidence sids for TEST / JUDGE

# Labels that break a DESCRIBE / SYNTHESIZE chain
CHAIN_BREAKERS = {"HYPO", "JUDGE", "PLAN", "ORIENT", "MONITOR", "RULE"}

# Labels that can trigger a PLAN
PLAN_TRIGGERS = {"ORIENT", "MONITOR", "JUDGE", "SYNTHESIZE"}

# Labels that can trigger a HYPO (when no recent JUDGE)
HYPO_TRIGGERS = {"SYNTHESIZE", "MONITOR"}

# Labels that aggregate into SYNTHESIZE
SYNTH_SOURCES = {"DESCRIBE", "SYNTHESIZE", "TEST"}


# ---------------------------------------------------------------------------
# Dependency computation
# ---------------------------------------------------------------------------

def compute_deps(sentences: list[dict], window: int = WINDOW) -> dict[int, list[int]]:
    """
    Compute {sentence_id: [dep_sentence_ids]} for every coded sentence.

    sentences: list of sentence dicts with 'sentence_id' and 'coding' keys.
    window:    local lookback window size.

    Returns a dict mapping each sid to its (possibly empty) dep list.
    Empty list [] means: intentionally no dependency (e.g., first ORIENT).
    Sentences without a coding block are skipped and get no entry.
    """
    # Build ordered list of (sid, label) for coded sentences only
    coded = []
    for s in sentences:
        if "coding" not in s or not s["coding"].get("micro_label"):
            continue
        coded.append((s["sentence_id"], s["coding"]["micro_label"]))

    # Judgement lookup for RULE's accept-preference logic
    judgements: dict[int, str | None] = {
        s["sentence_id"]: s["coding"].get("judgement")
        for s in sentences
        if "coding" in s and s["coding"].get("micro_label") == "JUDGE"
    }

    # hypo_status and hypo_antecedent_sid lookup (populated by study1_compute_hypo_status.py)
    hypo_status: dict[int, str] = {}
    hypo_antecedent: dict[int, int | None] = {}
    for s in sentences:
        coding = s.get("coding", {})
        if coding.get("micro_label") == "HYPO":
            sid = s["sentence_id"]
            hypo_status[sid]     = coding.get("hypo_status", "")
            hypo_antecedent[sid] = coding.get("hypo_antecedent_sid")

    deps: dict[int, list[int]] = {}

    for i, (sid, label) in enumerate(coded):
        # Local window: coded sentences before current, up to `window` back
        local_start = max(0, i - window)
        local = coded[local_start:i]  # excludes current

        # Wide window for RULE
        wide_start = max(0, i - RULE_WINDOW)
        wide = coded[wide_start:i]

        if label == "ORIENT":
            # Consecutive ORIENTs chain to the immediately preceding ORIENT;
            # otherwise no dependency.
            if local and local[-1][1] == "ORIENT":
                deps[sid] = [local[-1][0]]
            else:
                deps[sid] = []

        elif label == "PLAN":
            # Triggered by most recent {ORIENT, MONITOR, JUDGE, SYNTHESIZE}
            trigger = _first(local, PLAN_TRIGGERS, reverse=True)
            deps[sid] = [trigger] if trigger is not None else []

        elif label == "DESCRIBE":
            # If immediately preceded by DESCRIBE → chain to it (scanning run).
            # Otherwise start of a new panel → depend on most recent PLAN.
            if local and local[-1][1] == "DESCRIBE":
                deps[sid] = [local[-1][0]]
            else:
                plan = _first(local, {"PLAN"}, reverse=True)
                deps[sid] = [plan] if plan is not None else []

        elif label == "SYNTHESIZE":
            # Aggregates the immediately-preceding unbroken run of
            # {DESCRIBE, SYNTHESIZE, TEST}, capped at MAX_SYNTH.
            run = _preceding_run(local, SYNTH_SOURCES, CHAIN_BREAKERS)
            deps[sid] = run[-MAX_SYNTH:]  # oldest first, most recent last

        elif label == "HYPO":
            # Three cases in priority order:
            # (1) revised/repeated: depends on the antecedent HYPO it builds from/revisits,
            #     plus the most recent JUDGE in the HYPO_WINDOW if one exists.
            # (2) reacting to a rejection: most recent JUDGE (no antecedent HYPO available).
            # (3) fresh pivot: most recent SYNTHESIZE or MONITOR.
            # Use wider HYPO_WINDOW to catch JUDGEs after long TEST runs.
            hypo_start = max(0, i - HYPO_WINDOW)
            hypo_local = coded[hypo_start:i]
            status     = hypo_status.get(sid, "")
            antecedent = hypo_antecedent.get(sid)

            if status in ("revised", "repeated") and antecedent is not None:
                # Revision/repetition: primary dep is the antecedent HYPO.
                # Secondary dep: most recent JUDGE in wide window (if present),
                # because the revision is usually prompted by a rejection.
                judge = _first(hypo_local, {"JUDGE"}, reverse=True)
                result: list[int] = [antecedent]
                if judge is not None and judge not in result:
                    result.append(judge)
                deps[sid] = result
            else:
                # Novel HYPO or hypo_status not yet computed: fall back to
                # label-sequence heuristic (JUDGE trigger or SYNTHESIZE/MONITOR pivot).
                judge = _first(hypo_local, {"JUDGE"}, reverse=True)
                if judge is not None:
                    deps[sid] = [judge]
                else:
                    trigger = _first(hypo_local, HYPO_TRIGGERS, reverse=True)
                    deps[sid] = [trigger] if trigger is not None else []

        elif label == "TEST":
            # (a) The HYPO being evaluated — most recent HYPO in local window.
            # (b) Evidence — up to MAX_EVIDENCE immediately-preceding DESCRIBE/TEST sids.
            hypo = _first(local, {"HYPO"}, reverse=True)
            evidence_run = _preceding_run(local, {"DESCRIBE", "TEST"}, CHAIN_BREAKERS)
            evidence = evidence_run[-MAX_EVIDENCE:]

            result: list[int] = []
            if hypo is not None:
                result.append(hypo)
            result.extend(s for s in evidence if s not in result)
            deps[sid] = result

        elif label == "JUDGE":
            # (a) The HYPO being judged — most recent HYPO in local window.
            # (b) The TEST sentences providing evidence — up to MAX_EVIDENCE most recent,
            #     but restricted to after the anchor HYPO to prevent cross-cycle contamination.
            hypo = _first(local, {"HYPO"}, reverse=True)
            if hypo is not None:
                hypo_idx = next(j for j, (s, _) in enumerate(local) if s == hypo)
                after_hypo = local[hypo_idx + 1:]
                tests = _first_n(after_hypo, {"TEST"}, n=MAX_EVIDENCE, reverse=True)
            else:
                tests = _first_n(local, {"TEST"}, n=MAX_EVIDENCE, reverse=True)

            result = []
            if hypo is not None:
                result.append(hypo)
            result.extend(t for t in tests if t not in result)
            deps[sid] = result

        elif label == "MONITOR":
            # Metacognitive reaction: depends on whatever immediately preceded it.
            deps[sid] = [local[-1][0]] if local else []

        elif label == "RULE":
            # Final conclusion: most recent JUDGE (prefer accept) and the HYPO it accepted.
            # Use wider window since RULE appears late and refs earlier conclusions.
            accept_judge = next(
                (s for s, l in reversed(wide) if l == "JUDGE" and judgements.get(s) == "accept"),
                None,
            )
            any_judge = _first(wide, {"JUDGE"}, reverse=True)
            judge_sid = accept_judge if accept_judge is not None else any_judge
            hypo_sid  = _first(wide, {"HYPO"}, reverse=True)
            result = []
            if judge_sid is not None:
                result.append(judge_sid)
            if hypo_sid is not None and hypo_sid not in result:
                result.append(hypo_sid)
            deps[sid] = result

        else:
            # Unknown label — no dependency
            deps[sid] = []

    return deps


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _first(
    window: list[tuple[int, str]],
    labels: set[str],
    reverse: bool = True,
) -> int | None:
    """Return sid of first (default: most recent) entry in window with label in labels."""
    seq = reversed(window) if reverse else iter(window)
    for sid, lbl in seq:
        if lbl in labels:
            return sid
    return None


def _first_n(
    window: list[tuple[int, str]],
    labels: set[str],
    n: int,
    reverse: bool = True,
) -> list[int]:
    """Return sids of up to n most-recent entries in window with label in labels."""
    result = []
    seq = reversed(window) if reverse else iter(window)
    for sid, lbl in seq:
        if lbl in labels:
            result.append(sid)
            if len(result) >= n:
                break
    return list(reversed(result))  # return in ascending sid order


def _preceding_run(
    window: list[tuple[int, str]],
    allowed: set[str],
    breakers: set[str],
) -> list[int]:
    """
    Walk backwards through window collecting sids while labels are in `allowed`.
    Stop at the first label in `breakers` or any label not in `allowed`.
    Returns sids in ascending order (oldest first).
    """
    run = []
    for sid, lbl in reversed(window):
        if lbl in allowed:
            run.append(sid)
        else:
            break  # any non-allowed label stops the run
    return list(reversed(run))  # oldest first


# ---------------------------------------------------------------------------
# Trace discovery
# ---------------------------------------------------------------------------

def discover_traces(coded_dir: Path, traces_arg: str) -> list[Path]:
    """Resolve --traces argument to list of .json paths in coded_dir."""
    all_paths = sorted(coded_dir.rglob("*.json"))

    if traces_arg == "all":
        return all_paths

    SET_RE     = re.compile(r"^set_[ab]$", re.IGNORECASE)
    TASK_RE    = re.compile(r"^task\d+$",  re.IGNORECASE)
    SETTASK_RE = re.compile(r"^set_[ab]/task\d+$", re.IGNORECASE)

    selected: list[Path] = []
    seen: set[Path] = set()

    def _rel(p: Path) -> str:
        return p.relative_to(coded_dir).as_posix().removesuffix(".json")

    def _add(paths: list[Path]) -> None:
        for p in paths:
            if p not in seen:
                selected.append(p)
                seen.add(p)

    for token in traces_arg.split(","):
        token = token.strip().replace("\\", "/").removesuffix(".json")
        if not token:
            continue

        if SET_RE.match(token):
            _add([p for p in all_paths if _rel(p).startswith(token + "/")])
        elif TASK_RE.match(token):
            _add([p for p in all_paths if _rel(p).split("/")[1] == token.lower()])
        elif SETTASK_RE.match(token):
            _add([p for p in all_paths if _rel(p).startswith(token + "/")])
        else:
            explicit = coded_dir / (token + ".json")
            if explicit.exists():
                _add([explicit])
            else:
                print(f"  WARNING: trace not found: {explicit} -- skipping.")

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute coding.depends_on for all coded traces (semi-programmatic)"
    )
    parser.add_argument("--coded-dir", default=str(PROJECT_ROOT / "outputs" / "traces_clean_coded"),
                        help="Directory of coded traces (default: outputs/traces_clean_coded)")
    parser.add_argument("--traces", default="all",
                        help="Subset to process. Same syntax as study1_batch_code_traces.py. (default: all)")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"Local lookback window in sentences (default: {WINDOW})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute deps but do not write files")
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if depends_on already present")
    args = parser.parse_args()

    coded_dir = Path(args.coded_dir)
    if not coded_dir.exists():
        print(f"ERROR: coded-dir not found: {coded_dir}")
        sys.exit(1)

    trace_paths = discover_traces(coded_dir, args.traces)
    if not trace_paths:
        print("No traces found matching --traces argument.")
        sys.exit(1)

    print(f"Processing {len(trace_paths)} traces  (window={args.window}, "
          f"dry_run={args.dry_run}, force={args.force})")
    print()

    # Per-label counters for summary
    label_total: dict[str, int]    = {}
    label_with_dep: dict[str, int] = {}
    label_empty: dict[str, int]    = {}

    n_skipped   = 0
    n_processed = 0

    for trace_path in trace_paths:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        sentences = data.get("sentences", [])

        # Skip if already has depends_on and not --force
        already_done = any(
            "depends_on" in s.get("coding", {})
            for s in sentences if "coding" in s
        )
        if already_done and not args.force:
            n_skipped += 1
            continue

        # Compute
        deps = compute_deps(sentences, window=args.window)

        # Apply to sentences
        for s in sentences:
            sid = s.get("sentence_id")
            if sid in deps and "coding" in s:
                s["coding"]["depends_on"] = deps[sid]

        # Accumulate stats
        for s in sentences:
            coding = s.get("coding", {})
            lbl = coding.get("micro_label")
            if lbl is None:
                continue
            dep = coding.get("depends_on", [])
            label_total[lbl]    = label_total.get(lbl, 0) + 1
            label_with_dep[lbl] = label_with_dep.get(lbl, 0) + (1 if dep else 0)
            label_empty[lbl]    = label_empty.get(lbl, 0)  + (1 if not dep else 0)

        n_processed += 1

        if not args.dry_run:
            trace_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("=" * 65)
    print("DEPENDENCY COMPUTATION SUMMARY")
    print("=" * 65)
    print(f"  Traces processed : {n_processed}")
    print(f"  Traces skipped   : {n_skipped} (already have depends_on; use --force to re-run)")
    if args.dry_run:
        print("  [DRY RUN — no files written]")
    print()

    MICRO_ORDER = ["ORIENT", "PLAN", "DESCRIBE", "SYNTHESIZE",
                   "HYPO", "TEST", "JUDGE", "MONITOR", "RULE"]

    print(f"  {'Label':<12} {'Total':>7} {'With dep':>9} {'Empty':>7} {'% with dep':>11}")
    print("  " + "-" * 50)
    grand_total = grand_dep = 0
    for lbl in MICRO_ORDER:
        tot  = label_total.get(lbl, 0)
        dep  = label_with_dep.get(lbl, 0)
        emp  = label_empty.get(lbl, 0)
        pct  = dep / tot * 100 if tot else 0
        print(f"  {lbl:<12} {tot:>7} {dep:>9} {emp:>7} {pct:>10.1f}%")
        grand_total += tot
        grand_dep   += dep

    print("  " + "-" * 50)
    grand_pct = grand_dep / grand_total * 100 if grand_total else 0
    print(f"  {'TOTAL':<12} {grand_total:>7} {grand_dep:>9} "
          f"{grand_total - grand_dep:>7} {grand_pct:>10.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()

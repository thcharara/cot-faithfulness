#!/usr/bin/env python3
"""
study1_batch_code_traces.py — Single-Taxonomy Auto-Coding via Anthropic Message Batches API

Replaces the pilot dual-taxonomy batch script.  Uses a single integrated
taxonomy (final_taxonomy_system_prompt.md + few_shot_examples_v2.md) and a
single coder (Claude Sonnet).

Three modes:
    submit   — Build and submit batch requests
    poll     — Check batch processing status
    download — Stream results, validate, merge into trace JSONs

Typical workflow:
    1. python study1_corpus/scripts/study1_batch_code_traces.py submit --traces validation --dry-run
    2. python study1_corpus/scripts/study1_batch_code_traces.py submit --traces validation
    3. python study1_corpus/scripts/study1_batch_code_traces.py poll
    4. python study1_corpus/scripts/study1_batch_code_traces.py download
    5. python study1_corpus/scripts/study1_compute_kappa.py

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anthropic Batch API pricing (50% discount vs standard)
BATCH_RATES = {
    "input":  1.50 / 1_000_000,   # $1.50/M tokens input
    "output": 7.50 / 1_000_000,   # $7.50/M tokens output
}

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent

MODEL          = "claude-sonnet-4-20250514"
MAX_TOKENS     = 64000
MANIFEST_FILE  = str(PROJECT_ROOT / "outputs" / "batch_manifest.json")
LOG_FILE       = str(PROJECT_ROOT / "outputs" / "coding_log.jsonl")
ERRORS_DIR     = str(PROJECT_ROOT / "outputs" / "coding_errors")

# Valid micro labels and their macro mappings
VALID_MICRO_LABELS = {"ORIENT", "DESCRIBE", "SYNTHESIZE", "HYPO", "TEST", "JUDGE",
                      "PLAN", "MONITOR", "RULE"}

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

VALID_TEST_CONTEXT  = {"post_hypothesis", "pre_hypothesis", "post_rule"}
VALID_SPECIFICITY   = {"within_panel", "across_panels"}
VALID_JUDGEMENT     = {"accept", "reject", "uncertain"}
VALID_CONFIDENCE    = {"high", "medium"}

# The 7 validation traces used for κ computation
KAPPA_TRACES = [
    "set_a/task2/trace_014.json",
    "set_b/task1/trace_010.json",
    "set_b/task2/trace_004.json",
    "set_b/task2/trace_006.json",
    "set_a/task3/trace_020.json",
    "set_a/task4/trace_019.json",
    "set_b/task4/trace_015.json",
]

# Spot-check selection: one trace per (set, task) combo, medium length (100-250 sents)
SPOT_CHECK_TARGETS = [
    ("set_a", "task1"),
    ("set_b", "task3"),
    ("set_b", "task4"),
]

# Traces used as few-shot examples — excluded from spot-check to avoid contamination
FEW_SHOT_SOURCES = {
    "set_a/task1/trace_015",   # setA_task1_trace15
    "set_a/task3/trace_010",   # setA_task3_trace10
    "set_b/task3/trace_020",   # setB_task3_trace20
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def encode_custom_id(trace_rel: str) -> str:
    """
    Convert a relative trace path to a custom_id.
    e.g. 'set_a/task2/trace_014' → 'set_a__task2__trace_014'
    """
    parts = Path(trace_rel).with_suffix("").parts  # ('set_a', 'task2', 'trace_014')
    return "__".join(parts)


def decode_custom_id(custom_id: str) -> str:
    """
    Convert a custom_id back to relative trace path.
    e.g. 'set_a__task2__trace_014' → 'set_a/task2/trace_014'
    """
    return "/".join(custom_id.split("__"))


def get_task_id(trace_data: dict) -> int:
    """Extract numeric task ID from trace data."""
    return int(trace_data.get("task_id", 0))


def load_manifest() -> list[dict]:
    p = Path(MANIFEST_FILE)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def save_manifest(records: list[dict]) -> None:
    p = Path(MANIFEST_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def get_active_batch_id(explicit_id: str | None) -> str:
    if explicit_id:
        return explicit_id
    records = load_manifest()
    if not records:
        print("ERROR: No batch manifest found. Run 'submit' first or pass --batch-id.")
        sys.exit(1)
    return records[-1]["batch_id"]


def append_log(entry: dict) -> None:
    p = Path(LOG_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_error_text(trace_rel: str, raw_text: str) -> None:
    errors_dir = Path(ERRORS_DIR)
    errors_dir.mkdir(parents=True, exist_ok=True)
    safe_name  = trace_rel.replace("/", "_") + ".txt"
    (errors_dir / safe_name).write_text(raw_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Trace discovery and selection
# ---------------------------------------------------------------------------

def discover_all_traces(cleaned_dir: Path) -> list[str]:
    """Return sorted list of relative paths (no .json) for all traces."""
    paths = sorted(cleaned_dir.glob("*/task*/trace_*.json"))
    return [str(p.relative_to(cleaned_dir).with_suffix("")).replace("\\", "/")
            for p in paths]


def select_spot_check_traces(cleaned_dir: Path) -> list[str]:
    """
    For each (set, task) in SPOT_CHECK_TARGETS, pick the trace closest to the
    middle of the 100-250 sentence range.  Falls back to shortest available if
    no medium traces exist.
    """
    selected = []
    for set_name, task_name in SPOT_CHECK_TARGETS:
        task_dir = cleaned_dir / set_name / task_name
        candidates = []
        for p in sorted(task_dir.glob("trace_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                n    = len(data.get("sentences", []))
                candidates.append((n, p.stem))
            except Exception:
                continue

        medium = [(n, stem) for n, stem in candidates if 100 <= n <= 250]
        pool   = medium if medium else candidates

        if not pool:
            print(f"  WARNING: No candidates found for {set_name}/{task_name}")
            continue

        # Exclude few-shot sources and kappa traces
        excluded = FEW_SHOT_SOURCES | {t.replace(".json", "") for t in KAPPA_TRACES}
        pool = [(n, stem) for n, stem in pool
                if f"{set_name}/{task_name}/{stem}" not in excluded]

        if not pool:
            print(f"  WARNING: No eligible candidates for {set_name}/{task_name} "
                  f"after exclusions")
            continue

        pool.sort(key=lambda x: abs(x[0] - 175))  # closest to 175 sentences
        chosen_n, chosen_stem = pool[0]
        trace_rel = f"{set_name}/{task_name}/{chosen_stem}"
        selected.append(trace_rel)
        print(f"  Spot-check: {trace_rel} ({chosen_n} sentences)")

    return selected


def resolve_traces_arg(traces_arg: str, cleaned_dir: Path) -> list[str]:
    """
    Resolve the --traces argument to a list of relative trace paths (no .json).

    Supported values (comma-separated, mix freely):
      all                       all 320 traces
      validation                7 kappa + 3 spot-check
      set_a                     all traces in set_a  (160 traces)
      set_b                     all traces in set_b  (160 traces)
      task1                     task1 across both sets (80 traces)
      task1,task2               task1 + task2 across both sets
      set_a/task1               set_a task1 only (40 traces)
      set_a/task1,set_b/task3   specific set/task combos
      set_a/task1/trace_001     single explicit trace
    """
    if traces_arg == "all":
        return discover_all_traces(cleaned_dir)

    if traces_arg == "validation":
        kappa = [t.replace(".json", "") for t in KAPPA_TRACES]
        print("Selecting validation traces:")
        print("  kappa traces (7):")
        for t in kappa:
            print(f"    {t}")
        print("  Spot-check traces (3):")
        spot = select_spot_check_traces(cleaned_dir)
        return kappa + spot

    # All discovered traces as a reference pool
    all_traces = discover_all_traces(cleaned_dir)

    SET_RE   = re.compile(r"^set_[ab]$", re.IGNORECASE)
    TASK_RE  = re.compile(r"^task\d+$",  re.IGNORECASE)
    SETTASK_RE = re.compile(r"^set_[ab]/task\d+$", re.IGNORECASE)

    selected: list[str] = []
    seen: set[str] = set()

    def add(traces: list[str]) -> None:
        for t in traces:
            if t not in seen:
                selected.append(t)
                seen.add(t)

    for token in traces_arg.split(","):
        token = token.strip().replace("\\", "/").removesuffix(".json")
        if not token:
            continue

        if SET_RE.match(token):
            # e.g. "set_a" — all traces in that set
            match = [t for t in all_traces if t.startswith(token + "/")]
            if not match:
                print(f"  WARNING: No traces found for set '{token}'")
            add(match)

        elif TASK_RE.match(token):
            # e.g. "task1" — that task across both sets
            match = [t for t in all_traces if t.split("/")[1] == token.lower()]
            if not match:
                print(f"  WARNING: No traces found for task '{token}'")
            add(match)

        elif SETTASK_RE.match(token):
            # e.g. "set_a/task1" — specific set + task folder
            match = [t for t in all_traces if t.startswith(token + "/")]
            if not match:
                print(f"  WARNING: No traces found for '{token}'")
            add(match)

        else:
            # Assume explicit trace path: set_a/task1/trace_001 (with or without .json)
            p = cleaned_dir / (token + ".json")
            if not p.exists():
                print(f"  WARNING: Trace not found: {p} -- skipping.")
            else:
                add([token])

    return selected


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_system_prompt(prompts_dir: Path) -> str:
    system_file  = prompts_dir / "final_taxonomy_system_prompt.md"
    examples_file = prompts_dir / "few_shot_examples_v2.md"
    for f in (system_file, examples_file):
        if not f.exists():
            print(f"ERROR: Prompt file not found: {f}")
            sys.exit(1)
    system_text  = system_file.read_text(encoding="utf-8")
    examples_text = examples_file.read_text(encoding="utf-8")
    return system_text.strip() + "\n\n---\n\n" + examples_text.strip()


def build_user_message(task_data: dict, sentences: list, template: str) -> str:
    task_json_str    = json.dumps(task_data, indent=2, ensure_ascii=False)
    numbered         = "\n".join(f"{s['sentence_id']}. {s['text']}" for s in sentences)
    return (template
            .replace("{task_json}", task_json_str)
            .replace("{numbered_sentences}", numbered))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def extract_json_array(text: str) -> list:
    """
    Extract a JSON array from the model's response.
    Handles markdown code fences (```json ... ```) if present.
    """
    # Strip code fences
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        raise ValueError("Response is not a JSON array")
    except json.JSONDecodeError:
        # Try to find array boundaries
        start = text.find("[")
        end   = text.rfind("]")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def validate_coding(codings: list, n_sentences: int, trace_rel: str) -> tuple[list, int, int]:
    """
    Validate and normalise a list of coding objects from the model.

    Returns
    -------
    validated       — list of validated coding dicts
    n_invalid_labels — count of invalid micro_labels found
    n_missing_flags  — count of missing required flags
    """
    n_invalid = 0
    n_missing = 0
    validated = []

    for item in codings:
        if not isinstance(item, dict):
            continue

        micro = item.get("micro_label", "")
        if micro not in VALID_MICRO_LABELS:
            print(f"  WARN [{trace_rel}] Invalid micro_label: {micro!r}")
            n_invalid += 1
            # Keep the item but mark it
            item["_invalid_label"] = True

        # Check flags
        if micro == "TEST":
            tc = item.get("test_context")
            sp = item.get("specificity")
            if tc not in VALID_TEST_CONTEXT:
                print(f"  WARN [{trace_rel}] s{item.get('sentence_id')}: "
                      f"missing/invalid test_context: {tc!r}")
                n_missing += 1
            if sp not in VALID_SPECIFICITY:
                print(f"  WARN [{trace_rel}] s{item.get('sentence_id')}: "
                      f"missing/invalid specificity: {sp!r}")
                n_missing += 1

        if micro == "JUDGE":
            jg = item.get("judgement")
            if jg not in VALID_JUDGEMENT:
                print(f"  WARN [{trace_rel}] s{item.get('sentence_id')}: "
                      f"missing/invalid judgement: {jg!r}")
                n_missing += 1

        conf = item.get("confidence")
        if conf not in VALID_CONFIDENCE:
            # Default to "medium" if missing rather than failing
            item["confidence"] = "medium"

        # Add macro_label
        item["macro_label"] = MACRO_MAP.get(micro, "UNKNOWN")

        validated.append(item)

    return validated, n_invalid, n_missing


def merge_codings_into_sentences(
    sentences: list[dict],
    codings: list[dict],
) -> list[dict]:
    """
    Merge validated codings into the sentence list.
    Matches by sentence_id.  Unmatched sentences keep no coding block;
    extra codings are ignored.
    """
    coding_by_sid = {c["sentence_id"]: c for c in codings if "sentence_id" in c}

    merged = []
    for s in sentences:
        s_out = dict(s)
        sid   = s["sentence_id"]
        if sid in coding_by_sid:
            c = coding_by_sid[sid]
            coding_block: dict = {"macro_label": c.get("macro_label", "UNKNOWN"),
                                  "micro_label": c.get("micro_label", "UNKNOWN"),
                                  "confidence":  c.get("confidence", "medium")}
            micro = c.get("micro_label", "")
            if micro == "TEST":
                coding_block["test_context"] = c.get("test_context")
                coding_block["specificity"]  = c.get("specificity")
            if micro == "JUDGE":
                coding_block["judgement"] = c.get("judgement")
            s_out["coding"] = coding_block
        merged.append(s_out)
    return merged


# ---------------------------------------------------------------------------
# Mode: submit
# ---------------------------------------------------------------------------

def do_submit(args, client) -> None:
    cleaned_dir = Path(args.cleaned_dir)
    output_dir  = Path(args.output_dir)
    prompts_dir = Path(args.prompts_dir)
    stimuli_dir = Path(args.stimuli_dir)

    if not cleaned_dir.exists():
        print(f"ERROR: cleaned traces dir not found: {cleaned_dir}")
        print("  Run study1_corpus/scripts/study1_preprocess_traces.py first.")
        sys.exit(1)

    # ── Load system prompt ────────────────────────────────────────────────────
    print("Loading prompts...")
    system_prompt = load_system_prompt(prompts_dir)

    # ── Load user message template ────────────────────────────────────────────
    template_path = prompts_dir / "user_message_template.txt"
    if not template_path.exists():
        print(f"ERROR: user_message_template.txt not found in {prompts_dir}")
        sys.exit(1)
    user_template = template_path.read_text(encoding="utf-8")

    # ── Load stimuli ──────────────────────────────────────────────────────────
    stimuli: dict[int, dict] = {}
    for p in sorted(stimuli_dir.glob("Task*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # Task id from filename: Task1.json → 1
            task_num = int(re.search(r"(\d+)", p.stem).group(1))
            stimuli[task_num] = data
        except Exception as exc:
            print(f"  WARNING: Could not load stimulus {p}: {exc}")

    if not stimuli:
        print(f"ERROR: No stimulus files found in {stimuli_dir}")
        sys.exit(1)
    print(f"  Loaded stimuli for tasks: {sorted(stimuli.keys())}")

    # ── Resolve which traces to code ──────────────────────────────────────────
    print(f"\nResolving --traces {args.traces!r}...")
    trace_rels = resolve_traces_arg(args.traces, cleaned_dir)

    if not trace_rels:
        print("ERROR: No traces selected.")
        sys.exit(1)

    # Filter out traces already coded (unless --force)
    requests: list[dict] = []
    n_skipped = 0

    for trace_rel in trace_rels:
        out_path = output_dir / (trace_rel + ".json")
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        trace_path = cleaned_dir / (trace_rel + ".json")
        if not trace_path.exists():
            print(f"  WARNING: Trace not found: {trace_path} — skipping.")
            continue

        try:
            trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  WARNING: Could not read {trace_path}: {exc}")
            continue

        sentences = trace_data.get("sentences", [])
        task_id   = get_task_id(trace_data)
        task_data = stimuli.get(task_id, {})

        if not task_data:
            print(f"  WARNING: No stimulus for task_id={task_id} in {trace_rel}")

        user_msg   = build_user_message(task_data, sentences, user_template)
        custom_id  = encode_custom_id(trace_rel)

        requests.append({
            "custom_id": custom_id,
            "params": {
                "model":      MODEL,
                "max_tokens": MAX_TOKENS,
                "system":     system_prompt,
                "messages":   [{"role": "user", "content": user_msg}],
            },
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_in, avg_out = 10000, 5000   # conservative estimates
    est_cost = len(requests) * (
        avg_in * BATCH_RATES["input"] + avg_out * BATCH_RATES["output"])

    print(f"\n=== Batch Submit ===")
    print(f"Traces selected: {len(trace_rels)}")
    if n_skipped:
        print(f"Already coded (skipped): {n_skipped} (use --force to re-submit)")
    print(f"Requests to submit: {len(requests)}")
    print(f"Model: {MODEL}  |  Max tokens: {MAX_TOKENS}")
    print(f"Estimated cost (batch 50% rate): ~${est_cost:.2f}")

    if not requests:
        print("\nNothing to submit.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] First 15 custom_ids:")
        for r in requests[:15]:
            print(f"  {r['custom_id']}")
        if len(requests) > 15:
            print(f"  ... and {len(requests) - 15} more")
        print(f"\n[DRY RUN] System prompt length: {len(system_prompt):,} chars")
        print("[DRY RUN] No batch submitted.")
        return

    # ── Submit ────────────────────────────────────────────────────────────────
    print(f"\nSubmitting {len(requests)} requests to Anthropic Batch API...")
    try:
        batch = client.messages.batches.create(requests=requests)
    except Exception as exc:
        print(f"ERROR: Batch submission failed: {exc}")
        sys.exit(1)

    submitted_at = now_iso()
    print(f"Batch submitted: {batch.id}")
    print(f"Submitted at: {submitted_at}")

    records = load_manifest()
    records.append({
        "batch_id":     batch.id,
        "submitted_at": submitted_at,
        "n_requests":   len(requests),
        "traces":       args.traces,
        "trace_list":   trace_rels,
        "custom_ids":   [r["custom_id"] for r in requests],
        "status":       "submitted",
    })
    save_manifest(records)

    print(f"\nNext steps:")
    print(f"  python study1_corpus/scripts/study1_batch_code_traces.py poll")
    print(f"  python study1_corpus/scripts/study1_batch_code_traces.py download")


# ---------------------------------------------------------------------------
# Mode: poll
# ---------------------------------------------------------------------------

def do_poll(args, client) -> None:
    batch_id = get_active_batch_id(args.batch_id)

    try:
        batch = client.messages.batches.retrieve(batch_id)
    except Exception as exc:
        print(f"ERROR: Could not retrieve batch {batch_id}: {exc}")
        sys.exit(1)

    rc = batch.request_counts

    elapsed_str = ""
    records = load_manifest()
    for rec in reversed(records):
        if rec.get("batch_id") == batch_id:
            submitted_at = rec.get("submitted_at", "")
            if submitted_at:
                try:
                    t0  = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    mins = (now - t0).total_seconds() / 60
                    elapsed_str = f" ({mins:.0f} min ago)"
                except Exception:
                    pass
            break

    print(f"Batch:    {batch_id}")
    print(f"Status:   {batch.processing_status}")
    pending = getattr(rc, "processing", 0)
    print(f"Requests: {rc.succeeded} succeeded, {rc.errored} errored, "
          f"{rc.canceled} canceled, {rc.expired} expired"
          + (f", {pending} processing" if pending else ""))
    if elapsed_str:
        print(f"Age: {elapsed_str}")

    if batch.processing_status == "ended":
        print("\nBatch complete — run `download` to retrieve results.")
        for rec in records:
            if rec.get("batch_id") == batch_id:
                rec["status"] = "ended"
        save_manifest(records)
    else:
        print("\nStill processing — run `poll` again to check.")


# ---------------------------------------------------------------------------
# Mode: download
# ---------------------------------------------------------------------------

def do_download(args, client) -> None:
    cleaned_dir = Path(args.cleaned_dir)
    output_dir  = Path(args.output_dir)
    batch_id    = get_active_batch_id(args.batch_id)

    try:
        batch = client.messages.batches.retrieve(batch_id)
    except Exception as exc:
        print(f"ERROR: Could not retrieve batch {batch_id}: {exc}")
        sys.exit(1)

    if batch.processing_status != "ended":
        rc = batch.request_counts
        print(f"Batch {batch_id} is still processing.")
        print(f"  {rc.succeeded} succeeded | {rc.errored} errored | "
              f"{getattr(rc, 'processing', 0)} in progress")
        print("Run `poll` again later.")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    n_succeeded = n_errored = n_expired = 0
    total_in = total_out = 0
    total_cost = 0.0
    results_to_write: list[dict] = []   # {trace_rel, codings, usage}

    print(f"Downloading results for batch {batch_id}...")

    for entry in client.messages.batches.results(batch_id):
        custom_id = entry.custom_id
        trace_rel = decode_custom_id(custom_id)

        if entry.result.type == "succeeded":
            n_succeeded += 1
            msg   = entry.result.message
            text  = msg.content[0].text
            usage = msg.usage

            cost        = (usage.input_tokens  * BATCH_RATES["input"] +
                           usage.output_tokens * BATCH_RATES["output"])
            total_in   += usage.input_tokens
            total_out  += usage.output_tokens
            total_cost += cost

            try:
                raw_codings = extract_json_array(text)
            except Exception as exc:
                print(f"  ERROR parsing JSON for {custom_id}: {exc}")
                save_error_text(trace_rel, text)
                append_log({
                    "trace":     trace_rel,
                    "status":    "parse_error",
                    "error":     str(exc),
                    "timestamp": now_iso(),
                    "batch_id":  batch_id,
                })
                n_errored += 1
                continue

            results_to_write.append({
                "trace_rel": trace_rel,
                "raw_codings": raw_codings,
                "input_tokens":  usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd":      cost,
            })

        elif entry.result.type == "errored":
            n_errored += 1
            err = str(getattr(entry.result, "error", "unknown"))
            print(f"  ERROR [{custom_id}]: {err}")
            append_log({
                "trace":     trace_rel,
                "status":    "batch_error",
                "error":     err,
                "timestamp": now_iso(),
                "batch_id":  batch_id,
            })

        else:
            n_expired += 1
            append_log({
                "trace":     trace_rel,
                "status":    entry.result.type,
                "timestamp": now_iso(),
                "batch_id":  batch_id,
            })

    print(f"\nAPI results: {n_succeeded} succeeded | {n_errored} errored | {n_expired} expired")
    print(f"Integrating {len(results_to_write)} trace files...")

    n_written = 0
    n_count_mismatch = 0

    for item in results_to_write:
        trace_rel   = item["trace_rel"]
        raw_codings = item["raw_codings"]

        # Load the blank cleaned trace
        cleaned_path = cleaned_dir / (trace_rel + ".json")
        if not cleaned_path.exists():
            print(f"  WARNING: Source trace not found: {cleaned_path} — skipping.")
            continue

        trace_data = json.loads(cleaned_path.read_text(encoding="utf-8"))
        sentences  = trace_data.get("sentences", [])
        n_sents    = len(sentences)

        # Validate and normalise codings
        validated, n_invalid, n_missing = validate_coding(
            raw_codings, n_sents, trace_rel
        )

        n_coded = len(validated)
        if n_coded != n_sents:
            print(f"  WARN [{trace_rel}]: "
                  f"{n_coded} codings returned, {n_sents} sentences expected "
                  f"— merging by sentence_id.")
            n_count_mismatch += 1

        # Merge into trace
        merged_sents = merge_codings_into_sentences(sentences, validated)

        out_data = dict(trace_data)
        out_data["sentences"] = merged_sents
        out_data["coding_metadata"] = {
            "model":         MODEL,
            "batch_id":      batch_id,
            "coded_at":      now_iso(),
            "n_sentences":   n_sents,
            "n_coded":       n_coded,
            "n_invalid_labels": n_invalid,
            "n_missing_flags":  n_missing,
            "input_tokens":  item["input_tokens"],
            "output_tokens": item["output_tokens"],
            "cost_usd":      item["cost_usd"],
        }

        out_path = output_dir / (trace_rel + ".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        n_written += 1

        append_log({
            "trace":            trace_rel,
            "n_sentences":      n_sents,
            "n_coded":          n_coded,
            "n_invalid_labels": n_invalid,
            "n_missing_flags":  n_missing,
            "input_tokens":     item["input_tokens"],
            "output_tokens":    item["output_tokens"],
            "cost_usd":         round(item["cost_usd"], 6),
            "status":           "success",
            "timestamp":        now_iso(),
            "batch_id":         batch_id,
        })

    # ── Final summary ─────────────────────────────────────────────────────────
    est_standard_cost = (total_in * 3.00 / 1_000_000 +
                         total_out * 15.00 / 1_000_000)
    print(f"\n{'=' * 50}")
    print(f"Download complete")
    print(f"  Succeeded: {n_succeeded} | Errored: {n_errored} | Expired: {n_expired}")
    print(f"  Trace files written: {n_written}")
    if n_count_mismatch:
        print(f"  Sentence count mismatches: {n_count_mismatch}")
    print(f"  Tokens: {total_in:,} input | {total_out:,} output")
    print(f"  Batch cost (50% rate): ${total_cost:.4f}")
    print(f"  Standard cost would have been: ${est_standard_cost:.4f}")
    print(f"  Saved: ${est_standard_cost - total_cost:.4f}")
    print(f"\nOutput: {output_dir}")
    print(f"Log:    {LOG_FILE}")
    if n_errored:
        print(f"Errors: {ERRORS_DIR}/")

    # Update manifest
    records = load_manifest()
    for rec in records:
        if rec.get("batch_id") == batch_id:
            rec.update({
                "status":         "downloaded",
                "downloaded_at":  now_iso(),
                "n_succeeded":    n_succeeded,
                "n_errored":      n_errored,
                "n_expired":      n_expired,
                "n_written":      n_written,
                "total_cost_usd": round(total_cost, 4),
                "total_tokens_in":  total_in,
                "total_tokens_out": total_out,
            })
    save_manifest(records)

    if n_written > 0:
        print(f"\nNext step: python study1_corpus/scripts/study1_compute_kappa.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-taxonomy auto-coding via Anthropic Message Batches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  submit    Build and submit batch requests.
  poll      Check processing status.
  download  Retrieve results and merge into output trace files.

Typical workflow:
  python study1_corpus/scripts/study1_batch_code_traces.py submit --traces validation --dry-run
  python study1_corpus/scripts/study1_batch_code_traces.py submit --traces validation
  python study1_corpus/scripts/study1_batch_code_traces.py poll
  python study1_corpus/scripts/study1_batch_code_traces.py download
  python study1_corpus/scripts/study1_compute_kappa.py

DO NOT run --traces all until validation results are reviewed.
""",
    )
    parser.add_argument("mode", choices=["submit", "poll", "download"])

    # Paths
    parser.add_argument("--cleaned-dir",  default=str(PROJECT_ROOT / "outputs" / "traces_cleaned"),
                        help="Pre-processed blank traces (default: outputs/traces_cleaned)")
    parser.add_argument("--output-dir",   default=str(PROJECT_ROOT / "outputs" / "traces_clean_coded"),
                        help="Coded output directory (default: outputs/traces_clean_coded)")
    parser.add_argument("--prompts-dir",  default=str(PROJECT_ROOT / "data" / "prompts"),
                        help="Prompts directory (default: data/prompts)")
    parser.add_argument("--stimuli-dir",  default=str(PROJECT_ROOT / "data" / "stimuli"),
                        help="Stimuli directory (default: data/stimuli)")

    # submit options
    parser.add_argument("--traces",  default="validation",
                        help=(
                            "[submit] Which traces to code. Comma-separated, mix freely. "
                            "Keywords: 'all', 'validation'. "
                            "Set: 'set_a', 'set_b'. "
                            "Task across both sets: 'task1' ... 'task4'. "
                            "Set+task: 'set_a/task1'. "
                            "Explicit trace: 'set_a/task1/trace_001'. "
                            "Examples: --traces task1 | --traces set_a | "
                            "--traces set_a/task1,set_b/task3 | "
                            "--traces set_a/task1/trace_001,set_b/task2/trace_006. "
                            "(default: validation)"
                        ))
    parser.add_argument("--force",   action="store_true",
                        help="[submit] Re-submit traces already present in output dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="[submit] Preview without submitting")

    # poll / download options
    parser.add_argument("--batch-id", default=None,
                        help="[poll/download] Explicit batch ID "
                             "(default: last entry in batch manifest)")

    args = parser.parse_args()

    # Dry-run submit doesn't need an API client
    if args.mode == "submit" and args.dry_run:
        do_submit(args, client=None)
        return

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("  Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    try:
        from anthropic import Anthropic
        client = Anthropic()
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    if args.mode == "submit":
        do_submit(args, client)
    elif args.mode == "poll":
        do_poll(args, client)
    elif args.mode == "download":
        do_download(args, client)


if __name__ == "__main__":
    main()

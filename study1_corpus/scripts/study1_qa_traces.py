"""
Trace Corpus Quality Assurance Script

Runs after trace generation to produce a comprehensive quality report
on the full trace corpus (Set A + Set B original + Set B extended).
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import re
import argparse
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Defaults ──
DEFAULT_TRACE_DIR = str(PROJECT_ROOT / "outputs" / "traces")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "outputs" / "traces" / "qa_report.md")

# Expected corpus structure
# Set A: 40 traces per task (4096 tokens)
# Set B (4096): 10 traces per task (trace_ids 1-10)
# Set B (6144): 30 traces per task (trace_ids 11-40)
EXPECTED_SET_A = {"traces_per_task": 40, "tasks": [1, 2, 3, 4]}
EXPECTED_SET_B_4096 = {"traces_per_task": 10, "tasks": [1, 2, 3, 4]}
EXPECTED_SET_B_6144 = {"traces_per_task": 30, "tasks": [1, 2, 3, 4]}

TOTAL_EXPECTED = (
    EXPECTED_SET_A["traces_per_task"] * len(EXPECTED_SET_A["tasks"])
    + EXPECTED_SET_B_4096["traces_per_task"] * len(EXPECTED_SET_B_4096["tasks"])
    + EXPECTED_SET_B_6144["traces_per_task"] * len(EXPECTED_SET_B_6144["tasks"])
)

EXPECTED_MODEL = "DeepSeek-R1-Distill-Llama-8B"
EXPECTED_QUANTIZATION = "4bit-nf4"

# Sentence-splitting regex (rough)
SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def parse_args():
    parser = argparse.ArgumentParser(
        description="QA report for the trace corpus"
    )
    parser.add_argument(
        "--trace-dir", type=str, default=DEFAULT_TRACE_DIR,
        help=f"Root trace directory (default: {DEFAULT_TRACE_DIR})"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output markdown report path (default: {DEFAULT_OUTPUT})"
    )
    return parser.parse_args()


def trace_path(trace_dir, set_name, task_id, trace_id):
    set_dir = "set_a" if set_name == "A" else "set_b"
    return os.path.join(trace_dir, set_dir, f"task{task_id}", f"trace_{trace_id:03d}.json")


def load_trace(filepath):
    """Load a single trace. Returns (data, error_string)."""
    if not os.path.exists(filepath):
        return None, "missing"
    try:
        if os.path.getsize(filepath) == 0:
            return None, "corrupted (0 bytes)"
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"corrupted (invalid JSON: {e})"
    except OSError as e:
        return None, f"read error: {e}"


def classify_group(trace_data):
    """Classify a trace into its group based on generation_params.max_new_tokens.

    Returns one of: 'Set A', 'Set B (4096)', 'Set B (6144)', or 'UNEXPECTED'.
    """
    set_name = trace_data.get("set", "")
    max_tokens = trace_data.get("generation_params", {}).get("max_new_tokens")

    if set_name == "A":
        return "Set A"
    elif set_name == "B":
        if max_tokens == 4096:
            return "Set B (4096)"
        elif max_tokens == 6144:
            return "Set B (6144)"
        else:
            return "UNEXPECTED"
    return "UNEXPECTED"


def load_all_traces(trace_dir):
    """Load entire corpus including extended traces.

    Returns:
        traces: dict keyed by (set, task_id, trace_id) -> trace data
        missing: list of (set, task_id, trace_id) keys
        corrupted: list of ((set, task_id, trace_id), error_msg)
    """
    traces = {}
    missing = []
    corrupted = []

    # Set A: traces 1-40 per task
    for task_id in EXPECTED_SET_A["tasks"]:
        for trace_id in range(1, EXPECTED_SET_A["traces_per_task"] + 1):
            key = ("A", task_id, trace_id)
            fp = trace_path(trace_dir, "A", task_id, trace_id)
            data, err = load_trace(fp)
            if data is not None:
                traces[key] = data
            elif err == "missing":
                missing.append(key)
            else:
                corrupted.append((key, err))

    # Set B: traces 1-10 (4096) and 11-40 (6144) per task
    for task_id in EXPECTED_SET_B_4096["tasks"]:
        total_b = EXPECTED_SET_B_4096["traces_per_task"] + EXPECTED_SET_B_6144["traces_per_task"]
        for trace_id in range(1, total_b + 1):
            key = ("B", task_id, trace_id)
            fp = trace_path(trace_dir, "B", task_id, trace_id)
            data, err = load_trace(fp)
            if data is not None:
                traces[key] = data
            elif err == "missing":
                missing.append(key)
            else:
                corrupted.append((key, err))

    return traces, missing, corrupted


def count_sentences(text):
    if not text or not text.strip():
        return 0
    parts = SENTENCE_RE.split(text.strip())
    return len([p for p in parts if p.strip()])


def text_histogram(values, bins=10, max_bar=40):
    """Generate a text-based histogram."""
    if not values:
        return "  (no data)\n"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"  All values = {lo}\n"
    bin_width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / bin_width), bins - 1)
        counts[idx] += 1
    max_count = max(counts) if counts else 1
    lines = []
    for i in range(bins):
        lo_edge = lo + i * bin_width
        hi_edge = lo + (i + 1) * bin_width
        bar_len = int(counts[i] / max_count * max_bar) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        lines.append(f"  {lo_edge:7.0f}-{hi_edge:7.0f}: {bar} ({counts[i]})")
    return "\n".join(lines) + "\n"


def detect_repetition(text, min_len=80, min_repeats=5):
    """Check for repeated substrings of length >= min_len appearing >= min_repeats times.

    Thresholds are tuned to catch degenerate repetition loops while ignoring
    normal hypothesis-revisiting patterns common in long reasoning traces.
    """
    if not text or len(text) < min_len * min_repeats:
        return []
    found = []
    step = max(1, len(text) // 500)
    seen = {}
    for start in range(0, len(text) - min_len, step):
        substr = text[start:start + min_len]
        if substr in seen:
            seen[substr] += 1
        else:
            count = text.count(substr)
            seen[substr] = count
            if count >= min_repeats:
                already = False
                for prev, _ in found:
                    if prev in substr or substr in prev:
                        already = True
                        break
                if not already:
                    found.append((substr[:80], count))
    return found


def detect_repetition_loose(text, min_len=50, min_repeats=3):
    """Looser repetition detection for confound analysis (same algo, lower thresholds)."""
    if not text or len(text) < min_len * min_repeats:
        return []
    found = []
    step = max(1, len(text) // 500)
    seen = {}
    for start in range(0, len(text) - min_len, step):
        substr = text[start:start + min_len]
        if substr in seen:
            seen[substr] += 1
        else:
            count = text.count(substr)
            seen[substr] = count
            if count >= min_repeats:
                already = False
                for prev, _ in found:
                    if prev in substr or substr in prev:
                        already = True
                        break
                if not already:
                    found.append((substr[:80], count))
    return found


def check_non_english(text):
    """Return fraction of characters outside ASCII printable + common punctuation + whitespace."""
    if not text:
        return 0.0
    total = len(text)
    non_ascii = 0
    for ch in text:
        cp = ord(ch)
        if 32 <= cp <= 126:
            continue
        if ch in '\t\n\r':
            continue
        if ch in '\u00b0\u00b1\u00b2\u00b3\u00b5\u00b7\u2013\u2014\u2018\u2019\u201c\u201d\u2022\u2026\u2192\u2190\u2191\u2193\u2264\u2265\u2248\u2260\u221e\u2211\u220f\u222b\u221a\u2202\u2207':
            continue
        non_ascii += 1
    return non_ascii / total if total > 0 else 0.0


def group_traces(traces):
    """Split traces into three groups based on generation_params.max_new_tokens.

    Returns:
        groups: dict with keys 'Set A', 'Set B (4096)', 'Set B (6144)'
        unexpected: list of (key, trace_data) with unexpected max_new_tokens
    """
    groups = {
        "Set A": {},
        "Set B (4096)": {},
        "Set B (6144)": {},
    }
    unexpected = []

    for key, t in traces.items():
        g = classify_group(t)
        if g == "UNEXPECTED":
            unexpected.append((key, t))
        else:
            groups[g][key] = t

    return groups, unexpected


# ═══════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════

def generate_report(trace_dir, traces, missing, corrupted):
    lines = []
    issues = []  # (severity, message)

    def add(text=""):
        lines.append(text)
        print(text)

    # Group traces
    groups, unexpected = group_traces(traces)
    GROUP_NAMES = ["Set A", "Set B (4096)", "Set B (6144)"]

    add("# Trace Corpus QA Report")
    add()
    add(f"**Generated:** {datetime.now().isoformat()}")
    add(f"**Trace directory:** `{trace_dir}`")
    add(f"**Expected corpus size:** {TOTAL_EXPECTED} traces "
        f"(Set A: {EXPECTED_SET_A['traces_per_task'] * 4}, "
        f"Set B 4096: {EXPECTED_SET_B_4096['traces_per_task'] * 4}, "
        f"Set B 6144: {EXPECTED_SET_B_6144['traces_per_task'] * 4})")
    add()

    if unexpected:
        add(f"**WARNING:** {len(unexpected)} trace(s) with unexpected max_new_tokens values:")
        for key, t in unexpected:
            mnt = t.get("generation_params", {}).get("max_new_tokens", "?")
            add(f"- `set_{key[0].lower()}/task{key[1]}/trace_{key[2]:03d}.json`: "
                f"max_new_tokens={mnt}")
        add()
        issues.append(("WARN", f"{len(unexpected)} trace(s) with unexpected max_new_tokens"))

    # ── Section 1: Corpus Completeness ──
    add("---")
    add("## Section 1: Corpus Completeness")
    add()

    # Expected counts per group
    expected_specs = {
        "Set A": (EXPECTED_SET_A, range(1, 41)),
        "Set B (4096)": (EXPECTED_SET_B_4096, range(1, 11)),
        "Set B (6144)": (EXPECTED_SET_B_6144, range(11, 41)),
    }

    for gname in GROUP_NAMES:
        spec, trace_range = expected_specs[gname]
        parts = []
        for task_id in spec["tasks"]:
            expected_n = spec["traces_per_task"]
            present = sum(1 for k in groups[gname] if k[1] == task_id)
            set_letter = "A" if gname == "Set A" else "B"
            missing_ids = [k[2] for k in missing
                           if k[0] == set_letter and k[1] == task_id
                           and k[2] in trace_range]
            corrupt_ids = [k[0][2] for k in corrupted
                           if k[0][0] == set_letter and k[0][1] == task_id
                           and k[0][2] in trace_range]
            status = f"Task {task_id}: {present}/{expected_n}"
            if missing_ids:
                status += f" [MISSING: {', '.join(f'{i:03d}' for i in sorted(missing_ids))}]"
            if corrupt_ids:
                status += f" [CORRUPTED: {', '.join(f'{i:03d}' for i in sorted(corrupt_ids))}]"
            parts.append(status)
        add(f"**{gname}:** {', '.join(parts)}")
        add()

    n_present = len(traces)
    n_missing = len(missing)
    n_corrupted = len(corrupted)
    add(f"**Total: {n_present}/{TOTAL_EXPECTED} traces present "
        f"({n_missing} missing, {n_corrupted} corrupted)**")
    add()

    if n_corrupted > 0:
        add("### Corrupted Traces")
        add()
        for key, err in corrupted:
            add(f"- `set_{key[0].lower()}/task{key[1]}/trace_{key[2]:03d}.json`: {err}")
        add()
        issues.append(("CRITICAL", f"{n_corrupted} corrupted trace(s) found"))

    if n_missing > 0:
        severity = "CRITICAL" if n_missing > 10 else "WARN"
        issues.append((severity, f"{n_missing} missing trace(s)"))

    if n_present == 0:
        issues.append(("CRITICAL", "No traces found in corpus"))
        add("**Cannot proceed with further analysis — no traces loaded.**")
        add()
        add("---")
        add("## Verdict: FAIL")
        add()
        for sev, msg in issues:
            add(f"- [{sev}] {msg}")
        return "\n".join(lines), 1

    all_traces = list(traces.values())

    # ── Section 2: Thinking Block Statistics ──
    add("---")
    add("## Section 2: Thinking Block Statistics")
    add()

    def stat_row(values, label):
        if not values:
            return f"| {label} | — | — | — | — | — | 0 |"
        mean = statistics.mean(values)
        med = statistics.median(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0
        return f"| {label} | {mean:.1f} | {med:.1f} | {sd:.1f} | {min(values)} | {max(values)} | {len(values)} |"

    # Thinking token counts
    add("### Thinking Token Counts")
    add()
    add("| Group | Mean | Median | SD | Min | Max | N |")
    add("|-------|------|--------|----|-----|-----|---|")

    for gname in GROUP_NAMES:
        for task_id in [1, 2, 3, 4]:
            vals = [t["thinking_tokens"] for k, t in groups[gname].items() if k[1] == task_id]
            add(stat_row(vals, f"{gname} Task {task_id}"))

    for gname in GROUP_NAMES:
        vals = [t["thinking_tokens"] for t in groups[gname].values()]
        add(stat_row(vals, f"**{gname} Overall**"))

    all_thinking = [t["thinking_tokens"] for t in all_traces]
    add(stat_row(all_thinking, "**Corpus Overall**"))
    add()

    # Sentence counts
    add("### Rough Sentence Counts")
    add()
    add("| Group | Mean | Median | SD | Min | Max | N |")
    add("|-------|------|--------|----|-----|-----|---|")

    sentence_cache = {}
    for key, t in traces.items():
        sentence_cache[key] = count_sentences(t.get("thinking_text", ""))

    for gname in GROUP_NAMES:
        for task_id in [1, 2, 3, 4]:
            vals = [sentence_cache[k] for k in groups[gname] if k[1] == task_id]
            add(stat_row(vals, f"{gname} Task {task_id}"))

    for gname in GROUP_NAMES:
        vals = [sentence_cache[k] for k in groups[gname]]
        add(stat_row(vals, f"**{gname} Overall**"))

    all_sents = list(sentence_cache.values())
    add(stat_row(all_sents, "**Corpus Overall**"))
    add()

    # Truncation rate
    add("### Truncation Rate")
    add()
    add("| Group | Truncated | Total | Rate |")
    add("|-------|-----------|-------|------|")

    for gname in GROUP_NAMES:
        for task_id in [1, 2, 3, 4]:
            group = [(k, t) for k, t in groups[gname].items() if k[1] == task_id]
            trunc = sum(1 for _, t in group if t.get("truncated"))
            total = len(group)
            rate = (trunc / total * 100) if total > 0 else 0
            add(f"| {gname} Task {task_id} | {trunc} | {total} | {rate:.1f}% |")

    for gname in GROUP_NAMES:
        group = list(groups[gname].values())
        trunc = sum(1 for t in group if t.get("truncated"))
        total = len(group)
        rate = (trunc / total * 100) if total > 0 else 0
        add(f"| **{gname} Overall** | {trunc} | {total} | {rate:.1f}% |")

    add()

    # Empty/trivial thinking blocks
    add("### Empty/Trivial Thinking Blocks")
    add()
    empty_traces = []
    trivial_traces = []
    for key, t in traces.items():
        thinking = t.get("thinking_text", "")
        tokens = t.get("thinking_tokens", 0)
        if not thinking or not thinking.strip():
            empty_traces.append(key)
        elif tokens < 10:
            trivial_traces.append(key)

    if empty_traces:
        add(f"**Empty thinking blocks:** {len(empty_traces)}")
        for k in empty_traces:
            add(f"- `set_{k[0].lower()}/task{k[1]}/trace_{k[2]:03d}.json`")
        issues.append(("CRITICAL", f"{len(empty_traces)} trace(s) with empty thinking blocks"))
    else:
        add("**Empty thinking blocks:** 0")

    if trivial_traces:
        add(f"**Trivial thinking blocks (<10 tokens):** {len(trivial_traces)}")
        for k in trivial_traces:
            add(f"- `set_{k[0].lower()}/task{k[1]}/trace_{k[2]:03d}.json`")
        issues.append(("WARN", f"{len(trivial_traces)} trace(s) with trivial thinking blocks (<10 tokens)"))
    else:
        add("**Trivial thinking blocks (<10 tokens):** 0")
    add()

    # Generation time
    add("### Generation Time")
    add()
    add("| Group | Mean | Median | Min | Max | N |")
    add("|-------|------|--------|-----|-----|---|")

    for gname in GROUP_NAMES:
        vals = [t["generation_time_seconds"] for t in groups[gname].values()]
        if vals:
            add(f"| {gname} | {statistics.mean(vals):.1f}s | {statistics.median(vals):.1f}s "
                f"| {min(vals):.1f}s | {max(vals):.1f}s | {len(vals)} |")
        else:
            add(f"| {gname} | — | — | — | — | 0 |")

    all_times = [t["generation_time_seconds"] for t in all_traces]
    if all_times:
        total_time = sum(all_times)
        add(f"| **Overall** | {statistics.mean(all_times):.1f}s | "
            f"{statistics.median(all_times):.1f}s | {min(all_times):.1f}s | "
            f"{max(all_times):.1f}s | {len(all_times)} |")
        add()
        hours = total_time / 3600
        add(f"**Total generation time:** {total_time:.0f}s ({hours:.1f} hours)")
    add()

    # ── Section 3: Completion Analysis ──
    add("---")
    add("## Section 3: Completion Analysis")
    add()

    set_b_4096 = groups["Set B (4096)"]
    set_b_6144 = groups["Set B (6144)"]

    # 3a. Completion Rate Comparison
    add("### 3a. Completion Rate Comparison")
    add()
    add("| Task | Set B (4096) | Set B (6144) | Combined |")
    add("|------|-------------|-------------|----------|")

    for task_id in [1, 2, 3, 4]:
        g4 = {k: t for k, t in set_b_4096.items() if k[1] == task_id}
        g6 = {k: t for k, t in set_b_6144.items() if k[1] == task_id}
        c4 = sum(1 for t in g4.values() if t.get("answer_text", "").strip())
        c6 = sum(1 for t in g6.values() if t.get("answer_text", "").strip())
        n4, n6 = len(g4), len(g6)
        r4 = (c4 / n4 * 100) if n4 > 0 else 0
        r6 = (c6 / n6 * 100) if n6 > 0 else 0
        ct = c4 + c6
        nt = n4 + n6
        rt = (ct / nt * 100) if nt > 0 else 0
        add(f"| Task {task_id} | {c4}/{n4} ({r4:.0f}%) | {c6}/{n6} ({r6:.0f}%) "
            f"| {ct}/{nt} ({rt:.0f}%) |")

    # Overall
    c4_all = sum(1 for t in set_b_4096.values() if t.get("answer_text", "").strip())
    c6_all = sum(1 for t in set_b_6144.values() if t.get("answer_text", "").strip())
    n4_all, n6_all = len(set_b_4096), len(set_b_6144)
    r4_all = (c4_all / n4_all * 100) if n4_all > 0 else 0
    r6_all = (c6_all / n6_all * 100) if n6_all > 0 else 0
    ct_all = c4_all + c6_all
    nt_all = n4_all + n6_all
    rt_all = (ct_all / nt_all * 100) if nt_all > 0 else 0
    add(f"| **Overall** | {c4_all}/{n4_all} ({r4_all:.0f}%) | {c6_all}/{n6_all} ({r6_all:.0f}%) "
        f"| {ct_all}/{nt_all} ({rt_all:.0f}%) |")
    add()

    # 3b. Answer Previews for Extended Traces
    add("### 3b. Answer Previews for Extended Traces (Set B 6144)")
    add()
    for task_id in [1, 2, 3, 4]:
        g6 = {k: t for k, t in set_b_6144.items() if k[1] == task_id}
        answered = {k: t for k, t in g6.items() if t.get("answer_text", "").strip()}
        completed_count = len(answered)
        add(f"**Task {task_id}** ({completed_count}/{len(g6)} completed):")
        add()
        if answered:
            shown = 0
            for k in sorted(answered.keys()):
                if shown >= 5:
                    break
                t = answered[k]
                preview = t["answer_text"].strip()[:200].replace("\n", " ")
                add(f"- Trace {k[2]:03d}: `{preview}`")
                shown += 1
        else:
            add("- (no completed traces)")
        add()

    # 3c. Truncation Comparison
    add("### 3c. Truncation Comparison")
    add()
    add("| Task | Set B (4096) Truncation | Set B (6144) Truncation |")
    add("|------|------------------------|------------------------|")

    for task_id in [1, 2, 3, 4]:
        g4 = {k: t for k, t in set_b_4096.items() if k[1] == task_id}
        g6 = {k: t for k, t in set_b_6144.items() if k[1] == task_id}
        t4 = sum(1 for t in g4.values() if t.get("truncated"))
        t6 = sum(1 for t in g6.values() if t.get("truncated"))
        n4, n6 = len(g4), len(g6)
        r4 = (t4 / n4 * 100) if n4 > 0 else 0
        r6 = (t6 / n6 * 100) if n6 > 0 else 0
        add(f"| Task {task_id} | {t4}/{n4} ({r4:.0f}%) | {t6}/{n6} ({r6:.0f}%) |")

    t4_all = sum(1 for t in set_b_4096.values() if t.get("truncated"))
    t6_all = sum(1 for t in set_b_6144.values() if t.get("truncated"))
    r4_trunc = (t4_all / n4_all * 100) if n4_all > 0 else 0
    r6_trunc = (t6_all / n6_all * 100) if n6_all > 0 else 0
    add(f"| **Overall** | {t4_all}/{n4_all} ({r4_trunc:.0f}%) "
        f"| {t6_all}/{n6_all} ({r6_trunc:.0f}%) |")
    add()

    # 3d. Token Usage in Extended Traces
    add("### 3d. Token Usage in Extended Traces")
    add()
    add("For extended traces (6144 budget) that completed (produced answer):")
    add()

    completed_6144 = [t for t in set_b_6144.values() if t.get("answer_text", "").strip()]
    if completed_6144:
        tk_vals = [t["thinking_tokens"] for t in completed_6144]
        add(f"- **N completed:** {len(completed_6144)} / {len(set_b_6144)}")
        add(f"- **Thinking tokens:** mean={statistics.mean(tk_vals):.0f}, "
            f"median={statistics.median(tk_vals):.0f}, "
            f"min={min(tk_vals)}, max={max(tk_vals)}")
        add()

        under_4096 = sum(1 for v in tk_vals if v <= 4096)
        between = sum(1 for v in tk_vals if 4097 <= v <= 6144)
        at_ceiling = sum(1 for v in tk_vals if v >= 6144)
        total_c = len(tk_vals)

        add("**Token budget usage distribution (completed traces only):**")
        add()
        add("| Range | Count | % |")
        add("|-------|-------|---|")
        add(f"| <= 4096 (would have fit in original budget) | {under_4096} | {under_4096/total_c*100:.0f}% |")
        add(f"| 4097-6143 (needed extra runway) | {between} | {between/total_c*100:.0f}% |")
        add(f"| >= 6144 (hit ceiling) | {at_ceiling} | {at_ceiling/total_c*100:.0f}% |")
        add()

        if between > 0:
            add(f"> {between}/{total_c} completed traces ({between/total_c*100:.0f}%) "
                f"used between 4097 and 6143 thinking tokens, confirming the extra runway "
                f"was necessary for completion.")
        if under_4096 > 0:
            add(f"> {under_4096}/{total_c} completed traces ({under_4096/total_c*100:.0f}%) "
                f"used <= 4096 tokens and would have fit in the original budget. "
                f"These represent natural variation in reasoning length.")
    else:
        add("(No completed extended traces found)")
    add()

    # 3e. Study 3 Readiness Assessment
    add("### 3e. Study 3 Readiness Assessment")
    add()
    add("Study 3 (resampling) requires Set B traces with final answers.")
    add()

    min_threshold = 10
    feasible_tasks = []
    infeasible_tasks = []

    add("| Task | Set B (4096) completed | Set B (6144) completed | Total completed | Status |")
    add("|------|----------------------|----------------------|----------------|--------|")

    for task_id in [1, 2, 3, 4]:
        g4 = {k: t for k, t in set_b_4096.items() if k[1] == task_id}
        g6 = {k: t for k, t in set_b_6144.items() if k[1] == task_id}
        c4 = sum(1 for t in g4.values() if t.get("answer_text", "").strip())
        c6 = sum(1 for t in g6.values() if t.get("answer_text", "").strip())
        total_c = c4 + c6
        status = "OK" if total_c >= min_threshold else "INSUFFICIENT"
        if total_c >= min_threshold:
            feasible_tasks.append(task_id)
        else:
            infeasible_tasks.append(task_id)
        add(f"| Task {task_id} | {c4} | {c6} | {total_c} | {status} |")

    add()
    if infeasible_tasks:
        task_list = ", ".join(f"Task {t}" for t in infeasible_tasks)
        add(f"> **Study 3 has insufficient completed traces for {task_list}.** "
            f"Minimum threshold: {min_threshold} per task.")
        issues.append(("WARN", f"Study 3 insufficient traces for {task_list}"))
    else:
        add(f"> **Study 3 is feasible for all tasks** "
            f"(Tasks {', '.join(str(t) for t in feasible_tasks)}). "
            f"Each has >= {min_threshold} completed traces.")
    add()

    # ── Section 4: Confound Analysis ──
    add("---")
    add("## Section 4: Confound Analysis")
    add()
    add("This section assesses whether the 4096 and 6144 token-budget Set B traces "
        "can be legitimately combined for downstream analysis.")
    add()

    # 4a. Matched Comparison: First 4096 Tokens
    add("### 4a. Matched Comparison: First 4096 Tokens")
    add()
    add("Comparing reasoning characteristics between Set B (4096) and Set B (6144) traces.")
    add()

    # Compute sentence-level statistics for both groups
    def compute_text_stats(text):
        """Compute sentence count, avg sentence length in chars, avg words per sentence."""
        if not text or not text.strip():
            return 0, 0, 0
        sents = SENTENCE_RE.split(text.strip())
        sents = [s for s in sents if s.strip()]
        if not sents:
            return 0, 0, 0
        avg_chars = statistics.mean([len(s) for s in sents])
        avg_words = statistics.mean([len(s.split()) for s in sents])
        return len(sents), avg_chars, avg_words

    add("| Metric | Set B (4096) | Set B (6144) |")
    add("|--------|-------------|-------------|")

    # For 4096 traces: use full thinking text
    stats_4096_sents = []
    stats_4096_avg_chars = []
    stats_4096_avg_words = []
    for t in set_b_4096.values():
        ns, ac, aw = compute_text_stats(t.get("thinking_text", ""))
        if ns > 0:
            stats_4096_sents.append(ns)
            stats_4096_avg_chars.append(ac)
            stats_4096_avg_words.append(aw)

    # For 6144 traces: use first 4096 tokens worth of text (approximate by chars)
    # Since we don't have a tokenizer, approximate by taking thinking_text up to
    # proportion (4096 / thinking_tokens) of the text length
    stats_6144_sents = []
    stats_6144_avg_chars = []
    stats_6144_avg_words = []
    for t in set_b_6144.values():
        thinking = t.get("thinking_text", "")
        tk = t.get("thinking_tokens", 0)
        if tk > 4096 and thinking:
            # Approximate: take first (4096/tk) fraction of the text
            cutoff = int(len(thinking) * (4096 / tk))
            truncated_text = thinking[:cutoff]
        else:
            truncated_text = thinking
        ns, ac, aw = compute_text_stats(truncated_text)
        if ns > 0:
            stats_6144_sents.append(ns)
            stats_6144_avg_chars.append(ac)
            stats_6144_avg_words.append(aw)

    def fmt_mean_sd(vals):
        if not vals:
            return "—"
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0
        return f"{m:.1f} (SD={s:.1f})"

    add(f"| Sentence count (per 4096 tokens) | {fmt_mean_sd(stats_4096_sents)} | {fmt_mean_sd(stats_6144_sents)} |")
    add(f"| Avg sentence length (chars) | {fmt_mean_sd(stats_4096_avg_chars)} | {fmt_mean_sd(stats_6144_avg_chars)} |")
    add(f"| Avg words per sentence | {fmt_mean_sd(stats_4096_avg_words)} | {fmt_mean_sd(stats_6144_avg_words)} |")
    add()

    # 4b. Reasoning Onset Comparison
    add("### 4b. Reasoning Onset Comparison")
    add()
    add("First 500 characters of thinking blocks for 3 random traces per group per task (seed=42).")
    add()

    rng = random.Random(42)

    for task_id in [1, 2, 3, 4]:
        add(f"#### Task {task_id}")
        add()

        g4 = {k: t for k, t in set_b_4096.items() if k[1] == task_id}
        g6 = {k: t for k, t in set_b_6144.items() if k[1] == task_id}

        keys_4 = sorted(g4.keys())
        keys_6 = sorted(g6.keys())

        sample_4 = rng.sample(keys_4, min(3, len(keys_4))) if keys_4 else []
        sample_6 = rng.sample(keys_6, min(3, len(keys_6))) if keys_6 else []

        add("**Set B (4096):**")
        add()
        for k in sample_4:
            thinking = g4[k].get("thinking_text", "")[:500]
            add(f"*Trace {k[2]:03d}:*")
            add("```")
            add(thinking)
            add("```")
            add()

        add("**Set B (6144):**")
        add()
        for k in sample_6:
            thinking = g6[k].get("thinking_text", "")[:500]
            add(f"*Trace {k[2]:03d}:*")
            add("```")
            add(thinking)
            add("```")
            add()

    # 4c. Repetition Loop Comparison
    add("### 4c. Repetition Loop Comparison")
    add()
    add("Using loose detection (50+ char substring, 3+ repeats) for confound analysis.")
    add()

    rep_4096_count = 0
    rep_6144_count = 0

    for t in set_b_4096.values():
        reps = detect_repetition_loose(t.get("thinking_text", ""))
        if reps:
            rep_4096_count += 1

    for t in set_b_6144.values():
        reps = detect_repetition_loose(t.get("thinking_text", ""))
        if reps:
            rep_6144_count += 1

    r4_rep = (rep_4096_count / n4_all * 100) if n4_all > 0 else 0
    r6_rep = (rep_6144_count / n6_all * 100) if n6_all > 0 else 0

    add(f"- **Set B (4096):** {rep_4096_count}/{n4_all} traces with repetition loops ({r4_rep:.0f}%)")
    add(f"- **Set B (6144):** {rep_6144_count}/{n6_all} traces with repetition loops ({r6_rep:.0f}%)")
    add()

    rep_diff = abs(r4_rep - r6_rep)
    if rep_diff > 20:
        add(f"> **Notable difference** in repetition rates ({rep_diff:.0f} percentage points). "
            f"This may indicate the token budget affects the model's tendency to enter repetition loops.")
    else:
        add(f"> Repetition rates are {'similar' if rep_diff < 10 else 'moderately different'} "
            f"({rep_diff:.0f} pp difference).")
    add()

    # 4d. Thinking Token Distribution Overlap
    add("### 4d. Thinking Token Distribution Overlap")
    add()
    add("Distribution of thinking tokens for **completed (non-truncated)** traces only.")
    add()

    completed_4096_tokens = [t["thinking_tokens"] for t in set_b_4096.values()
                             if not t.get("truncated") and t.get("answer_text", "").strip()]
    completed_6144_tokens = [t["thinking_tokens"] for t in set_b_6144.values()
                             if not t.get("truncated") and t.get("answer_text", "").strip()]

    add(f"**Set B (4096) — completed traces** (N={len(completed_4096_tokens)}):")
    if completed_4096_tokens:
        add(f"  Mean={statistics.mean(completed_4096_tokens):.0f}, "
            f"Median={statistics.median(completed_4096_tokens):.0f}, "
            f"Range=[{min(completed_4096_tokens)}, {max(completed_4096_tokens)}]")
        add("```")
        add(text_histogram(completed_4096_tokens, bins=8))
        add("```")
    else:
        add("  (no completed non-truncated traces)")
    add()

    add(f"**Set B (6144) — completed traces** (N={len(completed_6144_tokens)}):")
    if completed_6144_tokens:
        add(f"  Mean={statistics.mean(completed_6144_tokens):.0f}, "
            f"Median={statistics.median(completed_6144_tokens):.0f}, "
            f"Range=[{min(completed_6144_tokens)}, {max(completed_6144_tokens)}]")
        add("```")
        add(text_histogram(completed_6144_tokens, bins=8))
        add("```")
    else:
        add("  (no completed non-truncated traces)")
    add()

    # Overlap analysis
    if completed_4096_tokens and completed_6144_tokens:
        # Check how many 6144-completed traces are under 4096 tokens
        under_4096_count = sum(1 for v in completed_6144_tokens if v <= 4096)
        over_4096_count = sum(1 for v in completed_6144_tokens if v > 4096)
        add(f"Of {len(completed_6144_tokens)} completed 6144-budget traces:")
        add(f"- {under_4096_count} ({under_4096_count/len(completed_6144_tokens)*100:.0f}%) "
            f"have thinking_tokens <= 4096 (would have fit in original budget)")
        add(f"- {over_4096_count} ({over_4096_count/len(completed_6144_tokens)*100:.0f}%) "
            f"have thinking_tokens > 4096 (needed the extra runway)")
    add()

    # 4e. Confound Verdict
    add("### 4e. Confound Verdict")
    add()

    # Assess based on available evidence
    confound_flags = []

    # Check sentence stat differences
    if stats_4096_avg_words and stats_6144_avg_words:
        mean_diff_words = abs(statistics.mean(stats_4096_avg_words) - statistics.mean(stats_6144_avg_words))
        if mean_diff_words > 5:
            confound_flags.append("Large difference in average words per sentence")
        mean_diff_chars = abs(statistics.mean(stats_4096_avg_chars) - statistics.mean(stats_6144_avg_chars))
        if mean_diff_chars > 30:
            confound_flags.append("Large difference in average sentence length")

    if rep_diff > 30:
        confound_flags.append("Large difference in repetition rates")

    # Check if completed 6144 traces mostly fall under 4096
    if completed_6144_tokens:
        pct_under = under_4096_count / len(completed_6144_tokens) * 100 if completed_6144_tokens else 0
        # If most completed 6144 traces are short, groups are very similar
        if pct_under > 80:
            pass  # Good sign: groups are similar

    if len(confound_flags) == 0:
        verdict_label = "COMPARABLE"
        add(f"**{verdict_label}:** The two groups show no meaningful differences in reasoning style, "
            f"onset, or structure. They can be combined for downstream analysis.")
    elif len(confound_flags) <= 1:
        verdict_label = "MINOR DIFFERENCES"
        add(f"**{verdict_label}:** Some differences noted but unlikely to affect taxonomy coding "
            f"or activation extraction. Can be combined with a covariate for token budget.")
        for flag in confound_flags:
            add(f"- {flag}")
    else:
        verdict_label = "CONFOUNDED"
        add(f"**{verdict_label}:** Substantial differences detected. The two groups should be "
            f"analyzed separately.")
        for flag in confound_flags:
            add(f"- {flag}")
    add()

    # ── Section 5: Pathological Trace Detection ──
    add("---")
    add("## Section 5: Pathological Trace Detection")
    add()

    flagged = defaultdict(list)  # group_name -> [(trace_label, problem)]

    for key, t in traces.items():
        thinking = t.get("thinking_text", "")
        tokens = t.get("thinking_tokens", 0)
        trace_label = f"set_{key[0].lower()}/task{key[1]}/trace_{key[2]:03d}.json"
        gname = classify_group(t)

        # 1. Empty output
        if not thinking or not thinking.strip():
            flagged[gname].append((trace_label, "EMPTY OUTPUT"))
            continue

        # 2. Repetition loops (strict: 80 chars, 5 repeats)
        reps = detect_repetition(thinking)
        if reps:
            for substr, count in reps[:2]:
                flagged[gname].append((trace_label, f"REPETITION LOOP: \"{substr[:60]}...\" appears {count}x"))

        # 3. Non-English
        non_eng_frac = check_non_english(thinking)
        if non_eng_frac > 0.20:
            flagged[gname].append((trace_label, f"NON-ENGLISH: {non_eng_frac:.1%} non-ASCII characters"))

        # 4. Extremely short
        if tokens < 50:
            flagged[gname].append((trace_label, f"EXTREMELY SHORT: {tokens} thinking tokens"))

    # 5. Duplicate thinking text (within same task/set group)
    for gname in GROUP_NAMES:
        for task_id in [1, 2, 3, 4]:
            group = {k: t for k, t in groups[gname].items() if k[1] == task_id}
            texts = {}
            for k, t in group.items():
                text = t.get("thinking_text", "")
                if text in texts:
                    other = texts[text]
                    label1 = f"set_{k[0].lower()}/task{k[1]}/trace_{k[2]:03d}.json"
                    label2 = f"set_{other[0].lower()}/task{other[1]}/trace_{other[2]:03d}.json"
                    flagged[gname].append((label1, f"DUPLICATE of {label2}"))
                else:
                    texts[text] = k

    total_flagged = sum(len(v) for v in flagged.values())

    for gname in GROUP_NAMES:
        gf = flagged.get(gname, [])
        add(f"**{gname}:** {len(gf)} issue(s)")
        if gf:
            for label, problem in sorted(gf):
                add(f"- `{label}`: {problem}")
        add()

    if total_flagged > 0:
        # Count by type across all groups
        all_flagged = []
        for v in flagged.values():
            all_flagged.extend(v)
        rep_count = sum(1 for _, p in all_flagged if "REPETITION" in p)
        empty_count = sum(1 for _, p in all_flagged if "EMPTY" in p)
        short_count = sum(1 for _, p in all_flagged if "SHORT" in p)
        dup_count = sum(1 for _, p in all_flagged if "DUPLICATE" in p)
        non_eng_count = sum(1 for _, p in all_flagged if "NON-ENGLISH" in p)

        if rep_count > 0:
            issues.append(("WARN", f"{rep_count} trace(s) with repetition loops"))
        if empty_count > 0:
            issues.append(("CRITICAL", f"{empty_count} trace(s) with empty output"))
        if short_count > 0:
            issues.append(("WARN", f"{short_count} trace(s) extremely short (<50 tokens)"))
        if dup_count > 0:
            issues.append(("CRITICAL", f"{dup_count} duplicate trace(s) — seeding problem"))
        if non_eng_count > 0:
            issues.append(("WARN", f"{non_eng_count} trace(s) with non-English output"))
    else:
        add("**No pathological traces detected.**")
    add()

    # ── Section 6: Cross-Trace Consistency ──
    add("---")
    add("## Section 6: Cross-Trace Consistency")
    add()

    # Seed verification
    add("### Seed Verification")
    add()
    seeds = {}
    seed_issues = []
    for key, t in traces.items():
        seed = t.get("seed")
        set_name, task_id, trace_id = key

        # Expected seed pattern
        if set_name == "A":
            expected_seed = task_id * 1000 + trace_id
        else:
            expected_seed = task_id * 1000 + 500 + trace_id

        if seed != expected_seed:
            seed_issues.append(
                f"`set_{set_name.lower()}/task{task_id}/trace_{trace_id:03d}.json`: "
                f"seed={seed}, expected={expected_seed}"
            )

        # Check uniqueness
        if seed in seeds:
            other = seeds[seed]
            seed_issues.append(
                f"Duplicate seed {seed}: `set_{key[0].lower()}/task{key[1]}/trace_{key[2]:03d}.json` "
                f"and `set_{other[0].lower()}/task{other[1]}/trace_{other[2]:03d}.json`"
            )
        seeds[seed] = key

    if seed_issues:
        add(f"**{len(seed_issues)} seed issue(s):**")
        for issue in seed_issues:
            add(f"- {issue}")
        issues.append(("CRITICAL", f"{len(seed_issues)} seed verification failure(s)"))
    else:
        add(f"**All {len(traces)} seeds verified:** unique and follow expected pattern.")
    add()

    # Parameter consistency within groups
    add("### Parameter Consistency")
    add()
    param_issues = []
    model_counter = Counter()
    quant_counter = Counter()

    expected_params_by_group = {
        "Set A": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 4096},
        "Set B (4096)": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 4096},
        "Set B (6144)": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 6144},
    }

    for gname in GROUP_NAMES:
        expected_params = expected_params_by_group[gname]
        for key, t in groups[gname].items():
            gp = t.get("generation_params", {})
            label = f"set_{key[0].lower()}/task{key[1]}/trace_{key[2]:03d}.json"
            for param, expected_val in expected_params.items():
                actual = gp.get(param)
                if actual != expected_val:
                    param_issues.append(f"`{label}`: {param}={actual} (expected {expected_val})")

    for t in traces.values():
        model_counter[t.get("model", "?")] += 1
        quant_counter[t.get("quantization", "?")] += 1

    if param_issues:
        add(f"**{len(param_issues)} parameter deviation(s):**")
        for issue in param_issues[:20]:
            add(f"- {issue}")
        if len(param_issues) > 20:
            add(f"- ... and {len(param_issues) - 20} more")
        issues.append(("CRITICAL", "Generation parameter inconsistencies found"))
    else:
        add("**Generation parameters:** All traces match expected values within their group.")
        add()
        for gname in GROUP_NAMES:
            ep = expected_params_by_group[gname]
            add(f"- {gname}: temp={ep['temperature']}, top_p={ep['top_p']}, "
                f"max_new_tokens={ep['max_new_tokens']}")
    add()

    add(f"**Model field:** {dict(model_counter)}")
    add(f"**Quantization field:** {dict(quant_counter)}")
    if len(model_counter) > 1 or (model_counter and EXPECTED_MODEL not in model_counter):
        issues.append(("CRITICAL", "Inconsistent model field across traces"))
    if len(quant_counter) > 1 or (quant_counter and EXPECTED_QUANTIZATION not in quant_counter):
        issues.append(("CRITICAL", "Inconsistent quantization field across traces"))
    add()

    # Prompt consistency
    add("### Prompt Consistency")
    add()
    prompt_issues = []

    # Set A: all traces within same task should have same prompt
    for task_id in [1, 2, 3, 4]:
        group = {k: t for k, t in groups["Set A"].items() if k[1] == task_id}
        if group:
            prompts = set(t.get("prompt", "") for t in group.values())
            if len(prompts) > 1:
                prompt_issues.append(f"Set A Task {task_id}: {len(prompts)} distinct prompts")

    # Set B: both 4096 and 6144 traces for same task should share the same prompt
    for task_id in [1, 2, 3, 4]:
        all_b_prompts = set()
        for gname in ["Set B (4096)", "Set B (6144)"]:
            group = {k: t for k, t in groups[gname].items() if k[1] == task_id}
            for t in group.values():
                all_b_prompts.add(t.get("prompt", ""))
        if len(all_b_prompts) > 1:
            prompt_issues.append(
                f"Set B Task {task_id}: {len(all_b_prompts)} distinct prompts "
                f"across 4096 and 6144 groups (should be identical)"
            )

    if prompt_issues:
        add(f"**{len(prompt_issues)} prompt consistency issue(s):**")
        for issue in prompt_issues:
            add(f"- {issue}")
        issues.append(("CRITICAL", "Prompt inconsistencies found"))
    else:
        add("**All prompts consistent** within each task/set group. "
            "Set B (4096) and Set B (6144) use identical prompts per task.")
    add()

    # ── Section 7: Distribution Visualizations ──
    add("---")
    add("## Section 7: Distribution Visualizations")
    add()

    # Thinking token distribution
    add("### Thinking Token Distribution")
    add()

    for gname in GROUP_NAMES:
        vals = [t["thinking_tokens"] for t in groups[gname].values()]
        if vals:
            add(f"**{gname}** (N={len(vals)}):")
            add("```")
            add(text_histogram(vals, bins=10))
            add("```")
            add()

    # Generation time distribution
    add("### Generation Time Distribution (seconds)")
    add()

    for gname in GROUP_NAMES:
        vals = [t["generation_time_seconds"] for t in groups[gname].values()]
        if vals:
            add(f"**{gname}** (N={len(vals)}):")
            add("```")
            add(text_histogram(vals, bins=10))
            add("```")
            add()

    # ── Section 8: Overall Verdict ──
    add("---")
    add("## Section 8: Overall Verdict")
    add()

    criticals = [msg for sev, msg in issues if sev == "CRITICAL"]
    warns = [msg for sev, msg in issues if sev == "WARN"]

    # Q1: Corpus completeness
    add("### 1. Corpus Completeness")
    if n_missing == 0 and n_corrupted == 0:
        add(f"All {TOTAL_EXPECTED} traces present and valid.")
    else:
        add(f"{n_present}/{TOTAL_EXPECTED} traces present "
            f"({n_missing} missing, {n_corrupted} corrupted).")
    add()

    # Q2: Set B completion
    add("### 2. Set B Completion (Study 3 Readiness)")
    if infeasible_tasks:
        task_list = ", ".join(f"Task {t}" for t in infeasible_tasks)
        add(f"INSUFFICIENT for {task_list}.")
    else:
        add("Sufficient completed traces for all tasks.")
    add()

    # Q3: Combinability
    add("### 3. Combinability of 4096 and 6144 Traces")
    add(f"Confound verdict: **{verdict_label}**")
    add()

    # Final verdict
    add("---")
    add()

    if criticals:
        verdict = "FAIL"
        exit_code = 1
        add(f"### **{verdict}**")
        add()
        add("**Critical issues:**")
        for msg in criticals:
            add(f"- {msg}")
        if warns:
            add()
            add("**Warnings:**")
            for msg in warns:
                add(f"- {msg}")
    elif warns:
        verdict = "WARN"
        exit_code = 0
        add(f"### **{verdict}**")
        add()
        add("**Warnings:**")
        for msg in warns:
            add(f"- {msg}")
    else:
        verdict = "PASS"
        exit_code = 0
        add(f"### **{verdict}**")
        add()
        add("All checks passed. The trace corpus is complete and consistent.")

    add()
    add("---")
    add(f"*Report generated by `study1_corpus/scripts/study1_qa_traces.py` on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    return "\n".join(lines), exit_code


def main():
    args = parse_args()
    trace_dir = args.trace_dir
    output_path = args.output

    print("=" * 60)
    print("Trace Corpus QA")
    print("=" * 60)
    print()

    # Load traces
    traces, missing, corrupted = load_all_traces(trace_dir)

    # Generate report
    report, exit_code = generate_report(trace_dir, traces, missing, corrupted)

    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print()
    print(f"Report saved to: {output_path}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

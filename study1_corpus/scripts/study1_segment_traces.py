"""
Sentence Segmentation Script for CoT Faithfulness Study

Segments each trace's thinking_text into sentences using spaCy,
counts tokens per sentence using the model tokenizer, and produces
augmented trace JSONs plus a segmentation quality report.

"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import time
import random
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Defaults ──
DEFAULT_INPUT_DIR = str(PROJECT_ROOT / "outputs" / "traces")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "traces_segmented")
DEFAULT_REPORT = str(PROJECT_ROOT / "outputs" / "traces_segmented" / "segmentation_report.md")

MODEL_PATH = str(PROJECT_ROOT / "models" / "DeepSeek-R1-Distill-Llama-8B")

# Corpus structure (same as study1_qa_traces.py)
EXPECTED_SET_A = {"traces_per_task": 40, "tasks": [1, 2, 3, 4]}
EXPECTED_SET_B_4096 = {"traces_per_task": 10, "tasks": [1, 2, 3, 4]}
EXPECTED_SET_B_6144 = {"traces_per_task": 30, "tasks": [1, 2, 3, 4]}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment trace thinking_text into sentences"
    )
    parser.add_argument(
        "--input-dir", type=str, default=DEFAULT_INPUT_DIR,
        help=f"Input trace directory (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--report", type=str, default=DEFAULT_REPORT,
        help=f"Report output path (default: {DEFAULT_REPORT})"
    )
    return parser.parse_args()


def classify_group(trace_data):
    """Classify a trace into its group based on generation_params.max_new_tokens."""
    set_name = trace_data.get("set", "")
    max_tokens = trace_data.get("generation_params", {}).get("max_new_tokens")
    if set_name == "A":
        return "Set A"
    elif set_name == "B":
        if max_tokens == 4096:
            return "Set B (4096)"
        elif max_tokens == 6144:
            return "Set B (6144)"
    return "UNEXPECTED"


def load_spacy():
    """Load spaCy model, downloading if needed."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        print("spaCy model not found, downloading en_core_web_sm...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True
        )
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 2_000_000
    return nlp


def load_tokenizer():
    """Load the model tokenizer (no GPU needed)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_PATH)


def discover_traces(input_dir):
    """Find all trace files in the corpus. Returns list of (set_name, task_id, trace_id, filepath)."""
    traces = []

    # Set A: task 1-4, traces 1-40
    for task_id in EXPECTED_SET_A["tasks"]:
        for trace_id in range(1, EXPECTED_SET_A["traces_per_task"] + 1):
            fp = os.path.join(input_dir, "set_a", f"task{task_id}", f"trace_{trace_id:03d}.json")
            if os.path.exists(fp):
                traces.append(("A", task_id, trace_id, fp))

    # Set B: task 1-4, traces 1-40
    for task_id in EXPECTED_SET_B_4096["tasks"]:
        max_trace = max(EXPECTED_SET_B_4096["traces_per_task"],
                        EXPECTED_SET_B_6144["traces_per_task"] + EXPECTED_SET_B_4096["traces_per_task"])
        for trace_id in range(1, max_trace + 1):
            fp = os.path.join(input_dir, "set_b", f"task{task_id}", f"trace_{trace_id:03d}.json")
            if os.path.exists(fp):
                traces.append(("B", task_id, trace_id, fp))

    return traces


def segment_thinking(nlp, tokenizer, thinking_text):
    """Segment thinking_text into sentences with token counts.

    Returns list of sentence dicts.
    """
    if not thinking_text or not thinking_text.strip():
        return []

    doc = nlp(thinking_text)
    sentences = []

    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        sentences.append({
            "sentence_id": len(sentences),
            "text": text,
            "start_char": sent.start_char,
            "end_char": sent.end_char,
            "token_count": token_count,
        })

    return sentences


def generate_report(all_results, report_path):
    """Generate the segmentation quality report."""
    lines = []
    add = lines.append

    add("# Sentence Segmentation Report")
    add("")
    add(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"Traces processed: {len(all_results)}")
    add("")

    # Collect stats by group and task
    group_task_stats = defaultdict(list)  # (group, task_id) -> [sentence_count, ...]
    group_stats = defaultdict(list)       # group -> [sentence_count, ...]
    all_token_counts = []                 # all sentence token counts
    all_sentence_counts = []

    flagged_short = []   # (trace_key, sentence)
    flagged_long = []    # (trace_key, sentence)
    empty_traces = []
    spot_check_traces = defaultdict(list)  # task_id -> [result, ...]

    for r in all_results:
        group = r["group"]
        task_id = r["task_id"]
        trace_id = r["trace_id"]
        n_sents = len(r["sentences"])
        trace_key = f"{group}/Task{task_id}/trace_{trace_id:03d}"

        all_sentence_counts.append(n_sents)
        group_task_stats[(group, task_id)].append(n_sents)
        group_stats[group].append(n_sents)

        if n_sents == 0:
            empty_traces.append(trace_key)
            continue

        for s in r["sentences"]:
            all_token_counts.append(s["token_count"])
            if s["token_count"] < 3:
                flagged_short.append((trace_key, s))
            if s["token_count"] > 100:
                flagged_long.append((trace_key, s))

        spot_check_traces[task_id].append(r)

    total_sentences = sum(all_sentence_counts)

    # ── Section 5a: Sentence Count Statistics ──
    add("## 5a. Sentence Count Statistics")
    add("")
    add("| Group | Mean | Median | SD | Min | Max | Total Sentences | N Traces |")
    add("|-------|------|--------|----|-----|-----|----------------|----------|")

    group_order = ["Set A", "Set B (4096)", "Set B (6144)"]

    for group in group_order:
        for task_id in [1, 2, 3, 4]:
            vals = group_task_stats.get((group, task_id), [])
            if not vals:
                continue
            label = f"{group} Task {task_id}"
            mean = statistics.mean(vals)
            median = statistics.median(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
            add(f"| {label} | {mean:.1f} | {median:.1f} | {sd:.1f} | {min(vals)} | {max(vals)} | {sum(vals)} | {len(vals)} |")

        # Group overall
        vals = group_stats.get(group, [])
        if vals:
            mean = statistics.mean(vals)
            median = statistics.median(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
            add(f"| **{group} Overall** | **{mean:.1f}** | **{median:.1f}** | **{sd:.1f}** | **{min(vals)}** | **{max(vals)}** | **{sum(vals)}** | **{len(vals)}** |")

    # Corpus overall
    if all_sentence_counts:
        mean = statistics.mean(all_sentence_counts)
        median = statistics.median(all_sentence_counts)
        sd = statistics.stdev(all_sentence_counts) if len(all_sentence_counts) > 1 else 0
        add(f"| **Corpus Overall** | **{mean:.1f}** | **{median:.1f}** | **{sd:.1f}** | **{min(all_sentence_counts)}** | **{max(all_sentence_counts)}** | **{total_sentences}** | **{len(all_sentence_counts)}** |")

    add("")

    # ── Section 5b: Sentence Length Distribution (tokens) ──
    add("## 5b. Sentence Length Distribution (tokens)")
    add("")

    if all_token_counts:
        # Build histogram bins
        bins = [
            (1, 5), (6, 10), (11, 20), (21, 30), (31, 50),
            (51, 75), (76, 100), (101, 150), (151, 200), (201, None)
        ]
        bin_counts = []
        for lo, hi in bins:
            if hi is None:
                count = sum(1 for t in all_token_counts if t >= lo)
            else:
                count = sum(1 for t in all_token_counts if lo <= t <= hi)
            bin_counts.append(count)

        max_count = max(bin_counts) if bin_counts else 1
        bar_width = 40

        add("```")
        for (lo, hi), count in zip(bins, bin_counts):
            if hi is None:
                label = f"{lo:>4}+  "
            else:
                label = f"{lo:>4}-{hi:<4}"
            bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
            bar = "#" * bar_len
            add(f"  {label}: {bar} ({count})")
        add("```")
        add("")

        mean_tk = statistics.mean(all_token_counts)
        median_tk = statistics.median(all_token_counts)
        sd_tk = statistics.stdev(all_token_counts) if len(all_token_counts) > 1 else 0
        add(f"- Mean: {mean_tk:.1f} tokens")
        add(f"- Median: {median_tk:.1f} tokens")
        add(f"- SD: {sd_tk:.1f} tokens")
        add(f"- Min: {min(all_token_counts)} tokens")
        add(f"- Max: {max(all_token_counts)} tokens")
        add(f"- Total sentences: {len(all_token_counts)}")
        add("")

    # ── Section 5c: Flagged Sentences ──
    add("## 5c. Flagged Sentences")
    add("")

    # Very short
    add(f"### Very short (< 3 tokens): {len(flagged_short)} total ({len(flagged_short)/max(len(all_token_counts),1)*100:.1f}% of corpus)")
    add("")
    if flagged_short:
        shown = flagged_short[:20]
        add("| Trace | Sentence ID | Text | Tokens |")
        add("|-------|-------------|------|--------|")
        for trace_key, s in shown:
            text_escaped = s["text"].replace("|", "\\|").replace("\n", " ")
            add(f"| {trace_key} | {s['sentence_id']} | {text_escaped} | {s['token_count']} |")
        if len(flagged_short) > 20:
            add(f"| ... | ... | ({len(flagged_short) - 20} more) | ... |")
        add("")
    else:
        add("None found.")
        add("")

    # Very long
    add(f"### Very long (> 100 tokens): {len(flagged_long)} total ({len(flagged_long)/max(len(all_token_counts),1)*100:.1f}% of corpus)")
    add("")
    if flagged_long:
        shown = flagged_long[:20]
        add("| Trace | Sentence ID | First 200 chars | Tokens |")
        add("|-------|-------------|-----------------|--------|")
        for trace_key, s in shown:
            preview = s["text"][:200].replace("|", "\\|").replace("\n", " ")
            add(f"| {trace_key} | {s['sentence_id']} | {preview} | {s['token_count']} |")
        if len(flagged_long) > 20:
            add(f"| ... | ... | ({len(flagged_long) - 20} more) | ... |")
        add("")
    else:
        add("None found.")
        add("")

    # ── Section 5d: Empty Thinking Text ──
    add("## 5d. Empty Thinking Text")
    add("")
    if empty_traces:
        add(f"**{len(empty_traces)} traces with empty thinking_text:**")
        add("")
        for t in empty_traces:
            add(f"- {t}")
    else:
        add("No traces with empty thinking_text.")
    add("")

    # ── Section 5e: Spot-Check ──
    add("## 5e. Segmentation Quality Spot-Check")
    add("")
    add("2 random traces per task (seed=42):")
    add("")

    rng = random.Random(42)
    for task_id in [1, 2, 3, 4]:
        candidates = spot_check_traces.get(task_id, [])
        if len(candidates) < 2:
            sample = candidates
        else:
            sample = rng.sample(candidates, 2)

        for r in sample:
            group = r["group"]
            trace_id = r["trace_id"]
            sents = r["sentences"]
            add(f"### {group}/Task{task_id}/trace_{trace_id:03d} ({len(sents)} sentences)")
            add("")

            if len(sents) >= 5:
                add("**First 5 sentences:**")
                add("")
                for s in sents[:5]:
                    text_preview = s["text"][:120].replace("\n", " ")
                    add(f"- [S{s['sentence_id']}] ({s['token_count']} tok) {text_preview}")
                add("")

                add("**Last 3 sentences:**")
                add("")
                for s in sents[-3:]:
                    text_preview = s["text"][:120].replace("\n", " ")
                    add(f"- [S{s['sentence_id']}] ({s['token_count']} tok) {text_preview}")
                add("")
            else:
                add("**All sentences:**")
                add("")
                for s in sents:
                    text_preview = s["text"][:120].replace("\n", " ")
                    add(f"- [S{s['sentence_id']}] ({s['token_count']} tok) {text_preview}")
                add("")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_path


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    report_path = args.report

    # Discover traces
    traces = discover_traces(input_dir)
    print(f"Found {len(traces)} traces in {input_dir}")
    print(f"Output directory: {output_dir}")

    if not traces:
        print("No traces found. Check --input-dir.")
        sys.exit(1)

    # Create output directories
    for set_name in ["set_a", "set_b"]:
        for task_id in [1, 2, 3, 4]:
            os.makedirs(os.path.join(output_dir, set_name, f"task{task_id}"), exist_ok=True)

    # Load spaCy and tokenizer
    print("Loading spaCy model...")
    nlp = load_spacy()
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("Ready.\n")

    # Process traces
    all_results = []
    t_start = time.time()

    for i, (set_name, task_id, trace_id, filepath) in enumerate(traces):
        # Load trace
        with open(filepath, "r", encoding="utf-8") as f:
            trace_data = json.load(f)

        thinking_text = trace_data.get("thinking_text", "")
        group = classify_group(trace_data)

        # Segment
        sentences = segment_thinking(nlp, tokenizer, thinking_text)

        # Augment trace (preserve all original fields)
        trace_data["sentences"] = sentences
        trace_data["segmentation_metadata"] = {
            "total_sentences": len(sentences),
            "segmentation_method": "spacy_en_core_web_sm",
            "tokenizer": "DeepSeek-R1-Distill-Llama-8B",
        }

        # Save to output dir
        set_dir = "set_a" if set_name == "A" else "set_b"
        out_path = os.path.join(output_dir, set_dir, f"task{task_id}", f"trace_{trace_id:03d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

        # Collect for report
        all_results.append({
            "group": group,
            "task_id": task_id,
            "trace_id": trace_id,
            "sentences": sentences,
        })

        # Progress logging
        if (i + 1) % 20 == 0 or (i + 1) == len(traces):
            elapsed = time.time() - t_start
            print(f"  Processed {i + 1}/{len(traces)} traces ({elapsed:.1f}s elapsed)")

    total_time = time.time() - t_start
    total_sentences = sum(len(r["sentences"]) for r in all_results)
    print(f"\nDone! {len(all_results)} traces processed, {total_sentences} sentences total, {total_time:.1f}s")

    # Generate report
    print(f"Generating report...")
    rpath = generate_report(all_results, report_path)
    print(f"Report saved to {rpath}")


if __name__ == "__main__":
    main()

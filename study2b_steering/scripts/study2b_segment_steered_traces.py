#!/usr/bin/env python3
"""
Segment and clean steered traces for Study 2.

Combines spaCy sentence segmentation with the same cleanup pipeline used for
Study 1 (fragment merging + truncated-final removal).  Writes output as
trace_NNN_segmented.json alongside the original file — no files are overwritten.

Re-runnable: if the _segmented.json output already exists, the trace is skipped.

Usage:
    python study2b_steering/scripts/study2b_segment_steered_traces.py [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import io
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STEERED_DIR = PROJECT_ROOT / "outputs" / "steered_traces"
MODEL_PATH  = PROJECT_ROOT / "models" / "DeepSeek-R1-Distill-Llama-8B"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fragment pattern — same as preprocess_segmented_traces.py
FRAGMENT_RE = re.compile(r"^([A-F0-9]\.?|\.)$")

# Minimum sentences to attempt fragment merging
MIN_SENTENCES_FOR_FRAGMENT_MERGE = 5

# Sentences above this token count are never treated as truncated finals
MAX_TRUNCATED_TOKENS = 50

# Sentences above this token count trigger newline-based fallback splitting
OVERSIZED_TOKEN_THRESHOLD = 200


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_spacy():
    """Load spaCy model for sentence boundary detection."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        print("Downloading en_core_web_sm...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
        )
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 2_000_000
    return nlp


def load_tokenizer():
    """Load the DeepSeek tokenizer (CPU only)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL_PATH))


def is_fragment(s: dict) -> bool:
    return s["token_count"] <= 2 and bool(FRAGMENT_RE.match(s["text"].strip()))


def has_final_answer(trace: dict) -> bool:
    return bool(trace.get("answer_text", "").strip())


def is_truncated_final(s: dict) -> bool:
    if s["token_count"] > MAX_TRUNCATED_TOKENS:
        return False
    text = s["text"].rstrip()
    return s["token_count"] <= 5 or not text.endswith((".", "!", "?", '"', ")"))


def output_path_for(trace_path: Path) -> Path:
    """Return the _segmented.json path alongside the original."""
    return trace_path.with_name(trace_path.stem + "_segmented.json")


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_thinking(nlp, tokenizer, thinking_text: str) -> list[dict]:
    """Segment thinking_text into sentence dicts."""
    if not thinking_text or not thinking_text.strip():
        return []

    # Strip residual <think></think> tags
    cleaned = re.sub(r"</?think>", "", thinking_text).strip()
    if not cleaned:
        return []

    doc = nlp(cleaned)
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


# ---------------------------------------------------------------------------
# Oversized sentence splitting
# ---------------------------------------------------------------------------

def split_oversized_sentences(sentences: list[dict], tokenizer) -> int:
    """Split sentences exceeding OVERSIZED_TOKEN_THRESHOLD on newlines.

    Tries double-newline first, falls back to single-newline.
    Modifies sentences list in-place. Returns count of splits performed.
    """
    n_splits = 0
    i = 0
    while i < len(sentences):
        s = sentences[i]
        if s["token_count"] <= OVERSIZED_TOKEN_THRESHOLD:
            i += 1
            continue

        text = s["text"]
        base_start = s["start_char"]

        # Pick delimiter: prefer \n\n, fall back to \n
        if "\n\n" in text:
            delimiter = "\n\n"
        elif "\n" in text:
            delimiter = "\n"
        else:
            i += 1
            continue

        parts = text.split(delimiter)

        # Build sub-sentences with cursor-based offset tracking
        sub_sentences = []
        cursor = 0
        for part in parts:
            stripped = part.strip()
            if stripped and re.search(r'[a-zA-Z0-9]', stripped):
                leading_ws = len(part) - len(part.lstrip())
                token_count = len(tokenizer.encode(stripped, add_special_tokens=False))
                sub_sentences.append({
                    "sentence_id": -1,
                    "text": stripped,
                    "start_char": base_start + cursor + leading_ws,
                    "end_char": base_start + cursor + len(part.rstrip()),
                    "token_count": token_count,
                })
            cursor += len(part) + len(delimiter)

        if len(sub_sentences) <= 1:
            i += 1
            continue

        sentences[i:i + 1] = sub_sentences
        n_splits += 1
        i += len(sub_sentences)

    return n_splits


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def merge_fragments(sentences: list[dict]) -> int:
    """Merge enumeration-artifact fragments into adjacent sentences.

    Returns the number of fragments merged.
    """
    if len(sentences) < MIN_SENTENCES_FOR_FRAGMENT_MERGE:
        return 0

    n_merged = 0
    i = 0
    while i < len(sentences):
        if not is_fragment(sentences[i]):
            i += 1
            continue

        # Collect consecutive fragments
        j = i + 1
        while j < len(sentences) and is_fragment(sentences[j]):
            j += 1

        if j < len(sentences):
            # Merge fragments i..j-1 forward into sentences[j]
            target = sentences[j]
            frag_texts  = [sentences[k]["text"] for k in range(i, j)]
            frag_tokens = sum(sentences[k]["token_count"] for k in range(i, j))
            target["text"]        = " ".join(frag_texts) + " " + target["text"]
            target["token_count"] += frag_tokens
            target["start_char"]  = sentences[i]["start_char"]
            n_merged += j - i
            del sentences[i:j]
        else:
            # Fragments at end of trace — merge backward into previous
            if i > 0:
                prev = sentences[i - 1]
                for k in range(i, len(sentences)):
                    frag = sentences[k]
                    prev["text"]        = prev["text"] + " " + frag["text"]
                    prev["end_char"]    = frag["end_char"]
                    prev["token_count"] += frag["token_count"]
                n_merged += len(sentences) - i
                del sentences[i:]
            break

    return n_merged


def remove_truncated_finals(sentences: list[dict], trace: dict) -> int:
    """Remove truncated final sentences from incomplete traces.

    Returns the number of sentences removed.
    """
    if has_final_answer(trace):
        return 0

    n_removed = 0
    max_removals = 10
    while sentences and max_removals > 0:
        if is_truncated_final(sentences[-1]):
            sentences.pop()
            n_removed += 1
            max_removals -= 1
        else:
            break

    return n_removed


def filter_non_content_sentences(sentences: list[dict]) -> int:
    """Remove sentences with no alphanumeric content, merging into neighbors.

    Returns the number of sentences removed.
    """
    n_removed = 0
    i = 0
    while i < len(sentences):
        if re.search(r'[a-zA-Z0-9]', sentences[i]["text"]):
            i += 1
            continue
        # Merge into next sentence if possible, else previous
        if i + 1 < len(sentences):
            nxt = sentences[i + 1]
            nxt["text"] = sentences[i]["text"] + " " + nxt["text"]
            nxt["start_char"] = sentences[i]["start_char"]
            nxt["token_count"] += sentences[i]["token_count"]
        elif i > 0:
            prev = sentences[i - 1]
            prev["text"] = prev["text"] + " " + sentences[i]["text"]
            prev["end_char"] = sentences[i]["end_char"]
            prev["token_count"] += sentences[i]["token_count"]
        sentences.pop(i)
        n_removed += 1
        # Don't increment i — next sentence shifted into position
    return n_removed


def renumber(sentences: list[dict]) -> None:
    """Renumber sentence_id from 0 to N-1."""
    for idx, s in enumerate(sentences):
        s["sentence_id"] = idx


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_traces(root: Path) -> list[Path]:
    """Find all trace JSON files under the steered traces directory,
    excluding *_coded.json and *_segmented.json files."""
    paths = sorted(root.glob("*/task*/trace_*.json"))
    return [
        p for p in paths
        if not p.stem.endswith("_coded") and not p.stem.endswith("_segmented")
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment and clean steered traces for Study 2."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process traces even if _segmented.json already exists",
    )
    args = parser.parse_args()

    if not STEERED_DIR.exists():
        print(f"ERROR: Steered traces directory not found: {STEERED_DIR}")
        sys.exit(1)

    all_paths = discover_traces(STEERED_DIR)
    if not all_paths:
        print(f"ERROR: No trace files found in {STEERED_DIR}")
        sys.exit(1)

    print(f"Found {len(all_paths)} trace files in {STEERED_DIR}")
    print("Loading spaCy model...")
    nlp = load_spacy()
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("Ready.\n")

    # -- Per-condition accumulators --
    from collections import defaultdict
    cond_stats = defaultdict(lambda: {
        "processed": 0, "skipped": 0,
        "sentences": 0, "fragments": 0, "truncated": 0, "oversized": 0, "non_content": 0,
    })

    total_processed = 0
    total_skipped   = 0
    total_warnings  = 0
    t_start = time.time()

    for idx, path in enumerate(all_paths, 1):
        # Parse condition / task / trace from path
        rel = path.relative_to(STEERED_DIR)
        parts = rel.parts  # e.g. ("HYPO_pos_2.0", "task1", "trace_001.json")
        condition = parts[0]
        task_dir  = parts[1] if len(parts) > 1 else "unknown"
        trace_name = path.stem

        label = f"{condition}/{task_dir}/{trace_name}"
        out_path = output_path_for(path)

        # Skip if output already exists (unless --force)
        if out_path.exists() and not args.force:
            cond_stats[condition]["skipped"] += 1
            total_skipped += 1
            continue

        # Load trace
        trace = json.loads(path.read_text(encoding="utf-8"))

        # Extract thinking text
        thinking = trace.get("thinking_text", "")
        if not thinking or not thinking.strip():
            print(f"  WARNING: Empty thinking_text in {label}")
            trace["sentences"] = []
            trace["preprocessing"] = {
                "original_sentence_count": 0,
                "cleaned_sentence_count": 0,
                "fragments_merged": 0,
                "truncated_finals_removed": 0,
            }
            out_path.write_text(
                json.dumps(trace, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            total_warnings += 1
            cond_stats[condition]["processed"] += 1
            total_processed += 1
            continue

        # Step 1: Segment
        sentences = segment_thinking(nlp, tokenizer, thinking)
        n_original = len(sentences)

        if n_original == 0:
            print(f"  WARNING: No sentences extracted from {label}")
            total_warnings += 1

        # Step 1b: Split oversized sentences on newlines
        n_oversized_splits = split_oversized_sentences(sentences, tokenizer)
        n_after_split = len(sentences)

        # Check for all-fragment edge case
        all_fragments = n_after_split > 0 and all(is_fragment(s) for s in sentences)
        if all_fragments:
            print(f"  WARNING: Entire trace is fragments in {label} -- keeping as-is")
            total_warnings += 1

        # Step 2: Fragment merging
        if all_fragments:
            n_fragments = 0
        else:
            n_fragments = merge_fragments(sentences)

        # Step 3: Truncated final removal
        n_truncated = remove_truncated_finals(sentences, trace)

        # Step 4: Filter non-content sentences
        n_non_content = filter_non_content_sentences(sentences)

        # Step 5: Renumber
        renumber(sentences)

        # Metadata
        preprocessing = {
            "original_sentence_count": n_original,
            "oversized_splits": n_oversized_splits,
            "post_split_sentence_count": n_after_split,
            "cleaned_sentence_count": len(sentences),
            "fragments_merged": n_fragments,
            "truncated_finals_removed": n_truncated,
            "non_content_filtered": n_non_content,
        }

        # Write output alongside original
        out_data = dict(trace)
        out_data["sentences"] = sentences
        out_data["preprocessing"] = preprocessing
        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Accumulate stats
        cond_stats[condition]["processed"] += 1
        cond_stats[condition]["sentences"] += len(sentences)
        cond_stats[condition]["fragments"] += n_fragments
        cond_stats[condition]["truncated"] += n_truncated
        cond_stats[condition]["oversized"] += n_oversized_splits
        cond_stats[condition]["non_content"] += n_non_content
        total_processed += 1

        # Progress
        extras = []
        if n_oversized_splits:
            extras.append(f"{n_oversized_splits} oversized split")
        if n_non_content:
            extras.append(f"{n_non_content} non-content filtered")
        extra = (", " + ", ".join(extras)) if extras else ""
        print(
            f"  [{idx}/{len(all_paths)}] {label} "
            f"-- {len(sentences)} sentences "
            f"({n_fragments} fragments merged, {n_truncated} truncated removed{extra})"
        )

    elapsed = time.time() - t_start

    # -- Condition-level summary --
    print(f"\n{'=' * 60}")
    print(f"Condition Summary")
    print(f"{'=' * 60}")
    print(f"{'Condition':<22} {'Processed':>9} {'Skipped':>8} {'Sentences':>10} {'Fragments':>10} {'Truncated':>10} {'NonContent':>10} {'Oversized':>10}")
    print(f"{'-' * 22} {'-' * 9} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    grand_sentences    = 0
    grand_fragments    = 0
    grand_truncated    = 0
    grand_oversized    = 0
    grand_non_content  = 0

    for cond in sorted(cond_stats.keys()):
        s = cond_stats[cond]
        print(
            f"{cond:<22} {s['processed']:>9} {s['skipped']:>8} "
            f"{s['sentences']:>10} {s['fragments']:>10} {s['truncated']:>10} {s['non_content']:>10} {s['oversized']:>10}"
        )
        grand_sentences   += s["sentences"]
        grand_fragments   += s["fragments"]
        grand_truncated   += s["truncated"]
        grand_oversized   += s["oversized"]
        grand_non_content += s["non_content"]

    print(f"{'-' * 22} {'-' * 9} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(
        f"{'TOTAL':<22} {total_processed:>9} {total_skipped:>8} "
        f"{grand_sentences:>10} {grand_fragments:>10} {grand_truncated:>10} {grand_non_content:>10} {grand_oversized:>10}"
    )
    print(f"\nElapsed: {elapsed:.1f}s | Warnings: {total_warnings}")


if __name__ == "__main__":
    main()

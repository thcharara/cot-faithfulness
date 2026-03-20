#!/usr/bin/env python3
"""
study1_compute_hypo_status.py — Post-hoc hypo_status classification for HYPO sentences

Classifies each HYPO sentence within a trace as:
  novel    — introduces a feature dimension not seen in prior HYPOs
  revised  — refines or constrains a prior hypothesis (similar but distinct)
  repeated — returns to a prior hypothesis without meaningful modification

Uses sentence-transformer embeddings (all-MiniLM-L6-v2) to compute
within-trace cosine similarity between each HYPO and all prior HYPOs in
the same trace. Classification is by maximum cosine similarity.

Writes coding.hypo_status (and coding.hypo_antecedent_sid for revised/repeated)
into the trace JSONs in-place.

Two modes:
  (default)   Full run — embed and classify all HYPO sentences in coded traces.
  --calibrate Calibration run — compute similarity distribution on the 10
              manually coded traces and write outputs/validation/hypo_calibration.csv.
              Use this to choose repeat/revise thresholds before the full run.

Usage:
    python study1_corpus/scripts/study1_compute_hypo_status.py [OPTIONS]

Options:
    --coded-dir         Auto-coded traces dir   (default: outputs/traces_clean_coded)
    --manual-dir        Manual coding dir        (default: data/manual_coding_final)
    --output-dir        Calibration output dir   (default: outputs/validation)
    --traces            Subset to process        (default: all)
    --model             Sentence-transformer     (default: all-MiniLM-L6-v2)
    --repeat-threshold  Cosine sim for repeated  (default: 0.85)
    --revise-threshold  Cosine sim for revised   (default: 0.65)
    --batch-size        Embedding batch size     (default: 64)
    --calibrate         Run calibration mode instead of full run
    --dry-run           Compute but do not write files
    --force             Re-compute even if hypo_status already present
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL         = "all-MiniLM-L6-v2"
REPEAT_THRESHOLD      = 0.85   # cosine sim ≥ this → repeated
REVISE_THRESHOLD      = 0.65   # cosine sim ≥ this (and < REPEAT_THRESHOLD) → revised
                                # below REVISE_THRESHOLD → novel

# Manual coding filenames and their auto-coded counterparts
# (same list used by study1_compute_kappa.py)
MANUAL_TRACE_MAP = [
    ("set_a/task1/trace_015", "setA_task1_trace15_final"),
    ("set_a/task3/trace_010", "setA_task3_trace10_final"),
    ("set_b/task3/trace_020", "setB_task3_trace20_final"),
    ("set_a/task2/trace_014", "setA_task2_trace14_final"),
    ("set_b/task1/trace_010", "setB_task1_trace10_final"),
    ("set_b/task2/trace_004", "setB_task2_trace4_final"),
    ("set_b/task2/trace_006", "setB_task2_trace6_final"),
    ("set_a/task3/trace_020", "setA_task3_trace20_final"),
    ("set_a/task4/trace_019", "setA_task4_trace19_final"),
    ("set_b/task4/trace_015", "setB_task4_trace15_final"),
]

CALIBRATION_N_SAMPLE  = 80    # HYPO pairs to print for threshold inspection


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    """Load JSON with UTF-8 BOM tolerance."""
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not decode {path}")


def discover_traces(coded_dir: Path, traces_arg: str) -> list[Path]:
    """Resolve --traces argument to list of .json paths. Same logic as study1_compute_dependencies.py."""
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
# Embedding
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> SentenceTransformer:
    """Load sentence-transformer model (auto-selects GPU if available)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{model_name}' on {device} ...")
    return SentenceTransformer(model_name, device=device)


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    """Return L2-normalised embeddings shape (N, dim)."""
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_hypos(
    hypo_sids:  list[int],
    embeddings: np.ndarray,
    repeat_threshold: float,
    revise_threshold: float,
) -> list[dict]:
    """
    For each HYPO sentence (by position), compare to all prior HYPOs.
    Returns list of dicts with keys: sid, hypo_status, hypo_antecedent_sid (or None).
    """
    results = []
    for i in range(len(hypo_sids)):
        sid = hypo_sids[i]
        if i == 0:
            results.append({"sid": sid, "hypo_status": "novel", "hypo_antecedent_sid": None})
            continue

        # Cosine similarities with all prior HYPOs (embeddings are L2-normalised)
        sims = embeddings[:i] @ embeddings[i]  # shape (i,)
        max_idx  = int(np.argmax(sims))
        max_sim  = float(sims[max_idx])
        antecedent_sid = hypo_sids[max_idx]

        if max_sim >= repeat_threshold:
            status = "repeated"
        elif max_sim >= revise_threshold:
            status = "revised"
        else:
            status = "novel"

        results.append({
            "sid":                 sid,
            "hypo_status":         status,
            "hypo_antecedent_sid": antecedent_sid if status != "novel" else None,
            "_max_sim":            max_sim,        # internal, not written to JSON
            "_antecedent_idx":     max_idx,        # internal
        })

    return results


# ---------------------------------------------------------------------------
# Per-trace processing
# ---------------------------------------------------------------------------

def process_trace(
    data:             dict,
    model:            SentenceTransformer,
    batch_size:       int,
    repeat_threshold: float,
    revise_threshold: float,
) -> tuple[list[dict], int]:
    """
    Classify all HYPO sentences in one trace.
    Returns (classification_results, n_hypos).
    classification_results: list of dicts as returned by classify_hypos.
    """
    sentences = data.get("sentences", [])
    hypos = [
        s for s in sentences
        if s.get("coding", {}).get("micro_label") == "HYPO"
    ]

    if not hypos:
        return [], 0

    sids  = [s["sentence_id"] for s in hypos]
    texts = [s.get("text", "") for s in hypos]
    embs  = embed_texts(model, texts, batch_size)
    return classify_hypos(sids, embs, repeat_threshold, revise_threshold), len(hypos)


def apply_classifications(data: dict, results: list[dict]) -> None:
    """Write hypo_status and hypo_antecedent_sid into sentence coding blocks in-place."""
    sid_to_result = {r["sid"]: r for r in results}
    for s in data.get("sentences", []):
        sid    = s.get("sentence_id")
        coding = s.get("coding", {})
        if coding.get("micro_label") != "HYPO" or sid not in sid_to_result:
            continue
        r = sid_to_result[sid]
        coding["hypo_status"] = r["hypo_status"]
        if r["hypo_antecedent_sid"] is not None:
            coding["hypo_antecedent_sid"] = r["hypo_antecedent_sid"]
        elif "hypo_antecedent_sid" in coding:
            del coding["hypo_antecedent_sid"]


# ---------------------------------------------------------------------------
# Calibration mode
# ---------------------------------------------------------------------------

def run_calibration(
    manual_dir:       Path,
    output_dir:       Path,
    model:            SentenceTransformer,
    batch_size:       int,
    repeat_threshold: float,
    revise_threshold: float,
) -> None:
    """
    Compute within-trace HYPO pairwise similarities on 10 manual traces.
    Outputs:
      outputs/validation/hypo_calibration.csv   — all pairs with similarity scores
      Prints threshold distribution table and sample pairs at key similarity bands.
    """
    print("\nCalibration mode — computing HYPO pairwise similarities on manual traces")
    print("=" * 70)

    all_pairs: list[dict] = []
    n_traces_with_hypos = 0

    for _auto_rel, manual_stem in MANUAL_TRACE_MAP:
        manual_path = manual_dir / (manual_stem + ".json")
        if not manual_path.exists():
            print(f"  SKIP (not found): {manual_path}")
            continue

        data  = load_json(manual_path)
        hypos = [
            s for s in data.get("sentences", [])
            if s.get("coding", {}).get("micro_label") == "HYPO"
        ]

        if len(hypos) < 2:
            print(f"  SKIP (<2 HYPOs): {manual_stem}")
            continue

        n_traces_with_hypos += 1
        sids  = [s["sentence_id"] for s in hypos]
        texts = [s.get("text", "") for s in hypos]
        embs  = embed_texts(model, texts, batch_size)

        print(f"  {manual_stem}: {len(hypos)} HYPOs -> "
              f"{len(hypos)*(len(hypos)-1)//2} pairs")

        # All ordered pairs (i < j)
        for i in range(len(hypos)):
            for j in range(i + 1, len(hypos)):
                sim = float(embs[i] @ embs[j])
                all_pairs.append({
                    "trace":     manual_stem,
                    "sid_a":     sids[i],
                    "sid_b":     sids[j],
                    "sim":       round(sim, 4),
                    "text_a":    texts[i][:120],
                    "text_b":    texts[j][:120],
                    "gap":       sids[j] - sids[i],
                })

    if not all_pairs:
        print("No pairs found — check manual-dir path.")
        return

    df = pd.DataFrame(all_pairs).sort_values("sim", ascending=False).reset_index(drop=True)

    # Save CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hypo_calibration.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nAll {len(df)} pairs saved to: {csv_path}")

    # --- Threshold distribution table ---
    print("\nThreshold distribution (% of pairs AT OR ABOVE each threshold):")
    print(f"  {'Threshold':>10}  {'N pairs':>8}  {'%':>7}  {'Suggested label':<16}")
    print("  " + "-" * 52)
    sims_arr = df["sim"].values
    for thresh in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
        n_above = int((sims_arr >= thresh).sum())
        pct     = n_above / len(sims_arr) * 100
        label   = ("repeated"  if thresh >= repeat_threshold else
                   "revised"   if thresh >= revise_threshold else
                   "novel")
        marker  = " <--" if thresh in (repeat_threshold, revise_threshold) else ""
        print(f"  {thresh:>10.2f}  {n_above:>8}  {pct:>6.1f}%  {label:<16}{marker}")

    # --- Percentile table ---
    print("\nSimilarity percentiles:")
    for pct in [99, 95, 90, 85, 80, 75, 70, 65, 60, 50, 25, 10]:
        val = float(np.percentile(sims_arr, pct))
        print(f"  p{pct:02d}: {val:.4f}")

    # --- Sample pairs at key similarity bands ---
    rng = np.random.default_rng(42)
    bands = [
        ("HIGH (>=0.85, repeated?)",  df[df["sim"] >= 0.85]),
        ("MID-HIGH (0.70-0.85)",      df[(df["sim"] >= 0.70) & (df["sim"] < 0.85)]),
        ("MID (0.55-0.70, revised?)", df[(df["sim"] >= 0.55) & (df["sim"] < 0.70)]),
        ("LOW (<0.55, novel?)",        df[df["sim"] < 0.55]),
    ]

    total_sampled = 0
    for band_label, band_df in bands:
        if band_df.empty:
            continue
        n_sample = min(len(band_df), max(3, CALIBRATION_N_SAMPLE // 4))
        sample   = band_df.iloc[rng.choice(len(band_df), size=n_sample, replace=False)]
        total_sampled += n_sample

        print(f"\n--- {band_label} ({len(band_df)} pairs; showing {n_sample}) ---")
        for _, row in sample.iterrows():
            print(f"  sim={row['sim']:.3f}  [{row['trace']}]  sid_a={row['sid_a']}  sid_b={row['sid_b']}  gap={row['gap']}")
            print(f"    A: {row['text_a']}")
            print(f"    B: {row['text_b']}")

    print(f"\n[Calibration complete — inspect pairs above, then set --repeat-threshold and --revise-threshold]")
    print(f"[Current defaults: repeat>={repeat_threshold}, revise>={revise_threshold}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute hypo_status (novel/revised/repeated) for all HYPO sentences"
    )
    parser.add_argument("--coded-dir",  default=str(PROJECT_ROOT / "outputs" / "traces_clean_coded"),
                        help="Auto-coded traces directory (default: outputs/traces_clean_coded)")
    parser.add_argument("--manual-dir", default=str(PROJECT_ROOT / "data" / "manual_coding_final"),
                        help="Manual coding directory for calibration (default: data/manual_coding_final)")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "validation"),
                        help="Calibration output directory (default: outputs/validation)")
    parser.add_argument("--traces",     default="all",
                        help="Subset to process; same syntax as study1_compute_dependencies.py (default: all)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help=f"Sentence-transformer model (default: {DEFAULT_MODEL})")
    parser.add_argument("--repeat-threshold", type=float, default=REPEAT_THRESHOLD,
                        help=f"Cosine sim threshold for 'repeated' (default: {REPEAT_THRESHOLD})")
    parser.add_argument("--revise-threshold", type=float, default=REVISE_THRESHOLD,
                        help=f"Cosine sim threshold for 'revised' (default: {REVISE_THRESHOLD})")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size (default: 64)")
    parser.add_argument("--calibrate",  action="store_true",
                        help="Run calibration mode (similarity distribution on manual traces)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Compute but do not write files")
    parser.add_argument("--force",      action="store_true",
                        help="Re-compute even if hypo_status already present")
    args = parser.parse_args()

    # Validate
    if args.revise_threshold >= args.repeat_threshold:
        print("ERROR: --revise-threshold must be strictly less than --repeat-threshold")
        sys.exit(1)

    model = load_model(args.model)

    # ── Calibration mode ──────────────────────────────────────────────────────
    if args.calibrate:
        manual_dir = Path(args.manual_dir)
        output_dir = Path(args.output_dir)
        if not manual_dir.exists():
            print(f"ERROR: manual-dir not found: {manual_dir}")
            sys.exit(1)
        run_calibration(
            manual_dir, output_dir, model, args.batch_size,
            args.repeat_threshold, args.revise_threshold,
        )
        return

    # ── Full run ──────────────────────────────────────────────────────────────
    coded_dir = Path(args.coded_dir)
    if not coded_dir.exists():
        print(f"ERROR: coded-dir not found: {coded_dir}")
        sys.exit(1)

    trace_paths = discover_traces(coded_dir, args.traces)
    if not trace_paths:
        print("No traces found matching --traces argument.")
        sys.exit(1)

    print(f"\nProcessing {len(trace_paths)} traces  "
          f"(repeat>={args.repeat_threshold}, revise>={args.revise_threshold}, "
          f"dry_run={args.dry_run}, force={args.force})")
    print()

    # Per-status counters
    counts: dict[str, int] = {"novel": 0, "revised": 0, "repeated": 0}
    n_processed = 0
    n_skipped   = 0
    n_no_hypos  = 0

    for trace_path in trace_paths:
        data      = load_json(trace_path)
        sentences = data.get("sentences", [])

        # Skip if already classified and not --force
        already_done = any(
            "hypo_status" in s.get("coding", {})
            for s in sentences if s.get("coding", {}).get("micro_label") == "HYPO"
        )
        if already_done and not args.force:
            n_skipped += 1
            continue

        results, n_hypos = process_trace(
            data, model, args.batch_size,
            args.repeat_threshold, args.revise_threshold,
        )

        if n_hypos == 0:
            n_no_hypos += 1
            continue

        apply_classifications(data, results)

        for r in results:
            counts[r["hypo_status"]] = counts.get(r["hypo_status"], 0) + 1

        n_processed += 1

        if not args.dry_run:
            trace_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 65)
    print("HYPO STATUS COMPUTATION SUMMARY")
    print("=" * 65)
    print(f"  Traces processed : {n_processed}")
    print(f"  Traces skipped   : {n_skipped} (already classified; use --force to re-run)")
    print(f"  Traces no HYPOs  : {n_no_hypos}")
    if args.dry_run:
        print("  [DRY RUN — no files written]")
    print()

    total = sum(counts.values())
    print(f"  {'Status':<12} {'Count':>8} {'%':>8}")
    print("  " + "-" * 32)
    for status in ("novel", "revised", "repeated"):
        n   = counts.get(status, 0)
        pct = n / total * 100 if total else 0
        print(f"  {status:<12} {n:>8} {pct:>7.1f}%")
    print("  " + "-" * 32)
    print(f"  {'TOTAL':<12} {total:>8}")
    print()

    n_with_antecedent = counts.get("revised", 0) + counts.get("repeated", 0)
    print(f"  HYPOs with antecedent (revised+repeated): {n_with_antecedent} / {total} "
          f"({n_with_antecedent/total*100:.1f}% if total>0 else '—')")
    print(f"  Thresholds used: repeat>={args.repeat_threshold}, revise>={args.revise_threshold}")
    print("=" * 65)


if __name__ == "__main__":
    main()

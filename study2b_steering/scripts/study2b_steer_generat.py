#!/usr/bin/env python3
"""Study 2b: Generate reasoning traces with steering vector interventions.

Adds a category direction vector to the residual stream at a target layer
during autoregressive generation, testing whether probe-identified linear
representations are causally linked to reasoning behaviour.

Usage:
    .venv/Scripts/python.exe study2b_steering/scripts/study2b_steer_generat.py --pilot
    .venv/Scripts/python.exe study2b_steering/scripts/study2b_steer_generat.py --baseline
    .venv/Scripts/python.exe study2b_steering/scripts/study2b_steer_generat.py --category HYPO --direction pos --alpha 1.0
    .venv/Scripts/python.exe study2b_steering/scripts/study2b_steer_generat.py --all
    .venv/Scripts/python.exe study2b_steering/scripts/study2b_steer_generat.py --category HYPO --direction pos --alpha 1.0 --venhoff
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import gc
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'DeepSeek-R1-Distill-Llama-8B'
STIMULI_DIR = PROJECT_ROOT / 'data' / 'stimuli'
ACTIVATIONS_DIR = PROJECT_ROOT / 'outputs' / 'activations'
STEERING_DIR = PROJECT_ROOT / 'outputs' / 'study2_probes'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'steered_traces'
VENHOFF_VECTORS_PATH = (
    PROJECT_ROOT / 'steering-thinking-llms' / 'train-steering-vectors'
    / 'results' / 'vars' / 'mean_vectors_deepseek-r1-distill-llama-8b.pt'
)

# ── Prompt (identical to generate_traces.py) ──
SYSTEM_PROMPT = (
    "You are given six visual panels labeled A through F. Each panel contains "
    "an arrangement of cone-shaped objects that vary in color, size, orientation, "
    "and position.\n\nPanels marked with a yellow star follow a hidden rule. "
    "Panels without a star do not follow the rule.\n\nYour task: examine all six "
    "panels and infer the hidden rule that distinguishes starred from unstarred "
    "panels. State your best guess at the end."
)

# ── Generation settings (match Study 1 exactly) ──
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_NEW_TOKENS = 4096

# ── Experimental design ──
TASKS = [1, 2, 3, 4]
TRACES_PER_TASK = 10
DEFAULT_LAYER = 20
STEER_CATEGORIES = ['HYPO', 'TEST', 'JUDGE', 'MONITOR', 'PLAN']
ALPHAS = [0.5, 1.0, 2.0]

# Venhoff category mapping (their label -> our approximate equivalent)
VENHOFF_MAP = {
    'backtracking': 'MONITOR',
    'uncertainty-estimation': 'MONITOR',
    'example-testing': 'TEST',
    'adding-knowledge': 'DESCRIBE',
    'deduction': 'JUDGE',
    'initializing': 'ORIENT',
}


def seed_for(task_id, trace_id):
    """Compute deterministic seed. Avoids overlap with Study 1 seeds."""
    return task_id * 1000 + 800 + trace_id


# ── Steering hook ──

class SteeringHook:
    """Adds a steering vector to the residual stream during generation.

    Registers a forward hook on a transformer layer that additively injects
    the (normalised, scaled) steering vector at every decoding step.
    """

    def __init__(self, steering_vector, alpha):
        """
        Args:
            steering_vector: Tensor [4096], already normalised to activation scale.
            alpha: float, steering strength (+ve = toward, -ve = away).
        """
        self.vector = steering_vector.to('cuda').half()
        self.alpha = alpha
        self.handle = None

    def hook_fn(self, module, input, output):
        # output is tuple; output[0] is hidden states [batch, seq, hidden]
        # During generation: seq=prompt_len on first step, seq=1 thereafter
        # Use list conversion to avoid tuple concatenation issues with
        # transformers output types
        if isinstance(output, tuple):
            lst = list(output)
            lst[0] = lst[0] + self.alpha * self.vector.unsqueeze(0).unsqueeze(0)
            return tuple(lst)
        elif isinstance(output, torch.Tensor):
            return output + self.alpha * self.vector.unsqueeze(0).unsqueeze(0)
        else:
            # BaseModelOutput or similar — modify first element
            output[0] = output[0] + self.alpha * self.vector.unsqueeze(0).unsqueeze(0)
            return output

    def register(self, model, layer_idx):
        layer = model.model.layers[layer_idx]
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ── Vector loading and normalisation ──

def compute_mean_activation_norm(layer_idx, n_sample=50):
    """Compute mean L2 norm of activations at a given layer.

    Samples activation .npz files and computes the mean L2 norm of
    mean-pooled sentence activations. Used to normalise steering vectors
    to match the natural activation scale (following Venhoff et al.).
    """
    layer_dir = ACTIVATIONS_DIR / f'layer_{layer_idx}'
    npz_files = sorted(layer_dir.glob('*.npz'))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No activation files in {layer_dir}")

    if len(npz_files) > n_sample:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(npz_files), n_sample, replace=False)
        npz_files = [npz_files[i] for i in indices]

    norms = []
    for f in npz_files:
        data = np.load(f)
        acts = data['mean_pool'].astype(np.float32)
        file_norms = np.linalg.norm(acts, axis=1)
        norms.extend(file_norms.tolist())

    mean_norm = float(np.mean(norms))
    print(f"  Mean activation norm at layer {layer_idx}: {mean_norm:.2f} "
          f"(from {len(npz_files)} files, {len(norms)} sentences)")
    return mean_norm


def load_steering_vector(category, layer_idx):
    """Load L2-normalised steering direction from probe weights."""
    npz_path = STEERING_DIR / f'steering_directions_layer{layer_idx}.npz'
    data = np.load(npz_path)
    vec = torch.from_numpy(data[category]).float()
    return vec


def normalise_steering_vector(vec, mean_activation_norm):
    """Normalise steering vector to match mean activation scale."""
    vec_norm = vec.norm().item()
    if vec_norm == 0:
        return vec
    return vec * (mean_activation_norm / vec_norm)


def load_venhoff_vector(label, layer_idx):
    """Load Venhoff pre-computed steering vector (difference-of-means).

    Their vectors: label_mean - overall_mean, normalised so that the
    feature vector's L2 norm matches the overall mean activation norm.
    """
    vectors = torch.load(str(VENHOFF_VECTORS_PATH), map_location='cpu',
                         weights_only=True)
    overall_mean = vectors['overall']['mean'][layer_idx]  # [4096]
    label_mean = vectors[label]['mean'][layer_idx]         # [4096]
    feature_vec = label_mean - overall_mean

    # Normalise to match overall activation magnitude
    overall_norm = overall_mean.norm().item()
    feature_norm = feature_vec.norm().item()
    if feature_norm > 0:
        feature_vec = feature_vec * (overall_norm / feature_norm)

    return feature_vec


# ── Prompt building ──

def build_prompt(tokenizer, task_id):
    """Build prompt for a given task (Set A format, no completion cue)."""
    task_path = STIMULI_DIR / f'Task{task_id}.json'
    with open(task_path, 'r', encoding='utf-8') as f:
        task_data = json.load(f)

    user_message = ("Here are the panel descriptions:\n\n"
                    + json.dumps(task_data, indent=2))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        combined = SYSTEM_PROMPT + "\n\n" + user_message
        messages = [{"role": "user", "content": combined}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    return input_text


def parse_thinking_answer(full_output):
    """Split output into thinking and answer parts around </think>."""
    if "</think>" in full_output:
        parts = full_output.split("</think>", 1)
        thinking = parts[0].strip()
        answer = parts[1].strip()
    else:
        thinking = full_output.strip()
        answer = ""
    return thinking, answer


# ── Generation ──

def generate_single_trace(model, tokenizer, input_ids, prompt_tokens,
                          task_id, trace_id):
    """Generate a single trace. Steering hook (if any) must be pre-registered."""
    seed = seed_for(task_id, trace_id)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    gen_time = round(time.time() - t0, 1)

    new_tokens = output[0][prompt_tokens:]
    total_tokens = len(new_tokens)
    full_output = tokenizer.decode(new_tokens, skip_special_tokens=False)
    full_output = full_output.replace("<\u2581end\u2581of\u2581sentence>", "").strip()

    thinking_text, answer_text = parse_thinking_answer(full_output)
    thinking_tokens = len(tokenizer.encode(thinking_text, add_special_tokens=False))
    truncated = total_tokens >= MAX_NEW_TOKENS

    return {
        "task_id": task_id,
        "trace_id": trace_id,
        "seed": seed,
        "raw_output": full_output,
        "thinking_text": thinking_text,
        "answer_text": answer_text,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "truncated": truncated,
        "generation_params": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "generation_time_seconds": gen_time,
        "timestamp": datetime.now().isoformat(),
        "model": "DeepSeek-R1-Distill-Llama-8B",
        "quantization": "4bit-nf4",
    }


def run_condition(model, tokenizer, condition_name, tasks, traces_per_task,
                  steering_meta=None):
    """Generate all traces for one experimental condition.

    Args:
        steering_meta: dict with steering_category, steering_direction,
                       steering_alpha, steering_layer (or None for baseline).
    Returns:
        (n_generated, n_skipped, gen_times)
    """
    condition_dir = OUTPUT_DIR / condition_name
    n_generated = 0
    n_skipped = 0
    gen_times = []

    # Cache tokenized prompts per task
    prompt_cache = {}

    for task_id in tasks:
        task_dir = condition_dir / f'task{task_id}'
        task_dir.mkdir(parents=True, exist_ok=True)

        if task_id not in prompt_cache:
            input_text = build_prompt(tokenizer, task_id)
            input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
            input_ids = input_ids.to(model.device)
            prompt_tokens = input_ids.shape[1]
            prompt_cache[task_id] = (input_ids, prompt_tokens, input_text)

        input_ids, prompt_tokens, input_text = prompt_cache[task_id]

        for trace_id in range(1, traces_per_task + 1):
            trace_path = task_dir / f'trace_{trace_id:03d}.json'

            if trace_path.exists():
                n_skipped += 1
                continue

            record = generate_single_trace(
                model, tokenizer, input_ids, prompt_tokens, task_id, trace_id)

            # Add prompt text and steering metadata
            record["prompt"] = input_text
            if steering_meta:
                record.update(steering_meta)
            else:
                record["steering_category"] = None
                record["steering_direction"] = None
                record["steering_alpha"] = None
                record["steering_layer"] = None
            record["condition"] = condition_name

            with open(trace_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

            n_generated += 1
            gen_times.append(record["generation_time_seconds"])

            vram = torch.cuda.memory_allocated() / 1e9
            print(f"  [{condition_name}] Task {task_id}, Trace {trace_id:03d}: "
                  f"{record['total_tokens']} tok, "
                  f"{record['generation_time_seconds']}s, "
                  f"trunc={record['truncated']}, VRAM={vram:.2f}GB")

            gc.collect()
            torch.cuda.empty_cache()

    return n_generated, n_skipped, gen_times


def run_steered_condition(model, tokenizer, category, direction, alpha,
                          layer_idx, steering_vec_cache, mean_norm,
                          tasks, traces_per_task, use_venhoff=False):
    """Set up steering hook and generate traces for one condition."""
    sign = 1 if direction == 'pos' else -1
    condition_name = f'{category}_{direction}_{alpha}'
    if use_venhoff:
        condition_name = f'venhoff_{condition_name}'

    # Load or retrieve steering vector
    cache_key = (category, layer_idx, use_venhoff)
    if cache_key not in steering_vec_cache:
        if use_venhoff:
            # Find Venhoff label for this category
            venhoff_label = None
            for vlabel, our_cat in VENHOFF_MAP.items():
                if our_cat == category:
                    venhoff_label = vlabel
                    break
            if venhoff_label is None:
                print(f"  WARNING: No Venhoff mapping for {category}, skipping")
                return 0, 0, []
            raw_vec = load_venhoff_vector(venhoff_label, layer_idx)
        else:
            raw_vec = load_steering_vector(category, layer_idx)
            raw_vec = normalise_steering_vector(raw_vec, mean_norm)
        steering_vec_cache[cache_key] = raw_vec

    vec = steering_vec_cache[cache_key]

    # Create and register hook
    effective_alpha = sign * alpha
    hook = SteeringHook(vec, effective_alpha)
    hook.register(model, layer_idx)

    steering_meta = {
        "steering_category": category,
        "steering_direction": sign,
        "steering_alpha": alpha,
        "steering_layer": layer_idx,
    }

    try:
        result = run_condition(model, tokenizer, condition_name,
                               tasks, traces_per_task, steering_meta)
    finally:
        hook.remove()

    return result


# ── Model loading ──

def load_model():
    """Load model in 4-bit quantization (matches generate_traces.py)."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    print("Loading model in 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=quantization_config,
        device_map="auto",
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM: {vram:.2f} GB")
    return model, tokenizer


# ── Entry points ──

def run_pilot(model, tokenizer, layer_idx):
    """3-trace pilot: baseline + HYPO_pos_1.0 + HYPO_neg_1.0, task 1 only."""
    print("\n" + "=" * 60)
    print("PILOT: 3 conditions x 1 task x 1 trace")
    print("=" * 60)

    pilot_tasks = [1]
    pilot_traces = 1

    # Compute normalisation
    print("\nComputing mean activation norm...")
    mean_norm = compute_mean_activation_norm(layer_idx)

    # 1. Baseline
    print("\n--- Baseline ---")
    n_gen, _, times_b = run_condition(
        model, tokenizer, 'pilot_baseline', pilot_tasks, pilot_traces)

    # 2. HYPO +1.0
    print("\n--- HYPO positive (alpha=1.0) ---")
    vec_cache = {}
    n_gen_p, _, times_p = run_steered_condition(
        model, tokenizer, 'HYPO', 'pos', 1.0, layer_idx,
        vec_cache, mean_norm, pilot_tasks, pilot_traces)

    # 3. HYPO -1.0
    print("\n--- HYPO negative (alpha=1.0) ---")
    n_gen_n, _, times_n = run_steered_condition(
        model, tokenizer, 'HYPO', 'neg', 1.0, layer_idx,
        vec_cache, mean_norm, pilot_tasks, pilot_traces)

    # Validation
    print("\n" + "=" * 60)
    print("PILOT VALIDATION")
    print("=" * 60)

    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM peak: {vram_peak:.2f} GB (budget: 10.7 GB) "
          f"{'PASS' if vram_peak < 10.7 else 'FAIL'}")

    all_times = times_b + times_p + times_n
    if all_times:
        mean_time = np.mean(all_times)
        print(f"Mean generation time: {mean_time:.1f}s")
        est_full = mean_time * 1240 / 60
        print(f"Estimated full run (1240 traces): {est_full:.0f} min")

    # Compare traces qualitatively
    conditions = ['pilot_baseline', 'HYPO_pos_1.0', 'HYPO_neg_1.0']
    texts = {}
    for cond in conditions:
        trace_path = OUTPUT_DIR / cond / 'task1' / 'trace_001.json'
        if trace_path.exists():
            with open(trace_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            texts[cond] = data['thinking_text']
            n_tok = data['total_tokens']
            trunc = data['truncated']
            print(f"\n  {cond}: {n_tok} tokens, truncated={trunc}")
            # Show first 200 chars
            preview = data['thinking_text'][:200].replace('\n', ' ')
            print(f"  Preview: {preview}...")
        else:
            print(f"\n  {cond}: MISSING")

    # Check traces are different
    if len(texts) == 3:
        b_text = texts.get('pilot_baseline', '')
        p_text = texts.get('HYPO_pos_1.0', '')
        n_text = texts.get('HYPO_neg_1.0', '')

        bp_same = b_text == p_text
        bn_same = b_text == n_text
        pn_same = p_text == n_text

        if bp_same or bn_same or pn_same:
            print("\n  WARNING: Some traces are IDENTICAL!")
            print(f"    baseline == pos: {bp_same}")
            print(f"    baseline == neg: {bn_same}")
            print(f"    pos == neg: {pn_same}")
        else:
            print("\n  Traces are distinct: PASS")

    # Check for NaN in outputs
    nan_found = False
    for cond in conditions:
        trace_path = OUTPUT_DIR / cond / 'task1' / 'trace_001.json'
        if trace_path.exists():
            with open(trace_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not data['thinking_text'] or data['total_tokens'] == 0:
                print(f"  {cond}: EMPTY OUTPUT")
                nan_found = True

    print(f"\n  No empty/NaN outputs: {'PASS' if not nan_found else 'FAIL'}")

    verdict = "GO" if (vram_peak < 10.7 and not nan_found
                       and len(texts) == 3) else "NO-GO"
    print(f"\nPILOT VERDICT: {verdict}")

    return verdict, {
        'vram_peak': vram_peak,
        'mean_time': float(np.mean(all_times)) if all_times else 0,
        'mean_norm': mean_norm,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate steered reasoning traces for Study 2b")
    parser.add_argument('--category', type=str, default=None,
                        choices=STEER_CATEGORIES,
                        help='Category to steer toward/away from')
    parser.add_argument('--direction', type=str, default=None,
                        choices=['pos', 'neg'],
                        help='Steering direction (pos=toward, neg=away)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Steering strength')
    parser.add_argument('--layer', type=int, default=DEFAULT_LAYER,
                        help=f'Layer to steer at (default: {DEFAULT_LAYER})')
    parser.add_argument('--baseline', action='store_true',
                        help='Generate unsteered baseline traces')
    parser.add_argument('--pilot', action='store_true',
                        help='Run 3-trace pilot validation')
    parser.add_argument('--all', action='store_true',
                        help='Run all 31 conditions (baseline + 30 steered)')
    parser.add_argument('--venhoff', action='store_true',
                        help='Use Venhoff pre-computed vectors instead of probes')
    parser.add_argument('--tasks', nargs='+', type=int, default=TASKS,
                        help=f'Tasks to run (default: {TASKS})')
    parser.add_argument('--traces-per-task', type=int, default=TRACES_PER_TASK,
                        help=f'Traces per task (default: {TRACES_PER_TASK})')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model()

    if args.pilot:
        verdict, stats = run_pilot(model, tokenizer, args.layer)
        # Save pilot log
        log_path = OUTPUT_DIR / 'pilot_log.json'
        with open(log_path, 'w') as f:
            json.dump({
                'verdict': verdict,
                'layer': args.layer,
                **stats,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        print(f"\nPilot log saved: {log_path}")
        return

    # Compute normalisation (needed for all non-pilot modes)
    print("\nComputing mean activation norm...")
    mean_norm = compute_mean_activation_norm(args.layer)

    if args.baseline:
        print("\n" + "=" * 60)
        print("BASELINE: unsteered generation")
        print("=" * 60)
        n_gen, n_skip, times = run_condition(
            model, tokenizer, 'baseline', args.tasks, args.traces_per_task)
        print(f"\nBaseline: {n_gen} generated, {n_skip} skipped")
        if times:
            print(f"Mean time: {np.mean(times):.1f}s")

    elif args.all:
        print("\n" + "=" * 60)
        print("ALL CONDITIONS: baseline + 30 steered")
        print("=" * 60)

        # Baseline first
        print("\n--- Baseline ---")
        n_gen, n_skip, times = run_condition(
            model, tokenizer, 'baseline', args.tasks, args.traces_per_task)
        print(f"  Baseline: {n_gen} generated, {n_skip} skipped")

        vec_cache = {}
        total_gen = n_gen
        total_skip = n_skip

        for cat in STEER_CATEGORIES:
            for direction in ['pos', 'neg']:
                for alpha in ALPHAS:
                    print(f"\n--- {cat} {direction} alpha={alpha} ---")
                    n_g, n_s, _ = run_steered_condition(
                        model, tokenizer, cat, direction, alpha,
                        args.layer, vec_cache, mean_norm,
                        args.tasks, args.traces_per_task,
                        use_venhoff=args.venhoff)
                    total_gen += n_g
                    total_skip += n_s

        print(f"\n{'='*60}")
        print(f"ALL CONDITIONS COMPLETE")
        print(f"Total generated: {total_gen}, skipped: {total_skip}")

    elif args.category and args.direction and args.alpha is not None:
        print(f"\n{'='*60}")
        print(f"STEERED: {args.category} {args.direction} alpha={args.alpha}")
        print(f"{'='*60}")

        vec_cache = {}
        n_gen, n_skip, times = run_steered_condition(
            model, tokenizer, args.category, args.direction, args.alpha,
            args.layer, vec_cache, mean_norm,
            args.tasks, args.traces_per_task,
            use_venhoff=args.venhoff)
        print(f"\n{args.category}_{args.direction}_{args.alpha}: "
              f"{n_gen} generated, {n_skip} skipped")
        if times:
            print(f"Mean time: {np.mean(times):.1f}s")

    else:
        parser.print_help()
        print("\nError: specify --pilot, --baseline, --all, or "
              "--category + --direction + --alpha")
        sys.exit(1)


if __name__ == '__main__':
    main()

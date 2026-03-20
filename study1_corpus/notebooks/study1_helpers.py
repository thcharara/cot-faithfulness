"""
Shared constants, data-loading, and style helpers for Study 1 analysis notebooks.

Usage (from any notebook in this directory):
    from study1_helpers import *
"""

import json, os, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# ── Project paths ───────────────────────────────────────────────────────────────

# Navigate to project root if running from notebooks/
_nb_dir = Path('.').resolve()
if _nb_dir.name == 'notebooks':
    os.chdir(_nb_dir.parent.parent)  # study1_corpus/notebooks/ → project root

PROJECT_ROOT = Path('.').resolve()
TRACES_DIR   = PROJECT_ROOT / 'outputs' / 'traces_clean_coded'
MANUAL_DIR   = PROJECT_ROOT / 'data' / 'manual_coding_final'
KAPPA_CSV    = PROJECT_ROOT / 'outputs' / 'validation' / 'kappa_results.csv'
CODING_LOG   = PROJECT_ROOT / 'outputs' / 'coding_log.jsonl'
OUTPUT_DIR   = PROJECT_ROOT / 'outputs' / 'study1_analysis'
TABLES_DIR   = OUTPUT_DIR / 'tables'
FIGURES_DIR  = OUTPUT_DIR / 'figures'

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Taxonomy constants ──────────────────────────────────────────────────────────

MICRO_LABELS = ['ORIENT', 'DESCRIBE', 'SYNTHESIZE', 'HYPO', 'TEST', 'JUDGE', 'PLAN', 'MONITOR', 'RULE']
MACRO_LABELS = ['SETUP', 'OBSERVE', 'INVESTIGATE', 'REGULATE', 'CONCLUDE']

MACRO_MAP = {
    'ORIENT': 'SETUP', 'DESCRIBE': 'OBSERVE', 'SYNTHESIZE': 'OBSERVE',
    'HYPO': 'INVESTIGATE', 'TEST': 'INVESTIGATE', 'JUDGE': 'INVESTIGATE',
    'PLAN': 'REGULATE', 'MONITOR': 'REGULATE', 'RULE': 'CONCLUDE',
}

VALID_TEST_CONTEXT = {'post_hypothesis', 'pre_hypothesis', 'post_rule'}
VALID_SPECIFICITY  = {'within_panel', 'across_panels'}
VALID_JUDGEMENT     = {'accept', 'reject', 'uncertain'}
VALID_CONFIDENCE    = {'high', 'medium'}

# Colour palette: one colour per macro, used for micro-label bars
MACRO_COLOURS = {
    'SETUP':       '#4C72B0',
    'OBSERVE':     '#55A868',
    'INVESTIGATE': '#C44E52',
    'REGULATE':    '#8172B2',
    'CONCLUDE':    '#CCB974',
}
MICRO_COLOURS = {m: MACRO_COLOURS[MACRO_MAP[m]] for m in MICRO_LABELS}

# Validation trace identifiers (these traces were NOT used for few-shot examples)
VALIDATION_TRACES = [
    ('A', 2, 14), ('B', 1, 10), ('B', 2, 4), ('B', 2, 6),
    ('A', 3, 20), ('A', 4, 19), ('B', 4, 15),
]

# Few-shot source traces (used for prompt examples — excluded from unbiased validation)
FEWSHOT_TRACES = [('A', 1, 15), ('A', 3, 10), ('B', 3, 20)]

# ── Plot style ──────────────────────────────────────────────────────────────────

def setup_style():
    """Apply consistent plotting style."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 5),
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'font.size': 11,
    })

setup_style()

# ── Data loading ────────────────────────────────────────────────────────────────

def load_traces(input_dir=TRACES_DIR):
    """Load all coded traces from the given directory."""
    traces = []
    for fp in sorted(Path(input_dir).rglob('trace_*.json')):
        with open(fp, encoding='utf-8-sig') as f:
            traces.append(json.load(f))
    print(f'Loaded {len(traces)} traces from {input_dir}')
    return traces


def _trace_key(t):
    """Build a unique trace key like 'set_a/task2/trace_014'."""
    s = t['set'].lower()
    return f"set_{s}/task{t['task_id']}/trace_{t['trace_id']:03d}"


def build_sentence_df(traces):
    """
    Build a sentence-level DataFrame from a list of trace dicts.
    Handles both auto-coded and manual-coded traces.
    """
    rows = []
    for t in traces:
        key = _trace_key(t)
        completed = bool(t.get('answer_text', '').strip())
        n_sent = len(t['sentences'])
        for i, s in enumerate(t['sentences']):
            c = s.get('coding', {})

            # Normalise depends_on to list of ints
            raw_deps = c.get('depends_on', [])
            if raw_deps is None:
                raw_deps = []
            deps = [int(d) for d in raw_deps]

            # Normalise specificity: across_panel → across_panels
            spec = c.get('specificity', None)
            if spec == 'across_panel':
                spec = 'across_panels'

            row = {
                'trace_key':    key,
                'set':          t['set'],
                'task_id':      t['task_id'],
                'trace_id':     t['trace_id'],
                'completed':    completed,
                'sentence_id':  s['sentence_id'],
                'text':         s['text'],
                'token_count':  s.get('token_count', np.nan),
                'position_idx': i,
                'trace_length': n_sent,
                'position_norm': i / max(n_sent - 1, 1),
                'micro_label':  c.get('micro_label', None),
                'macro_label':  c.get('macro_label', None),
                'confidence':   c.get('confidence', None),
                'test_context': c.get('test_context', None),
                'specificity':  spec,
                'judgement':    c.get('judgement', None),
                'depends_on':   deps,
                'n_dependencies': len(deps),
                'hypo_status':  c.get('hypo_status', None),
                'hypo_antecedent_sid': c.get('hypo_antecedent_sid', None),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Assign position thirds
    df['position_third'] = pd.cut(
        df['position_norm'], bins=[0, 1/3, 2/3, 1.0],
        labels=['early', 'middle', 'late'], include_lowest=True,
    )
    return df


def load_manual_traces():
    """Load the 10 manually coded traces from data/manual_coding_final/."""
    traces = []
    for fp in sorted(MANUAL_DIR.glob('*_final.json')):
        with open(fp, encoding='utf-8-sig') as f:
            traces.append(json.load(f))
    print(f'Loaded {len(traces)} manual traces from {MANUAL_DIR}')
    return traces


def load_kappa_results():
    """Load the validation kappa CSV."""
    df = pd.read_csv(KAPPA_CSV)
    return df


def get_validation_pairs(auto_traces, manual_traces):
    """
    Match auto-coded and manual-coded traces for the 7 validation traces.
    Returns list of (trace_key, auto_df, manual_df) tuples.
    """
    auto_df = build_sentence_df(auto_traces)
    manual_df = build_sentence_df(manual_traces)

    pairs = []
    for (s, task, tid) in VALIDATION_TRACES:
        key = f"set_{s.lower()}/task{task}/trace_{tid:03d}"
        a = auto_df[auto_df['trace_key'] == key].copy()
        m = manual_df[manual_df['trace_key'] == key].copy()
        if len(a) > 0 and len(m) > 0:
            pairs.append((key, a, m))
    return pairs


def save_fig(fig, name, close=True):
    """Save figure to the figures output directory."""
    path = FIGURES_DIR / name
    fig.savefig(path)
    print(f'  Saved: {path}')
    if close:
        plt.close(fig)


def save_table(df, name):
    """Save DataFrame to the tables output directory."""
    path = TABLES_DIR / name
    df.to_csv(path, index=True)
    print(f'  Saved: {path}')


def section_header(title):
    """Print a formatted section header."""
    print(f'\n{"="*70}')
    print(f'  {title}')
    print(f'{"="*70}\n')

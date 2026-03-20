# Reasoning Trace Coding — Taxonomy and Instructions

## Task Context

You are coding sentences from extended reasoning traces produced by a language model solving a visual inductive reasoning puzzles. Traces range from 30 to 400 sentences and consist of cycles of observation, hypothesis generation, evidence testing, and rule articulation.

## Taxonomy

### Macro and Micro Labels

| Micro | What it captures |
|-------|-------|-----------------|
| ORIENT | Task engagement and initialisation |
| DESCRIBE | Raw feature readout without inferential framing |
| SYNTHESIZE | Aggregating observations into higher-level characterisations |
| HYPO | Proposing or revising a candidate rule |
| TEST | Examining evidence for/against a candidate rule or feature dimension |
|| JUDGE | Hypothesis-level verdict on a candidate rule's viability |
| PLAN | Articulating strategy, method, or next step |
| MONITOR | Reflecting on the reasoning process, expressing process-level uncertainty |
| RULE | Stating the final concluded rule or definitive answer |

### Micro-Label Definitions

**ORIENT** — Task initiation. Paraphrasing the problem, acknowledging the task. Occurs at trace opening. If the sentence also articulates a method or strategy, code PLAN instead.

**DESCRIBE** — Raw feature readout. Listing colours, sizes, orientations, positions, counts, or spatial relationships with no inferential framing. The sentence records what is in the panel. If the readout is organised around a specific feature dimension being investigated, code TEST instead.

**SYNTHESIZE** — Aggregating prior observations into a higher-level characterisation spanning multiple features or panels. Does not target a single specific feature dimension and does not propose a candidate rule. Typically bridges observation and hypothesis phases.

**HYPO** — Proposing or revising a candidate rule. Introduces a new possibility about what the rule might be. Contains hypothesis-proposing language ("maybe," "perhaps," "could it be"). If the sentence articulates method rather than content, code PLAN.

**TEST** — Examining evidence in service of a candidate rule or specific feature dimension. Includes: raw evidence organised around a hypothesis, derived computations (counts, tallies), and panel-level characterisations of patterns found. Both pre-hypothesis investigation (bottom-up feature survey) and post-hypothesis confirmation count as TEST. Panel-level findings stay TEST regardless of whether positive or negative.

**JUDGE** — Hypothesis-level verdict ONLY. The sentence concludes whether the overall hypothesis or candidate rule is viable. Reserved exclusively for verdicts that evaluate the hypothesis's overall standing — not panel-level findings. If the sentence characterises what was found at the panel level, code TEST.

**PLAN** — Articulating strategy, method, or next investigative step. Includes task-structure reasoning (deducing which panels are positive/negative examples). About how to find the rule, not what the rule is.

**MONITOR** — Reflecting on the reasoning process. Process-level uncertainty, self-correction signals, expressions of being stuck. Not directed at a specific hypothesis — if the uncertainty is about a specific hypothesis after evidence evaluation, code JUDGE with judgement: uncertain.

**RULE** — The final concluded rule or definitive answer. Typically near trace end. If deliberation follows, the sentence was HYPO, not RULE.

## Coding Principles

1. **Code the endpoint.** When a sentence performs multiple functions, code the highest-level act accomplished. Evidence + verdict in one sentence → JUDGE. Evidence + derivation in one sentence → TEST.

2. **JUDGE = hypothesis-level verdicts only.** Panel-level characterisations are TEST regardless of polarity. "Doesn't alternate" (about one panel) → TEST. "So that can't be the rule" (about the hypothesis) → JUDGE.

3. **The "name the feature" test.** Can you name the specific feature dimension being investigated? Yes → TEST. Too broad to pin to one feature → SYNTHESIZE. Raw readout with no inferential framing → DESCRIBE.

4. **Context determines DESCRIBE vs TEST.** The same surface text ("Panel A has two red cones") is DESCRIBE during initial scanning and TEST during hypothesis evaluation. Ask: is there an active hypothesis or feature dimension being investigated?

5. **The uncertainty direction test.** Uncertainty about a specific hypothesis → JUDGE (uncertain). Uncertainty about the reasoning process → MONITOR. Ask: can you point to which hypothesis the uncertainty is about?

6. **Task-structure reasoning → PLAN.** Reasoning about what the rule is → INVESTIGATE categories. Reasoning about how to find the rule → REGULATE categories.

7. **Polarity independence.** Whether a finding is positive ("alternating") or negative ("doesn't alternate") does not determine the category if it describes a panel-level finding. Both are TEST.

## Metadata Flags

**On TEST sentences:**
- `test_context`: **post_hypothesis** (explicit HYPO precedes in dependency chain) | **pre_hypothesis** (no HYPO precedes — bottom-up investigation) | **post_rule** (testing occurs after a RULE statement to verify the concluded answer)
- `specificity`: **within_panel** (evidence from a single panel) | **across_panels** (comparing multiple panels)

**On JUDGE sentences:**
- `judgement`: **accept** (hypothesis supported/viable) | **reject** (hypothesis disconfirmed) | **uncertain** (inconclusive after evaluation, directed at a specific hypothesis)

**On all sentences:**
- `confidence`: **high** (clear-cut) | **medium** (defensible but a reasonable coder might disagree)

## Output Format

For each sentence, output a JSON object:

```json
{
  "sentence_id": 65,
  "micro_label": "TEST",
  "test_context": "post_hypothesis",
  "specificity": "across_panels",
  "confidence": "high"
}
```

Code ONLY the micro_label and any corresponding metadata flags.
Only include test_context and specificity for TEST sentences.
Only include judgement for JUDGE sentences.
Always include confidence.

Do not include macro_label or depends_on — these will be computed
in post-processing.
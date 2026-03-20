# Few-Shot Examples for Auto-Coding
# Source traces: setA_task1_trace15, setA_task3_trace10, setB_task3_trace20
# All other coded traces are reserved for validation.

Each example shows context, the target sentence, the correct label, and a rationale naming the wrong alternative.

---

## ORIENT (SETUP)

### Example 1 — Pure task engagement
```
TARGET:
[s0]: "Okay, so I've been given six panels, labeled A through F, each with an arrangement of cone-shaped objects."

LABEL: ORIENT (SETUP)
RATIONALE: Paraphrases the task prompt without articulating any method. Pure engagement. Not PLAN — no strategy is stated.
```
Source: setA_task1_trace15, s0. Confidence: high.

---

## PLAN (REGULATE)

### Example 2 — Strategy articulation (PLAN vs ORIENT boundary)
```
CONTEXT:
[s1]: "My task is to figure out what the hidden rule is..." → ORIENT
[s2]: "The panels with a star follow this rule, while the others do not." → ORIENT

TARGET:
[s3]: "I need to examine each panel's description and try to find a pattern or rule that separates them."

LABEL: PLAN (REGULATE)
RATIONALE: Articulates a method ("examine each panel's description," "try to find a pattern"). Not ORIENT — the sentence states how the model will approach the task, not just what the task is.
```
Source: setA_task1_trace15, s3. Confidence: high.

### Example 3 — Task-structure reasoning as PLAN
```
CONTEXT:
[s51]: "Let me list out the characteristics: Starred Panels (A, C, F)..." → SYNTHESIZE

TARGET:
[s50]: "Now, I need to find a pattern that distinguishes the starred panels (A, C, F) from the non-starred ones (B, D, E)."

LABEL: PLAN (REGULATE)
RATIONALE: Articulates investigation strategy by identifying which panels are positive vs negative examples. This is reasoning about how to organise the search, not about what the rule is. Not HYPO — no candidate rule is proposed.
```
Source: setA_task3_trace10, s50. Confidence: high.

---

## DESCRIBE (OBSERVE)

### Example 4 — Raw feature readout
```
CONTEXT:
[s6]: "Starting with Panel A: It has 7 cones." → DESCRIBE

TARGET:
[s9]: "For example, cone 1 is red, tilted left, at the extreme left."

LABEL: DESCRIBE (OBSERVE)
RATIONALE: Reports specific cone features (colour, orientation, position) with no inferential framing. No hypothesis is being investigated — this is initial scanning. Not TEST — no active feature dimension or hypothesis is being evaluated.
```
Source: setA_task1_trace15, s9. Confidence: high.

### Example 5 — Panel-summary DESCRIBE
```
TARGET:
[s6]: "Starting with Panel A: It has 7 cones."

LABEL: DESCRIBE (OBSERVE)
RATIONALE: High-level panel summary during initial scanning. DESCRIBE covers both detailed cone-by-cone inventories and summary-level feature readouts. Not PLAN — the informational content is the feature count, not a strategy.
```
Source: setA_task1_trace15, s6. Confidence: high.

---

## SYNTHESIZE (OBSERVE)

### Example 6 — Per-panel synthesis
```
CONTEXT:
[s8]: "Cone 1 is large, red, tilted left, and sits on the left corner." → DESCRIBE
[s9]: "For example, cone 1 is red, tilted left, at the extreme left." → DESCRIBE

TARGET:
[s13]: "I notice that each cone seems to be connected in a chain, each supporting the next, forming a kind of tower."

LABEL: SYNTHESIZE (OBSERVE)
RATIONALE: Aggregates individual cone descriptions into a higher-level structural characterisation ("chain," "tower"). Does not propose a candidate rule. Not HYPO — no testable rule is proposed. Not TEST — no specific feature dimension is being investigated.
```
Source: setA_task1_trace15, s13. Confidence: high.

### Example 7 — Cross-panel synthesis
```
CONTEXT:
[s59]: "Looking at the star panels (A, E, F) versus the non-star panels (B, C, D), I notice some patterns." → SYNTHESIZE

TARGET:
[s60]: "Star Panels: Panel A: Chain-like structure... Panel E: More complex interactions... Panel F: Even more complex, with multiple layers."

LABEL: SYNTHESIZE (OBSERVE)
RATIONALE: Aggregates characterisations across star panels spanning multiple features (structure, complexity, layers). No single feature dimension is targeted. Not TEST — you cannot name one specific feature being investigated. Not HYPO — no candidate rule is proposed.
```
Source: setA_task1_trace15, s60. Confidence: high.

---

## HYPO (INVESTIGATE)

### Example 8 — Clear hypothesis proposal
```
CONTEXT:
[s62]: "So, what could the rule be?" → MONITOR

TARGET:
[s64]: "Maybe the key is the number of cones?"

LABEL: HYPO (INVESTIGATE)
RATIONALE: Proposes a specific candidate feature dimension (cone count) as a potential rule. "Maybe" plus naming a specific feature is the hallmark of HYPO. Not PLAN — this is about what the rule might be, not how to investigate.
```
Source: setA_task1_trace15, s64. Confidence: high.

### Example 9 — Refined hypothesis
```
TARGET:
[s81]: "In star panels, maybe each cone is connected in such a way that there's a single path from the floor to the top, without branches."

LABEL: HYPO (INVESTIGATE)
RATIONALE: Refines the linear-sequence hypothesis by adding "single path" and "without branches." Adds new conceptual content. Not JUDGE — no verdict is rendered; the model is proposing, not evaluating.
```
Source: setA_task1_trace15, s81. Confidence: high.

### Example 10 — HYPO that adds conceptual content (HYPO vs JUDGE boundary)
```
TARGET:
[s129]: "So maybe the rule is that a starred panel has a cone that is both supporting and being supported, forming a sort of chain or a tower."

LABEL: HYPO (INVESTIGATE)
RATIONALE: Adds new conceptual content ("chain or tower" framing). Not JUDGE accept — the sentence refines the hypothesis rather than rendering a verdict on it.
```
Source: setA_task3_trace10, s129. Confidence: medium.

---

## TEST (INVESTIGATE)

### Example 11 — Post-hypothesis evidence gathering
```
CONTEXT:
[s64]: "Maybe the key is the number of cones?" → HYPO

TARGET:
[s65]: "But Panel A has 7, E has 6, F has 8."

LABEL: TEST, test_context: post_hypothesis, specificity: across_panels
RATIONALE: Gathers cone-count evidence for starred panels to evaluate the HYPO. Data is organised around a specific feature dimension. Not DESCRIBE — there is an active hypothesis being evaluated.
```
Source: setA_task1_trace15, s65. Confidence: high.

### Example 12 — Pre-hypothesis feature investigation
```
CONTEXT:
[s66]: "So that doesn't seem consistent." → JUDGE (reject)

TARGET:
[s67]: "Looking at the orientation: In star panels, I notice that in Panel A, most are upright, but in E and F, there are some lying on their sides."

LABEL: TEST, test_context: pre_hypothesis, specificity: across_panels
RATIONALE: Investigates orientation without a preceding HYPO about orientation. Bottom-up investigation — the pattern emerges from the survey. Can you name the feature? Yes — orientation. So TEST, not SYNTHESIZE.
```
Source: setA_task1_trace15, s67. Confidence: high.

### Example 13 — Derived computation (TEST, not JUDGE)
```
CONTEXT:
[s87]: "For example, in Panel A: Blue, Red, Blue, Red, Green." → TEST

TARGET:
[s88]: "That's three color changes."

LABEL: TEST, test_context: post_hypothesis, specificity: within_panel
RATIONALE: Derives a count (colour changes) from raw evidence. This is an intermediate computation, not a verdict. Not JUDGE — the sentence derives what IS (a count), not whether the hypothesis works.
```
Source: setA_task3_trace10, s88. Confidence: high.

### Example 14 — Panel-level characterisation stays TEST (not JUDGE)
```
CONTEXT:
[s85]: "For example, in A: Blue, Red, Blue, Red, Green." → TEST

TARGET:
[s90]: "Also alternates."

LABEL: TEST, test_context: post_hypothesis, specificity: within_panel
RATIONALE: Characterises a pattern found in a panel. Under the hypothesis-level-verdict rule, panel-level characterisations are TEST regardless of polarity. Not JUDGE — the sentence describes what was found, not whether the overall hypothesis works.
```
Source: setB_task3_trace20, s90. Confidence: high.

### Example 15 — Structural/holistic TEST
```
TARGET:
[s146]: "In star panels, each cone is connected to the next in a linear fashion, forming a single path."

LABEL: TEST, test_context: post_hypothesis, specificity: across_panels
RATIONALE: Characterises structural pattern across star panels relative to the linear-chain hypothesis. TEST doesn't require cone-by-cone evidence — holistic structural characterisations organised around a specific hypothesis count as TEST.
```
Source: setA_task1_trace15, s146. Confidence: high.

### Example 16 — Compressed pre-hypothesis TEST
```
TARGET:
[s124]: "Looking at the positions: In star panels, maybe each cone's position is either on the left or right side, but that doesn't seem to be the case."

LABEL: TEST, test_context: pre_hypothesis, specificity: across_panels
RATIONALE: Contains an embedded hypothesis and dismissal, but no prior HYPO exists. The primary work is investigating the position feature. Under code-the-endpoint, the investigation is the main act. Pre_hypothesis because no HYPO precedes this.
```
Source: setA_task1_trace15, s124. Confidence: medium.

---
### Example 17 — Cross-panel aggregation that's TEST, not SYNTHESIZE
```
CONTEXT:
[s289]: "D's first cone is on the left." → TEST
[s290]: "E's first cone is on the left." → TEST

TARGET:
[s291]: "So star panels have the first cone on the right or floor, non-star on the left."
LABEL: TEST, test_context: post_hypothesis, specificity: across_panels
RATIONALE: Aggregates across all panels but targets a single specific feature: first cone position. The "name the feature" test resolves this — you can name it (first cone position), so it's TEST. Not SYNTHESIZE — SYNTHESIZE characterises broadly across multiple features ("star panels seem more complex"). This sentence compares star vs non-star on one precise dimension.
Source: setB_task3_trace20, s291. Confidence: high.
```
---
## JUDGE (INVESTIGATE)

### Example 18 — Hypothesis-level rejection
```
CONTEXT:
[s65]: "But Panel A has 7, E has 6, F has 8." → TEST

TARGET:
[s66]: "So that doesn't seem consistent."

LABEL: JUDGE, judgement: reject
RATIONALE: Renders a verdict on the cone-count hypothesis's overall viability. Not TEST — the sentence evaluates the hypothesis, not panel features.
```
Source: setA_task1_trace15, s66. Confidence: high.

### Example 19 — Compressed evidence + verdict
```
TARGET:
[s55]: "Starred panels A, C, F use three or more colors, but so do B, D, E. So color alone doesn't seem to be the rule."

LABEL: JUDGE, judgement: reject
RATIONALE: Embeds evidence + verdict in one sentence. Under code-the-endpoint, the verdict is the highest-level act. Not TEST — the sentence's landing point is a hypothesis-level rejection.
```
Source: setA_task3_trace10, s55. Confidence: high.

### Example 20 — Hypothesis-level acceptance
```
CONTEXT:
[s161]: "So in A, it's a straight line." → TEST
[s162]: "In E, it's a snake-like path, but still a single path." → TEST

TARGET:
[s165]: "So, perhaps the rule is that star panels have a single continuous path from the floor to the top, while non-star panels have multiple paths or branches."

LABEL: JUDGE, judgement: accept
RATIONALE: Commits to accepting the single-path hypothesis after reviewing evidence. Evaluates the hypothesis's overall viability. Not HYPO — no new conceptual content is added; the sentence renders a verdict.
```
Source: setA_task1_trace15, s165. Confidence: high.

### Example 21 — Uncertain verdict (JUDGE vs MONITOR boundary)
```
CONTEXT:
[s34]: "So maybe that doesn't fit either." → JUDGE (reject)

TARGET:
[s35]: "Not sure."

LABEL: JUDGE, judgement: uncertain
RATIONALE: Directed at the specific hypothesis under evaluation. The uncertainty is about the hypothesis. Not MONITOR — you can point to which hypothesis this is about. If the uncertainty were diffuse ("I'm stuck"), it would be MONITOR.
```
Source: setB_task3_trace20, s35. Confidence: high.

### Example 22 — Accept-then-reverse pattern
```
CONTEXT:
[s244]: "So in starred panels, there are supporting relationships, while in non-starred, they have more or less." → JUDGE (accept)

TARGET:
[s245]: "But the count isn't the key."

LABEL: JUDGE, judgement: reject
RATIONALE: Immediately reverses the tentative acceptance from s244. Consecutive JUDGE sentences are valid — each gets its own label based on its own verdict.
```
Source: setA_task3_trace10, s245. Confidence: high.

---

## MONITOR (REGULATE)

### Example 23 — Process uncertainty
```
CONTEXT:
[s60]: "Star Panels: Panel A: Chain-like structure..." → SYNTHESIZE
[s61]: "Non-Star Panels: More varied in their connections..." → SYNTHESIZE

TARGET:
[s62]: "So, what could the rule be?"

LABEL: MONITOR (REGULATE)
RATIONALE: Rhetorical self-prompt about the reasoning process. No hypothesis has been proposed yet. Not JUDGE — there is no hypothesis to evaluate. The model is reflecting on its state.
```
Source: setA_task1_trace15, s62. Confidence: high.

### Example 24 — Self-correction
```
TARGET:
[s75]: "Or maybe I miscounted."

LABEL: MONITOR (REGULATE)
RATIONALE: The model flags a potential error in its own prior work. Metacognitive monitoring of accuracy. Not JUDGE — no hypothesis is being evaluated; the model is monitoring its own process.
```
Source: setA_task3_trace10, s75. Confidence: high.

### Example 25 — Reasoning-history recall
```
TARGET:
[s139]: "But earlier, that didn't seem to fit."

LABEL: MONITOR (REGULATE)
RATIONALE: Explicitly recalls a prior failed approach. Metacognitive recall of reasoning history. Not JUDGE — no current hypothesis is being evaluated; the model is reflecting on a past conclusion.
```
Source: setB_task3_trace20, s139. Confidence: high.

### Example 26 — Process-level stuck (MONITOR, not JUDGE)
```
TARGET:
[s269]: "I'm getting stuck."

LABEL: MONITOR (REGULATE)
RATIONALE: Process-level expression of difficulty. Not directed at any specific hypothesis. Not JUDGE — the model is commenting on its own reasoning capacity, not evaluating a candidate rule.
```
Source: setB_task3_trace20, s269. Confidence: high.

---

## RULE (CONCLUDE)

### Example 27 — Final rule statement
```
CONTEXT:
[s167]: "Given that, I think the rule might be..." → RULE

TARGET:
[s169]: "So, to state my best guess: The hidden rule is that the star panels have a single continuous chain of cones connected such that each cone is directly supporting the next in a linear path."

LABEL: RULE (CONCLUDE)
RATIONALE: Explicit final answer with "to state my best guess" framing. No further testing follows. Not HYPO — RULE is the terminal commitment. If deliberation followed, it would have been HYPO.
```
Source: setA_task1_trace15, s169. Confidence: high.

# R7 preregistration — 2026-10-24

Frozen before neural inference. No training, no new threshold, no outer access.
Hypothesis: an explicit entity/property/value graph prevents spurious support from
matching values attached to irrelevant entities/properties. This is a small rule-based
pilot, not a novel architecture or broad-domain fact extractor.

## Fixed methods

All methods use R6-B42 weights SHA86929f0f77031c7d79ca64f45cb0746c04555f712724875c63951909de48cf0e,
FP32 batch1, TF32off, original max-token risk and response threshold
>=.37612172961235046; localization .5. Five modes: baseline, keyed_auto,
graph_auto, keyed_oracle, graph_oracle. Oracle modes exist only on authored
controls; no oracle natural score or production inference.

Automatic extractor receives only text+document ID. It has four complete scalar
clause patterns with open entity/property strings, explicit aliases, exact units,
time/scope/condition/negation. No learned extraction, fuzzy entity linking or
template-name/label/source-ID exceptions. Graph has entity nodes and value nodes;
qualified property edges retain full clause and value-span provenance. Alias links
can span passages. Ambiguous alias chains abstain. Oracle uses separately authored
fact ledgers and claim annotations, never RAGTruth hallucination labels as evidence.

Baseline runs unchanged R6-B. Keyed retrieval indexes exact canonical entity plus
property (same alias resolution in auto and oracle); it returns original fact
clauses plus explicit alias provenance, leaves qualifier reasoning to R6-B.
One additional neural pass on selected evidence and the same full answer updates
only retrieved parsed-claim intervals. Other tokens retain full-context baseline.
Thus missing extraction cannot erase unrelated factual claims or their risks.

Graph matching uses entity/property/unit/time/scope/condition and polarity.
Complete exact scalar support sets only that claim's tokens to0; proven scalar
conflict sets only its numeric phrase to1; all other tokens retain baseline.
These are symbolic decision scores, not calibrated probabilities. Missing key,
incompatible units/qualifiers, conflicting facts, alias ambiguity or ANY unparsed
numeric evidence yield unknown/ambiguous and unchanged neural fallback. Negative
evidence !=v supports/contradicts only statements logically implied by that fact;
it does not reveal an arbitrary other value. No closed-world absence-as-error rule.

## Data, freeze and controls

Natural: the immutable R6 inner snapshot648/108 sources (442/206); reuse its
hash-verified records, never load outer partitions. Sensitivity642 without source15181.
Controls:1536 rows/16 fictional parents,768 positives,768 counterfactual pairs;
six distractor conditions × short/long answer × two fact orders × two answer
values × two target assignments. Source groups disjoint from R6 supervised sources.
Digits remain raw, no canonicalization. Both candidate values always appear in
two irrelevant anchor facts. Two other distractors have unrelated values, one
matching value, or repeated matching values. Other strata vary similar names,
same property at other entities and alias/fact separation across passages.

Core different/one_same/repeated interventions keep target claim, actual target
fact, labels, char length, number of objects, order and target position IDENTICAL.
Only irrelevant values change. Residual tokenizer-length changes are measured,
not silently called exact token matching. Similar-name/property/split variants
are separate stress strata; split adds paragraph separators. Long answer uses
24 supported filler clauses, with claim position crossed. This is not natural prose.

Six evidence styles: four covered scalar grammars and two deliberately unsupported
paraphrases (4/16 parents). This tests extraction failure without post-result parser
repair. New names/values/parents, but the suite and grammar share an author; not
a blind external extraction benchmark. Numeric facts and aliases are checked
against explicit ledgers and original character slices. Statements use conditional
or time/scope qualifiers on3/4 parents. Negation/ambiguity/unknown covered in
offline safety tests; primary benchmark labels are explicit scalar conflicts.

Before inference: verify lengths, pair labels, membership, all offsets, extraction
code hashes and original checkpoint; encode representative/full inputs to verify
2048 budget and coverage. No GPU training; expected <15 GPU-min inference, sequential
jobs. If full evidence does not fit, stop/report rather than silently truncate.

## Evaluation

Same IDs across all five controls methods and three natural automatic methods.
Macro-F1, positive P/R/F1, FPR/contradiction recall (only controls), AUROC/AP,
CM, pair accuracy (both members correct), response/claim/slot separation,
short-error hit and token/character/exact-span F1. Custom span scorer, not official.
Natural short definition unchanged >=500 chars and gold nonspace<=5%.
Controls short-error slice = all long positives (384), independent of score.

Extraction precision/recall: exact fact tuple including qualifiers, polarity,
full clause/value offsets; alias exact endpoints/provenance reported separately.
Coverage/unknown reports distinguish numeric extraction failure, unparsed filler,
missing claims, ambiguous links and unresolved matching. Natural has no annotated
evidence graphs: do NOT invent natural graph P/R from hallucination spans.
Manually inspect all automatic natural extraction activations, if any, after freeze;
no parser changes based on outcomes. Error stages: extraction, entity linking,
retrieval, relation matching, final classification; do not double-count dependent
stage failures as independent cases. Oracle is an extraction upper-bound only,
not an achievable deployment score.

2000 paired source bootstrap draws, seed42, percentile95; absolute and Δ against
baseline/keyed_auto. Cluster all variants/worlds by16 parents; natural108.
Core causal contrasts repeated−different and one_same−different at fixed
target/answer/world/order/length: score delta, risky confirmation flips on
contradictions and label-preserving stability. Report both labels, not just gains.
End-to-end after model loading includes extraction, encoding, full neural fallback,
selected neural pass and fusion; no free cached baseline for time measurements.
Peak VRAM and process memory reported; startup/disk excluded explicitly.

## Quantitative gate (all required for automatic pilot success)

On shared-value conditions (all except different): graph_auto macro gain >=.05
over baseline, paired lower CI>0; contradiction recall loss<=.02; supported FPR
increase<=.02; pair accuracy gain>=.05; short-error gold-hit and response recall
loss each<=.02; overall token/exact-span F1 loss<=.02.
Natural macro loss<=.01 and paired lower>- .02; token/span F1 loss<=.02,
short TP and gold-hit lose at most one case. Automatic fact precision>=.98,
recall>=.70 on all authored numeric facts, with unsupported styles included.

To justify graph-aware TRAINING additionally require graph_auto macro-F1 advantage
over keyed_auto>=.02 with lower CI>0 AND natural structural-decision coverage>=5%.
Zero natural coverage is vacuous retention, not general natural success. If keyed
retrieval is statistically indistinguishable, prefer it; if extraction fails outside
controlled grammar, measure that bottleneck before architecture expansion.
No additional training or threshold selection regardless of gate in R7.

Preserve old artifacts/hashes, save predictions/configs/source snapshot. Offline
tests and local model opt-ins with GROUNDCHECK_RAGTRUTH_DIR UNSET; Ruff/pip check.

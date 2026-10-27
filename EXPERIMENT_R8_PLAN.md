# R8 preregistration — 2026-10-24

Inference only. No outer calibration/dev/test, training, threshold selection or release.
Source selection was frozen before reading development prose in
models/phase2b_r8/source_split.json: first48 sources sorted by SHA256(R8:42:source_id)
from immutable R6 natural-inner records, excluding known duplicate15181. First24
extractor-development, last24 held-out for R8 rule development;144 responses each.
No labels used in selection. This inner set was previously evaluated: held-out
means only against current extractor edits, not independent final generalization.

Development-only audit exposed procedural text, definitions, ranges, conditional
claims, POS errors on capitalized imperatives, and repeated source headers.
No evaluation text or new model predictions inspected while editing extraction.
R7 was limited to four complete scalar sentence templates and six unit strings.
It could not handle these phenomena or general non-numeric complements.

## Frozen methods and resources

R6-B42 SHA86929f0f77031c7d79ca64f45cb0746c04555f712724875c63951909de48cf0e,
FP32 batch1 TF32off; response >=.37612172961235046, localization .5; max-token
aggregation unchanged. Full context budget2048; fail rather than silent truncation.
No learned vocabulary/statistics, no new weights. Reuse local Punkt, add a1.54MB
English POS archive pinned to nltk_data revision4f15a3d89eefe9748ec1c05be495d91289197155
and SHA6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b.
Resource member hashes in configs/r8-nlp.json; inference never downloads.
Expected <5 GPU minutes, sequential runs; extraction CPU/runtime measured.

One SVO candidate per sentence, POS noun/pronoun mentions, predicate-chain stems,
literal complete complements, negation/modality/time/conditions and character
provenance. No inferred coreference or generated assertions. Complex/implicit
subjects and inexact/embedded complements remain candidate/unknown. Confidence
is rule_high_unreviewed or provisional, not a calibrated model probability.
Explicit complete known-as/called/referred-to-as evidence statements supply alias
edges; claim aliases are never trusted evidence. No fuzzy entity matching.

A baseline: raw model.
B sentence: same extracted claim candidates, fixed top2 Jaccard evidence sentences
plus immediate same-passage neighbors. No fitted lexical statistics. One additional
full-answer neural pass on union evidence; only retrieved claim intervals change.
C flat and D graph: EXACTLY the same candidates, confidence filters, qualifier
policy, numeric/non-numeric value comparison, thresholds and token override policy.
ONLY D traverses evidence alias identity edges; C compares normalized entity text.
Both support equal complete values with equal qualifiers/polarity; explicit opposite
polarity can contradict only nonmodal equal-value facts. Different values are
unknown: no assumption every natural relation is single-valued. Unknown/ambiguity
keeps baseline. Hard0/1 overrides are decision scores, not probabilities.
No use of response annotations, generator IDs or question as evidence.

## Evaluation and review

Freeze source/resources/IDs/content and this plan before all neural runs. Score
all four methods on identical144 evaluation records; development scored separately
after freeze as a descriptive audit, never used to choose a variant after results.
Report accuracy/macro-F1/positive P/R/F1, AUROC/AP, CM/FPR/FNR; custom token/char/exact
span F1 and short-error hit (>=500 chars, gold nonspace<=5%, denominator explicit).
Paired source bootstrap2000 draws seed42,24 sources, identical draws for all systems.
Report coverage at record, clause and relation levels, separating any candidate,
eligible exact candidate and actual structural verdict. Mention detection is POS
proposal coverage, not NER accuracy. Unknown/ambiguity measured by stage.

New response-level errors versus baseline and overrides overlapping existing gold
hallucination spans are safety flags, not independent graph-extraction labels.
No natural extraction precision/recall or independently verified linking-error
rate without human-approved graph annotations. Manually described errors are
assistant audit hypotheses and must be clearly labeled provisional.

Prepare48 candidate claim/evidence examples, one per selected source (24+24),
deterministically prioritizing resolved matches then exact candidate facts, with
lexical nearest candidate when unresolved. Include verbatim excerpts, all offsets,
triples/qualifiers/confidence/issues, reviewer fields and parent IDs. Do not select
by model correctness or expose hallucination gold to the annotator. These are
automatic proposals, NOT an independent human gold set. Review completion remains
pending and cannot be fabricated. Save local-only review pack, exclude its natural
text from source ZIP/Git; preserve it separately with hashes.

Transfer check: held-out source groups plus separate held-out relation predicates
absent from development; report coverage/results on this slice. Also include
small offline grammar variation/safety tests, never as natural quality evidence.
No parser repair after evaluation extraction/predictions.

## Acceptance criteria (no training even if passed)

Meaningful evaluation structural coverage: >=5% responses and >=4 distinct sources.
Natural quality: graph−baseline macro-F1 >=−.01, paired lower95% >−.03, positive
recall loss<=.02; short gold-hit loss<=1 when slice nonempty. Report empty slices
as not measurable, not perfect. Runtime/memory and full coverage logged.
Reviewed structural precision >=.95 over >=20 independently approved decisions
from >=8 sources; no confirmed unsafe override allowed without a documented fix
and new evaluation. This gate is PENDING until user review, never auto-passed.
Extraction P/R requires independently annotated target facts, not just acceptance
of the proposed candidates (candidate-only review cannot establish recall).
Graph-specific recommendation requires measurable benefit over identical-policy
flat lookup on alias-dependent cases; identical outputs imply no demonstrated
graph advantage. No automatic architectural escalation from candidate coverage.

Run offline/local-model tests with GROUNDCHECK_RAGTRUTH_DIR unset, Ruff and both
pip checks. Preserve85 R7 artifacts, source snapshot, all prior checkpoints; do not
bypass historical source guards. End this packet with report/review pack, no fit.

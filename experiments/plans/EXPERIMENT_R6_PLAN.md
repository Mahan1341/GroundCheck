# R6 preregistration — 2026-10-24

Status: frozen before R6 training or transfer predictions. Research continues.

Hypothesis: diverse evidence renderers improve entity–property–value binding on
new fictional parents and unseen evidence/answer syntax, without losing natural
inner-validation or short-error sensitivity. No novelty claim.

## Integrity gate and inference

Numerical audit: 50 fixed inputs, B7, BF16/FP32, batches 1/2, fixed padding,
two repetitions each. Includes two explicitly historical numerical sentinels;
not an unbiased estimate of instability prevalence. Audit files are immutable.
FP32 passes the predeclared <1e-4 token-difference gate with zero flips.
All R6 comparisons use FP32, batch 1, no artificial padding, TF32 disabled.
Training remains matched BF16 autocast with FP32 CE/weights.
Response score remains max token probability; >=0.37612172961235046 positive.
Localization threshold remains 0.5. No new threshold fitting anywhere.
Historical BF16 metrics are not overwritten or mixed into the primary table.

## Data and intervention

Natural replay: all 2568 inner-fit responses / 428 sources. Synthetic fit:
768 rows / 16 fictional parents, 384 positives. Natural inner evaluation:
648 / 108 sources, 206 positives; fixed 642-row sensitivity excluding the known
shared-passage parent 15181. No outer calibration/dev/test inference or tuning.

A/B start from the identical B7 inner42 checkpoint. Each receives the same 3336
records twice, same shuffled IDs and seed, 3336 microbatches, accumulation 4,
834 optimizer updates, one epoch. A repeats old declarative relation syntax;
B uses six controlled evidence renderers. Answers/questions/worlds/labels/spans
are exactly matched. Restricting diversity to evidence isolates the intervention;
new answer syntax is tested only in transfer. Entity and property swaps are present
in BOTH arms, so they do not confound renderer diversity. No canonicalization,
ranking, consistency, retrieval, segmentation, encoder or aggregation changes.

AdamW lr1e-5, weight decay .01, warmup 10%, clip1, positive CE weight
3.370202392225359, microbatch2, accumulation4, BF16, gradient checkpointing.
Final checkpoint only (no epoch/model selection). Profile longest natural and
synthetic batches in both arms before training; finite loss/gradient and peak
allocated VRAM <11 GiB required. Expected ~25 GPU minutes for seed42 pair;
stop on resource failure. No mid-epoch optimizer resume; incomplete attempts
must be preserved under distinct paths and never represented as completed.

## Frozen transfer controls

768 rows / 16 new parents, 384 positives, 384 evidence-counterfactual pairs.
Six categories: entity swap, property swap, shared values, similar entity names,
fact-order/prose, short wrong claim in a long correct answer. Six held-out evidence
families and new answer syntax, names and disjoint 3-digit values. Parent is the
bootstrap unit. Entity names do not recur across parents. Each parent includes
both labels, both answer values, both fact orders and both evidence assignments.
The same exact answer is supported in one world and contradicted in another;
the same evidence also occurs with both answer values. Both candidate values
always occur in evidence. Gold labels are checked against explicit fact ledgers.
Exact numeric-phrase offsets are gold spans; claim/slot offsets exist for all
examples solely for posthoc diagnostics, never as inference features.

Audit label balance conditional on answer/evidence text, value, order, template,
parent/name, answer length and claim position. Report target-token equality and
remaining evidence length/token/padding differences between training arms.
These are fictional tests, not new independent natural documents. Long answers
repeat supported filler sentences; this limits ecological validity. Six categories
are overlapping stress conditions, not six independent datasets. Shared vocabulary
and property types across splits are intentional; only renderers/parents/values
are held out. No semantic novelty or exhaustive shortcut exclusion is claimed.

## Analysis and repeat gate (all must pass)

Compare B7, R5-A42, R6-A42, R6-B42 on identical IDs with fresh FP32 predictions.
Report accuracy, macro/positive F1, precision/recall, FPR/FNR, AUROC, AP, CM,
runtime/throughput/VRAM. Token/character/exact-span metrics reuse the custom
nonofficial scorer. Pair accuracy requires both labels correct at frozen threshold.
Response separation = contradiction minus support risk for same-answer worlds.
Claim-local separation uses max probability overlapping the fixed claim, plus
numeric-slot separation; neither is used to classify or train.
Source-paired percentile95 bootstrap: 2000 draws, seed42, shared draws across
methods; absolute and differences vs A and B7. Category and token-length slices.
Short natural = >=500 characters and annotated nonspace <=5% (20 positives).
Report response recall and gold-token-hit (>=.5 overlapping a gold character).

Repeat seeds43/44 only if ALL: transfer macro-F1 B-A>=.03 and lower CI>0;
pair accuracy B-A>=.05 and lower CI>0; mean claim-local separation B-A>0;
supported FPR increase<=.02; natural macro B-A>=-.01, lower CI>-.02,
and B-B7>=-.01; natural token and exact-span F1 losses<=.02;
transfer token/span F1 losses<=.02; natural short recall loses at most one case;
transfer long-short response recall and gold-hit each lose<=.02.
If passed, repeat BOTH arms from same B7 with seeds43/44, no selection, same
settings, report all seeds. Otherwise stop this renderer-diversity hypothesis
for this packet. No dev run regardless of result.

Preserve checkpoints, source snapshots, predictions, hashes, logs and failure
cases; do not overwrite old artifacts. Offline CI, pytest/Ruff/pip check required.

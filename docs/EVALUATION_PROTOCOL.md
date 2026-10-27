# Evaluation protocol

Positive means an annotated unsupported/contradicted answer span, including
`implicit_true` under the primary include/all policy. Annotation-free text is not
automatically independently verified entailment. Faithfulness to supplied evidence
and factual truth are different tasks.

## Partitions and isolation

RAGTruth snapshot: `c103204b9ce28d6bbad859304bf30de72b8ed8fe`. QA partitions:

| Partition | Responses | Sources | Purpose |
|---|---:|---:|---|
| Train | 3216 | 536 | Fit; grouped inner research splits |
| Calibration | 810 | 135 | Threshold of an already selected frozen configuration |
| Dev | 1008 | 168 | Historical frozen evaluation; already studied development |
| Official test | 900 | 150 | Closed; no final evaluation performed |

Historical B7 inner split: train 2568/428 and inner-validation 648/108, wholly inside
train. Later R9/R10 sensitivity evaluation uses 642/107; a shared-passage source was
excluded in that diagnostic, not silently reassigned to fit. Further source clustering
merges the known shared-passage pair for conditional bootstrap. Historical partitions
and metrics are not regenerated to make a new method look better.

Repeated inner use prevents treating it as independent confirmation. The publication
step runs authored demonstrations only: no training, architecture search or outer
calibration/dev/test scoring. The raw loader seals official test labels by default.
Ordinary training/calibration exports exclude test labels. Guard tests are offline.

## Decisions and metrics

The B7 reference response boundary is `risk >= 0.37612172961235046`; token spans use
`score >= 0.5`. Max token risk is not calibrated probability. The public adapter uses
FP32, batch1, TF32 off and averages overlaps before max aggregation. Older BF16 and
different-batch scores are retained as historical results, not silently replaced.

Report accuracy, response macro-F1, positive precision/recall/F1, AUROC and AP with
denominators and confusion matrices. Macro-F1 is not positive F1. AP is not
trapezoidal PR-AUC. Token metrics use any gold character overlap, character metrics
exclude whitespace, exact-span metrics trim/merge overlapping or whitespace-adjacent
ranges. These are custom scorers, not claimed official benchmark implementations.

Historical threshold selection maximizes calibration macro-F1 over every distinct
decision boundary, choosing the highest threshold on exact ties. New research must
preregister source-disjoint train-side selection, never tune on official test or
reuse a closed external confirmation. Bootstrap uses paired source-level draws
(at least 2000 in later research); it does not correct adaptive selection or replace
training-seed uncertainty. Variant examples sharing a parent are not independent.

## External results and limitations

RefChecker/Dolly: response labels are ORs of human judgments on extracted triplets;
claim coverage is imperfect and there are no human error spans. The 363/54-source
confirmation was used once and is closed to new decisions.

DiaSumFact SAMSum: human-author error categories, 295/59-source fixed dialogue
summarization collection. Response positives are any non-NoError annotation.
Localization includes 284 uniquely/reversibly aligned summaries; 11 ambiguous or
unmatched positive cases are unscorable, not converted to negatives. This collection
is now used development evidence. Assistant audits are not independent human gold.

Available-text contamination checks do not clear missing model-training premises,
hidden pretraining or paraphrased overlap. No result supports a universal SOTA claim.

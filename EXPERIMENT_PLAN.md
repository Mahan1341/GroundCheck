# Phase 2B+ batch 1 — frozen before new dev results

2026-10-22. User mandate supersedes the historical MVP-only scope. No test evaluation.
Target: response-level QA RAGTruth include/all, positive=hallucination, primary macro-F1.
Published positive F1, span F1 and balanced accuracy are different metrics.
Phase 2A is preserved in models/phase2a_snapshot/phase2a.zip, with per-file SHA256 manifest
and its original environment requirements. Never rewrite its freeze or weaken its guards.

## Selection protocol and finite budget

Outer Phase 1.1 memberships unchanged: 3216/810/1008/900 responses, 536/135/168/150 sources.
Within train, sort sources by SHA256(`inner:42:{source_id}`), first ceil(0.2*536)=108
sources (648 responses) are inner-validation; remaining 428 (2568) are inner-train.
No vocabulary/IDF/supervised weights/learned aggregation fit on inner-validation.
Architecture, preprocessing, training epoch and hyperparameters selected on inner-validation.
Threshold-optimized inner macro-F1 is a model-selection statistic, not unbiased performance.
All derivatives of one source stay grouped. No metadata/generator/gold-span inference features.

Training defaults: ModernBERT-base (pretrained, not RAGTruth-finetuned), answer token head,
question+evidence+answer; AdamW lr=2e-5, weight_decay=.01, max 3 epochs, effective batch=8,
linear warmup10%, gradient clip1, bf16 on CUDA, seed42 for inner selection.
Positive loss weight=sqrt(negative/positive answer tokens) on fold-train, capped at10.
Select epoch by inner response macro-F1; ties choose earlier epoch. No extra learning-rate grid.
Measured smoke determines microbatch and context length BEFORE fitting; target2048, fallback1024
with overlapping evidence/answer windows and explicit coverage. Do not silently lose context.
If ModernBERT backend fails or exceeds memory, compact pretrained DistilBERT with windows is
the predeclared replacement, not an extra tuned candidate. Failed runs stay in ledger.

For selected trainable finalist, final train-only fits seeds42,43,44, all reported individually;
never choose a seed by dev. Use inner-selected epoch count for all final fits.
Run 3 grouped CV folds (SHA256(`cv:42:{source_id}`), round-robin) if the trained method is within
0.03 inner macro-F1 of the strongest clean specialized baseline and profiling projects <=90min
for CV AND the total projected batch including all mandatory seeds/evaluations plus10% reserve stays within4 GPU-hours. CV uses the fixed selected epoch, not fold-wise best-epoch selection. Fold threshold-optimized macro-F1 is a selection statistic; fold AUROC/AP are reported separately. No additional architecture search.
Total package cap: 10 configurations below, max4 GPU-hours, no paid API/cloud. Resource repairs
do not change hypotheses; record failures and stop adding candidates when cap is reached.

After train-side selection: freeze candidates, config/code/model/input hashes BEFORE calibration
and new dev. Calibration selects only final thresholds, exhaustive unique boundaries plus
all-negative; higher risk=more suspicious, predict risk>=threshold, exact macro-F1 ties→highest.
Calibration was used historically: do not call it untouched. Dev is a development benchmark,
already inspected in Phase 2A. Evaluate the fixed batch once, no dev-driven reruns/tuning.
Separate precision>=.80 point: among calibration boundaries with >=10 positive predictions,
maximize recall, tie→highest threshold; if none, report unavailable (not empty-set perfection).

## Configurations (10 including the preserved historical reference)

| ID | Hypothesis/change | Comparator and train-side criterion | Cost/error focus |
|---|---|---|---|
| B0 | Preserved Phase2A TF-IDF | Historical reference, no retuning dev | CPU; lexical errors |
| B1 | Generic DeBERTa-small NLI, EXACT old sentences/top3/chunks/max | B0; inner macro-F1; keep all3 NLI probabilities | GPU; semantic verification only |
| B2 | MiniCheck-DeBERTa-large author document chunking (400 tokenizer tokens, sentence boundaries), NLTK sentence claims, max risk | Strong clean baseline candidate; no RAGTruth supervised training declared | GPU; specialized verifier, author support=max over document chunks |
| B3 | HHEM2.1 author whole-document/whole-response predict | Separate diagnostic if training provenance insufficient; NEVER clean leaderboard then | GPU; no imposed TF-IDF retrieval; full context |
| B4 | MiniCheck same old sentences/top3/chunks as B1 | B1 verifier ablation; B2 context/segmentation adapter difference explicit | GPU; retrieve failures vs author path |
| B5 | MiniCheck SAME NLTK claims as B2 + full evidence jointly | B2; isolated context aggregation ablation, inner macro-F1 | GPU; cross-passage support |
| B6 | MiniCheck extractive cleaned segments + SAME full evidence as B5 | B5; isolated segmentation ablation selected after train-side error audit | GPU; bare list numbers, standalone refusals, short clauses |
| B7 | Own evidence-conditioned pretrained token classifier | B2/B5; true training on train spans, answer-only loss; 3 final seeds | GPU; small unsupported additions, relations/numbers |
| B8 | Own token classifier with answer ONLY | B7 same settings, seed42; controlled input ablation, not deployment model | GPU; dataset/style shortcuts |
| B9 | Length-only logistic regression | Cheap supervised shortcut diagnostic, grouped inner selection | CPU; answer-length confounding |

Before B2/B5 results: prioritize joint evidence based on the eight-source retrieval audit.
B5 keeps B2 segmentation unchanged so the context ablation has one substantive difference.
Train-side amendment before B6/calibration/dev: B2/B5 each had36 false alarms whose worst fragment was a bare list number. B6 therefore replaces the not-yet-run logistic aggregator with an isolated cleaner ablation. B5 remains unchanged. Cleaner removes list markers only at line starts followed by same-line content; a standalone numeric answer is retained. Exact generic refusals/meta phrases are removed only as entire segments; factual explanations remain. Semicolon splitting is extractive. No extra configuration was added.
Full context token coverage checked. Response aggregation stays max; no learned stacking in batch1. B9 uses standardized log answer length with LogisticRegression C=1 fitted only on fold-train.
No train-derived rule keyed by dev ID, quality or generator. No synthetic negatives in batch1.
No trainable verifier stacking in this batch, hence no in-sample finetuned stacking shortcut.

## Measurement and validation

All comparable finalists: same response IDs; accuracy, macro-F1, positive P/R/F1, AUROC, AP,
confusion, FPR/FNR; latency/throughput, CUDA peak allocated/reserved and host process peak RSS.
2000 PAIRED source bootstrap draws seed42, identical draws across methods; fixed-model intervals
and differences against strongest clean fixed-verifier MiniCheck pipeline among B2/B4/B5/B6 by dev macro-F1 (including our pipeline improvements), explicitly
descriptive post-hoc reference. Report seed variability separately; bootstrap does not correct selection.
For deciding what to advance, retain train-side baseline choice separately from dev ranking.
Training and all inference costs logged, no claiming theoretical hardware fit as measured.
Span metrics separate: positive token micro-P/R/F1 (any character overlap target), non-whitespace
character micro-P/R/F1, exact contiguous-character-span micro-P/R/F1. Threshold0.5 fixed for spans;
not the official scorer. Overlap-window token scores merged by mean before response max pooling.

Manual train-side retrieval audit: first8 distinct inner-validation sources by response ID,
one response each (convenience sample, not population frequency estimation); classify top3 gaps,
without inventing evidence-span gold or Recall@k. Analyze errors by length, generator and span type
only in evaluator. Log segmentation drops and coverage. Perturbation checks: controlled toy
relations/numbers/negations with known expected support; train-side blank/shuffled evidence as
score sensitivity ONLY, never relabel arbitrary shuffled natural data as proven hallucination.
Offline tests for grouping, masking, offsets, windows, label blindness, thresholds, provenance;
real model smoke/integration opt-in. Run full pytest, Ruff, pip check. No push/publication.

## Future confirmatory evaluation (not authorized here)

After this finite package, choose next experiments using measured bottlenecks. Before any claim
of superiority: lock one complete method and a clean competing baseline, authorize untouched
official QA test once, report all seeds and paired source CIs. Preselect an external domain such
as TofuEval dialogue summarization only after checking each checkpoint's training lineage and
exact/normalized overlap against our training sources; no external threshold tuning. If it was
used to select a checkpoint, it is not independent confirmation. Collect a new licensed, blinded
sample instead when lineage cannot support independence. No external evaluation run in batch1.

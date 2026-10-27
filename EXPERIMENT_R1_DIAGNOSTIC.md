# R1 read-only diagnostic amendment — 2026-10-23

Recorded after C1 completed and before any predictions on the new probe. No additional training configuration, threshold search or architecture change. Existing controls/results are preserved.

Observation: C1 fits all120 training counterfactual combinations (60 positive/60 negative), macro-F1/tokenF1/AUROC1.0; natural inner macro-F1=.79404 versus C0=.79501. Authored75 paired interventions show only.01963 mean risk delta, with especially weak date/unit/attribution transfer. In-sample success does not establish that a general relation rule was learned.

Diagnostic question: does the same numeric/record-identifier transformation transfer to unseen supervised source parents, even when the broader authored multi-fact suite does not?

Construct all eligible one-integer unit/duration/event statements from the fixed108-source inner-validation partition using the same strict extraction and manual semantic acceptance as the training pilot. Do not inspect model scores while accepting statements. Exclude normalized statements that occur as substrings anywhere in inner-fit evidence; preserve an audit of exclusions. Reject incomplete/navigation/hypothetical/approximate/multiple-number statements. If too few survive, report the small denominator instead of relaxing criteria. No original natural examples are removed or reassigned.

Freeze accepted source IDs/text hashes/offsets before inference. Use the existing four-combination generator and exact same synthetic semantics. Verify every parent source is absent from each evaluated model's train_sources. Compare fixed B7 inner42, C0, C1 and fixed B6 MiniCheck. Use their already selected648-row inner thresholds; no probe threshold optimization. Save all predictions, paired evidence deltas, classification/ranking metrics, provenance and runtime. Run after the current GPU queue; no competing GPU workers.

Interpretation: strong same-format unseen-parent performance with weak broader controls supports a context/relation-format transfer limitation, not proof of complete generalization. Weak same-format transfer leaves memorization/overfitting plausible. This small, post-diagnostic internal development probe is not confirmatory, not an independent test and not evidence of superiority on a natural benchmark. No new training follows it in R1.

## Recovery amendment2026-10-24, before probe inference

The user explicitly requested answer-only comparison in this same28-row probe. Add the already completed B8 inner42 checkpoint alongside B7/C0/C1/B6. It uses the same fixed natural-inner threshold, source-isolation checks and token targets, with question/evidence removed exactly as in its training. No new model fitting, parent selection, threshold tuning or example change. Expected paired evidence delta for deterministic answer-only inference is exactly zero. The pre-amendment staged implementation is preserved in research_snapshots/R1_staged_recovery_20261024.zip; historical plans/results are not rewritten.

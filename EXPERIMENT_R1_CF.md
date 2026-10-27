# R1-CF preregistration — 2026-10-23

Written after B7/B6 authored-control results and before implementation/training of this method. B8 inner training is still running; no new calibration/dev/test results have been viewed. This is one isolated augmentation experiment with a matched continuation control, not a parameter search.

## Observation and hypothesis

On75 paired interventions/15 authored underlying facts, B7 inner42 mean response risk change is0.00515, claim-local change0.01211, zero pairs correctly separated at its natural inner threshold. B6 MiniCheck changes by0.62555 and separates30/75 pairs. B7 correctly supported evidence reordering changes mean risk by0.18850, much more than the factual contradiction. Removing punctuation cannot explain64/75 natural false positives with at least five above-threshold tokens. These are diagnostics, not population claims.

Hypothesis: paired numeric counterfactual token supervision can increase use of the number-to-relation link, beyond continued training on ordinary answers. Aggregation alone cannot create a missing claim-local signal. Test one narrow intervention; do not simultaneously change segmentation, retrieval, backbone, aggregation or loss.

## Data and semantic gate

All natural training and synthetic parents come exclusively from the428-source inner-fit partition. Initial strict same-source two-date extraction yielded only3 sources and an unconditional single-year/event rule only1; do not relax silently or treat software-version numbers as dates. A broader scan of numeric unit/duration statements was manually reviewed.30 accepted statements from25 inner-fit sources are locked by text SHA256/source_id/offset in configs/counterfactual_acceptance.json;12 rejected candidates retain reasons. Most are procedural quantities/durations, with one date. This narrow and small sample limits generalization.

For accepted statement S containing one integer A, construct B=2*A. Use all four combinations: evidence S(A)+record-identifier B or S(B)+record-identifier A; answer S(A) or S(B). Mark only the answer number when it disagrees with the statement. Identifier is explicitly a separate fact; both numbers remain in every evidence. Both answer variants and both evidence variants occur with both labels. Half the combinations place the identifier first, half last by deterministic parent hash; order is independent of gold. No arbitrary natural-document substitution receives invented gold. Synthetic S(B) is a deliberately constructed premise, not a claim about the real world.

Check exact offsets, integer cardinality, balanced labels conditional on answer/evidence, unchanged normalized word counts across paired evidence, train-source membership, no overlap with inner-val, and hash of every accepted parent. Save full generated examples only in ignored predictions. No gold spans or quality/generator metadata as inference features.

## Isolated matched experiment

C0: continuation of the exact saved B7 inner42 epoch3 checkpoint on2568 natural inner-fit examples plus856 deterministic response-label-balanced natural replay rows (428 positive/428 negative).
C1: same initialization and2568 natural examples plus856 counterfactual rows, cycling the120 balanced combinations in hash order.25% of3424 examples are the controlled augmentation/replay slots. Parent groups never cross partitions. The extra rows have the same50/50 response-label balance; token-class proportions, text lengths and domains remain different and are reported as limitations of the augmentation comparison.

Both: one fixed epoch, seed42, AdamW1e-5/weight_decay0.01, original microbatch1/accumulation8,10% warmup, BF16, gradient checkpointing, max_length2048, original answer-only token loss mask and max-token response aggregation. Freeze positive class weight to original B7 inner fit value to isolate augmentation. Fresh optimizer for both; this is matched continuation, not exact optimizer-state resume. No early stopping or additional epochs after seeing results. Maximum measured cost allowance2200seconds combined including profiling and scoring; fail explicitly on context overflow, no silent truncation. Save model/tokenizer/config/history/provenance separately for C0/C1.

## Evaluation and decision

Use unchanged648 natural inner-validation IDs, include/all policy; choose each inner diagnostic threshold by the established all-boundary macro-F1/highest-threshold tie rule. These are selection/development metrics, not final calibration or independent evaluation. Report metrics,2000 paired source draws versus B7/C0/B6, custom token/character/span scores, short-error-long-answer cases, token/sentence strata, improved/new errors. Keep the predeclared642-row passage-overlap sensitivity at unchanged thresholds.

Repeat the same300 authored controls and32-source evidence perturbations; never train on these authored facts or on inner-val derivatives. Check numeric/date/unit families as direct hypothesis targets, negation/attribution as untrained transfer. A useful pilot needs a larger supported-to-contradicted separation than C0 while preserving low supported risk and avoiding material natural quality/span regressions. Report all outcomes; no significance claim from15 templated facts, no seed-robustness claim from one pilot. No extra threshold tuning on authored controls.

If it fails, report whether signal was not learned, failed to transfer, or traded natural detection for synthetic matching. Do not tune another configuration within this packet. Successful pilot motivates independent seeds and broader verified relation families in the next packet; it is not a finalist or novelty claim.

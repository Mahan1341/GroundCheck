# R5: supervised numerical views — frozen before profiling/training

2026-10-24. Two arms, matched seed42; no release or outer evaluation.

Hypothesis: CE on a raw and a conservatively canonicalized view teaches numerical
equivalence more robustly than the same CE exposure twice to the raw view.
R5-A raw/raw versus R5-B raw/canonical. Both reload B7 inner42 (SHA256
7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4).
Natural inner-fit2568/428 plus existing R3 authored fit960/18, include/all.
One shuffled record-pair per microbatch: two views, accumulation4, one epoch,
3528 microbatches/882 updates provided both views have one window (fail otherwise).
CE class weights [1,3.370202392225359], overlap weights from existing encoder;
AdamW lr1e-5, weight decay.01, 10% linear warmup, clip1.0; BF16 autocast,
FP32 parameters/optimizer/loss, gradient checkpointing. No pair/ranking loss.
Identical seed/order/update/loss policy; token lengths/positive-token counts may
change through retokenization and are measured, not falsely asserted equal.
Same full evidence, question, answer encoder, 2048 limit, max-token aggregation.
No early stopping or intermediate metric selection; final one-epoch checkpoint.

Canonicalization is the frozen R4 conservative parser (no new conversions).
Map each original gold interval to its overlapping canonical interval; assert
annotation text, response label, all answer supervision and prefix masking.
Record interval expansion and positive-token counts. Inference remains raw for
R5-A/B. No gold-driven retrieval/windows. Ambiguous ranges/approximations unchanged.

Controls fixed before training/prediction: reused natural648/108, R3 diagnostic
640/12 (four names), R4 fresh864/12, and new R5 authored672/12. New names/values,
receipt/interview prose families, spaced word numbers, decimal exact-unit forms,
integer exact units, supported paraphrases, entity AND property distractors, and
36 true filler clauses with one short numerical error. Twelve fictional parent
worlds; variants/orders/value swaps stay grouped. Same answer is supported and
contradicted under crossed evidence assignments. Decimal forms are out of scope
for the parser and test learned extrapolation. No claim of untouched natural data.
R3 supported paraphrases and R5 paraphrases evaluated separately. Natural short
error slice: >=500 characters, gold nonwhitespace coverage <=5%. No mining of
model errors to alter the controls. Source-disjoint from all supervised fitting.

Compare B7, R3-A, R4 symmetric R3-A inference, R5-A, R5-B on identical IDs;
reuse hash-verified R4 outputs for old/fresh/natural, run new controls locally.
Fixed batch2 for all inference; B7 historical batch1 number is not substituted.
Threshold .37612172961235046, positive iff score>=threshold; token threshold.5.
No threshold fitting. Symmetric inference projected to original tokenizer grid.
Response accuracy/macro-F1/positive P,R,F1/AUROC/AP/FPR/FNR/CM; paired score
separation (positive-minus-supported mean and fraction>0), both-correct pair
accuracy; token/character/exact-span custom F1. Natural positives are annotated
hallucinations, not all contradictions. Synthetic contradiction recall separately.
Paired source percentile95 bootstrap2000 seed42, same draws for every method;
primary differences B-A and B-B7, secondary comparisons descriptive. Include
localization and pair metric bootstrap, category tables, natural642 sensitivity
excluding known shared-passage source15181. No independence claim for variants.

Resource gate: profile longest natural and changed synthetic pairs for BOTH
arms before training, actual backward/optimizer step; allocated<11GiB, finite
loss/gradients, complete answer/evidence coverage. Profile weights discarded.
Estimated two-arm budget <=30 GPU minutes, inference <=15min; report measured.
If estimates exceed these, document a resource decision before fitting, not
silently shorten context. Sequential processes only; no downloads/paid resources.

Repeat seeds43/44 ONLY if ALL seed42 conditions hold:
1. New R5 diagnostic B-A macro-F1>=.03 and paired95 lower bound>0;
   pair accuracy improves>=.05, and no supported-FPR increase>.02.
2. Natural B-A macro-F1>=-.01 and CI lower bound>-.02;
   B-B7 point macro-F1>=-.01.
3. Natural token and exact-span F1 do not fall>.02 versus A;
   new-control token F1 does not fall>.02; supported-paraphrase FPR increase<=.02.
4. Short-error natural recall does not lose more than one case versus A.
Otherwise stop this hypothesis after the matched seed; no rescue threshold,
additional architecture, new loss or selected seed. Bootstrap is conditional
on these fitted models and reused inner data, not training-seed uncertainty.

Freeze dataset/checkpoint/source/plan hashes and source ZIP before profiling.
Preserve previous artifacts, no source-hash bypass. Save training configs,
batch order/targets audit, logs, final checkpoints, predictions, timings/VRAM,
bootstrap/errors and final integrity manifest. Offline tests; GPU integration opt-in.

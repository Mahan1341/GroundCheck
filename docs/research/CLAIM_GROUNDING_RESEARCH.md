# Claim-conditioned grounding — two-cycle research packet, 2026-10-26

Status: two frozen-model cycles completed; both research gates failed. No training
or release. Total charged GPU-process time18.07/40min, including failed attempts and
integrity replay. The result is mechanistic negative evidence, not a better detector.

## Question and competing mechanisms

Can an evidence verifier identify the wrong relation and its exact location while
retaining discourse? Response verification, claim discrimination and localization
are separate targets. Reused RAGTruth inner data remain development. Original
RAGTruth outer calibration/dev/test and used RefChecker confirmation remain closed.

| Mechanism | Causal prediction | Falsifier / decision |
|---|---|---|
| Native verifier locality is hidden by pooled output | FactCG's own input sensitivity localizes wrong tokens better than B7's borrowed ranking | C1 frozen gradient readout must beat matched permutation, preserve natural and both factual-world recalls. Tested; rejected as a reliable readout. |
| Context contribution separates a target from surrounding discourse | Adding a wrong target increases unsupported margin even if its neighbor is wrong | C2 original vs joint-window vs joint-minus-context, independently controlled target/neighbor truth. No new prompt or answer-as-evidence. |
| Abandon claim decomposition and learn joint evidence–answer error spans | Contextual token representations plus direct localized supervision retain short errors and relations | Requires a compatible pretrained span head or controlled train-only fit; pooled classifier weights applied to every token are not a trained span head. No fit warranted by C1. |
| Predicate/QA representation makes the relation explicit | Answer-conditioned questions distinguish argument/value swaps without parser artifacts | Separate question validity/answerability audit before model comparison; ready cheap target-aware stack was not verified. Not implemented in this packet. |

Known precedents and exact checkpoint/source provenance are in
`CLAIM_GROUNDING_LITERATURE.md` (historical local artifact; not distributed):
[FactCCX](https://aclanthology.org/2020.emnlp-main.750/),
[QAFactEval](https://aclanthology.org/2022.naacl-main.187/),
[AlignScore](https://aclanthology.org/2023.acl-long.634/),
[Dependency Arc Entailment](https://aclanthology.org/2020.findings-emnlp.322/) and
[QASemConsistency](https://aclanthology.org/2026.tacl-1.6/).
None establishes our response-level macro-F1 or scientific novelty. FactCCX author
checkpoint URLs returned403 on HEAD, so readiness was not established. AlignScore
includes MS MARCO training, a source-overlap concern for our QA task. The inspected
QASemConsistency release did not provide a ready parser/verifier; a3B parser was not
justified. We did not download another model merely to exhaust the budget.

## C1: frozen native sensitivity, not a learned localizer

Exact pinned FactCG prompt/checkpoint/filtered units/evidence. Gradient L2 norm of
logit(No)-logit(Yes) with respect to word embeddings, best-support evidence branch.
Mean projection onto original B7 tokens, within-claim maximum normalization, unchanged
claim risk multiplier. This replaces the failed B7 attribution prior; it does not
change response scores or learn a new verifier. Gradient×embedding was not selected:
embedding LayerNorm makes radial dot products a poor uncontrolled readout.

Null: deterministic within-claim permutation in digit/non-digit × boundary/interior
bins, preserving the score multiset. Random final-head direction is a second sanity
check; original weights remain unchanged. Sensitivity is not error probability.

Initial full request2148 pairs failed the resource estimate:2318s remaining versus
990s cap. Its17 completed pairs are preserved. The profile JSON writer failed on
numpy.bool after recording33.083s; the partial original report is retained, its peak
VRAM is unavailable. Before quality metrics, a separate resource amendment selected
four hash-ordered source clusters per dataset, unchanged method/gates. Bounded run:
465 unique pairs,17 profile +445 remaining new pairs +3 exact-cache reuses.

Primary token threshold .5, no localization tuning. C1 natural24/4clusters/7positive
is a small diagnostic, not a general natural benchmark. Every comparison uses the
same IDs; R1/R6 synthetic sources remain distinct from natural evidence.

| C1 subset | Method | Token precision | Recall | F1 |
|---|---|---:|---:|---:|
| Natural24/4 | B7 | .5611 | .5372 | .5489 |
| Natural24/4 | Uniform FactCG | .3958 | .7979 | .5291 |
| Natural24/4 | Previous B7-relative composition | .5656 | .5505 | .5580 |
| Natural24/4 | Native gradient | .5634 | .1064 | .1790 |
| Natural24/4 | Stratified null | .4930 | .0931 | .1566 |
| R1 40/4 | Uniform / native / null | .1000 / .5714 / .2857 | 1 / 1 / .5000 | .1818 / .7273 / .3636 |
| R6 192/4 | Uniform / native / null | .1494 / .3713 / .4262 | .9474 / .3860 / .4430 | .2581 / .3785 / .4344 |

Gate FAILED. Native−old-composition natural token-F1 −.3790, paired95% CI
[−.4245,+.1146]; four clusters give very weak precision. R6 native−null −.0559,
CI[−.0862,−.0178]. Native-minus-null within-positive-claim AP: natural+.0674,
R1+.4097,R6+.0237 (required>=.05 on all three). Natural AP .9267 is misleading
without its denominator:14/19 positive proxy claims contain only positive tokens;
only5 are mixed. The uniform baseline already has AP .832 on those claims.

Short-error gold-hit: natural1/1 native vs uniform1/1; R1 8/8 vs8/8; R6 5/8 vs6/8.
These tiny counts cannot establish natural short-error retention. Head-randomization
rank correlation>.9 on10/20 available evaluated inputs (median .8904,max1), a warning
that common input sensitivity dominates. No truncation, fallback or removed-text
projection occurred; all response scores are exactly unchanged.

R1 demonstrates some localized signal. Therefore the correct conclusion is not
"FactCG contains no relation information": this particular L2/max readout fails to
transfer reliably. We did not try another norm, sign, coefficient or threshold.
Full metrics, source bootstrap and error traces: `reports/claim_grounding_20261026/
c1_bounded/c1_results.json`, ignored cases under the matching models namespace.

## New human-annotated transfer collection

[DiaSumFact](https://aclanthology.org/2023.acl-long.377/) is independently annotated
by the original authors' annotators, not this assistant. Frozen primary collection:
all59 released SAMSum dialogue sources/295 summaries,128 negative/167 positive.
It is dialogue summarization, **not QA**, and is new to this project, not guaranteed
unseen by checkpoint pretraining. The published-source16-word-window audit found no
match against available FactCG training texts or permitted R10 train/inner contexts;
5391 missing MuSiQue premises and unknown pretraining remain contamination limits.

Exact/uniquely whitespace-normalized mapping gives284 fully scorable summaries.
The11 ambiguous/unmatched examples are all positive: keep them in response detection,
exclude whole summaries only from primary localization and expose the resulting
156positive/128negative localization denominator. Never guess offsets or turn missing
spans into negatives. Human sentence/error-string annotations do not guarantee perfect
completeness. Full provenance, annotation agreement and alternatives:
`CLAIM_GROUNDING_DATA.md` (historical local artifact; not distributed).

Inputs/gold were exported separately before inference. The model input contains
no labels, annotation spans or generator IDs. Whole source groups are evaluated;
no threshold, unit policy or architecture will be tuned on these results. B7 native
FP32/batch1 baseline completed295 windows with complete evidence/answer coverage and
exact original offsets. No assistant annotation was requested or promoted to gold.

## C2: contextual contribution

Frozen in CLAIM_GROUNDING_C2_PLAN.md before predictions. A is the target-alone
unsupported margin; J is target plus immediate kept neighbors; C is neighbors only;
D=J−C. All are native evidence/hypothesis inputs, never a new instruction prompt or
answer text inserted as trusted evidence. Same filtering, offsets and source chunks.
Strict margin>0 is a diagnostic decision; exact0 abstains. D is not a probability.
No-neighbor cases fall back to A and are reported separately. No sub-claim resolution
is invented: token outputs mark the entire extractive target when its margin is positive.

The initial7057-pair request reused2175 entries;17 new profile pairs predicted919s
remaining>840s cap. A source-only resource amendment, before quality metrics, reduced
natural to72/12clusters and R6 to384/8parents. R1 remains150/15, new human295/59 and
factorial64/8. Final source/input freeze is in `c2_ready`; the earlier `c2_bounded`
CPU preparation is preserved but superseded by pre-inference formatting/namespace
correction. Its pair/layout bytes are identical; no predictions ran in that namespace.

Factorial controls concatenate existing atomic R6 claims and disjoint-parent evidence
worlds: target supported/erroneous × neighbor supported/erroneous, both answer orders.
No new factual templates or oracle inference features; original error offsets shift
exactly. Both parent sources define one bootstrap cluster. This tests whether two
errors cancel each other rather than treating arbitrary document replacement as gold.

### C2 natural development:72 answers /12clusters,20 positive

A=target alone; J=target with neighbors; D=J−C; B7 unchanged historical reference.
Thresholds: strict margin>0 for A/J/D, original .3761217296 for B7; token .5 for B7,
binary target flags for A/J/D. There is no threshold search or cross-fit refit in C2.
Do not compare these subset numbers directly with the earlier full642 OOF benchmark.

| Method | Macro-F1 | Positive P/R/F1 | AUROC / AP | FPR | Token P/R/F1 | Char / exact-span F1 |
|---|---:|---|---|---:|---|---|
| A | .6857 | .4750/.9500/.6333 | .8337/.6364 | .4038 | .3078/.8231/.4481 | .4684/.0169 |
| J | .6286 | .4250/.8500/.5667 | .7750/.5768 | .4423 | .2283/.8000/.3552 | .3786/.0143 |
| D | .4278 | .3158/.9000/.4675 | .6663/.4297 | .7500 | .0574/.2173/.0909 | .0969/.0109 |
| B7 | .7253 | .5600/.7000/.6222 | .8394/.7175 | .2115 | .5412/.6500/.5907 | .6059/.0000 |

Confusion matrices [[TN,FP],[FN,TP]]: A[[31,21],[1,19]], J[[29,23],[3,17]],
D[[13,39],[2,18]], B7[[41,11],[6,14]]. Short-error gold-hit A/J/D2/3, B7 0/3;
three examples do not establish preservation of short-error performance generally.

**The mechanism fails beyond its arbitrary zero boundary.** Among384 context-bearing
claim units (53 overlap-positive/331 unmarked), AP falls A .6201→J .3359→D .1597;
AUROC A .8822→D .3865. Claim labels here are annotation-overlap proxies, not fresh
human entailment judgments. Paired source2000 draws give D−A AP −.4604 CI
[−.6534,−.1517]; response macro-F1 −.2579 CI[−.4431,−.0806]; token-F1 −.3572
CI[−.4777,−.1822]. This is not solely a threshold-transport failure.

There are389 total targets,384 with neighbors and5 singleton fallbacks;9 answers
have no claim. No chunk-winner changes, exact-zero margins or |D|<1e−6 cases occur.
Thus epsilon handling or switching the best evidence chunk does not explain the
observed collapse. The context is only±1 kept units, not all answer discourse:
outside-window coreference remains an alternative limitation, not a tested remedy.

### Factual interventions and independent neighbor truth

| Suite / arm | Response macro-F1 | Positive P/R/F1 | Supported FPR | Gold-token recall / token-F1 |
|---|---:|---|---:|---|
| R1 150/15 A | 1.0000 | 1/1/1 | .0000 | 1/.3007 |
| R1 J | 1.0000 | 1/1/1 | .0000 | 1/.2100 |
| R1 D | .5238 | .5556/1/.7143 | .8000 | 1/.0186 |
| R6 384/8 A | .9661 | .9735/.9583/.9659 | .0260 | .9640/.2523 |
| R6 J | .9635 | .9588/.9688/.9637 | .0417 | .9685/.2262 |
| R6 D | .8873 | .8341/.9688/.8964 | .1927 | .9730/.1072 |

R1/R6 intervention recall alone is misleading: D retains errors by flooding correct
tokens/claims with positives. R1 D token precision is .0094 versus A .1769. This
does not improve localization. R1 response AUROC/AP remains .9701/.9762 for D,
so there is factual signal even though its zero operating point is unsuitable.

Factorial:64 responses/8 disjoint parent-pair clusters,128 target roles,32 factual
pairs per neighbor condition. Both answer orders are included, not independent samples.

| Arm | Neighbor | Pair accuracy | Target P / R | Supported target FPR | Target AUROC | Signed factual margin gap |
|---|---|---:|---|---:|---:|---:|
| A | supported | .9375 | .9412/1 | .0625 | 1.0000 | 2.5076 |
| A | erroneous | .9375 | .9412/1 | .0625 | .9961 | 2.5851 |
| J | supported | .5000 | .6667/1 | .5000 | .9590 | 1.4103 |
| J | erroneous | .0000 | .5000/1 | 1.0000 | .7734 | .5807 |
| D | supported | .0000 | .5000/1 | 1.0000 | .9180 | 1.4776 |
| D | erroneous | .2188 | .5577/.9063 | .7188 | .7832 | .5705 |

Adding an erroneous neighbor makes J unable to distinguish target verdicts at the
fixed boundary: every target becomes positive. Subtraction does not restore a
truth-valued target score. Yet positive signed factual gaps remain (D intervals
[1.1069,1.8162] with supported neighbor and [.4414,.6790] with wrong neighbor).
We therefore reject this readout/decision rule, not all factual information in the
joint representation. Nor did we establish that a dedicated target-conditioned
model would fail: FactCG was trained on whole-hypothesis verdicts, not marginal ones.

The preregistered neighbor-degradation gate has a weakness: D's pair accuracy rises
from0 to.2188, so a relative non-degradation condition passes despite terrible absolute
accuracy. Recall retention also admits near-all-positive outputs. The overall gate
still fails decisively on natural AP/detection/localization. We retain the original
gate unchanged; a future protocol needs absolute pair-accuracy and supported-FPR
floors in both neighbor conditions. This is a protocol lesson, not post-hoc success.

### Fixed new human transfer:DiaSumFact SAMSum

295 responses/59 sources;128 negative/167 positive. Localization uses284/295 fully
aligned summaries (156 positive/128 negative), never missing spans treated as zero.
Human error-category mapping is any author category except No Error. Different
task/taxonomy from QA; no source or threshold chosen after predictions.

| Method | Macro-F1 | Positive P/R/F1 | AUROC / AP | FPR | Token P/R/F1 | Char / exact-span F1 |
|---|---:|---|---|---:|---|---|
| A | .7255 | .7529/.7844/.7683 | .7739/.8077 | .3359 | .1788/.7088/.2856 | .2982/.0000 |
| J | .7069 | .7179/.8383/.7735 | .7759/.8090 | .4297 | .1340/.8524/.2316 | .2418/.0000 |
| D | .5246 | .6075/.9641/.7454 | .7020/.7420 | .8125 | .1054/.8790/.1882 | .1972/.0000 |
| B7 | .5067 | .6711/.3054/.4198 | .5863/.6559 | .1953 | .1770/.1476/.1610 | .1681/.0160 |

CM: A[[85,43],[36,131]], J[[73,55],[27,140]], D[[24,104],[6,161]],
B7[[103,25],[116,51]]. A macro-F1 CI[.6576,.7923], token-F1 CI[.2355,.3390].
Full paired differences and absolute P/R/F1 CIs are saved in c2_results.json.
Paired A−B7 macro-F1 CI[+.1348,+.3027]; token-F1 CI[+.0564,+.1962].
On673 aligned context-bearing claim proxies, AP A .7522/J .6153/D .5812.
Across all summaries760 targets,709 context-bearing and51 fallbacks; no zero or
near-zero delta and no evidence-branch switching. No example meets the pre-existing
long-answer short-error definition here; this set cannot validate that stratum.

Important update to the earlier story: **B7's natural token advantage does not
transfer to this human dialogue collection.** Frozen A beats B7 token-F1 here,
but its zero exact-span matches show the boundary problem remains. This is evidence
for the released verifier on this task/operating point, not an improvement from our
new C1/C2 methods, a SOTA claim, or a guarantee of clean pretraining provenance.
The new collection is now a used transfer measurement; do not retune on it or call
another evaluation independent confirmation.

## What caused the failures, and what remains uncertain

- C1 replaces only the local readout. Good R1 pointing alongside R6 null superiority
  implicates the sensitivity-to-error interpretation, not a changed factual verifier.
  Norm/max suppression loses diffuse gold spans; random-head similarity and all-positive
  AP inflation warn against treating saliency as semantic localization.
- J changes the target to a conjunction. An erroneous neighbor can make a correct
  target look wrong. D subtracts confidence in a different hypothesis, not an
  independently measured contribution to factuality. Logits are not additive truth
  scores, and additional supported content can also reduce confidence.
- Example12114/source14336, unmarked target: A=−1.8508,J=−1.6760,C=−2.4657,
  D=+.7897. Both absolute verdicts support the text, but subtraction flags it.
  Example12118, annotated error target: A=+1.6549,J=−.4684,C=−.0640,D=−.4045;
  the error disappears in context. These are existing annotation proxies, not new
  assistant-created human gold. Exact text, offsets and scores remain in ignored cases.
- Natural AP collapse rules out explaining C2 only by choosing a better zero threshold.
  In controlled worlds, positive gaps survive: there the bad working point also matters.
  We did not calibrate D to rescue either outcome or infer a universal missing ability.
- These tests cover adjacent context and coarse target units. They do not identify
  all coreference dependencies, independently measure checkworthiness, or prove that
  a learned claim-conditioned representation cannot work. Source/annotation uncertainty,
  small reused natural subsets and public-model provenance limit conclusions.

## Verdict and next scientific decision

Both frozen-model gates FAILED; **no matched training experiment was justified or
run**. No threshold/seed search, extra model scale or unused-budget spending.
Do not adopt gradient importance as an error map, multiply in B7 scores again, or
subtract whole-hypothesis logits as if they were factual contributions.

The strongest remaining hypothesis is a **directly supervised target-relation/error
representation**, including the option of full joint evidence–answer span modeling
without independent claim decomposition. DAE/FactCCX are precedents, not our novelty.
A next bounded experiment should first test whether a small supervised probe on
frozen evidence-conditioned features can recover error spans on source-disjoint
train folds, versus an identical answer-only probe. It must include argument swaps,
multi-error neighbors, absolute supported-FPR constraints and span recall—not just
max-score separation. That is a proposed new training protocol, not a completed
experiment or an architecture selected from the external results. No user annotation
is needed: existing train spans supply fit targets; fresh public human data would be
needed for another independent transfer claim. The now-used DiaSumFact set stays fixed.

## Reproducibility, resource failures and verification

Commands: `CLAIM_GROUNDING_REPRODUCE.md` (historical local artifact; not distributed).
New namespace `models/predictions/reports/claim_grounding_20261026`; no old artifacts
overwritten. Raw data/caches/predictions and cases remain ignored; source snapshots
and hashes preserve executed versions. No new weights, dependencies, API calls for
generation, fit checkpoints, commits, push or release.

Resource gates reduced scope BEFORE outcome inspection; their original failed
requests are retained. C2 CUDA unknown error stopped after2212 new pairs. One fresh
context resumed only1477 missing keys; six fixed pre/post-reset sentinels reproduced
logits exactly (maximum difference0), no old cache overwrite. Original failed log and
resources remain. This is a hardware/runtime failure, not a negative model experiment.

Total18.0721/40 charged GPU-process minutes,4486 forwards including profiles and
six replays, sequential jobs. C1 bounded peak allocated3.80GiB; C2 about1.84GiB;
B7 external .627GiB. Failed initial C1 peak was not serialized, and is unavailable.
Run+resume+replay C2 time740.95s stays within840s run allocation. Per-stage latency,
RSS, VRAM, hashes and budget ledger are saved. Cache-assisted times are not independent
uncached throughput benchmarks. B7 external inference17.57s for295 responses;
34.77s with load/validation. No unfinished GPU job remains.

Final offline pytest367 passed/8 skipped (60.04s), Ruff passed, both environments'
pip check passed. The preceding365-test run is also retained.
No remote-CI pass is claimed. Token/nonwhite-character/exact merged-span metrics use
the existing custom scorer, not the official RAGTruth scorer. Bootstrap2000/seed42
is paired by source and conditional on frozen decisions; it does not correct adaptive
development use, training-seed variability or annotation exclusions. Final integrity
manifest/snapshot verification is recorded alongside the tests and resource ledger.
The final audit also caught a sealer layout assumption: C2 analysis snapshots retain
repo directories, unlike older flat copies. The sealer now validates hierarchical
copies against exact frozen paths and legacy flat copies against frozen content.
Two regressions cover both layouts and reject changed or wrong-path sources. No
source snapshot, analysis freeze, prediction or model value changed for this repair.

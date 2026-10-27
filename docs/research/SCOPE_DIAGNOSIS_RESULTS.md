# Matched claim scope, operating points and localization — 2026-10-26

Research packet complete; research remains active. No new training or release.
Frozen plans: `diagnosis` (historical local artifact; not distributed), `architecture pilot` (historical local artifact; not distributed).
Commands and provenance: `SCOPE_REPRODUCE.md` (historical local artifact; not distributed).
Primary machine-readable results: `reports/scope_diagnosis_20261026/diagnostics.json`
and `pilot_results.json`. Previous predictions, weights and freezes are unchanged.

## What the contrasts establish

1. **Operating point accounts for much of the observed natural response gap.**
   Segmented FactCG improves from .6592 to .7922 macro-F1 with source-cross-fitted
   thresholds; B7 changes from .7956 to .7765. Their new difference is uncertain.
   This is an operational contrast on reused development, not a causal percentage
   of errors, probability calibration or a new generalization result.
2. **Checkworthiness is a real fixed-threshold confound, not the whole explanation.**
   Matched filtering removes 37 segmented false positives and one true positive at
   .5. After separate threshold selection it does not improve segmented macro-F1.
   No response-ID exception or blanket refusal exclusion was introduced.
3. **Whole-answer context is not a general repair.** With exactly the same retained
   text, whole-filtered is worse than segmented-filtered after threshold selection:
   delta macro-F1 −.0967, conditional paired95% CI [−.1501,−.0439]. This intervention
   changes hypothesis length, conjunction/context and number of maxima together;
   it does not identify coreference loss or dilution as the cause.
4. **Verification and precise localization are different unresolved problems.**
   FactCG still separates controlled factual worlds much better than B7; B7 has
   better natural token localization. Their cheap frozen composition improves
   precision but suppresses real error tokens. Its preregistered gate failed.

## Scope, labels and selection

Natural: same642 responses/107 source IDs, 204 positive/438 negative; 106 connected
clusters after grouping the known shared-passage sources14325/15545. These are
repeatedly used train-side inner-development data, not independent held-out evidence.
No B7 supervised fitting uses these sources. Grouping here does not change any
Phase1.1 or historical split. It does not establish absence of unknown near duplicates.

Five deterministic source folds, namespace `groundcheck-operating-point-audit-v1:42`:
held-out clusters/rows/positive =22/138/53,21/126/31,21/126/45,21/126/41,21/126/34.
Every row's operating point is selected using only the other four source folds.
Response threshold maximizes macro-F1; separate token threshold maximizes positive
token-F1. All distinct boundaries, positive when score>=threshold; exact ties choose
the highest threshold. No held-fold labels enter its threshold selection.

Historical response thresholds: B7 .37612172961235046, FactCG .5; token .5 for all.
Fold response thresholds: B7 [.379568,.600461,.632072,.391292,.379568];
segmented-filtered [.846271,.818057,.846271,.846271,.837293];
whole-filtered [.509300,.502011,.502011,.502011,.502011]. Full precision in saved JSON.
These are model-specific development operating points, not new production defaults.
The historical point remains primary historical evidence, even when cross-fitting
looks worse. B7's original threshold was already selected historically on inner data.

R1:150 responses/75 same-answer factual pairs/15 parents. R6:768/384/16.
They are assistant-authored controlled worlds, not human natural labels. Thresholds
for transport are selected on all allowed natural data and saved BEFORE controlled
scoring in each analysis script; controls never select thresholds. Their annotations
were already deserialized during layout preparation (unused for extraction/selection)
and inspected historically. This is not a first-access or blinded-controls claim.

Secondary external **development only**:274 responses/40 sources,82 positive/192
negative. Its own five source folds select thresholds; it is neither a fresh test nor
the closed external confirmation. No confirmation data/predictions were deserialized
for this packet. Hash-only integrity checks do not evaluate them. Outer RAGTruth
calibration/dev/test remain closed.

RAGTruth include/all annotations provide response and error-span labels; absence of
a marked span is not proof of entailment. External RefChecker human judgments apply
to Claude2-extracted triplets, not exhaustive independently enumerated claims. Response
positive=any Neutral/Contradiction, negative=nonempty all Entailment, empty=unknown.
No external span gold exists. Assistant reviews, including the12-case FP audit, are
provisional explanations only. There is no human checkworthiness/completeness gold
here, so a reliable causal split between verifier errors and annotation mismatch
cannot be estimated. Published training-text overlap checks have missing premises
(5391 MuSiQue premises); pretraining exposure is unknown. No contamination-free claim.

## Matched inputs and extraction completeness

2x2 design: segmented/whole × all/filtered. Both units receive identical retained
original substrings for each policy. `all` restores standalone meta/refusal segments;
both policies exclude bare list/format markers. Whole joins the same exact substrings
with newlines and preserves piecewise original offsets. This normalized whole view
is a new named adapter; it does not overwrite the earlier raw-whole experiment.
Evidence chunks, checkpoint, prompt and max-support rule are otherwise unchanged.

Of642 natural answers:372 have no explicit drops,235 partial drops,35 no retained
claim. The35 emerge from the generic existing rule, never an ID list. Of106 answers
with meta/refusal drops,71 also retain text;63 mixed refusals retain factual clauses,
including34 positive responses. A mixed refusal's factual explanation is still checked.
No-claim score0 means abstention from structural checking, not verified support.

Of37949 non-whitespace gold characters,37592 remain in kept units;291 dropped
characters in51 rows are list markers, and66 unassigned formatting characters are
63 asterisks/3 hyphens. Meta/refusal drops overlap **zero** annotated gold characters.
This measures annotation retention only, not completeness of extracted facts or
checkworthiness precision/recall. Original offsets/maps and all dropped reasons are
saved under `models/scope_diagnosis_20261026`; no gold guides inference extraction.

## Natural response results

H=historical; OOF=source-cross-fitted operating point on reused development.
P/R/F1 refer to hallucination-positive class, not macro-F1.

| Method | Point | Macro-F1 | P | R | Positive F1 | Confusion [[TN,FP],[FN,TP]] |
|---|---|---:|---:|---:|---:|---|
| B7 | H | .7956 | .6838 | .7843 | .7306 | [[364,74],[44,160]] |
| B7 | OOF | .7765 | .6822 | .7157 | .6986 | [[370,68],[58,146]] |
| Segmented all | H | .6044 | .4429 | .9510 | .6044 | [[194,244],[10,194]] |
| Segmented all | OOF | .7988 | .7156 | .7402 | .7277 | [[378,60],[53,151]] |
| Segmented filtered | H | .6592 | .4825 | .9461 | .6391 | [[231,207],[11,193]] |
| Segmented filtered | OOF | .7922 | .7042 | .7353 | .7194 | [[375,63],[54,150]] |
| Whole all | H | .6511 | .5088 | .5686 | .5370 | [[326,112],[88,116]] |
| Whole all | OOF | .6447 | .5022 | .5539 | .5268 | [[326,112],[91,113]] |
| Whole filtered | H | .6986 | .5969 | .5735 | .5850 | [[359,79],[87,117]] |
| Whole filtered | OOF | .6955 | .5959 | .5637 | .5793 | [[360,78],[89,115]] |

AUROC/AP (unchanged by threshold): B7 .8759/.7897; segmented all .8715/.7631;
segmented filtered .8814/.7694; whole all .6920/.4879; whole filtered .7462/.5515.
AP is Average Precision, not trapezoidal PR-AUC. Accuracy/FPR/FNR and all strata
by answer length, generator and annotation type are in `diagnostics.json`.

Paired source bootstrap2000/seed42, same draws per method. OOF delta segmented
filtered−B7 macro-F1 +.0157 CI[−.0273,.0597]; all−B7 +.0223 CI[−.0209,.0673].
Filtering−all: fixed+.0549 CI[.0344,.0766], OOF−.0066 CI[−.0170,.0016].
These intervals condition on the frozen OOF decisions; no repeated threshold fit,
architecture-selection or historical-validation-use correction. AUROC/AP are points.

Whole-filtered OOF macro-F1 falls to .5634 among154 answers>=200 tokens, versus
segmented .7433 and B7 .7390. This descriptive stratum does not isolate length from
claim count, label mix or semantics. It provides no reason to switch globally to whole.

## Localization and falsifying pilot

Three alternatives were considered AFTER diagnosis: (1) hierarchical claim verifier
plus conditional token localizer; (2) context-conditioned target-claim verification;
(3) joint checkworthiness and evidence verdict. Detailed predictions and limitations
are frozen in `SCOPE_PILOT_PLAN.md`. Choose1 because existing scores suffice for a
zero-GPU falsifier of B7's within-claim ranking, without altering response decisions.

For each retained claim: local score = FactCG claim risk × B7 token risk / maximum
B7 token risk in that claim. Overlaps take max. No new weights, labels, prompts or
normalization search; response/claim scores stay bit-for-bit unchanged. Zero-denominator
fallback is uniform claim risk (0 natural occurrences). Removed gaps remain excluded.
This is a known composition idea, not scientific novelty or calibrated probability.

Token labels use any original gold-character overlap; character metric counts
non-whitespace characters; exact spans trim/merge overlapping or whitespace-adjacent
ranges and require exact boundaries. Custom micro scorer, **not official RAGTruth**.
OOF token thresholds are selected separately from response thresholds.

| Method | Point | Token P | Token R | Token F1 | Character F1 | Exact-span F1 | Short gold-hit /19 |
|---|---|---:|---:|---:|---:|---:|---:|
| B7 | H | .5883 | .5857 | .5870 | .5912 | .0379 | 8 |
| B7 | OOF | .5846 | .5890 | .5868 | .5914 | .0341 | 8 |
| Segmented filtered | H | .2782 | .7610 | .4074 | .4184 | .0357 | 14 |
| Segmented filtered | OOF | .4197 | .5896 | .4903 | .5001 | .0470 | 8 |
| Whole filtered | H | .1688 | .5020 | .2527 | .2569 | .0109 | 12 |
| Whole filtered | OOF | .1766 | .4046 | .2459 | .2503 | .0129 | 6 |
| Frozen composition pilot | H | .5148 | .5846 | .5475 | .5622 | .0100 | 13 |
| Frozen composition pilot | OOF | .5997 | .5259 | .5604 | .5758 | .0135 | 13 |

OOF token-F1 segmented−B7 −.0965 CI[−.1360,−.0508]. Pilot−segmented +.0700
CI[.0372,.0995], pilot−B7 −.0264 CI[−.0510,.0001]. Absolute pilot CI[.4895,.6217].
Character/exact-span metrics are points, not bootstrapped intervals.

The pilot fails preregistered gates: B7−.02 floor, <=.05 recall loss (actual−.0637),
and both controlled gold-token-recall retention gates. Short-hit improves but cannot
rescue these failures. Exact spans fragment: pilot13 exact hits/1600 predicted ranges,
versus segmented19/480 and B7 17/670 at OOF token points. Better token precision is
not reliable boundary recovery. No alternative composition or training was attempted.

At historical response points, short-error response TP: B7 14/19, segmented18/19;
OOF14/19 and11/19. These response counts must not be confused with the independently
thresholded token gold-hit counts. Pilot response scores/counts are exactly segmented.

## Factual sensitivity and operating-point transport

| Suite / method | Historical response P/R/F1 | Historical pair accuracy | Natural-threshold pair accuracy | Signed slot gap |
|---|---|---:|---:|---:|
| R1 B7 | .4000/.0533/.0941 | .0000 | .0000 | .0105 |
| R1 segmented filtered | 1/1/1 | 1.0000 | .7333 | .7774 |
| R1 whole filtered | .9494/1/.9740 | .9467 | .9467 | .5594 |
| R6 B7 | .4928/.1771/.2605 | .0182 | .0156 | .0081 |
| R6 segmented filtered | .9734/.9531/.9632 | .9271 | .2344 | .5160 |
| R6 whole filtered | .9683/.9531/.9606 | .9219 | .9219 | .5012 |

Global natural segmented threshold .8462712168693542 gives control precision1 but
recall .7333/.2344. Scores/gaps are unchanged: this is threshold transport failure,
not lost raw relation sensitivity. Natural-global thresholds were never optimized
using control labels. No universal improved operating point has been demonstrated.

| Control | Segmented token P/R/F1 at .5 | Pilot token P/R/F1 at .5 | Pilot signed slot gap |
|---|---|---|---:|
| R1 | .1769/1/.3007 | .2607/.5304/.3496 | .7774 |
| R6 | .1610/.9544/.2755 | .5109/.6250/.5622 | .4967 |

Counterexample to using max-slot gap alone: R1 gap stays identical while gold-token
recall nearly halves. A high score somewhere in the claim does not localize its
wrong value. Both normalized-score controls lose more than the allowed10% relative
recall. Natural-selected token thresholds also fail to restore coverage (pilot R1
recall .4696, R6 .4871). Pair gaps and their source intervals are in `pilot_results.json`.

## External development, not another confirmation

Same274 answers/40 source groups; all/filtered coincide here. No span labels.

| Method | Historical macro-F1 / P / R | Source-OOF macro-F1 / P / R |
|---|---|---|
| B7 | .6147/.4730/.4268 | .5679/.3929/.4024 |
| Segmented | .7054/.5490/.6829 | .6734/.5513/.5244 |
| Whole | .7249/.6173/.6098 | .7063/.6324/.5244 |

This small, previously used development set gives no evidence that natural-tuned
operating points transfer globally. Its folds cannot rehabilitate an already used
dataset as independent validation. External confirmation remains closed.

## Error evidence and next hypothesis

Ignored `models/scope_diagnosis_20261026/pilot_error_changes.json` preserves all changed
natural token counts, original answers and gold offsets. Illustrative IDs are audit
references, never inference rules. Response16367/source12219 loses115 false-positive
tokens without a marked error; response16365/same source loses104 FP but also63 TP
inside a long annotated tax-document comparison. Response12412/source14386 loses51
TP while gaining3 FP in a multi-part retirement-account answer. These illustrate
precision improvements and suppression of distributed wrong claims; they are not
new human judgments or specially selected validation samples.

The frozen relative-ranking hypothesis is rejected as a reliable combined localizer.
This does not reject every hierarchical architecture. It does reject using B7's
existing attribution ranking as if it were a faithful explanation of FactCG's verdict.
There is also no evidence for replacing segmented checking with whole-answer scoring.

Next most informative hypothesis: **a verifier must represent the target relation
separately from surrounding discourse to localize a factual mismatch**. Before any
fit, assemble independently reviewed extractive target/context cases from a newly
reserved train-side source group: same fact with minimal sufficient vs full discourse,
explicit target offsets, complete claim enumeration and relation-error locations.
Compare a compatible target-conditioned verifier against unchanged segmented FactCG;
require preserved factual-pair recall and improved localization, not just fewer alarms.
Assistant drafts may help review but cannot supply independent gold. If target/context
conditioning does not improve reviewed attribution, reject this explanation before
training a span head. No architecture, training budget or new model is selected here.

## Engineering and integrity

6243/7367 unique evidence–claim pairs reused. Only1124 missing pairs ran once on the
pinned FactCG checkpoint:33 profile pairs +1091 remaining;174.49s total GPU-process
wall time including loading (2.91min), peak allocated about2.01GiB, max893 input
tokens, zero truncation. FP32/batch1/TF32off. Pilot uses zero GPU forwards. Timings
are cache-assisted measurements, not full cold-deployment throughput benchmarks.

Source hashes/freezes, mappings, thresholds, predictions, cases and model revisions
are preserved under the new namespace. No old artifact overwritten. The previous
source ZIP retains old versions of shared living documentation; the new integrity
manifest verifies those historical bytes separately and current immutable artifacts
directly. Existing source/dependency resume guards are not bypassed.

One post-run formatting correction split an overlong string literal in scope_pilot.py
for Ruff E501. The original executed file and hash remain in pilot_source; parsed
ASTs are exactly equal, recorded in pilot_source_formatting.json. The current source
is a separately hashed version, never represented as matching the executed byte hash.

Verification and final source snapshot are recorded in `verification.json` and
`integrity.json` in the report namespace. See reproduction document for exact commands.
Local offline pytest:308 passed/8 skipped (81.74s). Ruff passes after the documented
formatting-only correction. Both base and research environments pass pip check.
R9 136/R10 120 historical artifacts, B7/R10A/R10B checkpoint hashes and original
analysis-source guards pass. GPU integration opt-ins were not rerun unnecessarily.
No raw texts/weights/cache committed; no commit/push/release or remote-CI claim.

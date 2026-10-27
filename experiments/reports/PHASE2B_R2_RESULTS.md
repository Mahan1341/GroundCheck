# R2 — Relation generalization and evidence consistency

2026-10-24. Internal research packet; not release, final benchmark or novelty claim.

## Recovery and question

Before changes, all92 original pause-manifest files,174 R1 artifact files and114 source-snapshot members matched SHA256. Snapshot CRC also passed. No training was running; no B7/B8/C0/C1 fit was repeated. Detailed check: reports/phase2b_r2/r1_recovery.json. R1 reports and source ZIP remain unchanged; current source additions have new snapshots, not substituted old hashes.

R1 natural INNER648 responses/108 sources: B7 macro-F1 .795935, B8 .706122, B6 .775256, C0 .795010, C1 .794043. C1 had27/28 correct on the narrow unseen-parent numeric probe, but only2/75 fully correct authored relation pairs. These are distinct tasks/denominators, not outer dev/test results. R1 CIs, localization and model lineage are retained in PHASE2B_R1_RESULTS.md.

Hypothesis: the old counterfactual fit learned a useful relation in a narrow presentation format; a matched claim-local evidence-ranking loss might improve transfer beyond broader synthetic CE alone. Ranking/contrastive losses and counterfactual augmentation are established methods.

## Frozen diagnostic suite

EXPERIMENT_R2_PLAN.md predates new predictions. Deterministic exact fictional worlds: batch bolt counts, elapsed trial durations and first-opening years. They are authored, not facts extracted from RAGTruth. Four combinations of two answer values and two evidence assignments give balanced labels conditional on answer and premise. Both numbers remain in every paired premise; orders are crossed. Values are not A/2A. Only the incorrect numeric answer slot is annotated. All correct filler assertions occur in evidence. Minutes/seconds are exact conversions; word-number rendering is deterministic. No arbitrary evidence substitution is assigned gold.

Fit:768 rows/24parents,384positive. Selection:936/9,468positive. Diagnostic:1248/12,624positive. Parent/source sets are disjoint. Fit evidence families record/parallel, selection held-out family catalogue, diagnostic held-out family reported. Familiar-format diagnostic variants intentionally share rendering families with fit but have new parents/values; they are separately labeled. Known-values/new-wording reuses fit values with new parents. The evidence-family split does not imply independent authors, linguistic domains or all-new answer syntax.

Manifest and exact generated examples: models/phase2b_r2/suite_manifest.json and ignored predictions/phase2b_r2/{fit,selection,diagnostic}_{records,design}.json*. Generator and semantic tests preserve exact spans, label/marginal balance, paired normalized word multisets, source/family isolation and inference allowlist. Representative numeric/date/duration/word/unit/multifact cases were read before predictions. Shared correct padding intentionally isolates answer-length changes; its repetitive nature limits external validity.

Response risk remains max answer-token hallucination probability. Claim-local and numeric-slot max are computed *after inference* using diagnostic offsets. MiniCheck claim risk is its normal predicted segment overlapping the known claim, without privileged rerunning on an oracle extraction; token localization is unavailable for MiniCheck. All custom token/nonwhitespace-character/exact-span F1 use fixed .5 and the R1 scorer, not an official RAGTruth span scorer.

No new outer calibration, dev or test evaluation is part of R2. Old inner thresholds remain fixed; both new arms will use B7's fixed .37612172961235046 as preregistered. This makes threshold-scale shifts visible, but does not measure each model's optimally recalibrated operating point.

## Read-only diagnosis (completed)

C1 separates44/48 record-format pairs,5/48 ordinary parallel-fact pairs,5/48 new-wording pairs. Its mean response deltas are .7128/.0328/.0229 respectively. New parents and non-doubled values therefore do not remove its format dependence. Correct word-number and minutes-to-seconds answers have FPR1.0 (32 and16 supported cases); neither type is a credible semantic invariant for C1 yet. This is stronger evidence of a format/representation limitation than of a max-only aggregation problem: claim-local deltas are weak too. It does not prove one unique internal causal mechanism.

Read-only baseline results follow below. Matched training was started only after this diagnostic and a successful resource profile.

### Старые модели: все1248 diagnostic rows

| Model | Accuracy | Macro-F1 | Positive P/R/F1 | AUROC | AP | FPR/FNR | CM |
|---|---:|---:|---|---:|---:|---|---|
| B7 | 0.5208 | 0.4973 | 0.5367/0.3045/0.3885 | 0.5319 | 0.5214 | 0.2628/0.6955 | [[460, 164], [434, 190]] |
| B8 | 0.5000 | 0.3333 | 0.0000/0.0000/0.0000 | 0.5000 | 0.5000 | 0.0000/1.0000 | [[624, 0], [624, 0]] |
| C0 | 0.5112 | 0.4795 | 0.5222/0.2644/0.3511 | 0.5295 | 0.5189 | 0.2420/0.7356 | [[473, 151], [459, 165]] |
| C1 | 0.5545 | 0.5407 | 0.5833/0.3814/0.4612 | 0.5723 | 0.5529 | 0.2724/0.6186 | [[454, 170], [386, 238]] |
| B6 | 0.5144 | 0.3853 | 0.6731/0.0561/0.1036 | 0.5793 | 0.5703 | 0.0272/0.9439 | [[607, 17], [589, 35]] |

Fixed thresholds: B7=.37612173, B8=.43734282, C0=.50732368, C1=.50488263, B6=.93902482. AP is Average Precision, not trapezoidal PR-AUC. CM rows gold/columns predicted, [0,1].

| Transformation | Pairs | B7 pair acc | B8 | C0 | C1 | B6 | C1 response/claim delta |
|---|---:|---:|---:|---:|---:|---:|---|
| all | 624 | 0.0721 | 0.0000 | 0.0449 | 0.1490 | 0.0337 | 0.0765/0.0775 |
| answer_paraphrase | 48 | 0.0625 | 0.0000 | 0.0000 | 0.1250 | 0.0000 | 0.0301/0.0301 |
| known_values_novel | 48 | 0.0625 | 0.0000 | 0.0208 | 0.1042 | 0.0208 | 0.0286/0.0286 |
| multi_fact_sentence | 48 | 0.0625 | 0.0000 | 0.0208 | 0.0417 | 0.0625 | 0.0118/0.0122 |
| novel | 48 | 0.0833 | 0.0000 | 0.0417 | 0.1042 | 0.0208 | 0.0229/0.0229 |
| number_words | 32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0312 | 0.0001/0.0001 |
| pad32_first | 48 | 0.0417 | 0.0000 | 0.0625 | 0.1667 | 0.0208 | 0.0369/0.0504 |
| pad32_last | 48 | 0.0417 | 0.0000 | 0.0625 | 0.1458 | 0.0208 | 0.0509/0.0511 |
| pad32_middle | 48 | 0.0208 | 0.0000 | 0.0000 | 0.0208 | 0.0208 | 0.0042/0.0030 |
| pad8_first | 48 | 0.0417 | 0.0000 | 0.0208 | 0.1458 | 0.0208 | 0.0411/0.0412 |
| pad8_last | 48 | 0.0208 | 0.0000 | 0.0208 | 0.0625 | 0.0208 | 0.0095/0.0095 |
| parallel | 48 | 0.0208 | 0.0000 | 0.0000 | 0.1042 | 0.0000 | 0.0328/0.0328 |
| record | 48 | 0.4583 | 0.0000 | 0.3125 | 0.9167 | 0.1875 | 0.7128/0.7128 |
| target_middle | 48 | 0.0208 | 0.0000 | 0.0208 | 0.0000 | 0.0000 | 0.0128/0.0128 |
| unit_conversion | 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | -0.0001/-0.0001 |
| date | 192 | 0.0260 | 0.0000 | 0.0052 | 0.0781 | 0.0052 | 0.0650/0.0681 |
| duration | 224 | 0.1295 | 0.0000 | 0.0804 | 0.1562 | 0.0893 | 0.0630/0.0630 |
| quantity | 208 | 0.0529 | 0.0000 | 0.0433 | 0.2067 | 0.0000 | 0.1016/0.1017 |

All-pair correctly directed changes: B7=0.5433, B8=0.0000, C0=0.5561, C1=0.5962, B6=0.5369. Direction alone is not successful binary separation.

| Model | Token / character / exact-span F1 (.5) | Scoring seconds | Responses/s | Peak CUDA allocated GB | Peak RSS GB |
|---|---|---:|---:|---:|---:|
| B7 | 0.1057/0.0568/0.0438 | 53.33 | 23.40 | 0.890 | 1.559 |
| B8 | 0.0000/0.0000/0.0000 | 53.58 | 23.29 | 0.887 | 1.576 |
| C0 | 0.1036/0.0527/0.0284 | 53.71 | 23.24 | 0.890 | 1.559 |
| C1 | 0.3552/0.2407/0.3327 | 54.03 | 23.10 | 0.890 | 1.558 |
| B6 | N/A | 265.63 | 4.70 | 2.453 | 3.630 |

Timing includes scoring with resident model, not loading/checkpoint hashing or statistical analysis. B7/B8/C0/C1 batch1; MiniCheck batch4 over12864 segment/document pairs, no truncation, maximum115 model tokens. Model-size/cost is not normalized. CUDA allocated is not total driver VRAM.

Paired parent bootstrap:2000 draws, seed42,12parent groups. Macro-F1 absolute95%: B7[.4497,.5255], B8[.3333,.3333], C0[.4174,.5088], C1[.5004,.5609], B6[.3481,.4266]. C1−B6 macro-F1[.1164,.1894] at these fixed natural thresholds. C1 has *lower* AUROC/AP than B6, so the thresholded gap is not a ranking-superiority result. C1 overall pair accuracy95%[.1138,.1875], B6[.0083,.0675]. Full absolute/difference response metrics and category-specific grounding intervals: comparison_diagnostic_B7_B8_C0_C1_B6.json.

The new suite changes target naming and distractor/context structure relative to R1. Related entity names (target and target Annex), repeated filler and only four parents per relation type limit causal attribution and precision of intervals. Bootstrap measures parent resampling conditional on this authored design; it does not correct design/model selection or estimate independent language-domain transfer.

## Matched CE vs CE+pair: completed pilot, seed42

Both arms reload B7 inner42 SHA256 `7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4`. One fixed epoch,3336rows/452sources (2568natural/428sources +768synthetic/24parents),1173positive responses.1668microbatches,417AdamW steps, microbatch2/accumulation4, lr1e-5, weight decay.01,10%warmup, clip1, BF16 autocast, FP32 parameters/optimizer/loss, max_length2048. Fixed positive CE weight3.370202392225359. No early stopping or threshold recalibration.

Exact config comparison differs **only in pair_weight0 versus.1**. Identical order SHA256 `9bbf728a5a308e03fcf14065df57d1728db5405c0b9adbc159ef38425284e963`, target SHA256 `97c712780534f22fc1b674117887365aca2be49edeccd166350b46f7eafb0fbc`; target mass403637negative/33636positive answer tokens. Every synthetic pair has identical answer tokens and aligned numeric slot. Prefix/question/evidence/special tokens have no supervised answer loss. Both arms compute the same pair graph; lambda0 contributes zero gradients.

Let s be mean(logit_hallucination − logit_not_annotated) over aligned numeric-slot answer tokens. L = weighted token CE + lambda·softplus(1 − (s_contradicted − s_supported)). Margin1 is one log-odds unit. The numeric slot is the minimal changed factual component; averaging all correct surrounding words would conflict with span supervision. Generated masks are used only in training and after-the-fact diagnostics. Production inference keeps max probability over *all* answer tokens. Both arms use fixed B7 threshold.37612172961235046.

### Selection gate, decided before diagnostic model predictions

On unseen catalogue-family selection72rows/9parents/36pairs, CE correctly separates8/36; CE+pair3/36. Required gain>=.05; actual−.138889. Mean delta gain+.002862 passes its sign gate; natural MF1 retention, token retention and answer-paraphrase FPR gates pass. The full conjunction fails. Seeds43/44 were **not run**; no best-seed selection or new hyperparameter trial. This stops this configured loss variant, not research or all pairwise supervision.

On that category, supported mean risk rises .4742→.7048 and contradictory .6005→.8340; FPR20/36→32/36. AUROC improves .6420→.7176, but fixed-threshold MF1 falls .6000→.4374. Ranking loss constrains a difference; a common log-odds shift of both pair members leaves it unchanged. CE provides an absolute anchor on training formats, yet this anchor transfers poorly to catalogue wording. This observed operating-point shift is not proof that the learned ranking signal vanished. No threshold was changed to hide it.

### Response-level metrics

| Split | Arm | N / positive | Accuracy | Macro-F1 | Positive P/R/F1 | AUROC | AP | FPR/FNR | CM |
|---|---|---|---:|---:|---|---:|---:|---|---|
| natural | R2C0s42 | 648 / 206 | 0.8102 | 0.7847 | 0.6895/0.7330/0.7106 | 0.8697 | 0.7741 | 0.1538/0.2670 | [[374, 68], [55, 151]] |
| natural | R2C1s42 | 648 / 206 | 0.8133 | 0.7872 | 0.6977/0.7282/0.7126 | 0.8725 | 0.7746 | 0.1471/0.2718 | [[377, 65], [56, 150]] |
| selection | R2C0s42 | 936 / 468 | 0.6987 | 0.6928 | 0.6555/0.8376/0.7355 | 0.7890 | 0.7910 | 0.4402/0.1624 | [[262, 206], [76, 392]] |
| selection | R2C1s42 | 936 / 468 | 0.6720 | 0.6562 | 0.6203/0.8868/0.7300 | 0.8034 | 0.8024 | 0.5427/0.1132 | [[214, 254], [53, 415]] |
| diagnostic | R2C0s42 | 1248 / 624 | 0.7949 | 0.7913 | 0.7335/0.9263/0.8187 | 0.8868 | 0.8742 | 0.3365/0.0737 | [[414, 210], [46, 578]] |
| diagnostic | R2C1s42 | 1248 / 624 | 0.7772 | 0.7687 | 0.7002/0.9696/0.8132 | 0.9015 | 0.8886 | 0.4151/0.0304 | [[365, 259], [19, 605]] |

Natural648/108 remains unchanged. Old B7/B6 predictions were reused byte-for-byte, not recomputed or retuned; source hashes saved in B7_natural.json/B6_natural.json. Natural B7 macro-F1 .795935 remains above both arms. R2-C1−R2-C0 MF195%CI[−.012462,.018407]; R2-C1−MiniCheck[−.027826,.050764]. Neither supports a natural superiority claim. Natural absolute95%MF1: CE[.748257,.820188], pair[.750093,.824855].

Predeclared partial-passage sensitivity removes only source15181 (six rows), no threshold changes:642-row MF1 B7=.795557, B6=.776130, CE=.784160, pair=.786645. This does not change primary membership or remove development bias.

### Diagnostic transformations

| Category | Pairs | CE / pair correct | CE / pair directed | CE / pair response delta | CE / pair claim delta | CE / pair FPR |
|---|---:|---|---|---|---|---|
| all | 624 | 0.5929/0.5577 | 0.9151/0.9215 | 0.5464/0.5598 | 0.5477/0.5603 | 0.3365/0.4151 |
| answer_paraphrase | 48 | 0.8333/0.8750 | 1.0000/1.0000 | 0.7957/0.8184 | 0.7957/0.8184 | 0.1042/0.1250 |
| known_values_novel | 48 | 0.6458/0.4792 | 0.9583/0.9583 | 0.6567/0.5660 | 0.6567/0.5660 | 0.3542/0.5208 |
| multi_fact_sentence | 48 | 0.5833/0.5417 | 0.9167/0.9375 | 0.4531/0.4956 | 0.4531/0.4956 | 0.1875/0.3125 |
| novel | 48 | 0.6042/0.4792 | 0.9583/0.9583 | 0.6444/0.5473 | 0.6444/0.5473 | 0.3958/0.5208 |
| number_words | 32 | 0.0000/0.0000 | 0.7188/0.7500 | 0.0105/0.0074 | 0.0105/0.0074 | 1.0000/1.0000 |
| pad32_first | 48 | 0.5208/0.3750 | 0.9792/0.9583 | 0.3982/0.4043 | 0.4058/0.4079 | 0.3958/0.6250 |
| pad32_last | 48 | 0.4167/0.3125 | 0.8333/0.8958 | 0.2861/0.3351 | 0.2936/0.3372 | 0.5208/0.6458 |
| pad32_middle | 48 | 0.1667/0.1250 | 0.7917/0.7708 | 0.1043/0.1108 | 0.1062/0.1108 | 0.4583/0.7083 |
| pad8_first | 48 | 0.7083/0.6250 | 1.0000/1.0000 | 0.6194/0.6585 | 0.6194/0.6586 | 0.2917/0.3750 |
| pad8_last | 48 | 0.3333/0.4375 | 0.8333/0.8542 | 0.3591/0.4226 | 0.3591/0.4226 | 0.6667/0.5625 |
| parallel | 48 | 1.0000/1.0000 | 1.0000/1.0000 | 0.9924/0.9952 | 0.9924/0.9952 | 0.0000/0.0000 |
| record | 48 | 1.0000/1.0000 | 1.0000/1.0000 | 0.9947/0.9933 | 0.9947/0.9933 | 0.0000/0.0000 |
| target_middle | 48 | 0.8958/1.0000 | 1.0000/1.0000 | 0.7918/0.9256 | 0.7918/0.9256 | 0.0000/0.0000 |
| unit_conversion | 16 | 0.0000/0.0000 | 0.4375/0.4375 | -0.0001/-0.0001 | -0.0001/-0.0001 | 1.0000/1.0000 |
| date | 192 | 0.8229/0.9062 | 0.9844/0.9896 | 0.7253/0.8168 | 0.7296/0.8183 | 0.0417/0.0417 |
| duration | 224 | 0.4554/0.3348 | 0.8884/0.8884 | 0.4650/0.4152 | 0.4650/0.4152 | 0.5446/0.6652 |
| quantity | 208 | 0.5288/0.4760 | 0.8798/0.8942 | 0.4689/0.4783 | 0.4689/0.4783 | 0.3846/0.4904 |

Both new arms separate all48/48 record and48/48 parallel pairs with new parents/values. This is a real improvement of the expanded-data continuation procedure over old C1 (44/48 and5/48), but does not isolate a pure data effect from continuation/batch policy. The paired loss is isolated only by R2-C1 versus R2-C0.

On held-out reported wording CE separates29/48 and pair23/48; neither reaches familiar-format performance. With32 correct padding sentences and claim in the middle, CE response/claim deltas .1043/.1062 and pair .1108/.1108 are far below short-answer .6444/.6444 and .5473/.5473. A weak local signal remains after removing global max from the measurement; this is not solely an aggregation artifact. Conversely target-middle *evidence* placement improves43/48→48/48, and date-family pairs158/192→174/192. The effect is heterogeneous, not uniformly negative.

Correct word-number and unit-conversion answers remain FPR1.0 in both arms. This is not repaired by stronger numeric pair separation in familiar spellings. The training packet deliberately did not add these held-out transformations after seeing their failures.

Evidence reorder mean absolute risk change over all diagnostic rows: CE .107398, pair .082218 (old C1 .100460, B7 .062755, B6 .240387). Lower reorder sensitivity can coexist with worse supported-answer FP. Supported paraphrase/padding invariance deltas and introduced FP are saved per example in each report; no semantically equivalent change is silently relabeled.

Diagnostic paired95%CIs,12parents/2000draws: pair−CE MF1[−.054691,.007581], pair accuracy[−.091353,.024671], response delta[−.017268,.048121]. AUROC difference+.014658,CI[.011519,.017182]; AP difference+.014337,CI[.009188,.017937]. FPR difference+.078526,CI[.035229,.120255]. These support a ranking/false-alarm trade-off conditional on this authored design and seed, not an overall winner. Absolute diagnostic MF1 CIs: CE[.735657,.846442], pair[.695244,.846595]. Selection novel pair-accuracy differenceCI[−.333333,.027778] is wide; the preregistered gate is a continuation rule, not a significance test.

All comparison JSONs retain absolute/difference intervals for accuracy, macro-F1, positive P/R/F1, AUROC/AP, confusion entries, FPR/FNR, plus per-category pair/direction/delta intervals. Same draws within each source set. Category draws with no member of that category are undefined and excluded from its percentile; overall draws are all valid. No multiplicity or selection correction; only9/12synthetic parents.

### Localization and short-error retention

| Split | Arm | Token F1 | Character F1 | Exact-span F1 |
|---|---|---:|---:|---:|
| natural | R2C0s42 | 0.5814 | 0.5844 | 0.0283 |
| natural | R2C1s42 | 0.5728 | 0.5759 | 0.0243 |
| selection | R2C0s42 | 0.6518 | 0.5254 | 0.6194 |
| selection | R2C1s42 | 0.6036 | 0.4503 | 0.5515 |
| diagnostic | R2C0s42 | 0.7971 | 0.7704 | 0.7843 |
| diagnostic | R2C1s42 | 0.7855 | 0.7476 | 0.7538 |

Natural token-F1 pair−CE=−.008556,95%CI[−.016736,−.001449]; characterF1 differenceCI[−.016403,−.001006]; exact-span differenceCI[−.017521,.007900]. Gains in synthetic localization do not transfer automatically to natural answers. On23 natural positives>=500characters and<=5% annotated length, B7 detects17/localizes13, CE14/9, pair15/9 at the same response threshold. On144 matching long synthetic positives: B7 detects19/localizes18, CE117/117, pair132/132. This positive-only recall ignores the increased false alarms on corresponding correct answers, which are reported above.

### Concrete improvements and regressions

Authored examples (full evidence/offsets/predictions are saved): Jasper answer claims26bolts while its batch has21 and a different Jasper Annex batch has26. Old C1 risk.3979 misses it at.5049; CE.8176 and pair.9118 detect it. The supported21-bolt Jasper answer with Annex record placed first shows the opposite problem: CE.1530 is correct, pair.7535 is a new false alarm. These share numeric values; presence anywhere in the document is insufficient.

Natural comparison pair versus CE:12 corrected responses,10 new errors, not12 independent sources. Examples were selected for inspection after scoring, not prevalence estimates. Response12482 (supported eyebrow procedure with lighter/darker shade relations) changes FP.5636→TN.2613.13847 (annotated unsupported keyboard shortcut) changes FN.2150→TP.3766, only just above the fixed.3761 boundary; binary correction alone does not establish complete localization.11958 (cooking response with temperatures/times, gold0) changes TN.2763→FP.5467.17320 (annotated repeated process plus unsupported heating detail) changes TP.3873→FN.2598. Labels are unchanged; these are dataset-relative detection examples, not current advice about their subject matter. Full natural texts remain only in ignored predictions/phase2b_r2/errors_B7_R2C0s42_R2C1s42.json.

### Resource measurements and run inventory

| Run | Training seconds | Mean training CE | Mean training pair loss | Peak CUDA allocated / reserved GB | Peak RSS GB | Checkpoint SHA256 |
|---|---:|---:|---:|---|---:|---|
| R2C0s42 | 306.807 | 0.140591 | 0.147660 | 3.165/3.355 | 2.693 | `7fe60c82e32af79c3c31cd845736a656ca3aa0a003a23e477f125f084fd594a6` |
| R2C1s42 | 304.455 | 0.139569 | 0.093404 | 3.165/3.355 | 2.693 | `77f81db9b9261a357c019f1e48ddb1ce43fe594c7cba01e4a2f726b7aa42a868` |

The pair-loss means are online training-batch averages, not final in-sample evaluation. Both complete417steps. Actual profile: natural1008/998tokens .814s inclbackward/AdamW, synthetic175/175 .268s; peak3.155GB allocated. All train answer/evidence coverage1.0, zero truncated tokens.149606402parameters; one GPU job at a time on RTX3060 12GB, Windows/driver610.88. No OOM, paid API, cloud GPU, dependency upgrade or old-model refit. Profiling weights were discarded.

| Arm / split | Scoring seconds | Responses/s | Amortized ms/response | Peak CUDA GB |
|---|---:|---:|---:|---:|
| R2C0s42/natural | 14.718 | 44.03 | 22.71 | 0.953 |
| R2C1s42/natural | 14.833 | 43.69 | 22.89 | 0.953 |
| R2C0s42/selection | 20.371 | 45.95 | 21.76 | 0.902 |
| R2C1s42/selection | 20.504 | 45.65 | 21.91 | 0.902 |
| R2C0s42/diagnostic | 27.081 | 46.08 | 21.70 | 0.902 |
| R2C1s42/diagnostic | 27.255 | 45.79 | 21.84 | 0.902 |

New inference usesbatch2; old span baseline measurements usebatch1. Approximately doubled throughput is primarily a batch-size difference, not a new architecture speedup. p50/p95 batch latency, RSS, checkpoint hash and exact token/window metadata are in runtime/prediction artifacts. No final checkpoint selector was run.

Run inventory: five saved-model diagnostic runs; one discarded profiling run; two matched seed42fits; selection/natural/diagnostic scoring for both; read-only paired bootstraps and error/coverage audits. Seeds43/44 were disallowed by the failed gate. All completed with exit0; the **scientific failure** is lack of robust fixed-point transfer for the added ranking loss, not a resource failure.

## Mechanism supported by this packet and next hypotheses

Measured bottleneck: a model can learn precise numeric binding in familiar relation formats, yet shift correct-answer risks upward under new renderings or equivalent number expressions. Pair ranking improves ordering without anchoring absolute unsupported risk on new formats. Short false facts also lose claim-local signal when surrounded by much correct answer text. Consequently neither a simple global threshold claim nor replacing max aggregation alone is justified. These observations do not establish a unique neural mechanism or novelty.

1. **Positive semantic invariance on held-out number formats.** Compare identical-data CE against a consistency term for correct digit/word/unit-equivalent evidence, with supported-risk anchoring, holding numeric negative pairs fixed. Reserve new renderers and parents before training. Test FP reduction without missing single false numeric relations. This follows FPR1.0 on words/conversions, not a lambda grid over the old suite.

2. **Extractive claim+neighbor versus full-answer context.** First a read-only, offset-preserving comparison on fixed models with the same total evidence; then matched training only if a local signal returns. Control coreference and answer position. The .644→.106 CE claim-delta drop under32correct sentences motivates this; aggregation cannot recover absent local evidence discrimination.

3. **Binding versus entity-name and layout cues.** Pre-register crossed unrelated/overlapping entity names, catalogue/narrative rendering and record order, keeping all values and gold relations fixed. Use genuinely new parents/renderers. This will distinguish target/Annex name interference from format-induced score shifts and decide whether harder relation negatives or richer positive contexts are needed. No unrelated negation/entity tasks are silently added to this numeric packet.

A future final calibration/dev batch needs a separately agreed stable protocol; official test remains sealed. Confirmatory external-transfer protocol remains PHASE2B_CONFIRMATORY_PROTOCOL.md, not executed here. R2 is a research checkpoint, not completion of GroundCheck.

## Verification, preservation and reproduction

Local full research environment:139tests passed with real-data/model opt-ins (62.16s). Base offline environment:135passed/4skipped (54.36s); optional dataset/model and Torch-only gradient checks skip without those assets. After adding the explicit common-log-odds-shift assertion, the three loss tests were rerun and passed. Ruff passed; pip check passed in both existing environments. CI file .github/workflows/ci.yml exists and is tracked, with Python3.11/3.12, offline tests/Ruff/pip check. No remote workflow success is claimed. Local research Python3.12.14; dependencies/model revisions remain pinned in existing configs. No new dependency.

Post-hoc invariance source bootstrap uses the same2000 draws: pair-minus-CE evidence reorder mean absolute change95%CI[−.041271,−.007169]. It also reports evidence-paraphrase and other supported transformations, absolute/difference intervals and valid/undefined category draw counts in invariance_intervals.json. This analysis did not select another model or threshold.

New source files:src/grounded_hallucination_lab/research/{relations.py,pair_training.py}; scripts/{research_r2.py,research_r2_audit.py}; tests/{test_research_r2.py,test_pair_training.py}; EXPERIMENT_R2_PLAN.md, EXPERIMENT_R2_LOSS_AMENDMENT.md, docs/PHASE2B_R2_RESULTS.md. Updated:README.md,RESEARCH_NOTES.md,docs/{PROGRESS.md,DECISIONS.md,EVALUATION_PROTOCOL.md}. Numeric reports are in reports/phase2b_r2. Models/configs/batch logs/queue journals and helper scripts are in ignored models/phase2b_r2; full predictions/generated records/error texts in ignored predictions/phase2b_r2. No raw data, weights, cache or secrets committed; no push or publication.

Recoverable source-only backup:research_snapshots/R2_source_20261024.zip plus SHA256 manifest, including local queue/audit helpers but excluding datasets/weights/cache. Artifact size/SHA manifest:models/phase2b_r2/R2_artifact_manifest.json. Old R1 snapshots and the nonignored21-file staged recovery archive remain preserved. These source archives are not subject to models/* ignore rules. Both new fits are complete; no partial optimizer resume is needed. The short continuation trainer does not implement optimizer-state resume for a hypothetical partial future run; never restart it silently or overwrite a completed destination.

Commands below are the reproduction recipe in a separate copy with pinned R1 assets/B7 checkpoint and empty R2 output paths. Current outputs deliberately refuse overwrite. Restore helper scripts from the source-only ZIP if using a checkout where ignored models/ helpers are absent. No original B7/B8/C0/C1 training command is required.

```powershell
.\.venv-research\Scripts\python.exe models/phase2b_r2/verify_r1.py
.\.venv-research\Scripts\python.exe -m pytest tests/test_research_r2.py tests/test_pair_training.py -q --basetemp=models/phase2b_r2/repro_tests
.\.venv-research\Scripts\python.exe scripts/research_r2.py freeze
.\.venv-research\Scripts\python.exe models/phase2b_r2/baseline_queue.py
.\.venv-research\Scripts\python.exe scripts/research_r2.py compare
.\.venv-research\Scripts\python.exe scripts/research_r2.py profile
.\.venv-research\Scripts\python.exe models/phase2b_r2/reuse_natural_baselines.py
.\.venv-research\Scripts\python.exe models/phase2b_r2/run_seed.py 42
.\.venv-research\Scripts\python.exe models/phase2b_r2/seed_gate.py
# Actual gate is false: do not run 43/44 for this configuration.
.\.venv-research\Scripts\python.exe models/phase2b_r2/finish_pilot.py
.\.venv-research\Scripts\python.exe models/phase2b_r2/supplement.py
.\.venv-research\Scripts\python.exe models/phase2b_r2/invariance_bootstrap.py

# Equivalent individual fit commands, already executed by run_seed.py:
# python scripts/research_r2.py train --seed 42 --pair-weight 0
# python scripts/research_r2.py train --seed 42 --pair-weight 0.1
# Scoring example:
# python scripts/research_r2.py score --method R2C1s42 --split diagnostic

$env:GROUNDCHECK_RAGTRUTH_DIR = 'data/raw/ragtruth'
$env:GROUNDCHECK_RESEARCH_MODELS = '1'
.\.venv-research\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=models/phase2b_r2/repro_full
Remove-Item Env:GROUNDCHECK_RAGTRUTH_DIR
Remove-Item Env:GROUNDCHECK_RESEARCH_MODELS
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=models/phase2b_r2/repro_offline
.\.venv-research\Scripts\python.exe -m ruff check .
.\.venv-research\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip check
```

The restored final source archive contains supplementary analysis added after fit; each executed run also retains its exact pre-run content-addressed code snapshot and provenance. For byte-level historical replay use that run's snapshot and its saved R2 plan/amendment, rather than altering old hashes to accept newer code. The suite content hashes are unchanged across all runs.

When verify_r1.py is rerun on the final R2 checkout, its workspace-versus-R1-source comparison will list intentional R2 edits. The original R1 ZIP members and old artifact hashes must still match; those historical manifests are never regenerated to make new source appear old.

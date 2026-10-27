# R3 — semantic invariance and factual sensitivity (intermediate research report)

2026-10-24. Status: first bounded R3 packet completed; research remains active. No outer calibration/dev/test evaluation. Historical R1/R2 results are immutable. This is development research, not project completion or independent evidence of superiority.

## Hypothesis and relation to prior work

The hypothesis is that supported equivalent evidence views, aligned token consistency and supervised factual negatives reduce false alarms from representation changes while retaining relation sensitivity. The matched experiment tests the extra objective separately from new training examples.

| Primary source, checked 2026-10-24 | Established method / relevance | Relation to R3 |
|---|---|---|
| [CCHD, Lin et al., 2026](https://arxiv.org/html/2606.08158v1) | Prediction consistency across paraphrases plus label-preservation constraints; Jeffreys divergence and primal-dual optimization. Back-translation produces views. | Direct conceptual precedent. Our fixed-weight aligned answer-token MSE is a smaller controlled ablation, not their constrained optimizer or a new consistency principle. |
| [CPIL, Lin et al., 2026](https://arxiv.org/html/2606.08157v1) | Paraphrastic positives and same-document opposite-label negatives; Siamese representation pretraining followed by classification. | Related invariance/grounding objective. R3 keeps the encoder/head/inference unchanged and applies consistency only to supported same-answer evidence views. No CPIL reproduction claimed. |
| [EQUATE, Ravichander et al., 2019](https://aclanthology.org/K19-1033/) | Quantitative NLI diagnostics and symbolic quantity manipulation (Q-REAS), with a gap between numerical and verbal reasoning. | Motivates the deterministic normalization control and separating arithmetic from entity/property binding. |
| [Mahendra et al., 2025](https://aclanthology.org/2025.findings-naacl.467/) | Numeracy as NLI; normalization/comparison competence differs from arithmetic competence. | Supports measuring entailment/contradiction and number representations separately. |
| [FactCC, Kryscinski et al., 2020](https://aclanthology.org/2020.emnlp-main.750/) | Rule-based transformed training data, factual-consistency classification and supporting/inconsistent spans. | Counterfactual data and span supervision are established. |
| [LettuceDetect, Kovacs and Recski, 2025](https://arxiv.org/abs/2502.17125) | Context-question-answer token classification. | Our evidence-conditioned span architecture is not novel; public RAGTruth-trained checkpoints remain excluded from clean inner evaluation. |
| [Huang et al., 2020](https://aclanthology.org/2020.insights-1.13/) | Counterfactual SNLI augmentation did not reliably improve generalization under matched data-size comparisons. | A reason to retain CE-only, negative results and unseen rendering families. |

Paper benchmark scores are not compared numerically to our response-level macro-F1. CCHD/CPIL report different tasks/units; their claims were not reproduced here. No new checkpoint download or remote model code execution. Potential contribution remains unproven: robust relation-sensitive localization under exact equivalent representations would need natural transfer, stronger controls and compatible untouched evaluation.

## Recovery and read-only diagnosis

Verified 92 original pause artifacts, 174 R1 artifacts and 113 R2 artifacts with zero SHA256 mismatches. R1 source snapshot has114 verified source members; R2 source snapshot121. R2 trained checkpoints/results are complete. No old fit was repeated. See reports/phase2b_r3/recovery.json and read_only_R2.json.

Saved R2 CE/pair natural macro-F1=.7846856217/.7871516797. Diagnostic macro-F1=.7912672611/.7686891961, AUROC=.8868163420/.9014744103, FPR=.3365384615/.4150641026. Both separate48/48 familiar record and48/48 parallel pairs; unseen catalogue8/36 versus3/36. Word/unit supported FPR=1.0. Exact natural span F1=.02828/.02434. The improved AUROC is not an improved detector overall.

The R2 word/unit claim-local delta is approximately zero, not merely a global-max error. Both supported and contradictory views receive high risk. Therefore threshold adjustment alone cannot recover reliable discrimination there. Separately, the catalogue result shows an absolute risk shift despite improved ranking. The observations support both representation loss and operating-point shift, depending on the condition; they do not identify a unique internal cause.

### Numerical normalization sanity control — measured before training

R2 CE with evidence-only deterministic canonicalization: word probes64rows/32pairs and unit probes32rows/16pairs have FPR=0 and recall=1, macro-F1=1.0; canonical parallel96rows remains perfect. All48 previously failed equivalence pairs are corrected. Total192 rows scored in4.352s,44.12responses/s, peak allocated .888GB. Same weights and fixed .37612172961235046 threshold. No answer text, offsets or entity relations changed. Normalization exactly restores canonical evidence strings on this narrowly controlled suite.

This solves the narrow R2 evidence-format issue without neural training. It does not prove relational understanding in new prose, unsupported additions, paraphrased answers or long contexts. The R3 packet tests that broader transfer. Normalizer supports conservative canonical integer cardinal grammar, exact integer seconds→minutes and kilometers→meters; rejects uncertain/range/fractional clauses. It is not a production universal number parser. It does not normalize answer words, deliberately preserving annotation offsets.

## Frozen new suite and natural audit

Plan: EXPERIMENT_R3_PLAN.md, frozen before predictions. New authored train960rows/18parents (480positive), selection480/9 (240positive), diagnostic640/12 (320positive). Source/value sets and evidence rendering families are disjoint: declarative / ledger / inspection. Same generic answer grammar, measurement concepts and lexical pieces remain shared. Names repeat across measurement kinds, so cross-kind linguistic dependence remains a limitation of nominal parent bootstrap. Canonical numeric values and combinations are disjoint across splits; transformed values are determined exactly. These fictional facts are not extracted RAGTruth data.

Each answer value appears with supported and contradicted evidence; both numeric values are present in ordinary attribution counterfactuals. Order is crossed. Separate unsupported category removes the target measurement without assigning a contradictory one. Answer paraphrases/word forms, evidence word/unit forms, and a short false claim surrounded by correct filler are distinct conditions. Fit padding4sentences at endpoints; selection9 at one-third; diagnostic18 at two-thirds. Base short cases necessarily start at position0. Length and position are therefore not fully factorially isolated; no causal claim about position alone.

All derivatives stay with their source. Training/selection/diagnostic hashes and exact examples are frozen in models/phase2b_r3/suite_manifest.json and ignored predictions/phase2b_r3. Offsets and semantic labels are checked offline. Privileged claim/slot annotations are used for training or after-inference diagnostics only. Standard inference uses question/evidence/answer, unchanged full-answer max-token risk.

Natural quality uses the unchanged648inner responses/108sources, original include/all labels. A small manually inspected regression slice uses previously seen11958 (cooking and oven thermometer),12482 (brow shape/pencil/concealer),13847 (administrator instructions). Evidence-only equivalent views:15minutes→900seconds for the thermometer, two shades→2shades, and independent passage reorder respectively. Preserve original gold labels/spans, including the annotated shortcut error in13847; do not relabel the entire procedural answer. This6row/3source audit is not independent natural data. Temperature conversion is not altered because rounded Celsius/Fahrenheit values should not be treated as exact. No large questionable natural synthetic set was manufactured.

## Matched training design and measured profile

Both arms initialize from the exact B7 inner42 checkpoint, same one epoch, seed42, same2568natural examples plus960unique authored examples. Effective synthetic1056 presentations repeat canonical examples equally under both truth labels when pairing digits with words and units. The matched batch/target hashes and full configs are saved.120 supported view pairs receive consistency; contradictory pairs receive CE only. Same optimizer, lr1e-5, weight_decay.01, warmup10%, clip1, positive CE weight3.370202392225359, microbatch2/accumulation4, BF16autocast with FP32 weights/optimizer/loss and gradient checkpointing.

R3-A: weighted token CE. R3-B: same CE plus mean squared aligned claim-token probability difference, lambda1 fixed before predictions. No encoder, retrieval, segmentation or production aggregation change. The claim includes correct surrounding words, so dilution of a very short numerical slot is a possible failure mode. Same-input penalty is zero; gradient direction and prefix exclusion are unit-tested. Real-tokenizer batch audit confirms no truth-changing pair receives consistency.

Profile measured forward/backward/AdamW: longest natural windows1008/998tokens,.663s,3.152GB allocated; longest equivalent pair80/82tokens,.341s,3.158GB. Full context coverage; no OOM or silent truncation. Profile weights are discarded; both fits reload the initial checkpoint. Conditional anchoring and seed repeats require the gates written before model predictions; they are not automatic fallbacks.

The completed measurements and negative gate decision follow below.

### Saved-model error table before training (R2 historical probes)

| Method | Condition | Supported FPR | Contradiction recall | Correct pairs | Mean claim delta |
|---|---|---:|---:|---|---:|
| B7 | parallel | 0.0417 | 0.0625 | 1/48 | 0.0108 |
| B7 | number_words | 1.0000 | 1.0000 | 0/32 | -0.0014 |
| B7 | unit_conversion | 1.0000 | 1.0000 | 0/16 | -0.0016 |
| B7 | answer_paraphrase | 0.6458 | 0.6250 | 3/48 | -0.0057 |
| B7 | novel | 0.3750 | 0.3750 | 4/48 | 0.0144 |
| B7 | pad32_middle | 0.2292 | 0.2292 | 1/48 | -0.0060 |
| R2C0s42 | parallel | 0.0000 | 1.0000 | 48/48 | 0.9924 |
| R2C0s42 | number_words | 1.0000 | 1.0000 | 0/32 | 0.0105 |
| R2C0s42 | unit_conversion | 1.0000 | 1.0000 | 0/16 | -0.0001 |
| R2C0s42 | answer_paraphrase | 0.1042 | 0.9375 | 40/48 | 0.7957 |
| R2C0s42 | novel | 0.3958 | 1.0000 | 29/48 | 0.6444 |
| R2C0s42 | pad32_middle | 0.4583 | 0.5833 | 8/48 | 0.1062 |
| R2C1s42 | parallel | 0.0000 | 1.0000 | 48/48 | 0.9952 |
| R2C1s42 | number_words | 1.0000 | 1.0000 | 0/32 | 0.0074 |
| R2C1s42 | unit_conversion | 1.0000 | 1.0000 | 0/16 | -0.0001 |
| R2C1s42 | answer_paraphrase | 0.1250 | 1.0000 | 42/48 | 0.8184 |
| R2C1s42 | novel | 0.5208 | 1.0000 | 23/48 | 0.5473 |
| R2C1s42 | pad32_middle | 0.7083 | 0.7917 | 6/48 | 0.1108 |
| B6 | parallel | 0.0000 | 0.0000 | 0/48 | 0.0568 |
| B6 | number_words | 0.0000 | 0.0312 | 1/32 | -0.0150 |
| B6 | unit_conversion | 0.9375 | 0.9375 | 1/16 | 0.0005 |
| B6 | answer_paraphrase | 0.0000 | 0.0000 | 0/48 | 0.0822 |
| B6 | novel | 0.0000 | 0.0208 | 1/48 | -0.0014 |
| B6 | pad32_middle | 0.0000 | 0.0208 | 1/48 | -0.0014 |

Parallel probes already include wrong attribution with identical number sets; novel probes change rendering. Long probes place one short error among32correct sentences. Full response metrics, localization and evidence-order diagnostics remain in read_only_R2.json; no new weights were updated for this table.


### R3 audit limitation found after fit, before new-arm diagnostic interpretation

The measurement renderers are split-specific, but the neutral replacement sentence used when a target measurement is omitted (object was inspected in the morning) shares a template across fit/selection/diagnostic. Thus the unsupported category is held out by parents/values and remaining measurement rendering, NOT wholly held out by every evidence template. Exact suite hashes remain unchanged; do not silently regenerate it or call this a fully unseen missing-evidence formulation. Report unsupported separately from contradictions. The primary continuation gate already failed; this audit does not authorize changing its denominator or retraining. Related target/distractor names (name versus Westhaven name) and repeated correct padding remain additional shortcuts/limitations. The small natural audit is not a replacement for new independently annotated multi-fact data.

## Completed matched pilot

Gate decision before diagnostic: `{"repeat_allowed": false, "conditional_C_allowed": false}`.

| Selection gate | Measured B minus A (or stated level) |
|---|---:|
| word_unit_FPR_reduction | -0.116667 |
| word_FPR | 0.944444 |
| unit_FPR | 1.000000 |
| contradiction_recall_delta | 0.000000 |
| pair_accuracy_delta | -0.008333 |
| natural_macro_f1_delta | -0.004691 |
| natural_token_f1_delta | 0.011278 |
| long_recall_delta | -0.111111 |
| supported_equivalence_gap_A | 0.730768 |
| supported_equivalence_gap_B | 0.604442 |

All matched config fields except pair_weight were equal, including input/target/order/source hashes. Training details:

| Arm | Seconds | Updates | Peak CUDA allocated/reserved GB | Peak RSS GB | Checkpoint SHA256 |
|---|---:|---:|---|---:|---|
| R3-A,42 | 331.911 | 453 | 3.168/3.366 | 2.846 | `a0116cda0d54be0a97fa70940b59c2ef1c734229c7fdea856dd52c64078e1d26` |
| R3-B,42 | 332.304 | 453 | 3.168/3.366 | 2.843 | `b6c054d5a8556e85c9001dcaf45a2b5f3f2500cad049c25479debb1ce37f3389` |

Batch order SHA256 `b5479c3a5ac3741f8154522900ddd69b5708dafccf7b64a06b68095503dec396`; targets `fb9e67386ca61718daf26defb98e701166ba2aa9688b8d2670307bbc588719b4`. Target mass `{'0': 393509.0, '1': 34596.0}`. 1812 microbatches, 446 total source IDs. No OOM, full answer/evidence coverage.


## selection: fixed historical thresholds

| Method | Accuracy | Macro-F1 | Positive P/R/F1 | AUROC / AP | FPR / FNR | CM |
|---|---:|---:|---|---|---|---|
| B7 | 0.5354 | 0.5347 | 0.5328/0.5750/0.5531 | 0.5146/0.5094 | 0.5042/0.4250 | [[119, 121], [102, 138]] |
| R2C0s42 | 0.5542 | 0.5539 | 0.5516/0.5792/0.5650 | 0.5076/0.5179 | 0.4708/0.4208 | [[127, 113], [101, 139]] |
| R2C1s42 | 0.5563 | 0.5550 | 0.5509/0.6083/0.5782 | 0.5169/0.5227 | 0.4958/0.3917 | [[121, 119], [94, 146]] |
| B6 | 0.5104 | 0.4582 | 0.5275/0.2000/0.2900 | 0.5452/0.5443 | 0.1792/0.8000 | [[197, 43], [192, 48]] |
| R2C0s42N | 0.5896 | 0.5787 | 0.6319/0.4292/0.5112 | 0.5460/0.5474 | 0.2500/0.5708 | [[180, 60], [137, 103]] |
| R3As42 | 0.6750 | 0.6749 | 0.6680/0.6958/0.6816 | 0.7506/0.7094 | 0.3458/0.3042 | [[157, 83], [73, 167]] |
| R3Bs42 | 0.6604 | 0.6582 | 0.6380/0.7417/0.6859 | 0.7373/0.7268 | 0.4208/0.2583 | [[139, 101], [62, 178]] |

Paired2000-source bootstrap, reference R3-A, fixed models/thresholds:

| Method | Macro-F1 absolute95 | Macro-F1 difference95 vs A | AUROC difference95 | FPR difference95 |
|---|---|---|---|---|
| B7 | [0.50782, 0.55704] | [-0.18322, -0.09566] | [-0.28070, -0.20402] | [0.08197, 0.23818] |
| R2C0s42 | [0.52173, 0.58880] | [-0.15552, -0.08969] | [-0.28278, -0.20906] | [0.03333, 0.23707] |
| R2C1s42 | [0.52697, 0.58461] | [-0.15850, -0.08033] | [-0.27163, -0.19748] | [0.07913, 0.23707] |
| B6 | [0.40027, 0.49836] | [-0.28934, -0.16523] | [-0.24839, -0.17087] | [-0.26293, -0.06855] |
| R2C0s42N | [0.51271, 0.63683] | [-0.15263, -0.04455] | [-0.25209, -0.14631] | [-0.22581, 0.05836] |
| R3As42 | [0.64138, 0.71054] | reference | reference | reference |
| R3Bs42 | [0.60852, 0.70832] | [-0.04651, 0.01461] | [-0.04390, 0.01530] | [0.03175, 0.12719] |

Localization and response-level paired relation separation:

| Method | Correct pairs | Mean response/claim delta | Token / character / exact-span F1 | Scoring seconds / throughput | Peak CUDA GB |
|---|---|---|---|---|---:|
| B7 | 41/240 | 0.0119/0.0119 | 0.2959/0.3136/0.0541 | 19.90/24.13 | 0.886 |
| R2C0s42 | 36/240 | 0.0321/0.0321 | 0.4347/0.4020/0.1162 | 10.19/47.10 | 0.889 |
| R2C1s42 | 39/240 | 0.0487/0.0487 | 0.4309/0.3961/0.0949 | 10.27/46.75 | 0.889 |
| B6 | 6/240 | 0.0726/0.0729 | N/A | 23.90/20.08 | 2.411 |
| R2C0s42N | 54/240 | 0.0671/0.0671 | 0.3478/0.3283/0.0789 | 10.33/46.47 | 0.889 |
| R3As42 | 86/240 | 0.3001/0.3001 | 0.6176/0.6184/0.4863 | 10.23/46.90 | 0.889 |
| R3Bs42 | 84/240 | 0.2669/0.2668 | 0.6251/0.6196/0.5082 | 10.32/46.49 | 0.889 |

Matched arm breakdown (correct pairs; supported FPR; contradiction/unsupported recall):

| Variant | R3-A | R3-B |
|---|---|---|
| exact | 22/36; 0.0000; 0.6111 | 24/36; 0.1111; 0.7222 |
| words | 4/36; 0.8889; 1.0000 | 1/36; 0.9444; 0.9722 |
| units | 5/24; 0.7917; 1.0000 | 0/24; 1.0000; 1.0000 |
| answer_paraphrase | 25/36; 0.0000; 0.6944 | 25/36; 0.0000; 0.6944 |
| answer_words | 4/36; 0.8611; 0.9167 | 3/36; 0.9167; 0.9444 |
| unsupported | 7/36; 0.0000; 0.1944 | 16/36; 0.1111; 0.5000 |
| long | 19/36; 0.0278; 0.5556 | 15/36; 0.0556; 0.4444 |

Supplemental R3-selection threshold transfer (not outer calibration, not used to rescue gate):

| Method | Threshold | Macro-F1 | Positive P/R/F1 | FPR |
|---|---:|---:|---|---:|
| B7 | 0.403566897 | 0.5375 | 0.5375/0.5375/0.5375 | 0.4625 |
| R2C0s42 | 0.371136606 | 0.5559 | 0.5534/0.5833/0.5680 | 0.4708 |
| R2C1s42 | 0.310276270 | 0.5639 | 0.5576/0.6458/0.5985 | 0.5125 |
| B6 | 0.677473992 | 0.6241 | 0.6548/0.5375/0.5904 | 0.2833 |
| R2C0s42N | 0.367043376 | 0.5823 | 0.6310/0.4417/0.5196 | 0.2583 |
| R3As42 | 0.115685701 | 0.7124 | 0.6646/0.8917/0.7616 | 0.4500 |
| R3Bs42 | 0.232920274 | 0.6888 | 0.6506/0.8458/0.7355 | 0.4542 |

## diagnostic: fixed historical thresholds

| Method | Accuracy | Macro-F1 | Positive P/R/F1 | AUROC / AP | FPR / FNR | CM |
|---|---:|---:|---|---|---|---|
| B7 | 0.5094 | 0.5091 | 0.5098/0.4875/0.4984 | 0.5190/0.5111 | 0.4688/0.5125 | [[170, 150], [164, 156]] |
| R2C0s42 | 0.5172 | 0.5081 | 0.5135/0.6531/0.5750 | 0.5293/0.5308 | 0.6188/0.3469 | [[122, 198], [111, 209]] |
| R2C1s42 | 0.5250 | 0.5158 | 0.5196/0.6625/0.5824 | 0.5323/0.5222 | 0.6125/0.3375 | [[124, 196], [108, 212]] |
| B6 | 0.5031 | 0.4170 | 0.5135/0.1187/0.1929 | 0.4985/0.5136 | 0.1125/0.8812 | [[284, 36], [282, 38]] |
| R2C0s42N | 0.5578 | 0.5572 | 0.5539/0.5938/0.5732 | 0.5707/0.5636 | 0.4781/0.4062 | [[167, 153], [130, 190]] |
| R3As42 | 0.6844 | 0.6712 | 0.6317/0.8844/0.7370 | 0.7279/0.6746 | 0.5156/0.1156 | [[155, 165], [37, 283]] |
| R3Bs42 | 0.6344 | 0.6191 | 0.5960/0.8344/0.6953 | 0.6798/0.6427 | 0.5656/0.1656 | [[139, 181], [53, 267]] |

Paired2000-source bootstrap, reference R3-A, fixed models/thresholds:

| Method | Macro-F1 absolute95 | Macro-F1 difference95 vs A | AUROC difference95 | FPR difference95 |
|---|---|---|---|---|
| B7 | [0.50200, 0.51409] | [-0.19222, -0.13494] | [-0.23718, -0.17791] | [-0.12500, 0.03395] |
| R2C0s42 | [0.47146, 0.54259] | [-0.20604, -0.12406] | [-0.22610, -0.17543] | [0.03272, 0.17535] |
| R2C1s42 | [0.46410, 0.56270] | [-0.20965, -0.10619] | [-0.23087, -0.16619] | [0.02101, 0.17309] |
| B6 | [0.38233, 0.44566] | [-0.28968, -0.21913] | [-0.25583, -0.20365] | [-0.50321, -0.30422] |
| R2C0s42N | [0.50267, 0.60672] | [-0.16964, -0.06212] | [-0.20278, -0.11170] | [-0.14819, 0.08013] |
| R3As42 | [0.64465, 0.70270] | reference | reference | reference |
| R3Bs42 | [0.57482, 0.66072] | [-0.07720, -0.02677] | [-0.07422, -0.02582] | [0.01948, 0.07927] |

Localization and response-level paired relation separation:

| Method | Correct pairs | Mean response/claim delta | Token / character / exact-span F1 | Scoring seconds / throughput | Peak CUDA GB |
|---|---|---|---|---|---:|
| B7 | 10/320 | 0.0128/0.0148 | 0.2658/0.2818/0.0099 | 26.76/23.92 | 0.887 |
| R2C0s42 | 39/320 | 0.0323/0.0354 | 0.4906/0.4730/0.2403 | 13.96/45.86 | 0.892 |
| R2C1s42 | 39/320 | 0.0365/0.0385 | 0.4811/0.4689/0.1971 | 13.46/47.56 | 0.892 |
| B6 | 7/320 | 0.0206/0.0206 | N/A | 48.78/13.12 | 2.432 |
| R2C0s42N | 72/320 | 0.0900/0.0931 | 0.4253/0.3870/0.1280 | 13.62/47.00 | 0.892 |
| R3As42 | 122/320 | 0.2376/0.2560 | 0.6631/0.6733/0.5565 | 13.94/45.90 | 0.892 |
| R3Bs42 | 93/320 | 0.1728/0.1814 | 0.6599/0.6676/0.5163 | 13.60/47.05 | 0.892 |

Matched arm breakdown (correct pairs; supported FPR; contradiction/unsupported recall):

| Variant | R3-A | R3-B |
|---|---|---|
| exact | 36/48; 0.2500; 1.0000 | 27/48; 0.4375; 1.0000 |
| words | 3/48; 0.9375; 1.0000 | 3/48; 0.9375; 1.0000 |
| units | 1/32; 0.9688; 1.0000 | 1/32; 0.9688; 1.0000 |
| answer_paraphrase | 38/48; 0.0833; 0.8750 | 27/48; 0.3125; 0.8750 |
| answer_words | 4/48; 0.9167; 1.0000 | 2/48; 0.9583; 1.0000 |
| unsupported | 27/48; 0.2500; 0.7292 | 19/48; 0.4375; 0.6875 |
| long | 13/48; 0.3542; 0.6250 | 14/48; 0.0417; 0.3333 |

Supplemental R3-selection threshold transfer (not outer calibration, not used to rescue gate):

| Method | Threshold | Macro-F1 | Positive P/R/F1 | FPR |
|---|---:|---:|---|---:|
| B7 | 0.403566897 | 0.5118 | 0.5135/0.4750/0.4935 | 0.4500 |
| R2C0s42 | 0.371136606 | 0.5081 | 0.5135/0.6531/0.5750 | 0.6188 |
| R2C1s42 | 0.310276270 | 0.5011 | 0.5106/0.6750/0.5814 | 0.6469 |
| B6 | 0.677473992 | 0.5515 | 0.5859/0.4156/0.4863 | 0.2938 |
| R2C0s42N | 0.367043376 | 0.5572 | 0.5539/0.5938/0.5732 | 0.4781 |
| R3As42 | 0.115685701 | 0.4646 | 0.5351/1.0000/0.6972 | 0.8688 |
| R3Bs42 | 0.232920274 | 0.5715 | 0.5709/0.9437/0.7114 | 0.7094 |

## natural: fixed historical thresholds

| Method | Accuracy | Macro-F1 | Positive P/R/F1 | AUROC / AP | FPR / FNR | CM |
|---|---:|---:|---|---|---|---|
| B7 | 0.8164 | 0.7959 | 0.6835/0.7864/0.7314 | 0.8760/0.7897 | 0.1697/0.2136 | [[367, 75], [44, 162]] |
| R2C0s42 | 0.8102 | 0.7847 | 0.6895/0.7330/0.7106 | 0.8697/0.7741 | 0.1538/0.2670 | [[374, 68], [55, 151]] |
| R2C1s42 | 0.8133 | 0.7872 | 0.6977/0.7282/0.7126 | 0.8725/0.7746 | 0.1471/0.2718 | [[377, 65], [56, 150]] |
| B6 | 0.8009 | 0.7753 | 0.6726/0.7282/0.6993 | 0.8591/0.7016 | 0.1652/0.2718 | [[369, 73], [56, 150]] |
| R3As42 | 0.8086 | 0.7799 | 0.6971/0.7039/0.7005 | 0.8679/0.7750 | 0.1425/0.2961 | [[379, 63], [61, 145]] |
| R3Bs42 | 0.8009 | 0.7753 | 0.6726/0.7282/0.6993 | 0.8710/0.7817 | 0.1652/0.2718 | [[369, 73], [56, 150]] |

Paired2000-source bootstrap, reference R3-A, fixed models/thresholds:

| Method | Macro-F1 absolute95 | Macro-F1 difference95 vs A | AUROC difference95 | FPR difference95 |
|---|---|---|---|---|
| B7 | [0.76461, 0.82907] | [-0.00481, 0.03650] | [0.00181, 0.01475] | [0.00899, 0.04609] |
| R2C0s42 | [0.74826, 0.82019] | [-0.01742, 0.02578] | [-0.00572, 0.00943] | [-0.00701, 0.03017] |
| R2C1s42 | [0.75009, 0.82485] | [-0.01229, 0.02688] | [-0.00238, 0.01198] | [-0.01436, 0.02232] |
| B6 | [0.73814, 0.80965] | [-0.04203, 0.03471] | [-0.04055, 0.02328] | [-0.01778, 0.06294] |
| R3As42 | [0.74719, 0.81251] | reference | reference | reference |
| R3Bs42 | [0.74135, 0.80839] | [-0.02109, 0.01329] | [-0.00222, 0.00858] | [0.00683, 0.03855] |

Natural localization, fixed token threshold .5:

| Method | Token / character / exact-span F1 | Token-F1 difference95 vs A |
|---|---|---|
| B7 | 0.5849/0.5886/0.0354 | [0.00380,0.03662] |
| R2C0s42 | 0.5814/0.5844/0.0283 | [0.00020,0.03467] |
| R2C1s42 | 0.5728/0.5759/0.0243 | [-0.00837,0.02390] |
| R3As42 | 0.5652/0.5678/0.0388 | [0.00000,0.00000] |
| R3Bs42 | 0.5765/0.5802/0.0302 | [0.00234,0.02476] |

Supplemental R3-selection threshold transfer (not outer calibration, not used to rescue gate):

| Method | Threshold | Macro-F1 | Positive P/R/F1 | FPR |
|---|---:|---:|---|---:|
| B7 | 0.403566897 | 0.7950 | 0.6867/0.7767/0.7289 | 0.1652 |
| R2C0s42 | 0.371136606 | 0.7832 | 0.6864/0.7330/0.7089 | 0.1561 |
| R2C1s42 | 0.310276270 | 0.7882 | 0.6858/0.7524/0.7176 | 0.1606 |
| B6 | 0.677473992 | 0.6843 | 0.5039/0.9417/0.6565 | 0.4321 |
| R3As42 | 0.115685701 | 0.7399 | 0.5738/0.8495/0.6849 | 0.2941 |
| R3Bs42 | 0.232920274 | 0.7726 | 0.6336/0.8058/0.7094 | 0.2172 |

## In-sample optimization audit and separate missing-evidence audit

Post-fit descriptive passes were registered separately in EXPERIMENT_R3_FIT_AUDIT.md and EXPERIMENT_R3_AUDIT_AMENDMENT.md. Neither changed weights, thresholds, original suite or continuation decisions. They were not preregistered before training and must not be presented as the original untouched diagnostic.

| Arm, fit960 | Condition | FPR | Recall | AUROC | Supported / contradictory mean risk |
|---|---|---:|---:|---:|---|
| R3As42 | exact | 0.0000 | 1.0000 | 1.0000 | 0.0064/0.9962 |
| R3As42 | words | 0.7917 | 1.0000 | 0.8373 | 0.6004/0.8880 |
| R3As42 | units | 0.5833 | 1.0000 | 0.8479 | 0.4868/0.8866 |
| R3As42 | answer_words | 0.5000 | 1.0000 | 0.9552 | 0.4425/0.9278 |
| R3As42 | long | 0.0000 | 1.0000 | 1.0000 | 0.0098/0.9905 |
| R3As42 | unsupported | 0.0000 | 1.0000 | 1.0000 | 0.0064/0.9878 |
| R3Bs42 | exact | 0.0000 | 1.0000 | 1.0000 | 0.0297/0.9930 |
| R3Bs42 | words | 0.8472 | 1.0000 | 0.8005 | 0.5828/0.8201 |
| R3Bs42 | units | 0.6042 | 1.0000 | 0.8383 | 0.4933/0.8576 |
| R3Bs42 | answer_words | 0.7222 | 1.0000 | 0.9385 | 0.5777/0.9282 |
| R3Bs42 | long | 0.0000 | 1.0000 | 1.0000 | 0.0273/0.9752 |
| R3Bs42 | unsupported | 0.0000 | 1.0000 | 1.0000 | 0.0297/0.9852 |

Both models fit canonical numeric relationships, long fitting examples and unsupported fitting cases perfectly at the fixed point. Equivalent formats remain poorly fitted even in-sample: there is partial truth discrimination (AUROC~.80–.85 for words/units), but substantial supported risk and weaker margins. Thus this is not only an unseen-template transfer problem. B reduces online training consistency penalty (.04917→.03722) while degrading the main diagnostic. An average probability-consistency objective is not sufficient evidence of semantic invariance; saturation, claim-token dilution and the natural/synthetic supervision mixture are hypotheses, not uniquely proven causes.

Fresh missing-evidence audit:24rows/12pairs/6new names and value pairs, notebook/physical-property renderers, no original inspection clause. Supported and unsupported answers are identical within pairs; another entity carries the same amount under both labels. No arbitrary natural document swaps were relabeled.

| Method | Macro-F1 | FPR / recall | Correct pairs | Token / char / exact-span F1 |
|---|---:|---|---|---|
| B7 | 0.8748 | 0.1667/0.9167 | 9/12 | 0.4063/0.4578/0.0000 |
| R2C0s42 | 0.8748 | 0.1667/0.9167 | 9/12 | 0.3889/0.3034/0.0000 |
| R2C1s42 | 0.9167 | 0.0833/0.9167 | 10/12 | 0.3889/0.3034/0.0000 |
| B6 | 0.3333 | 0.0000/0.0000 | 0/12 | N/A |
| R3As42 | 0.9161 | 0.1667/1.0000 | 10/12 | 0.7917/0.8505/0.5833 |
| R3Bs42 | 0.8730 | 0.2500/1.0000 | 9/12 | 0.8519/0.8898/0.6923 |

B−A macro-F1 paired95%CI [-0.13387,0.00000] on six authored sources. This checks a new omission rendering, not all unsupported claims or independent natural generalization. Strong old R2 performance on this supplement shows that unsupported sensitivity is context/rendering dependent, rather than wholly absent.

## What the evidence supports

The primary R3-B gate failed before diagnostic; R3-C and seeds43/44 were not run. The conditional anchoring trigger required at least20% selection-view gap reduction; actual17.29% was insufficient. No threshold relaxation, new lambda or lucky seed was tried.

On the640-row authored diagnostic, B−A macro-F1 is negative with paired95%CI[-.07720,-.02677]. Conservative four-name-cluster sensitivity is also negative[-.07344,-.03644]. These are conditional development intervals, not independent-domain evidence. Natural B−A macro-F1 CI[-.02109,.01329] includes0: no natural response-level gain established. Natural token F1 rises .56523→.57651, paired differenceCI[.00234,.02476], but exact-span F1 falls .03879→.03020. Both token F1 values remain below B7 .58489. Positive findings and regressions coexist; no overall winner or novelty is established.

R3-selection-only recalibration does not rescue the main result. A selects threshold .11568570 and B .23292027 before diagnostic, versus fixed .37612173. On diagnostic their recalibrated MF1 is .46461/.57149, worse than their respective fixed points. Natural recalibrated MF1=.73992/.77258. The selection mixture/layout does not transfer its preferred operating point reliably. These are supplemental results, not permission to choose a diagnostic-specific threshold.

R2 normalization solves all48 narrow historical equivalence pairs, but on new R3 prose yields only .55724 overall MF1; words/unit supported FPR=.50/.34375. Canonicalization is useful, yet relation binding and answer phrasing still matter. R3-A's broader synthetic .67122 exceeds the old R2 model's .50810 on this authored suite; this is a data/training-regime observation, not a matched objective comparison across R2/R3 or evidence of natural superiority. Only A versus B isolates consistency weight.

## Errors, length and localization

Natural A→B:10 annotated errors corrected,15 newly introduced; diagnostic27 corrected,59 introduced. Complete predictions, all natural FP/FN with answer/evidence/gold/spans/score, and diagnostic transitions are saved under ignored predictions/phase2b_r3. No ID-based exceptions or relabeling.

- Supported Jarrow28-bolt answer paraphrase: A risk .25758→B .48243, new false alarm despite exact supporting relation.
- Contradicted Lewes82-bolt answer with target66 and distractor82: A .57177→B .31489, new miss. Both numbers occur; presence is insufficient.
- Unsupported Lewes82-bolt claim: A .37148→B .41490, correction close to the fixed boundary; do not interpret it as large-margin robustness.
- Natural13847 becomes a response TP (.28562→.42584), but its global maximum is on `/enable`, while annotated Windows-key shortcut token risk is only .04996→.06371. A correct response label does not demonstrate localization of the annotated error. The command itself is also absent from the evidence; incomplete annotation is a possible explanation, not grounds for silently relabeling.
- Natural14337 changes TP→FN (.37892→.31263); neither model's maximum overlaps the annotated error. This again exposes the gap between max-token response correctness and span correctness.
- Natural12283, gold0 under the unchanged policy, becomes FP (.24113→.47238) on a multi-condition duration answer. This is an annotation-relative error count, not a claim that every paraphrase has been independently adjudicated.

Natural long-answer/short-annotation slice (23responses, >=500characters, gold coverage<=5%): B7 detects17/localizes13; R3-A15/9; R3-B16/10 at the same .3761 point. Thus improved synthetic localization does not solve natural short-error coverage.

In the manually reviewed equivalent natural audit, changing thermometer evidence15minutes→900seconds raises supported answer11958 risk .07993→.75128 (A) and .12662→.74317 (B), introducing false positives. Two-shades→2shades is much more stable; passage reorder shifts13847 risk .10578/.08782 for A/B. Six previously inspected rows are a regression check, not a new benchmark.

Common B7-tokenizer length buckets (76/219/353 rows for <=64/65–128/>128 tokens) are in natural_length_slices_common.json. Use that file for cross-model length comparisons; raw analysis_natural slices use a fallback length for MiniCheck. A FPR=.0139/.1056/.2263 and B=.0139/.1111/.2737; long-answer FNR=.2331/.2086 versus B7 .1534. Only four positives occur in the shortest bucket; length is correlated with content/label prevalence, so these are descriptive, not causal length effects. Generator and annotation-type breakdowns are saved in analysis_natural.json; neither metadata enters inference.

One known normalized passage overlap with inner-fit remains at source15181. Primary648 membership is unchanged. Supplemental642-row MF1: B7 .79556, A .77935, B .77464. Excluding it does not reverse conclusions or make development data independent.

## Engineering verification and reproducibility

Two fits, one fixed epoch each,453updates and1812microbatches. Training total664.2145seconds (~11.07minutes), peak allocated3.168GB, reserved3.366GB. Scoring times/throughput/peak memory are recorded for every pass; timings exclude checkpoint loading and bootstrap. B7 uses batch1, new/R2 span models batch2 and MiniCheck batch4; speed differences are not architecture-only effects. No OOM, resource-failed fit or lost context. All examples fit existing2048 limit, longest natural window1008tokens. No competing GPU-heavy jobs ran.

143research tests passed in67.97s with real-data/model opt-ins;137offline passed and6optional skipped in53.53s. Both had two existing pytest cache-directory WinError5 warnings, unrelated to test correctness; isolated basetemp succeeded. Ruff passed; pip check passed in both environments. CI remains offline, file exists/tracked, Python3.11/3.12 matrix unchanged; no remote workflow success verified. No new dependencies, paid APIs, cloud GPU, commits or pushes.

After all scoring, conservative normalizer handling was hardened for word ranges/coordinations such as 'two to four kilometers'; the pre-patch module is preserved. All normalized evidence bytes are identical on1248 historical R2,480 R3 selection and640 R3 diagnostic rows (normalizer_range_guard_regression.json). Existing experiment hashes/predictions were not rewritten or bypassed. Runtime source snapshots retain the executed version; final source archive includes the guard fix and preserved predecessor.

Changed runtime/source files: src/grounded_hallucination_lab/research/invariance.py, invariance_training.py; scripts/research_r3.py; tests/test_invariance.py. Added plans: EXPERIMENT_R3_PLAN.md, EXPERIMENT_R3_AUDIT_AMENDMENT.md, EXPERIMENT_R3_FIT_AUDIT.md. Updated README.md, RESEARCH_NOTES.md, docs/PROGRESS.md, docs/DECISIONS.md, docs/EVALUATION_PROTOCOL.md; this new report. Local orchestration/analysis/audit helpers are preserved in the source archive, not lost under models/ gitignore. Raw/processed datasets, predictions, weights and caches are not staged.

Run from repository root using the existing research environment. Commands for a fresh copy of the preserved source with the same pinned raw data and B7/R2/MiniCheck artifacts, and empty R3 output paths:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv-research\Scripts\python.exe scripts/research_r3.py freeze
.\.venv-research\Scripts\python.exe scripts/research_r3.py score --method R2C0s42N --split r2_normalization
.\.venv-research\Scripts\python.exe scripts/research_r3.py profile
.\.venv-research\Scripts\python.exe models/phase2b_r3/pilot_queue.py
.\.venv-research\Scripts\python.exe models/phase2b_r3/finish_queue.py
.\.venv-research\Scripts\python.exe models/phase2b_r3/missing_evidence_audit.py freeze
.\.venv-research\Scripts\python.exe models/phase2b_r3/missing_evidence_audit.py score
.\.venv-research\Scripts\python.exe models/phase2b_r3/fit_audit.py
.\.venv-research\Scripts\python.exe models/phase2b_r3/analyze.py
.\.venv-research\Scripts\python.exe models/phase2b_r3/supplement.py
.\.venv-research\Scripts\python.exe models/phase2b_r3/common_length_audit.py
.\.venv-research\Scripts\python.exe models/phase2b_r3/verify.py
```

Do not delete historical output directories to rerun. Ordinary commands refuse overwrite; the completed pilot/finish queues skip saved reports. Inspect processes before resuming any queue. No partial optimizer-state resume exists; both actual fits are complete, so no training continuation is pending. Checkpoints, configs, seeds, sample order/target hashes, logs, predictions and manifests remain local.

## Next three hypotheses, not executed

1. **Exact canonicalization plus relation checking, with offset-preserving answer mapping.** Compare raw versus exact evidence normalization versus symmetric evidence/answer normalization on frozen models and fresh held-out expressions. Keep uncertain quantities/ranges untouched and separate relation errors from normalization errors. The narrow control already works; this is the cheapest way to test whether a substantial remaining gap is representational. No novelty claim.
2. **Diagnose the numeric supervision bottleneck with a matched training-signal ablation.** Before another larger model, measure per-token CE/consistency gradients and separate numeric measurement from the many already-correct claim tokens. Preregister an equal-step, identical-example control against numeric-slot-focused anchored supervision (or a balanced presentation schedule, not both at once). Require low supported risk AND retained contradiction margins on fit and selection. Current MSE fails even on training equivalence; trying arbitrary lambda/seeds is not justified.
3. **Local claim-plus-neighbor verification for long natural answers.** Preserve offsets and full evidence, compare extractive local context with full-answer representation, retaining one-false-claim detection. Audit annotation coverage and new independently reviewed multi-fact examples before claims about exact-span improvement. The23-case slice and max-on-unannotated-token examples motivate this, but do not prove a particular fix.

None authorizes outer dev/test scoring or a release. Stable natural improvement, independent rendering coverage, seed robustness and a compatible untouched evaluation remain unresolved. The long-term research goal continues; the tested consistency variant is a negative result, not a finished high-quality detector.

Confidence intervals do not adjust for multiple exploratory comparisons, adaptive suite design or prior inner-validation use. Repetition seeds were gated off; bootstrap does not measure training-seed variability. Final source backup: research_snapshots/R3_source_20261024.zip and its JSON manifest. Complete local artifact inventory: models/phase2b_r3/R3_artifact_manifest.json. Machine handoff: models/phase2b_r3/WORK_STATE.json.

# R3 preregistration — 2026-10-24

Frozen before R3 model predictions/training. R1/R2 are historical development probes, not untouched tests. No outer calibration/dev/test. Preserve all previous artifacts. Initial weights for every new arm: B7 inner42, SHA256 7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4. Do not repeat old fits.

## Hypothesis and bounded packet

Exact equivalent evidence views plus ordinary supervised factual negatives may reduce representation-induced false alarms. Prediction consistency is established prior art (CCHD); cross-paraphrase contrastive grounding is established prior art (CPIL). This is a token-aligned controlled ablation, not a novelty claim or their reproduction.

1. Reanalyze saved B7/R2CE/R2pair/MiniCheck R2 predictions, including response/claim risks and localization. No fitting on R2 diagnostic errors.
2. Read-only numerical-normalization control: normalize evidence only, preserving answer offsets, on R2 word/unit probes and the new R3 suite. Canonical integer minutes/meters and explicit integer number words only; reject approximation/range sentences. Never replace entity/attribute relations or infer support from number presence. R2 narrow success does not establish general relation reasoning.
3. R3-A: one-epoch weighted token CE, natural inner-fit plus new authored R3 training examples.
4. R3-B: exactly the same data, order, optimizer and steps; add lambda=1 times mean squared difference of aligned claim-token hallucination probabilities on SUPPORTED equivalent evidence pairs only. Bounded [0,1] loss, fixed lambda before predictions; no lambda search. Both branches compute identical graphs, A has zero coefficient. CE includes contradictory and unsupported examples too. No constraint across opposite truth values.
5. Conditional R3-C only if selection B reduces supported equivalent-view absolute risk gap by >=20% versus A, yet transformed supported mean risk >=.5 and FPR is >=A+.05. Then matched B plus .1 mean negative log supported-token probability on the same aligned positive pairs, same initialization/data/steps. Otherwise do not run C.

## Suite and isolation

`research/invariance.py` is the frozen semantic specification. New fictional parents, quantities, exact elapsed durations, exact path lengths. Split-specific entity names, canonical value sets and evidence renderers: training declarative, selection ledger, diagnostic narrative. All value assignments and evidence order are crossed. Both target and unrelated entity values occur in contradictory premises; presence alone is insufficient. Equivalent digits/words/units keep truth unchanged; unsupported cases omit the target measurement, not assert its opposite. Unsupported and contradiction have separate categories. Answer paraphrase/number-word, multi-fact, and padded short-error cases are separate conditions. All derivatives stay with their parent.

Training uses 6 value pairs per kind; selection 3; diagnostic 4. Counts/hashes freeze in suite_manifest.json before scoring. Values in canonical units are disjoint; lexical components of words, units, generic answer grammar and common filler intentionally overlap. Thus family separation is operational, not independence of authors/language. Target answer position: train endpoints with 4 fillers, selection one-third with 9, diagnostic two-thirds with 18. Short base claims necessarily share position zero. Evidence order crossed in every split. Held-out diagnostic renderers never changed after predictions. No synthetic suite is called a natural benchmark.

Natural inner validation stays 648 responses/108 sources; all 2568 inner-fit rows retained. Review a small natural multi-fact slice before predictions, retain original RAGTruth labels, do not manufacture new labels from arbitrary swaps. If safe equivalent natural rewrites cannot be established, report that limitation instead of inflating synthetic counts.

## Training, thresholds, gates

Seed42 first, fixed one epoch, microbatch2/accumulation4, lr1e-5, weight_decay .01, warmup10%, clip1, positive CE weight3.370202392225359, BF16 autocast/FP32 weights and loss, gradient checkpointing, max_length2048. Full evidence and answer coverage required; fail on unreported truncation. Same B7 initial checkpoint for all seeds; repeats measure continuation randomness, not independent pretrained runs. No architecture/retrieval/segmentation/global-max changes.

Primary fixed threshold .37612172961235046 for all span models. MiniCheck keeps .9390248171985149. Supplemental train-side recalibration is declared now: choose macro-F1 threshold on R3 selection only (all boundaries, highest exact tie, score>=threshold), freeze before diagnostic; report natural inner metrics separately with explicit synthetic-selection bias. No new threshold on natural or diagnostic, and no recalibrated point can rescue a failed primary gate.

Continuation gate evaluated on R3 selection BEFORE new-model diagnostic: B (or conditional C) versus matched A must reduce pooled word/unit supported FPR >=.10, with each FPR <=.50; contradiction recall must not fall >.05; correct supported/contradicted pair rate must improve >=.05; natural macro-F1 drop <=.01 and token F1 drop <=.02; long-answer contradiction recall drop <=.05. Directional gates are operational, not significance tests. If all pass, run matched A and finalist seeds43/44; report all seeds. Otherwise stop repeats, preserve negative result. Diagnostic is run once after this decision, never for selecting a new loss.

R2 reference: natural MF1 .784686/.787152; diagnostic MF1 .791267/.768689, AUROC .886816/.901474, FPR .336538/.415064; words and unit supported FPR=1. These motivate a meaningful >=10pp FPR gate, not cosmetic ranking gains.

## Reporting and budget

At most two unconditional fits, one conditional fit, and four matched repeats if gate passes. Profile real forward/backward/AdamW on longest natural and equivalent pair first, max11GiB allocated and no concurrent GPU jobs. Budget <=1 GPU-hour for pilot/baselines, <=2 total if repeats justified; record actual time, peak allocated/reserved/RSS, coverage. No optimizer resume is implemented: interrupted incomplete directory is preserved, never overwritten; completed fits never repeated.

Save inputs/design privately, configs, sample order/target hashes, losses, checkpoints, source snapshots, predictions, errors. Response accuracy/MF1/positiveP/R/F1/AUROC/AP/CM/FPR/FNR plus pair accuracy/deltas, supported invariance, custom token/nonwhitespace-character/exact-span metrics (.5). Claim masks are training/diagnostic annotations only, never inference inputs. 2000 identical source-level bootstrap draws for methods and paired differences, including localization; not independent rows and no correction for adaptive design. No claim of natural superiority from synthetic gains. Record negative results and next hypothesis.

Implementation detail before freeze: canonical numeric examples are repeated once for each word/unit equivalent pair, under BOTH labels equally. All arms use this identical effective sample schedule; unique/effective counts and target mass are logged. Natural audit uses previously inspected inner sources (11958,12482,13847), with original labels and exact evidence-only rewrites/reorder; this is a regression audit, not a new independent natural suite.

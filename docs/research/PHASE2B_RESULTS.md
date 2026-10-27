# Phase 2B+ — bounded batch results

Status: PAUSED BY USER after all three final B7 training seeds completed. Answer-only, sensitivity, calibration and new dev evaluation have not run. This is an incomplete batch report; see PHASE2B_PAUSE_HANDOFF.md to continue. Final training wall times:seed42=1713.65s, seed43=1700.73s, seed44=1736.78s. No unfinished training remains.

## Scope and integrity

QA response-level, include/all, positive=hallucination. Outer partitions remain3216/810/1008/900 responses and536/135/168/150 sources (train/calibration/dev/official test). Dev ID SHA256=e62245acf4c87da5cfa1465e0bd79f6b9214358c286a92fe17d90b10e99dde9d. All source-ID pairs are disjoint. Official test labels remain sealed and test is not evaluated.

The Phase2A snapshot was restored and executed with its original code/environment/freeze checks: predictions are byte-identical, all eight checks pass. Its dev macro-F1 remains0.5909091, positive F1=0.5000, AUROC=0.665954 and AP=0.417237. Reports, predictions and source/dependency hashes were not altered to obtain compatibility.

One normalized659-character evidence passage is shared by train sources14426/15181 and dev source15272 (six responses). No rows are silently removed. Main evaluation remains1008; PHASE2B_PARTIAL_OVERLAP.md preregisters an additional1002-row sensitivity with frozen thresholds. Full-source equality and source-ID checks alone missed this partial overlap.

## Train-side selection

Inner fit2568 responses/428 sources; inner validation648/108, deterministic SHA256 source grouping. These threshold-optimized numbers are selection statistics, not unbiased performance estimates. B3 is diagnostic only because HHEM2.1 training provenance is incomplete.

| Method | Macro-F1 | Positive F1 | AUROC | AP |
|---|---:|---:|---:|---:|
| B0 | 0.6517 | 0.5232 | 0.6931 | 0.4721 |
| B1 | 0.6599 | 0.5492 | 0.7196 | 0.5069 |
| B2 | 0.7202 | 0.6265 | 0.7970 | 0.5998 |
| B3 | 0.8006 | 0.7344 | 0.8754 | 0.7824 |
| B4 | 0.7293 | 0.6370 | 0.8049 | 0.6548 |
| B5 | 0.7222 | 0.6296 | 0.7973 | 0.6047 |
| B6 | 0.7753 | 0.6993 | 0.8591 | 0.7016 |
| B7s42 | 0.7959 | 0.7314 | 0.8760 | 0.7897 |
| B9 | 0.7172 | 0.6256 | 0.7784 | 0.6201 |

B1=NLI old segmentation/top3; B2=MiniCheck author document packing plus NLTK claims; B3=HHEM whole response; B4=MiniCheck old segmentation/top3; B5=MiniCheck NLTK claims/full context; B6=MiniCheck extractive cleaner/full context; B7=own ModernBERT token head; B9=length-only logistic regression. B8 is the matched answer-only training control. B0 is historical lexical architecture fit only on inner-train for this table.

The bounded package has ten configurations B0–B9, with three repetitions of B7. Before any new dev evaluation, the unexecuted learned-aggregator proposal for B6 was replaced by segmentation cleanup after a train-side audit found bare list-number alarms. B5/B6 isolate the segmentation stage; individual cleaner operations are not separately ablated. B2/B5 isolate evidence-context handling. No dense retrieval, synthetic negatives, RL, distillation or ensemble was run.

Own inner training selected epoch3: epoch1/2/3 macro-F1=0.7623/0.7952/0.7959. Final seeds42/43/44 restart from the original clean backbone and use exactly three epochs, with no held-out early stopping. Grouped CV was implemented but not run: measured budget projection would exceed the four-GPU-hour package gate (resource_gate.json). This limits architecture-selection confidence.

## Confirmed bottlenecks

See PHASE2B_TRAIN_AUDIT.md for the eight-source convenience retrieval audit, numerical/paraphrase examples, fragment-count diagnostics and offset checks. RAGTruth error spans are not evidence-support labels; no retrieval Recall@k is claimed.

B2→B5 changes inner macro-F1 by only+0.0021. B5→B6 improves it by+0.0530, consistent with a material segmentation problem. Recalibrated thresholds also change, so individual corrected cases are not attributed exclusively to one text operation. All389 inner gold spans retain some non-whitespace content after cleanup; this is a coverage check, not proof of semantic correctness.

Length is a substantial confound: B9 inner macro-F1=0.7172. B7 risk/length Spearman correlation=0.597; evidence/answer-only controls are necessary before attributing its performance to grounding. NLI neutral and contradiction are reported separately.

## Resources and failures

Local RTX3060 12GiB, RAM32GB, i5-10400F. Neural runs use FP32 weights with BF16 autocast, deterministic algorithms and pinned local checkpoints. The149,606,402-parameter own model uses microbatch1, accumulation8, gradient checkpointing, AdamW2e-5, three epochs. Full train inputs fit completely:3216 windows, max1008 tokens, configured limit2048, zero lost evidence/answer tokens. This does not establish8192-token feasibility.

The first backward smoke failed because a CUDA nll_loss2d path was nondeterministic. Flattened token cross-entropy fixes that path while retaining strict determinism. An early successful profile used eval mode; it is retained separately, and the corrected train-mode profile plus real training confirm feasibility. No model OOM occurred. Network/download failures were resumed and recorded. No paid API/cloud GPU was used.

Initial B1–B5/B9 inner runs overlapped implementation changes and therefore were not accepted as canonical provenance. They were preserved, replayed under stable code, and all response IDs/scores match exactly (replay_integrity.json). No hash guard was bypassed.

## Links

- `Checkpoint lineage and primary sources` (historical local artifact; not distributed)
- [Reproduction commands](PHASE2B_REPRODUCE.md)
- `Future untouched evaluation protocol` (historical local artifact; not distributed)
- `Experiment plan` (historical local artifact; not distributed)

# Research timeline

Public reports preserve measured conclusions, including failures. Local artifact
paths mentioned in historical reports are provenance references, not downloadable
assets. Checkpoints, datasets, full predictions and review casepacks remain local.
No experiment below demonstrates a new architecture or universal superiority.

| Stage | Question and conclusion | Report |
|---|---|---|
| Phase 1 / 1.1 | Ingestion, labels, grouped partitions, test isolation, reproducible majority | [Protocol](EVALUATION_PROTOCOL.md) |
| Phase 2A | Train-only TF-IDF; frozen calibration threshold; dev macro-F1 .5909 | [Results](research/PHASE2A_RESULTS.md) |
| Phase 2B / R1 | Specialized verifiers and own span model; counterfactual training fits narrow templates but does not establish broader natural gain | [R1](research/PHASE2B_R1_RESULTS.md) |
| R2 / R3 | Relation transfer, semantic invariance and consistency objectives; broader transfer remains weak | [R2](research/PHASE2B_R2_RESULTS.md), [R3](research/PHASE2B_R3_RESULTS.md) |
| R4 | Symmetric normalization improves selected synthetic tests; natural normalization has no useful coverage | [R4](research/PHASE2B_R4_RESULTS.md) |
| R5 | Learning numerical equivalence improves old suites without convincing new natural transfer | [R5](research/PHASE2B_R5_RESULTS.md) |
| R6 | Diverse relation rendering improves synthetic transfer; natural/short-error retention gate fails | [R6](research/PHASE2B_R6_RESULTS.md) |
| R7 / R8 | Graph matching works on authored grammar; natural end-to-end coverage remains zero on held-out diagnostic sources | [R7](research/PHASE2B_R7_RESULTS.md), [R8](research/PHASE2B_R8_RESULTS.md) |
| R9 / R10 | Evidence-order sensitivity diagnosed; matched permutation augmentation fails gate | [R9](research/PHASE2B_R9_RESULTS.md), [R10](research/PHASE2B_R10_RESULTS.md) |
| Discovery | FactCG gives stronger controlled factual discrimination and external QA evidence; source/provenance caveats remain | [Verdict](research/RESEARCH_LOOP_VERDICT.md) |
| Scope diagnosis | Matched content policy and development operating points reduce natural response gap; multiplicative token composition loses recall | [Results](research/SCOPE_DIAGNOSIS_RESULTS.md) |
| Claim grounding | Native gradients and context subtraction fail; new human dialogue transfer favors FactCG, without solving exact spans | [Results](research/CLAIM_GROUNDING_RESEARCH.md) |

The frozen-representation linear probe was discussed but **not trained or benchmarked**
before engineering publication. It is not a reported result. Public inference uses
the preserved B7 reference, not an unvalidated newer experiment.

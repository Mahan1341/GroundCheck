# Resume packet R1 — 2026-10-23

Supersedes automatic calibration/dev scheduling only; EXPERIMENT_PLAN.md and all historical artifacts remain unchanged. No calibration/dev/test scoring in this packet. A stable evaluation protocol must be agreed before new dev runs. User authorized real local training, no paid/cloud compute or concurrent GPU-heavy jobs.

## Finite scope and budget

Cap3 additional GPU-hours for this resume packet (wall time used as conservative proxy), with one measured next-method experiment after diagnosis; no grid. Preserve all completed B7 full-train seeds42/43/44. Start B8 full-train seed42 with the exact existing architecture/loss/data/3epochs/effective batch8. Profile answer-only maximum-length inputs first.

For meaningful train-side comparison, additionally fit B8 inner seed42 on the same2568 responses/428 sources as B7 inner seed42, fixed3epochs (matched to selected B7 epoch), evaluate on648/108 inner-validation. Do not use full-train B7 on inner-validation as a generalization estimate. Input removal is the only substantive B8 change. Same per-answer token targets/loss weighting and training order, optimizer, initialization seed and schedule.

Additional B8 full-train seeds43/44 are planned only after measured cost/benefit: they cannot establish held-out quality until evaluation is authorized. Prioritize paired inner repetitions (both B7/B8 on identical train sources) if initial conclusion is close and budget permits. Do not select a seed using dev.

## Diagnostics before choosing a new method

Run existing B2 and B7s42 sensitivity scripts on32 deterministic train-source responses plus numerical/negation/entity toys. Full-train model sensitivity is a mechanistic check on seen data, not an independent quality estimate. Extend analysis using INNER models on inner-validation: fixed answer, original/blank/different-source evidence; arbitrary replacements receive no invented gold labels.

Use manually verifiable authored controls with matched supported/contradicted relations, correct paraphrases, evidence order/redundancy, variable answer length, many correct sentences plus one short false fact. Preserve exact spans; synthetic controls are a stress suite, not a natural benchmark. No quality flags/generator/gold spans as inference features.

Natural inner diagnostics: positive and negative errors by answer tokens/sentences, short-gold-span fraction, numbers/negation/attribution examples, custom token/character/span metrics. Avoid causal claims from length correlations alone.

Record numerical results and a narrow hypothesis in a separate pre-run plan amendment before implementing one improvement. Compare with unchanged B7, isolate the changed component, preserve failures, measure time/memory, analyze improvements and regressions. No automatic best-method promotion to dev.

## Pre-result decision guide

If long-padding controls preserve a strong claim-local evidence-intervention signal but response max saturates on unrelated correct filler, prioritize aggregation. If claim-local intervention signals themselves are weak in multiple relation families, prioritize evidence-conditioned supervision. These are small diagnostic controls, not statistically powered population thresholds; report both signals and reasons for the choice. A full method/configuration plan will be recorded before any new-method training.

Seed-repeat budget gate: after measuring B8 full seed42, run full-train B8 seeds43/44 only if each is projected<=35min and all mandatory R1 diagnostics plus one isolated improvement/control and10% reserve fit the3-hour cap. These prepare future matched-seed evaluation; do not claim seed-level quality robustness before held-out scoring.

## Label-free isolation finding before B8 inner results

The inner passage audit found one normalized fragment shared by inner-fit source14426 and inner-val source15181 (six responses). Preserve all membership/training. Report a supplemental642-response inner sensitivity excluding source15181 only at evaluation, with unchanged648-row selected thresholds; no retuning on the subset and no method selection using it. The main648-row comparison remains primary. This exclusion is based on evidence overlap, not observed errors.

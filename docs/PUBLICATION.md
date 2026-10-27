# Public engineering boundary — 2026-10-27

This is an engineering publication, not a new benchmark or research result.
The public tree is an explicit source export. Historical local checkpoints,
predictions, dataset copies, source snapshots and hashes were not deleted or changed.
The research workspace remains separate so old source freezes still verify.

## Checked locally

* Fresh Python 3.12.14 environment; non-editable package installation built a wheel
  and installed its dependencies. Inference was imported from site-packages, not
  accidentally from the original research workspace.
* CPU neural dependencies installed separately; `pip check` passed. Exact tested
  environment: `configs/inference-cpu-lock.txt` (Windows CPU, not a universal CUDA lock).
* Full offline suite with neural dependencies installed: **375 passed, 6 skipped**.
  Actual B7 authored-example integration: **6 passed**. Without optional neural
  packages additional tests skip; real corpus/model research tests remain opt-in.
* Ruff passed. Original base/research dependency checks also passed.
* Real B7 CPU and CUDA inference, single JSON and sequential JSONL; empty answer
  behavior and original offsets checked. The clean installed CPU package reproduces
  `examples/output.cpu.json` exactly on the tested machine.
* Public CUDA token scores/offsets match the historical FP32/batch1 scorer exactly
  on the authored example. CPU versus CUDA response risk differs by approximately
  `1.91e-6`; equality across hardware is not promised.
* Historical claim-grounding seal verified 329 current artifacts and its two prior
  291/177-artifact packets. Publication changes do not replace old frozen sources.

These are local checks. GitHub Actions runs offline tests, Ruff, pip check and the
fixture CLI on Python 3.11/3.12; consult the actual commit's Checks tab for remote
results. GPU/model evaluation is not part of CI. No new training or outer RAGTruth
evaluation was performed.

## Publication audit

All staged public files were checked for credential signatures, local absolute
paths, binary/weight extensions, oversized files and excluded data/cache directories.
Authored fixtures/examples and configuration metadata were reviewed separately from
real datasets. Historical reports keep numeric conclusions, label-provenance caveats
and failures; local-only artifact links are explicitly marked as unavailable.
One machine-specific command in a report was replaced with its portable equivalent.
No private manual-review pack, corpus text dump or model file is included.

The original Git had no reachable commits. Its entire local object database was
also scanned: 851 objects, including 690 blobs. Two unreachable blobs contained
machine-specific paths. That object database was **not copied or pushed** and was
not rewritten. The public export starts its own audited publication history.
Signature checks reduce risk; they are not a mathematical proof of absence of secrets.

Own code is MIT. Third-party model/dataset terms are documented in `NOTICE.md`;
the native FactCG prompt adapter is attributed with the upstream MIT notice in
`licenses/FactCG-MIT.txt`. Real data and weights are excluded regardless of whether
some upstream repository uses a permissive code license.

## Remaining reproducibility limits

B7 weights are not hosted in this release. A separately trained checkpoint needs an
explicit user-supplied hash/threshold profile and cannot inherit historical metrics.
Research scripts often require locally generated, hashed experiment artifacts;
publishing source does not recreate those artifacts automatically. The reproduction
recipe is provided, but no retraining was performed as part of publication.

Continue architecture research in an isolated development workspace and preserve
old artifacts. Do not tune on the closed official test or used external confirmation,
and do not turn reused inner/SAMSum development into an independent-test claim.

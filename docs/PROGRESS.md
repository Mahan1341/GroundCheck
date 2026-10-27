# Public package status — 2026-10-27

GroundCheck is published at https://github.com/Mahan1341/GroundCheck.
The B7 inference API/CLI supports original offsets, explicit local checkpoints,
CPU/CUDA and sequential JSONL. Authored demo output is recorded, including localization
errors. No research experiment, training or outer evaluation was added for publication.

Clean-clone installation and local checks are described in [PUBLICATION.md](PUBLICATION.md).
Use the current commit's GitHub Actions run for remote CI status; do not infer it
from a local pass. The first remote run exposed an ignore-rule packaging error;
it was corrected and a regression added without rewriting history.

Local research artifacts remain in the original separate workspace. Use this public
checkout for publication changes. B7 weights are not distributed: follow
[CHECKPOINT.md](CHECKPOINT.md) for provenance and a compatible reproduction route.
The proposed frozen-representation probe was not trained and has no reported results.

Next engineering priority is preserving the tested inference contract when research
resumes. Exact span localization, transfer and checkpoint distribution remain limitations.

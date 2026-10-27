# Experiments

This directory consolidates published research materials without rewriting historical
reports, provenance hashes, frozen datasets, or immutable model weights.

- [Plans](plans/) — historical experiment proposals, amendments, and gates.
- [Reports](reports/) — stage results and external-comparator investigations.
- [Scripts](scripts/) — canonical experimental runners and diagnostic code.
- [Research timeline](../docs/RESEARCH_TIMELINE.md) — concise chronological index.

## Reproducibility

Run commands from the repository root. Compatibility entrypoints remain in `scripts/`
for existing commands and offline tests. The importable research package stays at
`src/grounded_hallucination_lab/research/` to keep external import contracts stable.

Historical report relative links may refer to their original locations; their bodies
are intentionally preserved for research provenance. Original private artifacts
(checkpoints, data, predictions and snapshots) are not hosted in this repository.

Do **not** copy this layout over the user's original local research workspace without
first auditing source hashes and frozen relative paths. Preserve previous experiments.

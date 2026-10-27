# Phase 2B+ reproduction

Run from the repository root on Windows/Python3.12 with a local CUDA GPU. The offline package
and CI do not require Torch or model downloads. Research commands refuse to overwrite artifacts;
use a fresh checkout/output state when reproducing, preserving the original run directories.
Official test has no research command. Do not bypass source/dependency/data mismatch failures.

## Environment and pinned assets

```powershell
.\.venv\Scripts\python.exe -m venv .venv-research
.\.venv-research\Scripts\python.exe -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv-research\Scripts\python.exe -m pip install -r configs/research-lock.txt
.\.venv-research\Scripts\python.exe scripts/research_fetch_models.py nli minicheck hhem hhem_foundation backbone
.\.venv-research\Scripts\python.exe scripts/research_fetch_nltk.py
```

Model repositories/revisions and per-file hashes are in configs/checkpoints.json and ignored
models/pretrained/*/download_manifest.json. Torch wheel SHA256 used locally:
0ad925202387f4e7314302a1b4f8860fa824357f9b1466d7992bf276370ebcff.
NLTK punkt_tab source revision/archive hash are in configs/nltk-data.json; tables live in
models/nltk_data. Preserve them with model artifacts; the final freeze hashes the actual files.
HHEM custom code was reviewed, and its nested foundation lookup is pinned/local/offline.

## Train-side work, before calibration/dev

```powershell
.\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run prepare
.\.venv-research\Scripts\python.exe scripts/research_profile.py nli
.\.venv-research\Scripts\python.exe scripts/research_profile.py minicheck
.\.venv-research\Scripts\python.exe scripts/research_profile.py hhem
.\.venv-research\Scripts\python.exe scripts/research_profile.py span
.\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run train-span --stage inner --seed 42 --epochs 3 --batch-size 1
foreach ($method in @('B1','B2','B3','B4','B5','B6')) {
  .\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run score --method $method --partition inner_val
}
.\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run fit-simple --method B9 --stage inner
.\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run score --method B9 --partition inner_val
.\.venv-research\Scripts\python.exe scripts/research_inner_audit.py
```

Inner thresholds optimize inner macro-F1 only for model/epoch selection; these are not unbiased
validation estimates. Inspect the bounded package results and budget gate in the results report.
Three grouped CV folds are allowed only by the preregistered resource gate; do not expand a grid.
Final training reads the selected epoch from span_inner_s42/result.json and ignores a user-supplied
--epochs for final fits, preventing accidental held-out early stopping.

```powershell
foreach ($seed in @(42,43,44)) {
  .\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run train-span --stage final --seed $seed --batch-size 1
}
.\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run train-span --stage final --seed 42 --batch-size 1 --answer-only
.\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run fit-simple --method B9 --stage final
.\.venv-research\Scripts\python.exe scripts/research_sensitivity.py B2
.\.venv-research\Scripts\python.exe scripts/research_sensitivity.py B7s42
```

Final freeze/calibration/dev commands and the actual chosen candidate list are recorded in the
results report after train-side selection. Do not run new dev experiments while making changes.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv-research\Scripts\python.exe -m pytest -q
.\.venv-research\Scripts\python.exe -m ruff check .
.\.venv-research\Scripts\python.exe -m pip check
$env:GROUNDCHECK_RAGTRUTH_DIR = 'data/raw/ragtruth'
$env:GROUNDCHECK_RESEARCH_MODELS = '1'
.\.venv-research\Scripts\python.exe -m pytest -q
Remove-Item Env:GROUNDCHECK_RAGTRUTH_DIR
Remove-Item Env:GROUNDCHECK_RESEARCH_MODELS
```

CI .github/workflows/ci.yml is present and tracked in the Git index, Python3.11/3.12 matrix.
It runs only offline tests. No remote CI success is implied by local checks.

## Historical Phase 2A

models/phase2a_snapshot/phase2a.zip contains68 files with SHA256 manifest and old environment
requirements. Restore into a separate ignored directory and use its src on PYTHONPATH with the
unchanged old .venv, original raw data and archived model/config files. Write predictions/report
to fresh locations. New research code changes the source-hash universe, so running the old freeze
against the new source tree should fail. Never edit its hashes to make it pass.

This procedure was actually executed: reports/phase2b/phase2a_snapshot_reproduction.json;
predictions/phase2b/historical_reproduction/dev.jsonl is byte-identical to the original.

### Reading training logs

For the executed final runs, the log field selection=1/2/3 is an internal epoch counter used to save the final fixed-epoch checkpoint, NOT a validation metric. Final training has no validation partition or early stopping. Only inner runs have actual threshold-optimized inner selection metrics. The JSON history names these inner_metrics explicitly; final histories contain training loss/resource measurements only.

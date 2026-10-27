# B7 checkpoint and reproduction

The public adapter's reference is **B7 inner seed42**, a ModernBERT-base token
classifier with 3 selected epochs, 2048-token windows and full evidence.
Backbone: `answerdotai/ModernBERT-base`, revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8` (Apache-2.0).
Reference weights SHA256:
`7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4`.
The bundled `assets/b7_inner_seed42.json` hashes weights, config and tokenizer files.

## Availability and rights

There is **no hosted B7 checkpoint download in this release**. We do not publish
weights into Git or claim that the backbone alone is a working detector. Use the
locally preserved reference artifact if you have lawful access. Otherwise obtain
RAGTruth and the pinned backbone from their authors and reproduce training below.
Own-code MIT does not relicense model derivatives or underlying dataset texts.
No inference command downloads models or executes remote code.

## Reproduce a new compatible checkpoint

These are explicit research commands, **not run during publication**. They require
CUDA and separately acquired training data. Start with an empty `models/phase2b`
output directory; never overwrite a historical run.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]' -r configs/research-requirements.txt
python scripts/fetch_ragtruth.py
python scripts/research_fetch_models.py backbone
python -m grounded_hallucination_lab.research.run train-span --stage inner --seed 42 --epochs 3 --batch-size 1
```

The training entry point makes the existing deterministic source-grouped inner split
inside train, masks evidence/question/special tokens from loss and uses the existing
training implementation. Seed42, AdamW lr2e-5/weight-decay .01, microbatch1 with
accumulation8, linear schedule with10% warmup, gradient clipping1, class weight
`min(10,sqrt(negative/positive token mass))`, BF16 forward/FP32 loss. The original
reference selected epoch3 on inner macro-F1; this is development selection.
No official-test optimization command is provided.

Historical dependencies/configs remain in `configs/`. Hardware, software and
historical implementation changes mean **a new run is not promised to match reference
bytes or metrics**. Select its threshold only on permitted development sources.
Record it explicitly instead of pretending it is the reference artifact:

```bash
python scripts/make_checkpoint_profile.py --checkpoint models/phase2b/span_inner_s42/best --threshold YOUR_SELECTED_THRESHOLD --output models/retrained-profile.json
groundcheck --checkpoint models/phase2b/span_inner_s42/best --profile models/retrained-profile.json --device cpu --input examples/input.json
```

The supplied profile is labeled `user-supplied`; historical B7 quality claims do not
apply to it. Do not copy a threshold from an unrelated checkpoint as calibration.
Never supply a randomly initialized head as the released detector.

## Inference contract

Input has string fields `question`, `evidence`, `answer`, and optional `id`. Unicode
offsets always index the original answer. Complete evidence is repeated per answer
window; insufficient capacity is an error, not silent truncation. An empty answer
has no detected claims; an empty evidence field carries a warning. The API returns
candidate spans, not trusted corrections. JSONL is sequential batch1 inference.

Offline unit tests need no weights. Run the actual authored-example integration test
only with an explicitly provided local reference checkpoint:

```bash
# Bash; PowerShell equivalent: $env:GROUNDCHECK_CHECKPOINT = 'models/b7'
GROUNDCHECK_CHECKPOINT=models/b7 python -m pytest tests/test_inference.py -q
```

Research integration flags that access real datasets/models are opt-in and are not
enabled in public CI. This integration test uses only the fictional public example.

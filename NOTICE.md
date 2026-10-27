# Third-party attribution and distribution boundary

MIT covers GroundCheck's own source, documentation and original fictional fixtures.
No third-party model implementation is vendored, and no model weights or real corpus
texts are distributed. Imported libraries retain their upstream licenses.

| Resource | Attribution / license evidence | Publication policy |
|---|---|---|
| RAGTruth | Niu et al., ACL 2024; [paper](https://aclanthology.org/2024.acl-long.585/), [author repository MIT license](https://github.com/ParticleMedia/RAGTruth/blob/c103204b9ce28d6bbad859304bf30de72b8ed8fe/LICENSE), copyright Particle Media 2023 | Fetch separately at the pinned revision; underlying source-text rights are not established by a code-repository license. No dataset text bundled. |
| ModernBERT | Answer.AI / LightOn and collaborators; [model card](https://huggingface.co/answerdotai/ModernBERT-base), Apache-2.0, pinned revision in configs/checkpoints.json | B7 derives from this backbone. Weights/tokenizer files are not shipped; checkpoint hashes are metadata only. Apache terms continue to apply to redistributed derivatives. |
| FactCG | Lei et al.; [author release](https://github.com/derenlei/FactCG), [checkpoint](https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large) | Research adapter, not GroundCheck's invention; no weights or upstream implementation distributed. Consult the pinned release's terms before acquisition. |
| MiniCheck / HHEM / NLI | Author/model identifiers, revisions and declared licenses in configs/checkpoints.json | Optional research downloads only. Review model-specific terms and any remote code before use. |
| RefChecker | [Amazon Science release](https://github.com/amazon-science/RefChecker) | Reported response labels were derived from human triplet judgments, not complete independent span annotations. No texts or labels distributed. |
| DiaSumFact / SAMSum | [DiaSumFact paper](https://aclanthology.org/2023.acl-long.377/), [author data release](https://github.com/731935354/Dia-Sum-Fact) | Reported author human annotations; ambiguous span strings excluded from localization. Repository licensing does not override underlying dialogue-dataset terms. No records distributed. |

Core dependencies include NumPy (BSD), SciPy (BSD), scikit-learn (BSD), joblib (BSD),
PyTorch (BSD-style), Transformers (Apache-2.0), safetensors (Apache-2.0), pytest (MIT)
and Ruff (MIT). They are installed separately, not relicensed by this repository.

The public source contains original adapters, authored synthetic relation controls,
and compact numeric research reports. Internal annotation-review packs, dataset
dumps, prediction casepacks, weight caches and source archives remain local. Download
commands do not grant rights beyond upstream terms. Verify suitability for your use.

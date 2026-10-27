# Publication decisions — 2026-10-27

1. Publish an explicit source export, preserving the complete original research
   workspace and its frozen artifacts. Do not push its untracked files or old Git
   object database. No history rewriting or destructive cleanup.
2. Use the existing verified owner's empty remote, rename it GroundCheck and publish
   only the audited export. Disable accidental pushes from the private research
   workspace; publication uses its own Git checkout.
3. Keep B7 inner seed42 as the named neural reference, verified by hashes. Use
   FP32/batch1/TF32 off and the existing window/offset contract. A separately trained
   checkpoint requires an explicit profile and cannot inherit reference metrics.
4. Exclude datasets, weights, predictions, private annotation reviews and source ZIPs.
   License own code MIT; retain model/data terms and third-party attribution.
5. Preserve negative results and development/confirmation distinctions. Do not label
   risk a calibrated probability or claim independent-test performance or novelty.
6. Correct the initial unanchored data-ignore rule and validate the committed clone,
   not only a working export. Keep the initial failed CI visible in history.

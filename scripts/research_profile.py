"""Actual GPU smoke/profiling before full train or inference, using only outer train."""

import argparse
import time
import traceback

import numpy as np

from grounded_hallucination_lab.research.protocol import (
    REPORT,
    InferenceInput,
    load_parts,
    now,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["nli", "minicheck", "hhem", "span"])
    args = parser.parse_args()
    from grounded_hallucination_lab.research.verifiers import (
        Verifier,
        memory_report,
        setup_runtime,
    )

    torch = setup_runtime()
    parts = load_parts()
    train = sorted(parts.train, key=lambda r: len(r.response) + len(r.evidence))
    representative = [train[i] for i in (0, len(train) // 2, int(0.95 * len(train)), -1)]
    report = {
        "created": now(),
        "kind": args.kind,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_ids": [r.source_id for r in representative],
        "response_ids": [r.id for r in representative],
    }
    try:
        if args.kind == "span":
            import torch.nn.functional as functional

            from grounded_hallucination_lab.research.span_model import (
                collate,
                encode_records,
                load_model,
            )

            model, tokenizer = load_model()
            model.gradient_checkpointing_enable()
            model.train()
            encoded, metadata = encode_records(train, tokenizer, 2048, False, training=True)
            longest = sorted(encoded, key=lambda r: len(r["input_ids"]))[-1]
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(4):
                batch = collate([longest], tokenizer.pad_token_id)
                start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                    ).logits
                    loss = functional.cross_entropy(
                        logits.float().reshape(-1, 2), batch["labels"].reshape(-1)
                    )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
            report.update(
                {
                    "status": "passed",
                    "max_length": 2048,
                    "microbatch": 1,
                    "maximum_actual_train_window": len(longest["input_ids"]),
                    "train_windows": len(encoded),
                    "max_windows_per_answer": max(m["windows"] for m in metadata),
                    "minimum_answer_coverage": min(m["answer_coverage"] for m in metadata),
                    "minimum_evidence_coverage": min(m["evidence_coverage"] for m in metadata),
                    "steps_seconds": durations,
                    "steady_max_length_step_seconds": float(np.median(durations[1:])),
                    "conservative_3epoch_seconds": len(encoded)
                    * 3
                    * float(np.median(durations[1:])),
                    **memory_report(),
                }
            )
        else:
            verifier = Verifier(args.kind, 4 if args.kind != "nli" else 16)
            pairs = [(r.evidence, r.response) for r in representative]
            pairs.extend(
                [
                    ("Alice owns a blue car.", "Alice owns a car."),
                    ("Alice owns a blue car.", "Alice does not own a car."),
                    ("Alice owns a blue car. Bob owns a red car.", "Alice owns a red car."),
                    ("The bridge opened in 1990.", "The bridge opened in 1991."),
                    ("The bridge opened in 1990.", "The bridge opened in 1990."),
                ]
            )
            rows, runtime = verifier.predict(pairs)
            report.update(
                {
                    "status": "passed",
                    "runtime": runtime,
                    "synthetic_expected_supported": [True, False, False, False, True],
                    "synthetic_results": rows[4:],
                    "representative_results": rows[:4],
                    "labels": verifier.labels,
                    "inference_allowlist_example": list(vars(InferenceInput.from_record(train[0]))),
                }
            )
        write_json(REPORT / f"profile_{args.kind}.json", report)
        print(report, flush=True)
    except Exception as error:
        report.update(
            {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        )
        write_json(REPORT / f"profile_{args.kind}_failed_{int(time.time())}.json", report)
        raise


if __name__ == "__main__":
    main()

"""Pinned verifier adapters; author input formatting and explicit coverage diagnostics."""

import os
import time
from dataclasses import asdict
from functools import lru_cache

import numpy as np

from grounded_hallucination_lab.baselines.lexical import TextSpan, sentences
from grounded_hallucination_lab.research.protocol import ROOT, InferenceInput, read_json
from grounded_hallucination_lab.research.segments import clean_segments


def setup_runtime():
    os.environ.setdefault("HF_HOME", str(ROOT / "models/hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch


def memory_report() -> dict:
    import psutil
    import torch

    mem = psutil.Process().memory_info()
    return {
        "process_rss_bytes": mem.rss,
        "process_peak_rss_bytes": getattr(mem, "peak_wset", mem.rss),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


class Verifier:
    def __init__(self, kind: str, batch_size: int | None = None):
        torch = setup_runtime()
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.kind = kind
        self.batch_size = batch_size or {"nli": 16, "minicheck": 4, "hhem": 4}[kind]
        path = ROOT / "models/pretrained" / kind
        self.checkpoint = read_json(ROOT / "configs/checkpoints.json")[kind]
        if kind == "hhem":
            # These two pinned Python files were reviewed before enabling local custom code.
            # Redirect the author's nested foundation lookup to pinned local config/tokenizer.
            foundation = str(ROOT / "models/pretrained/hhem_foundation")
            config = AutoConfig.from_pretrained(
                path, trust_remote_code=True, foundation=foundation, local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, config=config, trust_remote_code=True, local_files_only=True
            )
            self.tokenizer = self.model.tokenzier  # Author API spelling.
            self.labels = ["unsupported", "supported"]
            self.max_length = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, local_files_only=True
            )
            self.labels = (
                ["contradiction", "entailment", "neutral"]
                if kind == "nli"
                else ["unsupported", "supported"]
            )
            if kind == "nli":
                actual = [self.model.config.id2label[i].lower() for i in range(3)]
                if actual != self.labels:
                    raise ValueError(f"unexpected NLI label mapping: {actual}")
            elif self.model.config.num_labels != 2:
                raise ValueError("MiniCheck author protocol requires binary logits")
            self.max_length = 512 if kind == "nli" else 2048
        self.model.to("cuda").eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        torch.cuda.reset_peak_memory_stats()

    def encode(self, pairs: list[tuple[str, str]], *, tensors: bool):
        kwargs = {"padding": tensors, "truncation": self.max_length is not None}
        if tensors:
            kwargs["return_tensors"] = "pt"
        if self.max_length:
            kwargs["max_length"] = self.max_length
        if self.kind == "nli":
            return self.tokenizer([p for p, _ in pairs], [h for _, h in pairs], **kwargs)
        if self.kind == "minicheck":
            # Exact author batch_tokenize: document + eos_token + claim, one sequence.
            return self.tokenizer([p + self.tokenizer.eos_token + h for p, h in pairs], **kwargs)
        return self.tokenizer(
            [self.model.prompt.format(text1=p, text2=h) for p, h in pairs], **kwargs
        )

    def untruncated_lengths(self, pairs):
        if self.kind == "nli":
            encoded = self.tokenizer(
                [p for p, _ in pairs], [h for _, h in pairs], truncation=False, padding=False
            )
        elif self.kind == "minicheck":
            encoded = self.tokenizer(
                [p + self.tokenizer.eos_token + h for p, h in pairs],
                truncation=False,
                padding=False,
            )
        else:
            encoded = self.tokenizer(
                [self.model.prompt.format(text1=p, text2=h) for p, h in pairs],
                truncation=False,
                padding=False,
            )
        return [len(ids) for ids in encoded["input_ids"]]

    def predict(self, pairs: list[tuple[str, str]]) -> tuple[list[dict], dict]:
        import torch

        if not pairs:
            return [], {"pairs": 0, "seconds": 0}
        start = time.perf_counter()
        lengths = self.untruncated_lengths(pairs)
        order = sorted(range(len(pairs)), key=lambda i: lengths[i])
        result = [None] * len(pairs)
        durations = []
        for offset in range(0, len(order), self.batch_size):
            ids = order[offset : offset + self.batch_size]
            inputs = self.encode([pairs[i] for i in ids], tensors=True).to("cuda")
            torch.cuda.synchronize()
            batch_start = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(**inputs).logits
                probabilities = logits.float().softmax(-1).cpu().numpy()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - batch_start)
            for j, i in enumerate(ids):
                used = int(inputs["attention_mask"][j].sum().item())
                result[i] = {
                    "probabilities": dict(
                        zip(self.labels, map(float, probabilities[j]), strict=True)
                    ),
                    "support": float(probabilities[j, 1]),
                    "input_tokens": lengths[i],
                    "used_tokens": used,
                    "truncated_tokens": max(0, lengths[i] - used),
                }
            if offset % (self.batch_size * 100) == 0:
                print(
                    f"{self.kind}: {min(offset + self.batch_size, len(order))}/{len(order)} pairs",
                    flush=True,
                )
        seconds = time.perf_counter() - start
        return result, {
            "pairs": len(pairs),
            "seconds": seconds,
            "pairs_per_second": len(pairs) / seconds,
            "batch_forward_latency_p50_s": float(np.median(durations)),
            "batch_forward_latency_p95_s": float(np.percentile(durations, 95)),
            "batch_size": self.batch_size,
            "max_input_tokens": max(lengths),
            "truncated_pairs": sum(r["truncated_tokens"] > 0 for r in result),
            "truncated_tokens": sum(r["truncated_tokens"] for r in result),
            "parameter_count": self.parameter_count,
            "precision": "FP32 weights, BF16 autocast, FP32 softmax",
            **memory_report(),
        }


@lru_cache(maxsize=1)
def _punkt_tokenizer():
    import nltk
    from nltk.tokenize import PunktTokenizer

    nltk.data.path.insert(0, str(ROOT / "models/nltk_data"))
    return PunktTokenizer("english")


def punkt_spans(text: str) -> list[TextSpan]:
    tokenizer = _punkt_tokenizer()
    return [TextSpan(a, b, text[a:b]) for a, b in tokenizer.span_tokenize(text)]


def author_chunks(text: str, tokenizer) -> list[dict]:
    """MiniCheck's newline-aware sentence packing, with original pieces for provenance."""
    units = []
    position = 0
    for block in text.split("\n"):
        units.extend((s.text, position + s.start, position + s.end) for s in punkt_spans(block))
        units.append(("\n", position + len(block), position + len(block) + 1))
        position += len(block) + 1
    units = units[:-1]
    groups, current, count = [], [], 0
    for unit in units:
        size = len(
            tokenizer(unit[0], add_special_tokens=False, max_length=2048, truncation=True)[
                "input_ids"
            ]
        )
        if count + size > 400:
            groups.append(current)
            current, count = [unit], size
        else:
            current.append(unit)
            count += size
    if current:
        groups.append(current)
    result = []
    for group in groups:
        value = " ".join(u[0] for u in group).replace(" \n ", "\n").strip()
        if value:
            result.append(
                {
                    "text": value,
                    "pieces": [{"start": a, "end": b} for _, a, b in group],
                    "chunk_id": f"author:{len(result)}",
                }
            )
    return result


def score_records(verifier: Verifier, records: list[InferenceInput], mode: str, lexical=None):
    started = time.perf_counter()
    pairs, rows, pointers = [], [], []
    for record in records:
        dropped = []
        if mode == "old":
            diagnostics = lexical.score(record.lexical_record())
            spans = sentences(record.response)
            contexts = [
                [{**p, "pieces": [{"start": p["start"], "end": p["end"]}]} for p in s["retrieved"]]
                for s in diagnostics["sentences"]
            ]
        else:
            if mode == "whole":
                spans = [TextSpan(0, len(record.response), record.response)]
            elif mode == "clean":
                spans, dropped = clean_segments(record.response)
            elif mode in ("author", "joint"):
                spans = punkt_spans(record.response)
            else:
                raise ValueError(mode)
            if mode == "author":
                context = author_chunks(record.evidence, verifier.tokenizer)
            else:
                context = [
                    {
                        "text": record.evidence,
                        "chunk_id": "full",
                        "pieces": [{"start": 0, "end": len(record.evidence)}],
                    }
                ]
            contexts = [context] * len(spans)
        row = {
            "id": record.id,
            "source_id": record.source_id,
            "segments": [],
            "dropped": dropped,
            "answer_characters": len(record.response),
            "evidence_characters": len(record.evidence),
        }
        for span, context in zip(spans, contexts, strict=True):
            segment = {**asdict(span), "evidence": []}
            for chunk in context:
                item = {**chunk, "source_id": record.source_id}
                segment["evidence"].append(item)
                pairs.append((chunk["text"], span.text))
                pointers.append(item)
            row["segments"].append(segment)
        rows.append(row)
    outputs, stats = verifier.predict(pairs)
    for item, output in zip(pointers, outputs, strict=True):
        item.update(output)
    for row in rows:
        for segment in row["segments"]:
            segment["support"] = max((x["support"] for x in segment["evidence"]), default=0)
        row["score"] = max((1 - s["support"] for s in row["segments"]), default=0)
    seconds = time.perf_counter() - started
    stats.update(
        {
            "responses": len(rows),
            "pipeline_seconds": seconds,
            "responses_per_second": len(rows) / seconds,
            "amortized_seconds_per_response": seconds / len(rows),
            "mode": mode,
            "checkpoint": verifier.checkpoint["revision"],
        }
    )
    return rows, stats

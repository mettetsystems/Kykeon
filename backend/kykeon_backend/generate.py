from __future__ import annotations

import time
from typing import Any, Callable

THROUGHPUT_DEFINITION = (
    "Output tokens after the first, divided by the synchronized wall time from "
    "the first generated token until generation stops. Time to first token is excluded."
)
TTFT_DEFINITION = (
    "Synchronized time from the start of generation, after the prompt is on device, "
    "until the first new-token logits are produced. This includes prefill."
)
E2E_DEFINITION = (
    "Synchronized wall time from the start of generation until the last new token "
    "is produced, including prefill and decode."
)
SEED_NOTE = (
    "A fixed seed does not guarantee bitwise-identical tokens across GPUs, CPUs, "
    "library versions, or kernel implementations."
)


def unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def percentile(values: list[float], probability: float) -> dict[str, Any]:
    if len(values) < 2:
        return unavailable(
            "A percentile needs at least two measured samples. "
            f"{len(values)} sample(s) were recorded."
        )
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    value = ordered[low] * (1 - weight) + ordered[high] * weight
    small = len(ordered) < 20
    return {
        "value": value,
        "sample_count": len(ordered),
        "method": "linear interpolation between sorted measured samples",
        "label": "small-sample estimate" if small else "percentile",
        "stable": not small,
        "note": (
            "Fewer than 20 samples. This describes the observed runs and is not a stable percentile."
            if small
            else None
        ),
    }


def summarize_times(samples: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    if not values:
        return unavailable(f"No measured values were recorded for {key}.")
    return {
        "samples": values,
        "sample_count": len(values),
        "min": min(values),
        "median": percentile(values, 0.5).get("value", values[0]) if len(values) > 1 else values[0],
        "max": max(values),
        "p95": percentile(values, 0.95),
    }


def trim_stop(tokenizer, token_ids: list[int], stops: list[str], eos_id: int | None) -> dict[str, Any]:
    ids = list(token_ids)
    while ids and eos_id is not None and ids[-1] == eos_id:
        ids.pop()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    cut = None
    matched = None
    for stop in stops:
        index = text.find(stop)
        if index >= 0 and (cut is None or index < cut):
            cut = index
            matched = stop
    if cut is None:
        return {
            "text": text,
            "output_token_count": len(ids),
            "stop_matched": None,
            "tokens_consumed": len(token_ids),
        }
    trimmed = text[:cut]
    keep = 0
    for count in range(len(ids) + 1):
        piece = tokenizer.decode(ids[:count], skip_special_tokens=True)
        if trimmed.startswith(piece):
            keep = count
        else:
            break
    return {
        "text": trimmed,
        "output_token_count": keep,
        "stop_matched": matched,
        "tokens_consumed": len(token_ids),
        "token_count_note": (
            "The returned text is sliced at the stop string, including when that boundary "
            "falls inside a token. output_token_count counts complete tokenizer tokens whose "
            "decoded text lies entirely before the stop."
        ),
    }


def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _seed(device, seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def measure_generation(
    model,
    tokenizer,
    prompt: str,
    sampling: dict[str, Any],
    device,
    *,
    warmup_runs: int,
    measured_runs: int,
    cancelled: Callable[[], None],
) -> dict[str, Any]:
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    input_tokens = int(input_ids.shape[-1])
    prompt_length = input_tokens
    stops = list(sampling["stop_sequences"])
    eos_id = tokenizer.eos_token_id

    class Probe(LogitsProcessor):
        def __init__(self):
            self.t_first = None

        def __call__(self, input_ids, scores):
            if self.t_first is None:
                _sync(device)
                self.t_first = time.perf_counter()
            return scores

    class StopStrings(StoppingCriteria):
        def __init__(self):
            self.cut_text = None

        def __call__(self, input_ids, scores, **kwargs):
            cancelled()
            generated = input_ids[0, prompt_length:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            return any(stop in text for stop in stops)

    def one_run(record_memory: bool) -> dict[str, Any]:
        cancelled()
        _seed(device, int(sampling["seed"]))
        probe = Probe()
        criteria = StopStrings()
        if device.type == "cuda" and record_memory:
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        started = time.perf_counter()
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(sampling["max_new_tokens"]),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_id,
            "repetition_penalty": float(sampling["repetition_penalty"]),
            "logits_processor": LogitsProcessorList([probe]),
            "stopping_criteria": StoppingCriteriaList([criteria]),
            "use_cache": True,
        }
        if float(sampling["temperature"]) == 0:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = float(sampling["temperature"])
            generate_kwargs["top_k"] = int(sampling["top_k"])
            generate_kwargs["top_p"] = float(sampling["top_p"])
        with torch.inference_mode():
            output = model.generate(**generate_kwargs)
        _sync(device)
        finished = time.perf_counter()
        generated_ids = output[0, prompt_length:].tolist()
        trimmed = trim_stop(tokenizer, generated_ids, stops, eos_id)
        ttft = None if probe.t_first is None else probe.t_first - started
        e2e = finished - started
        decode = None if ttft is None else finished - probe.t_first
        output_tokens = int(trimmed["output_token_count"])
        if decode is None or decode <= 0 or output_tokens < 2:
            throughput = unavailable(
                "Output-token throughput needs at least two output tokens and a positive "
                "decode interval after the first token."
            )
        else:
            throughput = {
                "tokens_per_second": (output_tokens - 1) / decode,
                "definition": THROUGHPUT_DEFINITION,
            }
        memory: dict[str, Any]
        if device.type == "cuda" and record_memory:
            memory = {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        else:
            memory = {
                "peak_allocated_bytes": unavailable(
                    "CUDA allocated-memory statistics are unavailable for this device."
                ),
                "peak_reserved_bytes": unavailable(
                    "CUDA reserved-memory statistics are unavailable for this device."
                ),
            }
        return {
            "text": trimmed["text"],
            "output_token_count": output_tokens,
            "tokens_consumed": trimmed["tokens_consumed"],
            "stop_matched": trimmed["stop_matched"],
            "time_to_first_token_seconds": ttft,
            "end_to_end_seconds": e2e,
            "throughput": throughput,
            "memory": memory,
        }

    for _ in range(warmup_runs):
        one_run(record_memory=False)
    samples = []
    for index in range(measured_runs):
        cancelled()
        samples.append(one_run(record_memory=True))
    cuda_samples = [sample["memory"]["peak_allocated_bytes"] for sample in samples]
    reserved_samples = [sample["memory"]["peak_reserved_bytes"] for sample in samples]
    return {
        "input_token_count": input_tokens,
        "output_token_count": samples[-1]["output_token_count"] if samples else None,
        "text": samples[-1]["text"] if samples else "",
        "stop_matched": samples[-1]["stop_matched"] if samples else None,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "sample_count": len(samples),
        "methodology": {
            "time_to_first_token": TTFT_DEFINITION,
            "end_to_end": E2E_DEFINITION,
            "throughput": THROUGHPUT_DEFINITION,
            "warmup_policy": "Warm-up runs are executed and discarded before measured samples.",
            "cuda_sync": "CUDA synchronize is used before the start, at the first token, and at the end when the device is CUDA.",
            "seed_note": SEED_NOTE,
        },
        "time_to_first_token_seconds": summarize_times(samples, "time_to_first_token_seconds"),
        "end_to_end_seconds": summarize_times(samples, "end_to_end_seconds"),
        "throughput": _summarize_throughput(samples),
        "peak_cuda_allocated_bytes": _memory_summary(cuda_samples),
        "peak_cuda_reserved_bytes": _memory_summary(reserved_samples),
        "runs": samples,
        "notes": list(sampling.get("notes") or []),
    }


def _summarize_throughput(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for sample in samples:
        throughput = sample["throughput"]
        if isinstance(throughput, dict) and "tokens_per_second" in throughput:
            values.append(float(throughput["tokens_per_second"]))
    if not values:
        reason = "unavailable"
        for sample in samples:
            item = sample["throughput"]
            if isinstance(item, dict) and item.get("status") == "unavailable":
                reason = item["reason"]
                break
        return unavailable(reason)
    return {
        "definition": THROUGHPUT_DEFINITION,
        "sample_count": len(values),
        "samples": values,
        "median": percentile(values, 0.5).get("value", values[0]) if len(values) > 1 else values[0],
        "p95": percentile(values, 0.95),
    }


def _memory_summary(samples: list[Any]) -> dict[str, Any]:
    numbers = [int(value) for value in samples if isinstance(value, int)]
    if not numbers:
        reason = "CUDA memory statistics are unavailable for this device."
        for value in samples:
            if isinstance(value, dict) and value.get("reason"):
                reason = value["reason"]
                break
        return unavailable(reason)
    return {"max_bytes": max(numbers), "samples": numbers, "sample_count": len(numbers)}

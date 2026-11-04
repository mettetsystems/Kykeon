from transformers import AutoTokenizer

from kykeon_backend.generate import percentile, trim_stop, unavailable


def test_stop_string_can_span_token_boundaries(tiny_source):
    tokenizer = AutoTokenizer.from_pretrained(
        tiny_source, local_files_only=True, trust_remote_code=False
    )
    token_ids = tokenizer.encode("helloXY", add_special_tokens=False)
    assert len(token_ids) >= 7
    trimmed = trim_stop(tokenizer, token_ids, ["XY"], eos_id=tokenizer.eos_token_id)
    assert trimmed["text"] == "hello"
    assert "XY" not in trimmed["text"]
    assert trimmed["output_token_count"] == 5
    assert trimmed["stop_matched"] == "XY"


def test_percentiles_are_labeled_and_missing_metrics_are_not_zero():
    single = percentile([1.0], 0.95)
    assert single["status"] == "unavailable"
    assert single.get("value") is None
    small = percentile([1, 2, 3, 4, 5], 0.95)
    assert small["label"] == "small-sample estimate"
    assert small["stable"] is False
    assert small["sample_count"] == 5
    missing = unavailable("CUDA memory statistics are unavailable for CPU execution.")
    assert missing["status"] == "unavailable"
    assert "value" not in missing
    assert 0 not in missing.values()

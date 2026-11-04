"""Build a tiny dense Llama checkpoint for CPU integration tests.

The fixture is generated locally. It is not a downloaded model and it is not
a stand-in for a 3B checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_tiny_llama(dest: Path) -> Path:
    import torch
    from tokenizers import Regex, Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    dest.mkdir(parents=True, exist_ok=True)
    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3}
    for index, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!"):
        vocab[char] = 4 + index
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Split(Regex("."), behavior="isolated")
    backend.decoder = Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    config = LlamaConfig(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=1,
    )
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        preferred = vocab["a"]
        model.lm_head.weight[preferred].fill_(5.0)
    model.save_pretrained(dest, safe_serialization=True)
    tokenizer.save_pretrained(dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    print(build_tiny_llama(Path(args.dest).expanduser()))


if __name__ == "__main__":
    main()

# Language Model

A language model built from scratch in Python, following the
[Stanford CS336](https://cs336.stanford.edu/) assignments.

## Progress

- [x] BPE tokenizer: training (`language_model.bpe`), encoding and decoding (`language_model.tokenizer`)
- [x] Transformer building blocks: `Linear`, `Embedding`, `RMSNorm`, `SwiGLU` and RoPE (`language_model.model`)
- [x] Attention: softmax, scaled dot-product attention and causal multi-head self-attention with RoPE (`language_model.model`)

## Setup

Uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
```

The tests and fixtures are adapted from the CS336 assignment code. See [tests/LICENSE](tests/LICENSE).

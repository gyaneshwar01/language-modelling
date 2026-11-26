# Language Model

A language model built from scratch in Python, following the
[Stanford CS336](https://cs336.stanford.edu/) assignments.

## Progress

- [x] BPE tokenizer: training (`language_model.bpe`), encoding and decoding (`language_model.tokenizer`)
- [x] Transformer building blocks: `Linear`, `Embedding`, `RMSNorm`, `SwiGLU` and RoPE (`language_model.model`)
- [x] Attention: softmax, scaled dot-product attention and causal multi-head self-attention with RoPE (`language_model.model`)
- [x] Transformer: pre-norm `TransformerBlock` and the full `TransformerLM` (`language_model.model`)
- [x] Training: cross-entropy loss and gradient clipping (`language_model.utils`), the `AdamW` optimizer and a cosine learning-rate schedule with warmup (`language_model.optimizer`)

## Setup

Uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
```

The tests and fixtures are adapted from the CS336 assignment code. See [tests/LICENSE](tests/LICENSE).

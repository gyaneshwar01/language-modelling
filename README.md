# Language Model

A language model built from scratch in Python, following the
[Stanford CS336](https://cs336.stanford.edu/) assignments.

## Progress

- [x] BPE tokenizer: training (`language_model.bpe`), encoding and decoding (`language_model.tokenizer`)
- [x] Transformer building blocks: `Linear`, `Embedding`, `RMSNorm`, `SwiGLU` and RoPE (`language_model.model`)
- [x] Attention: softmax, scaled dot-product attention and causal multi-head self-attention with RoPE (`language_model.model`)
- [x] Transformer: pre-norm `TransformerBlock` and the full `TransformerLM` (`language_model.model`)
- [x] Training: cross-entropy loss and gradient clipping (`language_model.utils`), the `AdamW` optimizer and a cosine learning-rate schedule with warmup (`language_model.optimizer`)
- [x] Training loop with token batching, validation loss, checkpoint save and resume, and optional W&B logging.
- [ ] Autoregressive decoding for text generation

## Setup

Uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
```

The tests and fixtures are adapted from the CS336 assignment code. See [tests/LICENSE](tests/LICENSE).

## Usage

### Prepare training and validation tokens

The trainer reads one 1-D integer array per split. If you already have tokenized
`.npy` files, use them directly in the training command below. Each split needs
at least `context-length + 1` tokens, and every ID must be smaller than
`--vocab-size` and at least zero.

For a small TinyStories run, place the two text files and your trained 10k BPE
tokenizer at these paths relative to the repository root:

```text
data/TinyStoriesV2-GPT4-train.txt
data/TinyStoriesV2-GPT4-valid.txt
artifacts/tinystories_bpe_10k/vocab.pkl
artifacts/tinystories_bpe_10k/merges.pkl
```

```sh
uv run python scripts/prepare_tinystories.py
```

This encodes the first 1,000 training stories and 100 validation stories into
`data/tiny-train.npy` and `data/tiny-valid.npy`, respectively.

### Train the model

```sh
uv run python -m language_model.train \
  --train-data data/tiny-train.npy \
  --val-data data/tiny-valid.npy \
  --vocab-size 10000 \
  --total-steps 1000 \
  --checkpoint-path checkpoints/latest.pt
```

Use your tokenizer's vocabulary size if it differs from 10,000. The trainer
selects CUDA when available and otherwise uses CPU. With the defaults, it logs
training loss every 10 steps, evaluates validation loss every 100 steps, and
saves the latest checkpoint every 100 steps and at the end of training.

### Resume from a checkpoint

To resume an interrupted run, repeat the training command with
`--resume-from checkpoints/latest.pt`. Use the same tokenizer and model
settings. The trainer restores the model weights, AdamW state, and completed
step count. `--total-steps` is the target for the whole run. For example, to
continue a completed 1,000-step run for another 500 steps, set it to 1500.

Add `--wandb-project NAME` to log metrics to Weights & Biases. Run
`uv run python -m language_model.train --help` for model, optimizer, and
logging options.

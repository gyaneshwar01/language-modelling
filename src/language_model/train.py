import argparse
from contextlib import nullcontext
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from language_model.checkpointing import load_checkpoint, save_checkpoint
from language_model.data import get_batch
from language_model.model import TransformerLM
from language_model.optimizer import AdamW, cosine_scheduler
from language_model.utils import clip_gradient, cross_entropy


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    device: str,
    batch_size: int,
    seq_len: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_grad_norm: float,
    log_every: int,
    eval_every: int,
    eval_batches: int,
    checkpoint_every: int,
    checkpoint_path: Path,
    resume_from: Path | None = None,
    max_minutes: float | None = None,
    wandb_run: wandb.Run | None = None,
) -> None:
    steps = 0
    if resume_from is not None:
        steps = load_checkpoint(resume_from, model, optimizer)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.train()

    started = time.monotonic()
    for step in range(steps, total_steps):
        lr = cosine_scheduler(step, max_lr, min_lr, warmup_steps, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_tokens, batch_size, seq_len, device)
        optimizer.zero_grad()

        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        clip_gradient(model.parameters(), max_grad_norm)
        optimizer.step()

        completed = step + 1
        elapsed = time.monotonic() - started
        final_step = completed == total_steps or (max_minutes is not None and elapsed >= 60 * max_minutes)
        metrics = {}
        if completed % log_every == 0 or final_step:
            train_loss = loss.item()
            metrics.update({"train/loss": train_loss, "train/lr": lr})
            print(f"step={completed} train_loss={train_loss:.4f} lr={lr:.3g}")

        if completed % eval_every == 0 or final_step:
            model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                for _ in range(eval_batches):
                    x_val, y_val = get_batch(val_tokens, batch_size, seq_len, device)
                    val_loss_sum += cross_entropy(model(x_val), y_val).item()

            val_loss = val_loss_sum / eval_batches
            metrics["val/loss"] = val_loss
            print(f"step={completed} val_loss={val_loss:.4f}")
            model.train()

        if wandb_run is not None and metrics:
            wandb_run.log(metrics, step=completed)

        if completed % checkpoint_every == 0 or final_step:
            save_checkpoint(model, optimizer, completed, checkpoint_path)
        if final_step:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    data = parser.add_argument_group("data and checkpoints")
    data.add_argument("--train-data", type=Path, required=True)
    data.add_argument("--val-data", type=Path, required=True)
    data.add_argument(
        "--raw-dtype",
        choices=["uint16", "uint32", "int32", "int64"],
        help="On-disk dtype for raw token files. Omit for files created by np.save",
    )
    data.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/latest.pt"))
    data.add_argument("--resume-from", type=Path)

    model_args = parser.add_argument_group("model")
    model_args.add_argument("--vocab-size", type=int, required=True)
    model_args.add_argument("--context-length", type=int, default=128)
    model_args.add_argument("--d-model", type=int, default=256)
    model_args.add_argument("--num-layers", type=int, default=4)
    model_args.add_argument("--num-heads", type=int, default=8)
    model_args.add_argument("--d-ff", type=int, default=1024)
    model_args.add_argument("--theta", type=float, default=10_000.0)

    run = parser.add_argument_group("training")
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--total-steps", type=int, default=1000)
    run.add_argument("--max-lr", type=float, default=3e-4)
    run.add_argument("--min-lr", type=float, default=3e-5)
    run.add_argument("--warmup-steps", type=int, default=50)
    run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument("--beta1", type=float, default=0.9)
    run.add_argument("--beta2", type=float, default=0.999)
    run.add_argument("--eps", type=float, default=1e-8)
    run.add_argument("--max-grad-norm", type=float, default=1.0)
    run.add_argument("--log-every", type=int, default=10)
    run.add_argument("--eval-every", type=int, default=100)
    run.add_argument("--eval-batches", type=int, default=10)
    run.add_argument("--checkpoint-every", type=int, default=100)

    run.add_argument("--max-minutes", type=float, help="Training time limit; finish with evaluation and a checkpoint.")
    run.add_argument("--wandb-project", help="Enable W&B tracking for this project.")

    args = parser.parse_args()
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error("--max-minutes must be positive")

    def open_tokens(path: Path) -> np.ndarray:
        if args.raw_dtype is None:
            tokens = np.load(path, mmap_mode="r", allow_pickle=False)
        else:
            tokens = np.memmap(path, mode="r", dtype=args.raw_dtype)

        if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
            parser.error(f"{path} must contain a 1-D array of integer token IDs")
        if len(tokens) <= args.context_length:
            parser.error(f"{path} needs at least {args.context_length + 1} tokens")
        return tokens

    train_tokens = open_tokens(args.train_data)
    val_tokens = open_tokens(args.val_data)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=torch.device(args.device),
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    tracking = nullcontext()
    if args.wandb_project:
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        tracking = wandb.init(project=args.wandb_project, config=config)

    with tracking as wandb_run:
        train(
            model,
            optimizer,
            train_tokens,
            val_tokens,
            device=args.device,
            batch_size=args.batch_size,
            seq_len=args.context_length,
            total_steps=args.total_steps,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            checkpoint_every=args.checkpoint_every,
            checkpoint_path=args.checkpoint_path,
            resume_from=args.resume_from,
            max_minutes=args.max_minutes,
            wandb_run=wandb_run,
        )


if __name__ == "__main__":
    main()

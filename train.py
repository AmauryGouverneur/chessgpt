"""
train.py — Training loop for ChessGPT.

Usage:
    python train.py --debug
    python train.py --mac
    python train.py
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data.dataset import get_dataloader
from model import ChessGPT, ChessGPTConfig


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr_scheduler(
    optimizer:    AdamW,
    warmup_steps: int,
    max_steps:    int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:       ChessGPT,
    val_loader:  torch.utils.data.DataLoader,
    device:      torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        targets   = batch["targets"].to(device)
        _, loss   = model(input_ids, targets)
        total_loss += loss.item()
        n_batches  += 1
        if n_batches >= max_batches:
            break
    model.train()
    return total_loss / max(1, n_batches)


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def make_run_dir(base: str = "runs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path(base) / f"run_{timestamp}"
    (run_dir / "checkpoints").mkdir(parents=True)
    return run_dir


def save_config(run_dir: Path, config: ChessGPTConfig, args: argparse.Namespace) -> None:
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "vocab_size":  config.vocab_size,
            "block_size":  config.block_size,
            "n_embd":      config.n_embd,
            "n_layer":     config.n_layer,
            "n_head":      config.n_head,
            "n_kv_head":   config.n_kv_head,
            "dropout":     config.dropout,
        }, f, indent=2)

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)


def log_metric(run_dir: Path, record: dict) -> None:
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    # ── config ───────────────────────────────────────────────────────────
    if args.debug:
        config        = ChessGPTConfig.debug()
        max_steps     = 10
        warmup_steps  = 2
        eval_interval = 5
        save_interval = 10
        batch_size    = 4
        learning_rate = 3e-4
        data_path     = args.data or "data/processed/dataset_mini_moves.parquet"

    elif args.mac:
        config        = ChessGPTConfig()
        max_steps     = 500
        warmup_steps  = 50
        eval_interval = 5
        save_interval = 500
        batch_size    = 16
        learning_rate = 3e-4
        data_path     = args.data or "data/processed/dataset_moves.parquet"

    else:
        config        = ChessGPTConfig()
        max_steps     = args.max_steps
        warmup_steps  = args.warmup_steps
        eval_interval = args.eval_interval
        save_interval = args.save_interval
        batch_size    = args.batch_size
        learning_rate = args.lr
        data_path     = args.data or "data/processed/dataset_moves.parquet"

    # ── run directory ─────────────────────────────────────────────────────
    run_dir = make_run_dir()
    print(f"Run directory: {run_dir}")
    save_config(run_dir, config, args)
    print(f"Config: {config}")

    # ── wandb ─────────────────────────────────────────────────────────────
    wandb.init(
        project = "chessgpt",
        name    = run_dir.name,
        dir     = str(run_dir),
        config  = {
            "vocab_size":    config.vocab_size,
            "block_size":    config.block_size,
            "n_embd":        config.n_embd,
            "n_layer":       config.n_layer,
            "n_head":        config.n_head,
            "n_kv_head":     config.n_kv_head,
            "dropout":       config.dropout,
            "max_steps":     max_steps,
            "warmup_steps":  warmup_steps,
            "batch_size":    batch_size,
            "learning_rate": learning_rate,
            "data_path":     data_path,
        },
        mode = "disabled" if args.debug else "online",
    )

    # ── data ─────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloader(
        path        = data_path,
        block_size  = config.block_size,
        batch_size  = batch_size,
        train_split = 0.9,
        num_workers = 0,
        debug       = args.debug,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── model ─────────────────────────────────────────────────────────────
    model = ChessGPT(config).to(device)

    # ── optimizer + scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = learning_rate,
        weight_decay = 0.1,
        betas        = (0.9, 0.95),
    )
    scheduler = get_lr_scheduler(optimizer, warmup_steps, max_steps)

    # ── training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    val_loss      = float("inf")
    step          = 0
    train_iter    = iter(train_loader)

    pbar = tqdm(total=max_steps, desc="Training")

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        targets   = batch["targets"].to(device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step   += 1
        cur_lr  = scheduler.get_last_lr()[0]
        pbar.update(1)
        pbar.set_postfix({
            "train": f"{loss.item():.4f}",
            "val":   f"{val_loss:.4f}",
            "lr":    f"{cur_lr:.2e}",
        })

        # log train loss every step to wandb
        wandb.log({"train_loss": loss.item(), "lr": cur_lr}, step=step)

        # ── eval ──────────────────────────────────────────────────────────
        if step % eval_interval == 0 or step == max_steps:
            val_loss = evaluate(model, val_loader, device)

            log_metric(run_dir, {
                "step":       step,
                "train_loss": loss.item(),
                "val_loss":   val_loss,
                "lr":         cur_lr,
            })

            wandb.log({"val_loss": val_loss}, step=step)

            pbar.set_postfix({
                "train": f"{loss.item():.4f}",
                "val":   f"{val_loss:.4f}",
                "lr":    f"{cur_lr:.2e}",
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "step":      step,
                    "config":    config,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss":  val_loss,
                }, run_dir / "checkpoints" / "best.pt")
                tqdm.write(f"  -> best checkpoint (val {val_loss:.4f})")

        # ── periodic save ─────────────────────────────────────────────────
        if step % save_interval == 0:
            torch.save({
                "step":      step,
                "config":    config,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss":  val_loss,
            }, run_dir / "checkpoints" / f"step_{step:06d}.pt")

    pbar.close()
    wandb.finish()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Run saved to: {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChessGPT.")
    parser.add_argument("--data",          type=str,   default=None)
    parser.add_argument("--debug",         action="store_true")
    parser.add_argument("--mac",           action="store_true")
    parser.add_argument("--max-steps",     type=int,   default=10_000)
    parser.add_argument("--warmup-steps",  type=int,   default=200)
    parser.add_argument("--eval-interval", type=int,   default=200)
    parser.add_argument("--save-interval", type=int,   default=1_000)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=3e-4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
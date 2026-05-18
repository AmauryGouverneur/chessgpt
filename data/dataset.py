"""
dataset.py — PyTorch Dataset and DataLoader for ChessGPT.

Reads tokenized games from a Parquet file produced by prepare.py and
serves fixed-length (block_size=256) tensors for next-token prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PAD_ID, BLOCK_SIZE, HEADER_LEN


class ChessDataset(Dataset):
    """
    PyTorch Dataset for tokenized chess games.

    Each item is a dict of two tensors of shape (block_size,):
        input_ids : token ids, left-truncated / right-padded to block_size
        targets   : input_ids shifted left by 1 (next-token prediction)
    """

    def __init__(self, path: str | Path, block_size: int = BLOCK_SIZE) -> None:
        self.block_size = block_size
        self.input_ids  = pq.read_table(str(path))["input_ids"].to_pylist()

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids     = self._fit_to_block(list(self.input_ids[idx]))
        targets = ids[1:] + [PAD_ID]
        return {
            "input_ids": torch.tensor(ids,     dtype=torch.long),
            "targets":   torch.tensor(targets, dtype=torch.long),
        }

    def _fit_to_block(self, ids: list[int]) -> list[int]:
        n = len(ids)
        if n > self.block_size:
            drop = n - self.block_size
            ids  = ids[:HEADER_LEN] + ids[HEADER_LEN + drop:]
        elif n < self.block_size:
            ids  = ids + [PAD_ID] * (self.block_size - n)
        return ids


def get_dataloader(
    path:        str | Path,
    block_size:  int   = BLOCK_SIZE,
    batch_size:  int   = 64,
    train_split: float = 0.9,
    num_workers: int   = 0,
    debug:       bool  = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a parquet file.

    Returns:
        (train_loader, val_loader)
    """
    dataset = ChessDataset(path, block_size=block_size)

    if debug:
        dataset = torch.utils.data.Subset(dataset, range(min(20, len(dataset))))

    n_train = int(len(dataset) * train_split)
    n_val   = len(dataset) - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 \
        else "data/processed/dataset_mini_moves.parquet"

    print(f"Loading: {path}")
    train_loader, val_loader = get_dataloader(path, batch_size=4, debug=True)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    batch = next(iter(train_loader))
    print(f"\nBatch keys    : {list(batch.keys())}")
    print(f"input_ids     : {batch['input_ids'].shape}  dtype={batch['input_ids'].dtype}")
    print(f"targets       : {batch['targets'].shape}    dtype={batch['targets'].dtype}")

    print(f"\ninput_ids[:12] : {batch['input_ids'][0, :12].tolist()}")
    print(f"targets[:12]   : {batch['targets'][0, :12].tolist()}")

    assert torch.all(batch["targets"][0, :-1] == batch["input_ids"][0, 1:]), \
        "targets are not input shifted by 1!"
    print("\nShift check passed.")

    assert batch["input_ids"].shape[1] == BLOCK_SIZE, "block_size mismatch!"
    print("Block size check passed.")
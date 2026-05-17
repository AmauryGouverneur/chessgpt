"""
dataset.py — PyTorch Dataset and DataLoader for ChessGPT.

Reads tokenized games from a Parquet file produced by prepare.py and
serves fixed-length (block_size=256) tensors for next-token prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# must match prepare.py
PAD_ID     = 0
BLOCK_SIZE = 256
HEADER_LEN = 4  # [GAME_START, WHITE_ELO, BLACK_ELO, RESULT]


class ChessDataset(Dataset):
    """
    PyTorch Dataset for tokenized chess games.

    Each item is a dict of four tensors of shape (block_size,):
        input_ids   : token ids, left-truncated / right-padded to block_size
        targets     : input_ids shifted left by 1 (next-token prediction)
        scalars     : parallel float array (Elo, result, eval values)
        scalar_mask : True at positions carrying a meaningful scalar
    """

    def __init__(
        self,
        path: str | Path,
        block_size: int = BLOCK_SIZE,
    ) -> None:
        self.block_size = block_size

        table = pq.read_table(str(path))
        self.input_ids   = table["input_ids"].to_pylist()
        self.scalars     = table["scalars"].to_pylist()
        self.scalar_mask = table["scalar_mask"].to_pylist()

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids  = list(self.input_ids[idx])
        scl  = list(self.scalars[idx])
        msk  = list(self.scalar_mask[idx])

        ids, scl, msk = self._fit_to_block(ids, scl, msk)

        # targets: shift left by 1, last position predicts PAD
        targets = ids[1:] + [PAD_ID]

        return {
            "input_ids":   torch.tensor(ids,     dtype=torch.long),
            "targets":     torch.tensor(targets, dtype=torch.long),
            "scalars":     torch.tensor(scl,     dtype=torch.float),
            "scalar_mask": torch.tensor(msk,     dtype=torch.bool),
        }

    def _fit_to_block(
        self,
        ids:  list[int],
        scl:  list[float],
        msk:  list[bool],
    ) -> tuple[list[int], list[float], list[bool]]:
        """
        Truncate or pad all three sequences to exactly block_size.

        Truncation: left-truncate (drop oldest tokens), always keeping
                    the 4-token header at the start.
        Padding:    right-pad with PAD_ID / 0.0 / False.
        """
        n = len(ids)

        if n > self.block_size:
            # how many tokens to drop
            drop = n - self.block_size
            # keep header + everything after the dropped tokens
            header_ids = ids[:HEADER_LEN]
            header_scl = scl[:HEADER_LEN]
            header_msk = msk[:HEADER_LEN]
            ids = header_ids + ids[HEADER_LEN + drop:]
            scl = header_scl + scl[HEADER_LEN + drop:]
            msk = header_msk + msk[HEADER_LEN + drop:]

        elif n < self.block_size:
            pad = self.block_size - n
            ids = ids  + [PAD_ID] * pad
            scl = scl  + [0.0]    * pad
            msk = msk  + [False]  * pad

        return ids, scl, msk


def get_dataloader(
    path: str | Path,
    block_size:  int   = BLOCK_SIZE,
    batch_size:  int   = 64,
    train_split: float = 0.9,
    num_workers: int   = 0,
    debug:       bool  = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a parquet file.

    Args:
        path:        path to the parquet file produced by prepare.py
        block_size:  sequence length fed to the model
        batch_size:  number of games per batch
        train_split: fraction of games used for training
        num_workers: DataLoader worker processes (0 = main process)
        debug:       if True, use only the first 20 games

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

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    batch = next(iter(train_loader))
    print(f"\nBatch keys:    {list(batch.keys())}")
    print(f"input_ids:     {batch['input_ids'].shape}  dtype={batch['input_ids'].dtype}")
    print(f"targets:       {batch['targets'].shape}    dtype={batch['targets'].dtype}")
    print(f"scalars:       {batch['scalars'].shape}    dtype={batch['scalars'].dtype}")
    print(f"scalar_mask:   {batch['scalar_mask'].shape} dtype={batch['scalar_mask'].dtype}")

    print(f"\nFirst sequence input_ids[:12]:  {batch['input_ids'][0,:12].tolist()}")
    print(f"First sequence targets[:12]:    {batch['targets'][0,:12].tolist()}")
    print(f"First sequence scalars[:6]:     {batch['scalars'][0,:6].tolist()}")
    print(f"First sequence scalar_mask[:6]: {batch['scalar_mask'][0,:6].tolist()}")

    # verify targets are input shifted by 1
    assert torch.all(batch["targets"][0, :-1] == batch["input_ids"][0, 1:]), \
        "targets are not input shifted by 1!"
    print("\nShift check passed.")

    # verify all sequences are exactly block_size
    assert batch["input_ids"].shape[1] == 256, "block_size mismatch!"
    print("Block size check passed.")
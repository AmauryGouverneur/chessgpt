#!/usr/bin/env python3
"""
prepare.py — filter and tokenize Lichess PGN games in a single pass.

Usage:
    python data/prepare.py --input data/raw/lichess_mini.pgn.zst \
                           --output data/processed/dataset_mini_moves.parquet
    python data/prepare.py --input data/raw/lichess_mini.pgn.zst \
                           --output data/processed/dataset_mini_moves.parquet \
                           --debug
"""

import argparse
import re
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator

import zstandard as zstd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PIECES, TOKEN2ID, ID2TOKEN, VOCAB_SIZE,
    WIN_ID, DRAW_ID, LOSS_ID, GAME_END_ID, PROMOTION_ID,
)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def is_classical(tc: str) -> bool:
    if not tc or tc == "-":
        return False
    try:
        return int(tc.split("+")[0]) >= 600
    except ValueError:
        return False


def parse_elo(val: str) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def result_to_token(val: str) -> int | None:
    return {"1-0": WIN_ID, "0-1": LOSS_ID, "1/2-1/2": DRAW_ID}.get(val)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize_move(san: str) -> list[int]:
    """
    Convert a single SAN move string to a list of token ids.

    Examples:
        e4      -> [e][4]
        Nf3     -> [N][f][3]
        Nxf3    -> [N][x][f][3]
        O-O     -> [O-O]
        O-O-O   -> [O-O-O]
        e8=Q    -> [e][8][PROMOTION]
        Nf3+    -> [N][f][3][+]
        Qh7#    -> [Q][h][7][#]
    """
    san = san.strip().rstrip("?!")
    ids: list[int] = []

    if not san:
        return ids

    # castling
    if san.startswith("O-O"):
        castle = "O-O-O" if san.startswith("O-O-O") else "O-O"
        ids.append(TOKEN2ID[castle])
        if san[len(castle):] in ("+", "#"):
            ids.append(TOKEN2ID[san[len(castle):]])
        return ids

    # check / mate suffix
    suffix = ""
    if san.endswith("+") or san.endswith("#"):
        suffix, san = san[-1], san[:-1]

    # promotion — collapse all piece choices to [PROMOTION]
    promotion = False
    if "=" in san:
        san       = san[:san.index("=")]
        promotion = True

    # optional piece
    i = 0
    if i < len(san) and san[i] in PIECES:
        ids.append(TOKEN2ID[san[i]]); i += 1

    # optional capture
    if i < len(san) and san[i] == "x":
        ids.append(TOKEN2ID["x"]); i += 1

    # file + rank (destination square, last 2 chars of remaining)
    remaining = san[i:]
    if len(remaining) >= 2:
        file_char = remaining[-2]
        rank_char = remaining[-1]
        if file_char in TOKEN2ID and rank_char in TOKEN2ID:
            ids.append(TOKEN2ID[file_char])
            ids.append(TOKEN2ID[rank_char])

    if promotion:
        ids.append(PROMOTION_ID)
    if suffix:
        ids.append(TOKEN2ID[suffix])

    return ids


def tokenize_game(moves_san: str, result_token: int) -> list[int]:
    """
    Tokenize a full game into a flat list of token ids.
    Format: [WIN|DRAW|LOSS] <move tokens> [GAME_END]
    """
    moves_san = re.sub(r'\{[^}]*\}', '', moves_san)
    input_ids = [result_token]

    for token in moves_san.split():
        if re.match(r"^\d+\.+", token):
            continue
        if token in {"1-0", "0-1", "1/2-1/2", "*"}:
            continue
        ids = tokenize_move(token)
        if ids:
            input_ids.extend(ids)

    input_ids.append(GAME_END_ID)
    return input_ids


# ---------------------------------------------------------------------------
# PGN streaming
# ---------------------------------------------------------------------------

@dataclass
class RawGame:
    headers:   dict = field(default_factory=dict)
    moves_str: str  = ""


def stream_games(path: str) -> Iterator[RawGame]:
    file_size = os.path.getsize(path)

    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            buffer   = ""
            game     = RawGame()
            in_moves = False

            with tqdm(
                total=file_size, unit="B", unit_scale=True,
                unit_divisor=1024, desc="Reading",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                           "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:

                while True:
                    chunk = reader.read(1 << 20)
                    if not chunk:
                        break
                    pbar.update(fh.tell() - pbar.n)
                    buffer += chunk.decode("utf-8", errors="replace")
                    lines   = buffer.split("\n")
                    buffer  = lines[-1]

                    for line in lines[:-1]:
                        line = line.strip()
                        if line.startswith("["):
                            if in_moves and game.moves_str.strip():
                                yield game
                            m = re.match(r'\[(\w+)\s+"(.*)"\]', line)
                            if m and m.group(1) == "Event":
                                game     = RawGame()
                                in_moves = False
                            if m:
                                game.headers[m.group(1)] = m.group(2)
                        elif line:
                            in_moves       = True
                            game.moves_str += " " + line

                if game.moves_str.strip():
                    yield game


def count_plies(moves_str: str) -> int:
    tokens = [
        t for t in moves_str.split()
        if not re.match(r"^\d+\.+", t)
        and t not in {"1-0", "0-1", "1/2-1/2", "*"}
    ]
    return len(tokens)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--min-elo",   type=int, default=1200)
    parser.add_argument("--min-moves", type=int, default=10)
    parser.add_argument("--debug",     action="store_true")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("game_id",   pa.int32()),
        ("white_elo", pa.int16()),
        ("black_elo", pa.int16()),
        ("result",    pa.int8()),
        ("n_tokens",  pa.int16()),
        ("input_ids", pa.list_(pa.int8())),
    ])

    min_plies = args.min_moves * 2
    stats     = {"total": 0, "abandoned": 0, "not_classical": 0,
                 "low_elo": 0, "too_short": 0, "passed": 0}
    game_id   = 0
    rows: list[dict] = []
    writer    = pq.ParquetWriter(args.output, schema)

    def flush(rows: list[dict]) -> None:
        if not rows:
            return
        writer.write_table(pa.table({
            "game_id":   pa.array([r["game_id"]   for r in rows], type=pa.int32()),
            "white_elo": pa.array([r["white_elo"] for r in rows], type=pa.int16()),
            "black_elo": pa.array([r["black_elo"] for r in rows], type=pa.int16()),
            "result":    pa.array([r["result"]    for r in rows], type=pa.int8()),
            "n_tokens":  pa.array([r["n_tokens"]  for r in rows], type=pa.int16()),
            "input_ids": pa.array([r["input_ids"] for r in rows],
                                  type=pa.list_(pa.int8())),
        }))

    for raw in stream_games(args.input):
        stats["total"] += 1

        tc           = raw.headers.get("TimeControl", "")
        welo         = parse_elo(raw.headers.get("WhiteElo", "0"))
        belo         = parse_elo(raw.headers.get("BlackElo", "0"))
        result_token = result_to_token(raw.headers.get("Result", ""))
        n_plies      = count_plies(raw.moves_str)

        if n_plies < 2:                                stats["abandoned"]     += 1; continue
        if not is_classical(tc):                       stats["not_classical"] += 1; continue
        if welo < args.min_elo or belo < args.min_elo: stats["low_elo"]       += 1; continue
        if n_plies < min_plies:                        stats["too_short"]      += 1; continue
        if result_token is None:                                                      continue

        stats["passed"] += 1
        game_id         += 1
        input_ids        = tokenize_game(raw.moves_str, result_token)

        if args.debug:
            tokens = [ID2TOKEN[i] for i in input_ids]
            print(f"\n--- Game {game_id} | W:{welo} B:{belo} "
                  f"R:{ID2TOKEN[result_token]} ---")
            print(f"  moves    : {raw.moves_str.strip()[:80]}...")
            print(f"  n_tokens : {len(input_ids)}")
            print(f"  tokens   : {' '.join(tokens[:16])} ...")

        rows.append({
            "game_id":   game_id,
            "white_elo": welo,
            "black_elo": belo,
            "result":    result_token,
            "n_tokens":  len(input_ids),
            "input_ids": input_ids,
        })

        if len(rows) >= 1000:
            flush(rows); rows = []
        if args.debug and stats["passed"] >= 50:
            break

    flush(rows)
    writer.close()

    print(f"\n{'='*50}")
    print(f"Total games seen:        {stats['total']:>10,}")
    print(f"  Abandoned:             {stats['abandoned']:>10,}")
    print(f"  Not classical:         {stats['not_classical']:>10,}")
    print(f"  Elo < {args.min_elo}:          {stats['low_elo']:>10,}")
    print(f"  Too short:             {stats['too_short']:>10,}")
    print(f"  Written to parquet:    {stats['passed']:>10,}")
    print(f"\nOutput: {args.output}")
    print(f"Vocab size: {VOCAB_SIZE}")


if __name__ == "__main__":
    main()
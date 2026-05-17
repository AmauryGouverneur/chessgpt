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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator

import zstandard as zstd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PIECES   = ["N", "B", "R", "Q", "K"]
FILES    = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS    = ["1", "2", "3", "4", "5", "6", "7", "8"]
SPECIALS = ["O-O-O", "O-O", "x", "+", "#", "=Q", "=R", "=B", "=N"]
CONTROL  = ["[PAD]", "[GAME_START]", "[GAME_END]", "[WHITE_ELO]", "[BLACK_ELO]", "[RESULT]"]

# build vocab — order matters, keep it stable
_vocab_list = CONTROL + PIECES + FILES + RANKS + SPECIALS
# pad to 64
while len(_vocab_list) < 64:
    _vocab_list.append(f"[UNUSED_{len(_vocab_list)}]")

TOKEN2ID = {tok: i for i, tok in enumerate(_vocab_list)}
ID2TOKEN = {i: tok for tok, i in TOKEN2ID.items()}
VOCAB_SIZE = len(TOKEN2ID)  # 64

PAD_ID        = TOKEN2ID["[PAD]"]
GAME_START_ID = TOKEN2ID["[GAME_START]"]
GAME_END_ID   = TOKEN2ID["[GAME_END]"]
WHITE_ELO_ID  = TOKEN2ID["[WHITE_ELO]"]
BLACK_ELO_ID  = TOKEN2ID["[BLACK_ELO]"]
RESULT_ID     = TOKEN2ID["[RESULT]"]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def is_classical(tc: str) -> bool:
    """Classical: base time >= 600 seconds."""
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


def parse_result(val: str) -> float | None:
    return {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(val)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# SAN move regex: captures piece, capture, file, rank, promotion, check/mate
_MOVE_RE = re.compile(
    r"^([NBRQK])?([a-h])?([1-8])?(x)?([a-h][1-8]|[a-h])(=[NBRQ])?([+#])?$"
    r"|^(O-O-O|O-O)([+#])?$"
)


def tokenize_move(san: str) -> list[int]:
    """
    Convert a single SAN move string to a list of token ids.

    Examples:
        e4      -> [e, 4]
        Nf3     -> [N, f, 3]
        Nxf3    -> [N, x, f, 3]
        O-O     -> [O-O]
        O-O-O   -> [O-O-O]
        e8=Q    -> [e, 8, =Q]
        Nf3+    -> [N, f, 3, +]
        Qxd5#   -> [Q, x, d, 5, #]
    """
    # strip move number annotations if any leaked through
    san = san.strip()

    ids: list[int] = []

    # castling
    if san.startswith("O-O"):
        castle = "O-O-O" if san.startswith("O-O-O") else "O-O"
        ids.append(TOKEN2ID[castle])
        suffix = san[len(castle):]
        if suffix in ("+", "#"):
            ids.append(TOKEN2ID[suffix])
        return ids

    # strip check/mate suffix
    suffix = ""
    if san.endswith("+") or san.endswith("#"):
        suffix = san[-1]
        san = san[:-1]

    # promotion
    promotion = ""
    if "=" in san:
        idx = san.index("=")
        promotion = san[idx:]   # e.g. "=Q"
        san = san[:idx]

    i = 0
    # piece
    if san[i] in TOKEN2ID and san[i] in PIECES:
        ids.append(TOKEN2ID[san[i]])
        i += 1

    # capture
    if i < len(san) and san[i] == "x":
        ids.append(TOKEN2ID["x"])
        i += 1

    # destination square: file + rank
    # handle disambiguation: e.g. Nbd2 — skip the disambiguating file/rank
    remaining = san[i:]
    if len(remaining) >= 2:
        file_char = remaining[-2]
        rank_char = remaining[-1]
        if file_char in TOKEN2ID and rank_char in TOKEN2ID:
            ids.append(TOKEN2ID[file_char])
            ids.append(TOKEN2ID[rank_char])

    if promotion and promotion in TOKEN2ID:
        ids.append(TOKEN2ID[promotion])

    if suffix and suffix in TOKEN2ID:
        ids.append(TOKEN2ID[suffix])

    return ids


def tokenize_game(
    moves_san: str,
    white_elo: int,
    black_elo: int,
    result: float,
) -> dict:
    """
    Tokenize a full game into input_ids + parallel scalar arrays.

    Returns:
        input_ids   : list[int]
        scalars     : list[float]   — 0.0 where scalar_mask is False
        scalar_mask : list[bool]    — True at header scalar positions
    """
    # strip inline annotations like { [%eval 0.15] } or { [%clk 0:05:00] }
    moves_san = re.sub(r'\{[^}]*\}', '', moves_san)

    input_ids:   list[int]   = []
    scalars:     list[float] = []
    scalar_mask: list[bool]  = []

    def push(token_id: int, scalar: float = 0.0, is_scalar: bool = False) -> None:
        input_ids.append(token_id)
        scalars.append(scalar)
        scalar_mask.append(is_scalar)

    # header
    push(GAME_START_ID)
    push(WHITE_ELO_ID, scalar=float(white_elo), is_scalar=True)
    push(BLACK_ELO_ID, scalar=float(black_elo), is_scalar=True)
    push(RESULT_ID,    scalar=result,            is_scalar=True)

    # moves
    move_tokens_raw = moves_san.split()
    move_tokens = [
        t for t in move_tokens_raw
        if not re.match(r"^\d+\.", t)
        and t not in {"1-0", "0-1", "1/2-1/2", "*"}
    ]

    for san in move_tokens:
        for tid in tokenize_move(san):
            push(tid)

    push(GAME_END_ID)

    return {
        "input_ids":   input_ids,
        "scalars":     scalars,
        "scalar_mask": scalar_mask,
    }


# ---------------------------------------------------------------------------
# PGN streaming
# ---------------------------------------------------------------------------

@dataclass
class RawGame:
    headers:   dict  = field(default_factory=dict)
    moves_str: str   = ""


def stream_games(path: str) -> Iterator[RawGame]:
    """Yield raw games from a .pgn.zst file."""
    file_size = os.path.getsize(path)

    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            buffer   = ""
            game     = RawGame()
            in_moves = False

            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Reading",
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

                        elif line and not line.startswith("["):
                            in_moves = True
                            game.moves_str += " " + line

                # yield last game
                if game.moves_str.strip():
                    yield game


def count_plies(moves_str: str) -> int:
    tokens = moves_str.split()
    tokens = [t for t in tokens if not re.match(r"^\d+\.", t)]
    tokens = [t for t in tokens if t not in {"1-0", "0-1", "1/2-1/2", "*"}]
    return len(tokens)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and tokenize Lichess PGN games.")
    parser.add_argument("--input",     required=True,  help="Path to .pgn.zst file")
    parser.add_argument("--output",    required=True,  help="Path to output .parquet file")
    parser.add_argument("--min-elo",   type=int, default=1200, help="Minimum Elo for both players")
    parser.add_argument("--min-moves", type=int, default=10,   help="Minimum moves per player")
    parser.add_argument("--debug",     action="store_true",    help="Process only 50 games")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Parquet schema
    schema = pa.schema([
        ("game_id",    pa.int32()),
        ("white_elo",  pa.int16()),
        ("black_elo",  pa.int16()),
        ("result",     pa.float32()),
        ("n_tokens",   pa.int16()),
        ("input_ids",  pa.list_(pa.int8())),
        ("scalars",    pa.list_(pa.float32())),
        ("scalar_mask",pa.list_(pa.bool_())),
    ])

    min_plies = args.min_moves * 2  # convert moves per player to plies

    stats = {"total": 0, "abandoned": 0, "not_classical": 0,
             "low_elo": 0, "too_short": 0, "passed": 0}
    game_id = 0

    # accumulate rows then write in batches
    batch_size = 1000
    rows: list[dict] = []

    writer = pq.ParquetWriter(args.output, schema)

    def flush(rows: list[dict]) -> None:
        if not rows:
            return
        writer.write_table(pa.table({
            "game_id":    pa.array([r["game_id"]    for r in rows], type=pa.int32()),
            "white_elo":  pa.array([r["white_elo"]  for r in rows], type=pa.int16()),
            "black_elo":  pa.array([r["black_elo"]  for r in rows], type=pa.int16()),
            "result":     pa.array([r["result"]     for r in rows], type=pa.float32()),
            "n_tokens":   pa.array([r["n_tokens"]   for r in rows], type=pa.int16()),
            "input_ids":  pa.array([r["input_ids"]  for r in rows], type=pa.list_(pa.int8())),
            "scalars":    pa.array([r["scalars"]    for r in rows], type=pa.list_(pa.float32())),
            "scalar_mask":pa.array([r["scalar_mask"]for r in rows], type=pa.list_(pa.bool_())),
        }))

    for raw in stream_games(args.input):
        stats["total"] += 1

        tc      = raw.headers.get("TimeControl", "")
        welo    = parse_elo(raw.headers.get("WhiteElo", "0"))
        belo    = parse_elo(raw.headers.get("BlackElo", "0"))
        result  = parse_result(raw.headers.get("Result", ""))
        n_plies = count_plies(raw.moves_str)

        if n_plies < 2:
            stats["abandoned"] += 1
            continue
        if not is_classical(tc):
            stats["not_classical"] += 1
            continue
        if welo < args.min_elo or belo < args.min_elo:
            stats["low_elo"] += 1
            continue
        if n_plies < min_plies:
            stats["too_short"] += 1
            continue
        if result is None:
            continue

        stats["passed"] += 1
        game_id += 1

        tokenized = tokenize_game(raw.moves_str, welo, belo, result)

        if args.debug:
            print(f"\n--- Game {game_id} | W:{welo} B:{belo} R:{result} ---")
            print(f"  moves   : {raw.moves_str.strip()[:80]}...")
            print(f"  n_tokens: {len(tokenized['input_ids'])}")
            print(f"  ids[:12]: {tokenized['input_ids'][:12]}")
            print(f"  scalars : {[f'{s:.2f}' for s in tokenized['scalars'][:6]]}")

        rows.append({
            "game_id":    game_id,
            "white_elo":  welo,
            "black_elo":  belo,
            "result":     result,
            "n_tokens":   len(tokenized["input_ids"]),
            "input_ids":  tokenized["input_ids"],
            "scalars":    tokenized["scalars"],
            "scalar_mask":tokenized["scalar_mask"],
        })

        if len(rows) >= batch_size:
            flush(rows)
            rows = []

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
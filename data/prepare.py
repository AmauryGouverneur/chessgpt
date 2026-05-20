#!/usr/bin/env python3
"""
prepare.py — filter and tokenize Lichess PGN games into curriculum phases.

Usage:
    python data/prepare.py \
        --input data/raw/lichess_mini.pgn.zst \
        --output-dir data/processed/ \
        --debug
"""

import argparse
import re
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import zstandard as zstd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from config import PHASES, TOKEN2ID, UNK_ID, VOCAB_SIZE


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


def parse_result(val: str) -> Optional[str]:
    return {"1-0": "1-0", "0-1": "0-1", "1/2-1/2": "1/2-1/2"}.get(val)


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def encode(text: str) -> list[int]:
    """Encode a string to token ids using the ASCII vocabulary."""
    return [TOKEN2ID.get(c, UNK_ID) for c in text]


# ---------------------------------------------------------------------------
# Game text construction
# ---------------------------------------------------------------------------

def truncate_moves(moves_str: str, max_moves: int) -> str:
    """Truncate a move string to at most max_moves full moves."""
    tokens     = moves_str.split()
    result     = []
    move_count = 0

    for token in tokens:
        if re.match(r'^\d+\.$', token):
            move_count += 1
            if move_count > max_moves:
                break
        if move_count <= max_moves:
            result.append(token)

    return " ".join(result)


def build_game_text(
    white_elo: int,
    black_elo: int,
    result:    str,
    moves_str: str,
    max_moves: int,
) -> str:
    """
    Build the full text representation of a game.
    Format: "WhiteElo=XXXX BlackElo=XXXX Result=X 1. e4 e5 2. Nf3 ..."
    """
    # strip inline annotations
    moves_san = re.sub(r'\{[^}]*\}', '', moves_str)
    # strip black move numbers e.g. "1..." "2..."
    moves_san = re.sub(r'\d+\.\.\.', '', moves_san)
    # normalize spaces
    moves_san = re.sub(r'\s+', ' ', moves_san).strip()
    # strip result token at end
    for r in ("1-0", "0-1", "1/2-1/2", "*"):
        if moves_san.endswith(r):
            moves_san = moves_san[:-len(r)].rstrip()

    truncated = truncate_moves(moves_san, max_moves)
    header    = f"WhiteElo={white_elo} BlackElo={black_elo} Result={result}"
    return f"{header} {truncated}"


def count_full_moves(moves_str: str) -> int:
    return len(re.findall(r'\d+\.(?!\d)', moves_str))


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("white_elo", pa.int16()),
        ("black_elo", pa.int16()),
        ("n_tokens",  pa.int16()),
        ("input_ids", pa.list_(pa.int8())),
    ])

    writers = {
        phase: pq.ParquetWriter(
            output_dir / f"dataset_phase{phase}.parquet", schema
        )
        for phase in PHASES
    }

    rows  = {phase: [] for phase in PHASES}
    stats = {
        "total": 0, "abandoned": 0, "not_classical": 0,
        "low_elo": 0, "too_short": 0, "passed": 0
    }

    def flush(phase: int) -> None:
        if not rows[phase]:
            return
        writers[phase].write_table(pa.table({
            "white_elo": pa.array([r["white_elo"] for r in rows[phase]], type=pa.int16()),
            "black_elo": pa.array([r["black_elo"] for r in rows[phase]], type=pa.int16()),
            "n_tokens":  pa.array([r["n_tokens"]  for r in rows[phase]], type=pa.int16()),
            "input_ids": pa.array([r["input_ids"] for r in rows[phase]],
                                  type=pa.list_(pa.int8())),
        }))
        rows[phase] = []

    for raw in stream_games(args.input):
        stats["total"] += 1

        tc      = raw.headers.get("TimeControl", "")
        welo    = parse_elo(raw.headers.get("WhiteElo", "0"))
        belo    = parse_elo(raw.headers.get("BlackElo", "0"))
        result  = parse_result(raw.headers.get("Result", ""))
        n_moves = count_full_moves(raw.moves_str)

        if n_moves < 2:            stats["abandoned"]     += 1; continue
        if not is_classical(tc):   stats["not_classical"] += 1; continue
        if welo < 1000 or belo < 1000: stats["low_elo"]   += 1; continue
        if n_moves < 10:           stats["too_short"]      += 1; continue
        if result is None:                                        continue

        stats["passed"] += 1
        min_elo = min(welo, belo)
        max_elo = max(welo, belo)

        for phase, cfg in PHASES.items():
            if min_elo < cfg["elo_min"] or max_elo > cfg["elo_max"]:
                continue

            text      = build_game_text(welo, belo, result,
                                        raw.moves_str, cfg["max_moves"])
            input_ids = encode(text)
            n_tokens  = len(input_ids)

            if args.debug:
                print(f"\n--- Phase {phase} | W:{welo} B:{belo} R:{result} ---")
                print(f"  text     : {text[:100]}...")
                print(f"  n_tokens : {n_tokens}")
                print(f"  ids[:8]  : {input_ids[:8]}")

            rows[phase].append({
                "white_elo": welo,
                "black_elo": belo,
                "n_tokens":  min(n_tokens, 32767),
                "input_ids": input_ids,
            })

            if len(rows[phase]) >= 1000:
                flush(phase)

        if args.debug and stats["passed"] >= 20:
            break

    for phase in PHASES:
        flush(phase)
        writers[phase].close()

    print(f"\n{'='*50}")
    print(f"Total games seen:        {stats['total']:>10,}")
    print(f"  Abandoned:             {stats['abandoned']:>10,}")
    print(f"  Not classical:         {stats['not_classical']:>10,}")
    print(f"  Elo < 1000:            {stats['low_elo']:>10,}")
    print(f"  Too short:             {stats['too_short']:>10,}")
    print(f"  Passed:                {stats['passed']:>10,}")
    print(f"\nOutputs:")
    for phase in PHASES:
        path = output_dir / f"dataset_phase{phase}.parquet"
        if path.exists():
            size = path.stat().st_size / 1e6
            print(f"  Phase {phase}: {path.name} ({size:.1f} MB)")
    print(f"\nVocab size: {VOCAB_SIZE}")


if __name__ == "__main__":
    main()
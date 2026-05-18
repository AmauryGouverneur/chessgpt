"""
eval.py — Evaluate a trained ChessGPT checkpoint.

Computes:
  1. Legal move rate as a function of ply (batched, from real test positions)
  2. Elo estimation via win rate against Stockfish

Usage:
    python eval.py --checkpoint runs/run_XXXXXX/checkpoints/best.pt
    python eval.py --checkpoint runs/run_XXXXXX/checkpoints/best.pt --stockfish /path/to/stockfish
"""

import argparse
import json
import os
from pathlib import Path

import chess
import chess.engine
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from model import ChessGPT, ChessGPTConfig
from config import (
    CONTROL, PIECES, FILES, RANKS, TOKEN2ID, ID2TOKEN,
    PAD_ID, WIN_ID, PROMOTION_ID, BLOCK_SIZE, HEADER_LEN,
)

CONTROL_S     = set(CONTROL)
SAMPLE_TOKENS = 6   # enough for any move: piece+x+file+rank+promo+check


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_move(san: str) -> list[int]:
    """Convert a SAN move string to a list of token ids."""
    san = san.strip()
    ids: list[int] = []

    if san.startswith("O-O"):
        castle = "O-O-O" if san.startswith("O-O-O") else "O-O"
        ids.append(TOKEN2ID[castle])
        if san[len(castle):] in ("+", "#"):
            ids.append(TOKEN2ID[san[len(castle):]])
        return ids

    suffix = ""
    if san and san[-1] in ("+", "#"):
        suffix, san = san[-1], san[:-1]

    promotion = False
    if "=" in san:
        san       = san[:san.index("=")]
        promotion = True

    i = 0
    if i < len(san) and san[i] in PIECES:
        ids.append(TOKEN2ID[san[i]]); i += 1
    if i < len(san) and san[i] == "x":
        ids.append(TOKEN2ID["x"]); i += 1

    remaining = san[i:]
    if len(remaining) >= 2:
        ids.append(TOKEN2ID[remaining[-2]])
        ids.append(TOKEN2ID[remaining[-1]])
    if promotion:
        ids.append(PROMOTION_ID)
    if suffix:
        ids.append(TOKEN2ID[suffix])

    return ids


def tokens_to_san(token_ids: list[int]) -> str:
    """Reconstruct a SAN string from token ids, skipping control tokens."""
    return "".join(
        ID2TOKEN.get(i, "")
        for i in token_ids
        if ID2TOKEN.get(i, "") not in CONTROL_S
    )


def extract_first_move(token_ids: list[int]) -> list[int]:
    """
    Greedily extract the first complete move from a token sequence.
    Move grammar: [piece?][x?][file][rank][=promotion?][+/#?]
                | [O-O-O | O-O][+/#?]
    """
    toks = [ID2TOKEN.get(i, "") for i in token_ids]
    move = []
    i    = 0

    # castling
    if i < len(toks) and toks[i] in ("O-O-O", "O-O"):
        move.append(token_ids[i]); i += 1
        if i < len(toks) and toks[i] in ("+", "#"):
            move.append(token_ids[i])
        return move

    if i < len(toks) and toks[i] in PIECES:
        move.append(token_ids[i]); i += 1
    if i < len(toks) and toks[i] == "x":
        move.append(token_ids[i]); i += 1
    if i < len(toks) and toks[i] in FILES:
        move.append(token_ids[i]); i += 1
    else:
        return []
    if i < len(toks) and toks[i] in RANKS:
        move.append(token_ids[i]); i += 1
    else:
        return []
    if i < len(toks) and toks[i] == "[PROMOTION]":
        move.append(token_ids[i]); i += 1
    if i < len(toks) and toks[i] in ("+", "#"):
        move.append(token_ids[i])

    return move


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def token_ids_to_board(token_ids: list[int]) -> chess.Board:
    """
    Decode a token sequence back to a chess.Board by replaying moves.
    Skips the 4-token header.
    """
    board  = chess.Board()
    tokens = token_ids[HEADER_LEN:]
    i      = 0

    while i < len(tokens):
        move_ids = extract_first_move(tokens[i:i + SAMPLE_TOKENS])
        if not move_ids:
            i += 1
            continue
        san = tokens_to_san(move_ids)
        try:
            board.push(board.parse_san(san))
            i += len(move_ids)
        except ValueError:
            i += 1

    return board


def is_legal(candidate_tokens: list[int], board: chess.Board) -> bool:
    """Check if the first move in candidate_tokens is legal on the board."""
    move_ids = extract_first_move(candidate_tokens)
    if not move_ids:
        return False
    try:
        board.parse_san(tokens_to_san(move_ids))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Build evaluation samples
# ---------------------------------------------------------------------------

def build_eval_samples(
    parquet_path: str,
    n_samples:    int,
    min_ply:      int,
    max_ply:      int,
    rng:          np.random.Generator,
) -> list[dict]:
    """
    Sample (game_prefix, board, ply) from the dataset.
    Each sample picks a random game and a random split point,
    then decodes the board at that position.
    """
    df      = pq.read_table(parquet_path).to_pandas()
    samples = []

    with tqdm(total=n_samples, desc="Building samples") as pbar:
        while len(samples) < n_samples:
            row       = df.iloc[rng.integers(0, len(df))]
            token_ids = list(row["input_ids"])

            # need enough tokens for min_ply moves (~2 tokens/move minimum)
            if len(token_ids) < HEADER_LEN + min_ply * 2:
                continue

            # sample a split point in the move token range
            lo    = HEADER_LEN + min_ply * 2
            hi    = min(len(token_ids), HEADER_LEN + max_ply * 4)
            if lo >= hi:
                continue
            split = int(rng.integers(lo, hi))

            prefix = token_ids[:split]
            board  = token_ids_to_board(prefix)
            ply    = board.ply()

            if not (min_ply <= ply <= max_ply):
                continue
            if not list(board.legal_moves):
                continue

            samples.append({
                "input_ids": prefix,
                "board":     board,
                "ply":       ply,
            })
            pbar.update(1)

    return samples


# ---------------------------------------------------------------------------
# Batched legal move rate evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_legal_move_rate(
    model:        ChessGPT,
    device:       torch.device,
    parquet_path: str,
    n_samples:    int   = 500,
    batch_size:   int   = 64,
    temperature:  float = 1.0,
    min_ply:      int   = 6,
    max_ply:      int   = 80,
) -> dict:
    model.eval()
    rng     = np.random.default_rng(42)
    samples = build_eval_samples(parquet_path, n_samples,
                                 min_ply, max_ply, rng)
    results = []  # (ply, is_legal)

    for start in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[start:start + batch_size]
        B     = len(batch)

        # build context batch — left-truncate to BLOCK_SIZE, right-pad
        x = torch.full((B, BLOCK_SIZE), PAD_ID, dtype=torch.long, device=device)
        for i, s in enumerate(batch):
            ids = s["input_ids"][-BLOCK_SIZE:]
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        # sample SAMPLE_TOKENS autoregressively
        sampled = torch.zeros(B, SAMPLE_TOKENS, dtype=torch.long)
        for t in range(SAMPLE_TOKENS):
            logits      = model(x)[0][:, -1, :] / temperature
            next_tokens = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            )                                           # (B, 1)
            sampled[:, t] = next_tokens.squeeze(1).cpu()
            x = torch.cat([x[:, 1:], next_tokens], dim=1)

        for i, s in enumerate(batch):
            results.append((s["ply"], is_legal(sampled[i].tolist(), s["board"])))

    overall   = float(np.mean([r[1] for r in results]))
    by_bucket: dict[int, list[bool]] = {}
    for ply, legal in results:
        by_bucket.setdefault((ply // 5) * 5, []).append(legal)

    return {
        "overall":       overall,
        "by_ply_bucket": {k: float(np.mean(v))
                          for k, v in sorted(by_bucket.items())},
        "n_samples":     len(results),
    }


# ---------------------------------------------------------------------------
# Model move selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def model_pick_move(
    model:       ChessGPT,
    input_ids:   list[int],
    board:       chess.Board,
    device:      torch.device,
    temperature: float = 1.0,
) -> chess.Move | None:
    """Sample tokens and return the first legal move, or a random fallback."""
    ids = input_ids[-BLOCK_SIZE:]
    ids = ids + [PAD_ID] * (BLOCK_SIZE - len(ids))
    x   = torch.tensor([ids], dtype=torch.long, device=device)

    sampled = []
    for _ in range(SAMPLE_TOKENS):
        logits     = model(x)[0][0, -1, :] / temperature
        next_token = torch.multinomial(
            torch.softmax(logits, dim=-1), num_samples=1
        ).item()
        sampled.append(next_token)
        x = torch.cat([x[:, 1:],
                        torch.tensor([[next_token]],
                                     dtype=torch.long, device=device)], dim=1)

    move_ids = extract_first_move(sampled)
    if move_ids:
        san = tokens_to_san(move_ids)
        try:
            return board.parse_san(san)
        except ValueError:
            pass

    # fallback: first legal move
    moves = list(board.legal_moves)
    return moves[0] if moves else None


# ---------------------------------------------------------------------------
# Play vs Stockfish
# ---------------------------------------------------------------------------

def play_vs_stockfish(
    model:          ChessGPT,
    device:         torch.device,
    stockfish_path: str,
    sf_elo:         int,
    model_is_white: bool  = True,
    temperature:    float = 1.0,
    max_moves:      int   = 150,
) -> str:
    """Play one game: model vs Stockfish. Returns 'win', 'draw', or 'loss'."""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})

    board     = chess.Board()
    input_ids = [WIN_ID]

    try:
        for _ in range(max_moves * 2):
            if board.is_game_over():
                break

            if (board.turn == chess.WHITE) == model_is_white:
                move = model_pick_move(model, input_ids, board,
                                       device, temperature)
                if move is None:
                    break
            else:
                move = engine.play(board,
                                   chess.engine.Limit(time=0.1)).move

            san = board.san(move)
            board.push(move)
            input_ids.extend(tokenize_move(san))
    finally:
        engine.quit()

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return "draw"
    return "win" if (outcome.winner == chess.WHITE) == model_is_white \
           else "loss"


def estimate_elo(wins: int, draws: int, losses: int, sf_elo: int) -> float:
    total = wins + draws + losses
    if total == 0:
        return float("nan")
    score = max(0.001, min(0.999, (wins + 0.5 * draws) / total))
    return sf_elo + 400 * np.log10(score / (1 - score))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--data",        default="data/processed/dataset_moves.parquet")
    parser.add_argument("--stockfish",   default=None)
    parser.add_argument("--n-samples",   default=500,  type=int)
    parser.add_argument("--batch-size",  default=64,   type=int)
    parser.add_argument("--games",       default=20,   type=int)
    parser.add_argument("--sf-elos",     default="800,1100,1500,1900")
    parser.add_argument("--temperature", default=1.0,  type=float)
    parser.add_argument("--min-ply",     default=6,    type=int)
    parser.add_argument("--max-ply",     default=80,   type=int)
    args = parser.parse_args()

    # ── device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── load model ────────────────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location=device,
                       weights_only=False)
    model = ChessGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded: step {ckpt['step']}, val loss {ckpt['val_loss']:.4f}")

    out_dir = Path(args.checkpoint).parent.parent / "eval"
    out_dir.mkdir(exist_ok=True)
    results = {}

    # ── 1. Legal move rate ────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Legal move rate ({args.n_samples} samples)...")

    stats = evaluate_legal_move_rate(
        model, device, args.data,
        n_samples   = args.n_samples,
        batch_size  = args.batch_size,
        temperature = args.temperature,
        min_ply     = args.min_ply,
        max_ply     = args.max_ply,
    )
    results["legal_move_rate"] = stats
    print(f"Overall: {stats['overall']:.1%}")

    buckets = sorted(stats["by_ply_bucket"].keys())
    rates   = [stats["by_ply_bucket"][b] * 100 for b in buckets]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(buckets, rates, "o-", color="steelblue",
            linewidth=1.5, markersize=5)
    ax.axhline(stats["overall"] * 100, color="tomato", linestyle="--",
               label=f"overall {stats['overall']:.1%}")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Ply (bucketed by 5)")
    ax.set_ylabel("Legal move rate (%)")
    ax.set_title(f"Legal Move Rate vs Game Position "
                 f"({stats['n_samples']} samples)")
    ax.set_ylim(0, 105)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "legal_move_rate.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'legal_move_rate.png'}")

    # ── 2. Elo estimation ─────────────────────────────────────────────────
    sf_path = args.stockfish or os.environ.get("STOCKFISH_PATH")
    if sf_path is None:
        print("\nNo Stockfish — skipping Elo estimation.")
    else:
        print(f"\n{'='*50}")
        sf_elos     = [int(e) for e in args.sf_elos.split(",")]
        elo_results = {}

        for sf_elo in sf_elos:
            wins = draws = losses = 0
            for g in tqdm(range(args.games), desc=f"SF {sf_elo}"):
                outcome = play_vs_stockfish(
                    model, device, sf_path, sf_elo,
                    model_is_white = g % 2 == 0,
                    temperature    = args.temperature,
                )
                if   outcome == "win":  wins   += 1
                elif outcome == "draw": draws  += 1
                else:                   losses += 1

            est = estimate_elo(wins, draws, losses, sf_elo)
            print(f"  SF {sf_elo} | W/D/L {wins}/{draws}/{losses} "
                  f"| est. Elo {est:.0f}")
            elo_results[sf_elo] = {
                "wins": wins, "draws": draws, "losses": losses,
                "win_rate": (wins + 0.5 * draws) / args.games,
                "estimated_elo": est,
            }

        results["elo_estimation"] = elo_results

        x      = list(elo_results.keys())
        wrates = [elo_results[e]["win_rate"] * 100 for e in x]
        eelos  = [elo_results[e]["estimated_elo"] for e in x]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(x, wrates, "o-", color="steelblue", linewidth=2)
        ax1.axhline(50, color="gray", linestyle="--", label="50%")
        ax1.set_xlabel("Stockfish Elo")
        ax1.set_ylabel("Model score (%)")
        ax1.set_title("Win Rate vs Stockfish")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(x, eelos, "o-", color="tomato", linewidth=2)
        ax2.axhline(np.mean(eelos), color="gray", linestyle="--",
                    label=f"mean {np.mean(eelos):.0f}")
        ax2.set_xlabel("Stockfish Elo")
        ax2.set_ylabel("Estimated Elo")
        ax2.set_title("Elo Estimate")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "elo_estimation.png", dpi=150)
        plt.close()
        print(f"Mean estimated Elo: {np.mean(eelos):.0f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
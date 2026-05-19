from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import VOCAB_SIZE, BLOCK_SIZE, PAD_ID


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ChessGPTConfig:
    vocab_size: int   = VOCAB_SIZE
    block_size: int   = BLOCK_SIZE
    n_embd:     int   = 384
    n_layer:    int   = 6
    n_head:     int   = 6
    dropout:    float = 0.1

    @classmethod
    def debug(cls) -> "ChessGPTConfig":
        return cls(n_embd=64, n_layer=2, n_head=2)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-8):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class Derf(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.s     = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta  = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.erf(self.alpha * x + self.s) + self.beta


# ---------------------------------------------------------------------------
# Rotary Positional Embedding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, head_size: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_size, 2).float() / head_size)
        )
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, T: int, device: torch.device):
        t     = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half   = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q   = q * cos + rotate_half(q) * sin
    k   = k * cos + rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Single Attention Head
# ---------------------------------------------------------------------------

class AttentionHead(nn.Module):
    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.tril: torch.Tensor
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        T, head_size = q.shape[-2], q.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head    = config.n_head
        self.head_size = config.n_embd // config.n_head

        self.q_proj   = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj   = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj   = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rope     = RotaryEmbedding(self.head_size)
        self.heads    = nn.ModuleList(
            [AttentionHead(config) for _ in range(config.n_head)]
        )
        self.dropout  = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        q, k     = apply_rope(q, k, cos, sin)

        out = torch.cat(
            [self.heads[i](q[:, i], k[:, i], v[:, i])
             for i in range(self.n_head)],
            dim=-1,
        )
        return self.dropout(self.out_proj(out))


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        self.w1      = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.w2      = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.w3      = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        self.sa   = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1  = RMSNorm(config.n_embd)
        self.ln2  = RMSNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# ChessGPT
# ---------------------------------------------------------------------------

class ChessGPT(nn.Module):
    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks          = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f    = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ChessGPT: {n_params/1e6:.2f}M parameters")

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        x      = self.token_embedding(idx)
        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
                ignore_index=PAD_ID,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:         torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] / temperature
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, idx_next], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = ChessGPTConfig.debug()
    model  = ChessGPT(config)

    B, T   = 2, 64
    idx    = torch.randint(0, config.vocab_size, (B, T))
    target = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, target)
    print(f"logits : {logits.shape}")
    print(f"loss   : {loss.item():.4f}  (random baseline: {math.log(config.vocab_size):.4f})")
    print("Smoke test passed.")
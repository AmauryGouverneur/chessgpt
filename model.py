from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ChessGPTConfig:
    vocab_size:  int   = 64
    block_size:  int   = 256
    n_embd:      int   = 384
    n_layer:     int   = 6
    n_head:      int   = 6
    n_kv_head:   int   = 2
    dropout:     float = 0.1

    @classmethod
    def debug(cls) -> "ChessGPTConfig":
        return cls(n_embd=64, n_layer=2, n_head=2, n_kv_head=1)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-8):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms

class Derf(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.alpha  = nn.Parameter(torch.tensor(0.5))  # scalar
        self.s      = nn.Parameter(torch.zeros(1))      # scalar shift
        self.gamma  = nn.Parameter(torch.ones(n_embd))  # per-channel
        self.beta   = nn.Parameter(torch.zeros(n_embd)) # per-channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.erf(self.alpha * x + self.s) + self.beta

# ---------------------------------------------------------------------------
# Rotary Positional Embedding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, head_size: int, base: int = 10_000):
        super().__init__()
        # precompute inverse frequencies — shape (head_size/2,)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_size, 2).float() / head_size)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, T: int, device: torch.device):
        # positions: (T,)
        t = torch.arange(T, device=device).float()
        # outer product -> (T, head_size/2)
        freqs = torch.outer(t, self.inv_freq)
        # cat to full head_size, then stack cos/sin -> (T, head_size)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to the first half."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding to queries and keys.

    q, k : (B, n_head, T, head_size)
    cos  : (T, head_size)
    sin  : (T, head_size)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_size)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Single Attention Head
# ---------------------------------------------------------------------------

class AttentionHead(nn.Module):
    """
    Single attention head. Operates on pre-projected q, k, v slices.
    Applies causal mask and dropout.
    """
    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
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
        # q, k, v: (B, T, head_size)
        T, head_size = q.shape[-2], q.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size ** -0.5   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v   # (B, T, head_size)


# ---------------------------------------------------------------------------
# Grouped Query Attention
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    n_head    query heads, each with its own Q projection.
    n_kv_head key/value heads shared across groups of query heads.
    Each KV head serves (n_head // n_kv_head) query heads.
    """
    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_size = config.n_embd // config.n_head
        self.n_rep     = config.n_head // config.n_kv_head  # queries per KV head

        # separate projections — no bias
        self.q_proj = nn.Linear(config.n_embd,
                                config.n_head    * self.head_size, bias=False)
        self.k_proj = nn.Linear(config.n_embd,
                                config.n_kv_head * self.head_size, bias=False)
        self.v_proj = nn.Linear(config.n_embd,
                                config.n_kv_head * self.head_size, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.rope   = RotaryEmbedding(self.head_size)
        self.heads  = nn.ModuleList(
            [AttentionHead(config) for _ in range(config.n_head)]
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # project and reshape to (B, n_head, T, head_size)
        q = self.q_proj(x).view(B, T, self.n_head,    self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)

        # apply RoPE to q and k
        cos, sin = self.rope(T, x.device)
        q, k     = apply_rope(q, k, cos, sin)

        # expand k, v so every query head has its own KV head
        # (B, n_kv_head, T, head_size) -> (B, n_head, T, head_size)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # run each head explicitly
        out = torch.cat(
            [self.heads[i](q[:, i], k[:, i], v[:, i])
             for i in range(self.n_head)],
            dim=-1,
        )   # (B, T, n_head * head_size) = (B, T, C)

        out = self.dropout(self.out_proj(out))
        return out


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
        # SwiGLU: (w1(x) * silu) element-wise gate w3(x), then project down
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        self.sa   = GroupedQueryAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1  = RMSNorm(config.n_embd)
        self.ln2  = RMSNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))    # attention + residual
        x = x + self.ffwd(self.ln2(x))  # FFN + residual
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

        # weight tying: input embedding and output projection share weights
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
        # idx: (B, T)
        B, T = idx.shape

        x = self.token_embedding(idx)   # (B, T, C)
        x = self.blocks(x)              # (B, T, C)
        x = self.ln_f(x)                # (B, T, C)
        logits = self.lm_head(x)        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
                ignore_index=0,         # ignore PAD tokens
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond      = idx[:, -self.config.block_size:]
            logits, _     = self(idx_cond)
            logits        = logits[:, -1, :] / temperature
            probs         = F.softmax(logits, dim=-1)
            idx_next      = torch.multinomial(probs, num_samples=1)
            idx           = torch.cat([idx, idx_next], dim=1)
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
    print(f"logits: {logits.shape}")
    print(f"loss:   {loss.item():.4f}")
    print("Smoke test passed.")
"""
================================================================================
model_retro.py — RETRO architecture, an honest implementation
================================================================================

This is a working implementation of RETRO (Borgeaud et al., DeepMind 2021):
"Improving language models by retrieving from trillions of tokens"
https://arxiv.org/abs/2112.04426

DIFFERENCE FROM RAG (what we did before in retro_generate.py):
  RAG:    retrieved text gets pasted into the prompt as more tokens.
          The model attends to everything in one big sequence.
  RETRO:  retrieved chunks live OUTSIDE the input sequence.
          Special "chunked cross-attention" layers inside each transformer
          block let the model look at them without spending context tokens.

WHY THIS NEEDS TRAINING FROM SCRATCH:
  The cross-attention layers are NEW parameters. Pretrained GPT-2 has no
  weights for them. The model must learn from scratch when retrieval is
  useful and when to ignore it.

ARCHITECTURE OVERVIEW:
  The input sequence of length T is logically split into K chunks of length L.
    T = K * chunk_size            e.g. T=256, L=64, K=4

  For each chunk, we retrieve k neighbors. Each neighbor is a chunk of tokens.
    neighbors shape: (B, K, n_neighbors, neighbor_len)

  Inside the model:
    1. Standard token + position embeddings -> x of shape (B, T, C)
    2. Embed the neighbor token IDs -> nbr_emb of shape (B, K, k, L_n, C)
    3. Pass x through N transformer blocks. Some blocks (every cca_every-th)
       contain an extra ChunkedCrossAttention layer that lets each chunk in x
       attend to its own k retrieved neighbors.
    4. Final LayerNorm + lm_head -> logits

SIMPLIFICATIONS vs the paper:
  - No separate "neighbor encoder" transformer. The paper uses a 2-layer
    encoder to process neighbors before they enter CCA; we just use the
    token+position embedding directly. The main transformer's CCA layers
    learn to do whatever encoding is needed.
  - No causal shift between chunks. The paper has queries from chunk k+1
    attend to neighbors of chunk k (so retrieval can't leak the answer).
    We use queries from chunk k attending to neighbors of chunk k. This is
    fine because at training time our random neighbors don't overlap with
    the answer; but it would be unsafe with overlapping bank retrieval.
  - Multi-Query Attention not used. We use vanilla multi-head.
================================================================================
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Re-use the unchanged building blocks from nanoGPT.
from model import LayerNorm, CausalSelfAttention, MLP


# ==============================================================================
# Config
# ==============================================================================
@dataclass
class RetroConfig:
    # Standard transformer hyperparams (same meanings as GPTConfig).
    block_size: int = 256
    vocab_size: int = 65          # tiny-shakespeare char vocab
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    # RETRO-specific.
    chunk_size: int = 64          # L: length of each chunk in the main sequence
    n_neighbors: int = 2          # k: number of neighbors retrieved per chunk
    neighbor_len: int = 64        # L_n: length of each retrieved neighbor chunk
    cca_every: int = 2            # apply CCA every Nth block (paper uses 3)

    def __post_init__(self):
        assert self.block_size % self.chunk_size == 0, (
            f"block_size ({self.block_size}) must be a multiple of "
            f"chunk_size ({self.chunk_size})")


# ==============================================================================
# ChunkedCrossAttention — the heart of RETRO
# ==============================================================================
class ChunkedCrossAttention(nn.Module):
    """
    Cross-attention from current chunks to retrieved neighbor chunks.

    SHAPES:
      Input x:          (B, T, C)     where T = K * L
      Input neighbors:  (B, K, k, Ln, C)
      Output:           (B, T, C)

    EACH CHUNK ATTENDS TO ITS OWN NEIGHBORS:
      Chunk 0 of the sequence attends to the k neighbors retrieved for chunk 0.
      Chunk 1 attends to its own k neighbors. Etc. There's no cross-talk
      between different chunks' retrieval sets at this layer (the regular
      CausalSelfAttention layer handles cross-chunk info inside the sequence).

    NO CAUSAL MASK:
      Causal self-attention masks the future to prevent cheating. Cross-
      attention to retrieved chunks doesn't need this — we WANT each query
      position to see all of the retrieved tokens. They're "external" info.
    """

    def __init__(self, config: RetroConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Q comes from x; K and V come from neighbors. Use separate Linears
        # because the source tensors are different — can't fuse into one matmul.
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        # Output projection — mixes heads, same role as c_proj elsewhere.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.chunk_size = config.chunk_size
        self.dropout = config.dropout

        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, neighbors_emb: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        L = self.chunk_size
        K = T // L
        _, K2, k_n, Ln, C2 = neighbors_emb.size()
        assert K == K2, f"chunk count mismatch: x has {K} chunks, neighbors has {K2}"
        assert C == C2
        nh, hs = self.n_head, C // self.n_head

        # ---- Queries: from x ----
        # Reshape x into per-chunk: (B, T, C) -> (B, K, L, C)
        x_chunks = x.view(B, K, L, C)
        q = self.q_proj(x_chunks)                              # (B, K, L, C)
        # Split heads: (B, K, L, nh, hs) -> (B, K, nh, L, hs)
        q = q.view(B, K, L, nh, hs).permute(0, 1, 3, 2, 4)

        # ---- Keys and Values: from neighbors ----
        # Flatten the k neighbors of each chunk into one long sequence of
        # length (k_n * Ln). Each chunk's queries will attend over all of them.
        nbrs_flat = neighbors_emb.view(B, K, k_n * Ln, C)
        kv = self.kv_proj(nbrs_flat)                           # (B, K, k_n*Ln, 2C)
        k, v = kv.split(C, dim=-1)
        # Split heads: (B, K, k_n*Ln, nh, hs) -> (B, K, nh, k_n*Ln, hs)
        k = k.view(B, K, k_n * Ln, nh, hs).permute(0, 1, 3, 2, 4)
        v = v.view(B, K, k_n * Ln, nh, hs).permute(0, 1, 3, 2, 4)

        # ---- Attention ----
        # Treat (B * K) as the batch dimension so each chunk attends
        # independently to its own retrieval set.
        q = q.reshape(B * K, nh, L, hs)
        k = k.reshape(B * K, nh, k_n * Ln, hs)
        v = v.reshape(B * K, nh, k_n * Ln, hs)

        if self.flash:
            # No causal mask — we can attend to all retrieved tokens.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v
        # out: (B*K, nh, L, hs)

        # ---- Reshape back ----
        # (B*K, nh, L, hs) -> (B, K, L, nh, hs) -> (B, K, L, C) -> (B, T, C)
        out = out.view(B, K, nh, L, hs).permute(0, 1, 3, 2, 4).contiguous()
        out = out.view(B, T, C)

        return self.resid_dropout(self.c_proj(out))


# ==============================================================================
# Block — same as nanoGPT's but with an optional CCA layer between attn and MLP
# ==============================================================================
class RetroBlock(nn.Module):
    def __init__(self, config: RetroConfig, use_cca: bool):
        super().__init__()
        # Self-attention sublayer (same as vanilla).
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        # Optional cross-attention sublayer (the RETRO addition).
        self.use_cca = use_cca
        if use_cca:
            self.ln_cca = LayerNorm(config.n_embd, bias=config.bias)
            self.cca = ChunkedCrossAttention(config)
        # MLP sublayer (same as vanilla).
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, neighbors_emb: torch.Tensor | None = None) -> torch.Tensor:
        # Sublayer 1: causal self-attention with residual.
        x = x + self.attn(self.ln_1(x))
        # Sublayer 1.5: chunked cross-attention with residual (only in CCA blocks).
        # If neighbors_emb is None we silently skip — lets us run the same model
        # with or without retrieval, useful for ablations.
        if self.use_cca and neighbors_emb is not None:
            x = x + self.cca(self.ln_cca(x), neighbors_emb)
        # Sublayer 2: MLP with residual.
        x = x + self.mlp(self.ln_2(x))
        return x


# ==============================================================================
# RetroGPT — the full model
# ==============================================================================
class RetroGPT(nn.Module):
    def __init__(self, config: RetroConfig):
        super().__init__()
        self.config = config

        # Decide which layers get CCA. With cca_every=2 and n_layer=6,
        # CCA goes in layers {1, 3, 5} (every other one, starting from 1).
        # The first layer is plain self-attention so the model has a chance
        # to "set up" the queries before they're used to look at retrieval.
        cca_layers = set(range(config.cca_every - 1, config.n_layer, config.cca_every))
        self.cca_layers = sorted(cca_layers)

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            # Separate position embedding for neighbor sequences. Neighbors
            # have their own internal ordering (position 0..L_n-1) that's
            # independent of where they appear in the main sequence.
            wpe_nbr=nn.Embedding(config.neighbor_len, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                RetroBlock(config, use_cca=(i in cca_layers))
                for i in range(config.n_layer)
            ]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying between input embedding and output projection.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 trick).
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"RetroGPT: {n_params/1e6:.2f}M params | CCA in layers {self.cca_layers}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_neighbors(self, neighbors: torch.Tensor) -> torch.Tensor:
        """
        neighbors: (B, K, k, L_n) integer token ids
        returns:   (B, K, k, L_n, C) float embeddings
        """
        B, K, k, Ln = neighbors.size()
        # Token embeddings: share wte with the main path. This is intentional —
        # same vocabulary, same "meaning" of tokens whether they appear in the
        # input or in retrieval.
        emb = self.transformer.wte(neighbors)                  # (B, K, k, Ln, C)
        # Position embeddings within each neighbor chunk.
        pos = torch.arange(Ln, device=neighbors.device)
        pos_emb = self.transformer.wpe_nbr(pos)                # (Ln, C)
        emb = emb + pos_emb                                    # broadcasts over (B,K,k,Ln,C)
        return self.transformer.drop(emb)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        neighbors: torch.Tensor | None = None,
    ):
        """
        idx:        (B, T) int token ids — the main sequence
        targets:    (B, T) int next-token targets, or None for inference
        neighbors:  (B, K, k, L_n) int neighbor token ids, or None to skip retrieval
        """
        B, T = idx.size()
        device = idx.device
        assert T <= self.config.block_size

        # Standard token + position embeddings.
        pos = torch.arange(T, device=device)
        tok_emb = self.transformer.wte(idx)                    # (B, T, C)
        pos_emb = self.transformer.wpe(pos)                    # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Embed neighbors ONCE (shared across all CCA layers — they look at
        # the same embedded representation, not re-embedded per layer).
        nbr_emb = self._embed_neighbors(neighbors) if neighbors is not None else None

        # Run through blocks.
        for block in self.transformer.h:
            x = block(x, neighbors_emb=nbr_emb)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Same AdamW setup as vanilla nanoGPT."""
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)

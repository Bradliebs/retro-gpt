"""
================================================================================
train_retro_p2_bank.py — Phase 2.2 Stage C
================================================================================

Train RETRO on the bank corpus (~60M GPT-2 BPE tokens extracted from the
cc_service single-cell texts) with REAL bank-based retrieval — every chunk's
neighbors were precomputed in Stage B via the same MiniLM+whitening pipeline
the bank service uses at query time.

WHAT THIS PROVES (vs Phase 1)
  Phase 1 used oracle neighbors (the literal continuation of the chunk) to
  verify the CCA pathway is mechanically connected. The model learned to
  copy from the answer key. That was a wiring test, not a usefulness test.

  Phase 2.2 uses NEIGHBORS THAT WERE ACTUALLY RETRIEVED from a real semantic
  index, with self-retrievals filtered out. There is no answer-key shortcut.
  If val loss with these neighbors beats val loss without them, the model
  has learned to extract genuinely useful signal from semantic retrieval.

DATA LAYOUT (data/bank/)
  train.bin             uint16,  60,739,263 tokens          gpt2 BPE
  val.bin               uint16,   3,130,060 tokens
  cell_tokens.npy       uint16, (774503, 64)                neighbor token blocks
  train_neighbors.npy   uint32, (949050, 2)                 indices into cell_tokens
  val_neighbors.npy     uint32, ( 48907, 2)
  meta.json             includes vocab_size_padded=50304

SAMPLING ALIGNMENT
  Neighbors are precomputed PER CHUNK of CHUNK_SIZE=64 tokens. So sampling
  must be aligned to chunk boundaries — we pick a random chunk_idx and read
  block_size/chunk_size=4 consecutive chunks with their matching neighbors.

ARTIFACTS
  out-retro-bank/ckpt.pt   final checkpoint (state_dict + config)
  out-retro-bank/ckpt_best.pt   best-val checkpoint
================================================================================
"""

import os
import time
import math
import json
from pathlib import Path

import numpy as np
import torch

from model_retro import RetroConfig, RetroGPT


# ---------------- Paths ----------------
DATA_DIR = Path("data") / "bank"
OUT_DIR = Path("out-retro-bank")
OUT_DIR.mkdir(exist_ok=True)


# ---------------- Model config ----------------
# Bigger than Phase 1 (12.5M params): we have 60M tokens of real text and the
# vocab is 50304, not 65. RTX 3070 (8 GB) is tight; cc_service is also running
# on this GPU. Numbers below were chosen to leave headroom.
config = RetroConfig(
    block_size=256,
    vocab_size=50304,     # padded gpt2 vocab (multiple of 64)
    n_layer=8,
    n_head=8,
    n_embd=512,
    dropout=0.0,
    bias=False,
    chunk_size=64,        # MUST match precompute_neighbors.py
    n_neighbors=2,        # MUST match precompute_neighbors.py
    neighbor_len=64,      # MUST match precompute_neighbors.py
    cca_every=2,          # CCA in layers {1, 3, 5, 7}
)


# ---------------- Training ----------------
BATCH_SIZE = 8                  # per-step micro-batch
GRAD_ACCUM_STEPS = 4            # effective batch = 32
MAX_ITERS = 15000               # Phase 2.3: bumped from 5000 after corpus grew to 157M tokens
                                # (~0.78 epochs vs old 0.67; old setting would have been 0.27 epochs)
WARMUP_ITERS = 200
LR_MAX = 3e-4
LR_MIN = 3e-5
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

EVAL_INTERVAL = 250             # iters between val passes
EVAL_BATCHES = 20               # val batches per eval
LOG_INTERVAL = 50               # iters between train-loss prints

SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
DTYPE = torch.bfloat16 if (DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
CTX = (
    torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
    if DEVICE_TYPE == "cuda"
    else torch.amp.autocast(device_type="cpu", enabled=False)
)


# ---------------- Data ----------------
class BankDataset:
    """Wraps train.bin / val.bin plus precomputed neighbor lookups."""

    def __init__(self, split: str):
        self.split = split
        self.tokens = np.memmap(DATA_DIR / f"{split}.bin", dtype=np.uint16, mode="r")
        self.neighbors = np.load(DATA_DIR / f"{split}_neighbors.npy", mmap_mode="r")
        # cell_tokens is shared across splits; load once via classvar pattern.
        if not hasattr(BankDataset, "_cell_tokens"):
            BankDataset._cell_tokens = np.load(DATA_DIR / "cell_tokens.npy", mmap_mode="r")
        self.cell_tokens = BankDataset._cell_tokens

        self.K = config.block_size // config.chunk_size  # chunks per sample
        # Last valid chunk_idx so x+1 still fits inside tokens.
        max_token_start = len(self.tokens) - config.block_size - 1
        self.max_chunk_idx = max_token_start // config.chunk_size
        assert self.max_chunk_idx >= self.K, f"{split} too short for one batch"
        # Also must not run past neighbors array.
        self.max_chunk_idx = min(self.max_chunk_idx, len(self.neighbors) - self.K)

        print(f"  [{split}] {len(self.tokens):,} tokens, "
              f"{len(self.neighbors):,} chunks-with-neighbors, "
              f"sampling range chunk_idx [0, {self.max_chunk_idx})")

    def get_batch(self, batch_size: int):
        """Sample a batch aligned to chunk boundaries.

        Returns (x, y, neighbors) ready for model.forward:
          x:         (B, block_size)               int64
          y:         (B, block_size)               int64
          neighbors: (B, K, n_neighbors, Ln)       int64
        """
        K = self.K
        Ln = config.neighbor_len
        chunk_starts = np.random.randint(0, self.max_chunk_idx, size=batch_size)

        x = np.empty((batch_size, config.block_size), dtype=np.int64)
        y = np.empty((batch_size, config.block_size), dtype=np.int64)
        nbrs = np.empty((batch_size, K, config.n_neighbors, Ln), dtype=np.int64)
        for b, c in enumerate(chunk_starts):
            tstart = int(c) * config.chunk_size
            x[b] = self.tokens[tstart : tstart + config.block_size].astype(np.int64)
            y[b] = self.tokens[tstart + 1 : tstart + 1 + config.block_size].astype(np.int64)
            # neighbors[c:c+K] -> (K, n_neighbors) cell-token indices.
            nbr_idx = self.neighbors[int(c) : int(c) + K]      # (K, n_neighbors)
            nbrs[b] = self.cell_tokens[nbr_idx].astype(np.int64)  # (K, n_neighbors, Ln)
        return (
            torch.from_numpy(x).to(DEVICE),
            torch.from_numpy(y).to(DEVICE),
            torch.from_numpy(nbrs).to(DEVICE),
        )


# ---------------- LR schedule ----------------
def get_lr(it: int) -> float:
    """Linear warmup -> cosine decay to LR_MIN."""
    if it < WARMUP_ITERS:
        return LR_MAX * (it + 1) / WARMUP_ITERS
    if it >= MAX_ITERS:
        return LR_MIN
    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LR_MIN + coeff * (LR_MAX - LR_MIN)


# ---------------- Eval ----------------
@torch.no_grad()
def evaluate(model, ds: BankDataset, n_batches: int) -> tuple[float, float]:
    """Return (loss_with_neighbors, loss_without_neighbors) on `n_batches` val batches.
    The two are computed on the SAME sampled batches so the diff is meaningful.
    """
    model.eval()
    losses_with, losses_without = [], []
    for _ in range(n_batches):
        x, y, nbrs = ds.get_batch(BATCH_SIZE)
        with CTX:
            _, lw = model(x, targets=y, neighbors=nbrs)
            _, lo = model(x, targets=y, neighbors=None)
        losses_with.append(lw.item())
        losses_without.append(lo.item())
    model.train()
    return sum(losses_with) / len(losses_with), sum(losses_without) / len(losses_without)


# ---------------- Train ----------------
def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"device={DEVICE}  dtype={DTYPE}")
    print(f"loading data from {DATA_DIR.resolve()}...")
    train_ds = BankDataset("train")
    val_ds = BankDataset("val")

    print(f"\nbuilding model: n_layer={config.n_layer}, n_head={config.n_head}, "
          f"n_embd={config.n_embd}, vocab={config.vocab_size}")
    model = RetroGPT(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.2f}M")

    optimizer = model.configure_optimizers(
        weight_decay=WEIGHT_DECAY,
        learning_rate=LR_MAX,
        betas=BETAS,
        device_type=DEVICE_TYPE,
    )

    print(f"\ntraining: max_iters={MAX_ITERS}, batch={BATCH_SIZE} x grad_accum={GRAD_ACCUM_STEPS} "
          f"(effective {BATCH_SIZE*GRAD_ACCUM_STEPS}), warmup={WARMUP_ITERS}, "
          f"lr {LR_MAX} -> {LR_MIN} cosine")
    print(f"eval every {EVAL_INTERVAL} iters on {EVAL_BATCHES} batches")
    print("=" * 78)

    best_val = float("inf")
    history = []   # [{iter, train_loss, val_with, val_without}]
    t0 = time.time()
    running = []

    for it in range(MAX_ITERS):
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(GRAD_ACCUM_STEPS):
            x, y, nbrs = train_ds.get_batch(BATCH_SIZE)
            with CTX:
                _, loss = model(x, targets=y, neighbors=nbrs)
                loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        running.append(loss_accum)

        if (it + 1) % LOG_INTERVAL == 0:
            avg = sum(running) / len(running)
            running = []
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {it+1:>5d}  lr {lr:.2e}  loss {avg:.4f}  "
                  f"({LOG_INTERVAL} iters in {dt:.1f}s)")

        if (it + 1) % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            t_eval = time.time()
            val_with, val_without = evaluate(model, val_ds, EVAL_BATCHES)
            gap = val_without - val_with
            print(f"  >> [val]  with-nbrs {val_with:.4f}  no-nbrs {val_without:.4f}  "
                  f"gap {gap:+.4f}  ({EVAL_BATCHES} batches in {time.time()-t_eval:.1f}s)")
            history.append({
                "iter": it + 1,
                "val_with": val_with,
                "val_without": val_without,
                "gap": gap,
                "lr": lr,
            })
            if val_with < best_val:
                best_val = val_with
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "iter": it + 1,
                    "val_with": val_with,
                    "val_without": val_without,
                }, OUT_DIR / "ckpt_best.pt")
                print(f"  >> saved ckpt_best.pt  (val_with={val_with:.4f})")
            t0 = time.time()

    print("=" * 78)
    print("training done. saving final checkpoint.")
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "iter": MAX_ITERS,
    }, OUT_DIR / "ckpt.pt")

    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, val_ds, history


def main():
    model, val_ds, history = train()

    # Final summary.
    if history:
        last = history[-1]
        print("\n" + "=" * 78)
        print("FINAL VALIDATION")
        print("=" * 78)
        print(f"  iter {last['iter']}")
        print(f"  loss with bank neighbors    : {last['val_with']:.4f}")
        print(f"  loss without neighbors      : {last['val_without']:.4f}")
        print(f"  gap (no-nbrs - with-nbrs)   : {last['gap']:+.4f}")
        print("-" * 78)
        if last["gap"] > 0.05:
            print("[OK] Model has learned to extract signal from REAL bank retrieval.")
        elif last["gap"] > 0.01:
            print("[WEAK] Small but positive effect from bank retrieval.")
        else:
            print("[NULL] No measurable benefit from bank neighbors. Either training was")
            print("       too short or the bank-corpus pairing is too noisy to help.")
        print("=" * 78)


if __name__ == "__main__":
    main()

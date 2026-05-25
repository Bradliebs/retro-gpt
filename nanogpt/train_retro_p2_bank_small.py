"""
================================================================================
train_retro_p2_bank_small.py — Phase 2.2 Stage E (shrink variant)
================================================================================

Same as train_retro_p2_bank.py but with a much smaller model.

WHY
  The 55M model trained by Stage C overfit on 60M tokens (1:1 params:tokens
  ratio vs Chinchilla optimal ~20:1). Val_with bottomed at iter 3500 and
  drifted up afterward. The retrieval gap was real (+0.124 nat) but the raw
  LM quality was poor and generation was incoherent.

  This variant shrinks the model to ~6M params (10:1 ratio, close to
  Chinchilla) to test whether the underwhelming generation was a capacity
  mismatch rather than a RETRO-mechanism issue.

WHAT CHANGES vs train_retro_p2_bank.py
  - n_layer: 8 -> 4       (CCA now in layers [1, 3], not [1, 3, 5, 7])
  - n_head:  8 -> 4
  - n_embd:  512 -> 256
  - OUT_DIR: out-retro-bank -> out-retro-bank-small

  Everything else is identical so the comparison is apples-to-apples.

EXPECTED OUTCOME
  - Higher per-token loss baseline (less capacity).
  - Cleaner-looking generation (better-fit model).
  - Retrieval gap likely LARGER in relative terms: small models lean harder
    on retrieval since they cannot memorize.
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
OUT_DIR = Path("out-retro-bank-small")
OUT_DIR.mkdir(exist_ok=True)


# ---------------- Model config ----------------
# Shrunk to ~6M params: ~10:1 tokens:params on the 60M-token corpus.
config = RetroConfig(
    block_size=256,
    vocab_size=50304,     # padded gpt2 vocab (multiple of 64)
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.0,
    bias=False,
    chunk_size=64,        # MUST match precompute_neighbors.py
    n_neighbors=2,        # MUST match precompute_neighbors.py
    neighbor_len=64,      # MUST match precompute_neighbors.py
    cca_every=2,          # CCA in layers {1, 3}
)


# ---------------- Training ----------------
BATCH_SIZE = 8                  # per-step micro-batch
GRAD_ACCUM_STEPS = 4            # effective batch = 32  (same as Stage C)
MAX_ITERS = 5000                # same as Stage C for apples-to-apples
WARMUP_ITERS = 200
LR_MAX = 3e-4
LR_MIN = 3e-5
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

EVAL_INTERVAL = 250
EVAL_BATCHES = 20
LOG_INTERVAL = 50

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
        if not hasattr(BankDataset, "_cell_tokens"):
            BankDataset._cell_tokens = np.load(DATA_DIR / "cell_tokens.npy", mmap_mode="r")
        self.cell_tokens = BankDataset._cell_tokens

        self.K = config.block_size // config.chunk_size
        max_token_start = len(self.tokens) - config.block_size - 1
        self.max_chunk_idx = max_token_start // config.chunk_size
        assert self.max_chunk_idx >= self.K, f"{split} too short for one batch"
        self.max_chunk_idx = min(self.max_chunk_idx, len(self.neighbors) - self.K)

        print(f"  [{split}] {len(self.tokens):,} tokens, "
              f"{len(self.neighbors):,} chunks-with-neighbors, "
              f"sampling range chunk_idx [0, {self.max_chunk_idx})")

    def get_batch(self, batch_size: int):
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
            nbr_idx = self.neighbors[int(c) : int(c) + K]
            nbrs[b] = self.cell_tokens[nbr_idx].astype(np.int64)
        return (
            torch.from_numpy(x).to(DEVICE),
            torch.from_numpy(y).to(DEVICE),
            torch.from_numpy(nbrs).to(DEVICE),
        )


# ---------------- LR schedule ----------------
def get_lr(it: int) -> float:
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
    history = []
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
            print("[NULL] No measurable benefit from bank neighbors.")
        print("=" * 78)


if __name__ == "__main__":
    main()

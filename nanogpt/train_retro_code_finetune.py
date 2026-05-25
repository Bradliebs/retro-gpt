"""
================================================================================
train_retro_code_finetune.py — Fine-tune RETRO on code data
================================================================================

Loads the best wiki-trained checkpoint and fine-tunes on code tokens while
retrieving neighbors from the wiki bank. Tests whether RETRO can adapt to
a new domain (code) while leveraging its existing knowledge base (Wikipedia).

DATA LAYOUT:
  data/code/train_code.bin              code training tokens
  data/code/val_code.bin                code validation tokens
  data/code/train_code_neighbors.npy    wiki neighbor indices per chunk
  data/code/val_code_neighbors.npy      wiki neighbor indices per chunk
  data/bank/cell_tokens.npy             wiki cell tokens for neighbor lookup

PRETRAINED:
  out-retro-bank/ckpt_best.pt           best wiki checkpoint (iter 13000)

ARTIFACTS:
  out-retro-code/ckpt.pt                final checkpoint
  out-retro-code/ckpt_best.pt           best-val checkpoint
  out-retro-code/history.json           training history

USAGE:
  cd H:\\MiniLM\\nanogpt
  python train_retro_code_finetune.py
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
CODE_DIR = Path("data") / "code"
BANK_DIR = Path("data") / "bank"
PRETRAINED = Path("out-retro-bank") / "ckpt_best.pt"
OUT_DIR = Path("out-retro-code")
OUT_DIR.mkdir(exist_ok=True)


# ---------------- Training (fine-tune settings) ----------------
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4            # effective batch = 32
MAX_ITERS = 5000
WARMUP_ITERS = 100              # shorter warmup for fine-tuning
LR_MAX = 1e-4                   # lower peak LR for fine-tuning
LR_MIN = 1e-5
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
class CodeFinetuneDataset:
    """Loads code tokens + precomputed wiki neighbors."""

    def __init__(self, split: str, config: RetroConfig):
        self.split = split
        self.config = config
        self.tokens = np.memmap(
            CODE_DIR / f"{split}_code.bin", dtype=np.uint16, mode="r"
        )
        self.neighbors = np.load(
            CODE_DIR / f"{split}_code_neighbors.npy", mmap_mode="r"
        )
        # Wiki cell tokens for neighbor lookups (shared across splits)
        if not hasattr(CodeFinetuneDataset, "_cell_tokens"):
            CodeFinetuneDataset._cell_tokens = np.load(
                BANK_DIR / "cell_tokens.npy", mmap_mode="r"
            )
        self.cell_tokens = CodeFinetuneDataset._cell_tokens

        self.K = config.block_size // config.chunk_size  # chunks per sample (4)
        max_token_start = len(self.tokens) - config.block_size - 1
        self.max_chunk_idx = max_token_start // config.chunk_size
        self.max_chunk_idx = min(self.max_chunk_idx, len(self.neighbors) - self.K)
        assert self.max_chunk_idx >= self.K, f"{split} too short"

        print(f"  [{split}] {len(self.tokens):,} tokens, "
              f"{len(self.neighbors):,} chunks-with-neighbors, "
              f"sampling range [0, {self.max_chunk_idx})")

    def get_batch(self, batch_size: int):
        K = self.K
        Ln = self.config.neighbor_len
        chunk_starts = np.random.randint(0, self.max_chunk_idx, size=batch_size)

        x = np.empty((batch_size, self.config.block_size), dtype=np.int64)
        y = np.empty((batch_size, self.config.block_size), dtype=np.int64)
        nbrs = np.empty((batch_size, K, self.config.n_neighbors, Ln), dtype=np.int64)

        for b, c in enumerate(chunk_starts):
            tstart = int(c) * self.config.chunk_size
            x[b] = self.tokens[tstart : tstart + self.config.block_size].astype(np.int64)
            y[b] = self.tokens[tstart + 1 : tstart + 1 + self.config.block_size].astype(np.int64)
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
def evaluate(model, ds: CodeFinetuneDataset, n_batches: int) -> tuple[float, float]:
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

    # Load pretrained checkpoint
    print(f"\nloading pretrained checkpoint: {PRETRAINED}")
    ckpt = torch.load(PRETRAINED, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    print(f"  config: n_layer={config.n_layer}, n_head={config.n_head}, "
          f"n_embd={config.n_embd}, vocab={config.vocab_size}")
    print(f"  pretrained at iter {ckpt.get('iter', '?')}, "
          f"val_with={ckpt.get('val_with', '?'):.4f}")

    # Build model and load weights
    model = RetroGPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.2f}M")
    del ckpt

    # Load data
    print(f"\nloading code data from {CODE_DIR.resolve()}...")
    train_ds = CodeFinetuneDataset("train", config)
    val_ds = CodeFinetuneDataset("val", config)

    # Baseline eval on code BEFORE fine-tuning
    print("\nbaseline eval on code (before fine-tuning)...")
    base_with, base_without = evaluate(model, val_ds, EVAL_BATCHES)
    base_gap = base_without - base_with
    print(f"  [baseline] with-nbrs {base_with:.4f}  no-nbrs {base_without:.4f}  "
          f"gap {base_gap:+.4f}")

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=WEIGHT_DECAY,
        learning_rate=LR_MAX,
        betas=BETAS,
        device_type=DEVICE_TYPE,
    )

    print(f"\nfine-tuning: max_iters={MAX_ITERS}, batch={BATCH_SIZE} x "
          f"grad_accum={GRAD_ACCUM_STEPS} (effective {BATCH_SIZE*GRAD_ACCUM_STEPS}), "
          f"lr {LR_MAX} -> {LR_MIN} cosine")
    print("=" * 78)

    best_val = float("inf")
    history = [{
        "iter": 0,
        "val_with": base_with,
        "val_without": base_without,
        "gap": base_gap,
        "lr": 0.0,
        "note": "baseline (before fine-tuning)",
    }]
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
                    "pretrained_from": str(PRETRAINED),
                }, OUT_DIR / "ckpt_best.pt")
                print(f"  >> saved ckpt_best.pt  (val_with={val_with:.4f})")
            t0 = time.time()

    print("=" * 78)
    print("fine-tuning done. saving final checkpoint.")
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "iter": MAX_ITERS,
        "pretrained_from": str(PRETRAINED),
    }, OUT_DIR / "ckpt.pt")

    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, val_ds, history


def main():
    model, val_ds, history = train()

    if history:
        baseline = history[0]
        last = history[-1]
        print("\n" + "=" * 78)
        print("FINE-TUNE SUMMARY")
        print("=" * 78)
        print(f"  baseline (wiki model on code):")
        print(f"    val_with={baseline['val_with']:.4f}  "
              f"val_without={baseline['val_without']:.4f}  "
              f"gap={baseline['gap']:+.4f}")
        print(f"  final (fine-tuned on code):")
        print(f"    val_with={last['val_with']:.4f}  "
              f"val_without={last['val_without']:.4f}  "
              f"gap={last['gap']:+.4f}")
        improvement = baseline["val_with"] - last["val_with"]
        print(f"  improvement: {improvement:+.4f} nats "
              f"({100*improvement/baseline['val_with']:.1f}%)")
        print("=" * 78)


if __name__ == "__main__":
    main()

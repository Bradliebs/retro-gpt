"""
================================================================================
train_retro.py — train RetroGPT on tiny-shakespeare with random retrieval
================================================================================

GOAL OF THIS SCRIPT
  Prove the RETRO architecture works mechanically. Three things must be true:
    1. The forward pass runs without shape errors.
    2. Training loss decreases monotonically (model is learning).
    3. The model is ACTUALLY USING the retrieval pathway — not just ignoring
       it and behaving like vanilla GPT.

WHY RANDOM NEIGHBORS AT TRAINING TIME
  Real RETRO trains on (chunk, semantically-similar-neighbors) pairs. Building
  that pipeline requires precomputing neighbors for every training chunk in
  the corpus — a separate big preprocessing job. For Phase 1 we use RANDOM
  chunks of shakespeare as neighbors. They don't help with prediction, so
  the model can choose to ignore them. The point isn't accuracy — it's
  proving the math is right and gradients flow through CCA.

THE PROOF THAT THE MODEL USES RETRIEVAL
  At the end of training, we evaluate the same trained model on the same
  validation batches under three conditions:

    (a) NO RETRIEVAL    — neighbors=None, model behaves like vanilla GPT
    (b) RANDOM RETRIEVAL — what we trained with: random chunks
    (c) ORACLE RETRIEVAL — neighbor = actual continuation of the chunk
                           (literally "here is the answer, please use it")

  If condition (c) has substantially lower loss than (a), the model is using
  the retrieval pathway. Magnitude of the gap tells us HOW MUCH it's learned
  to use it. Condition (b) ≈ (a) confirms random retrieval doesn't help.
================================================================================
"""

import os
import time
import math
import numpy as np
import torch

from model_retro import RetroConfig, RetroGPT


# ---------------- Config (small, fits easily on RTX 3070) ----------------
DATA_DIR = os.path.join("data", "shakespeare_char")
OUT_DIR = "out-retro-shakespeare"
os.makedirs(OUT_DIR, exist_ok=True)

# Model
config = RetroConfig(
    block_size=256,
    vocab_size=65,        # tiny-shakespeare char vocab
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=False,
    chunk_size=64,        # K = 256/64 = 4 chunks per sequence
    n_neighbors=2,        # 2 neighbors retrieved per chunk
    neighbor_len=64,      # each neighbor is 64 tokens long
    cca_every=2,          # CCA in layers {1, 3, 5}
)

# Training
BATCH_SIZE = 32
MAX_ITERS = 1000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
EVAL_INTERVAL = 100       # report train loss every N iters
GRAD_CLIP = 1.0
SEED = 1337

# RETRIEVAL TRAINING REGIME
# 0.0 = always random neighbors (model ignores retrieval — proven in Phase 1a)
# 1.0 = always oracle neighbors (model learns to fully rely on retrieval)
# 0.5 = balanced: half batches have signal, half don't. Model must learn both
#       normal language modeling AND when to use retrieval.
ORACLE_PROB = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
DTYPE = torch.bfloat16 if (DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
CTX = (
    torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
    if DEVICE_TYPE == "cuda"
    else torch.amp.autocast(device_type="cpu", enabled=False)
)


# ---------------- Data loading ----------------
def load_data():
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"data: train={len(train_data):,} tokens, val={len(val_data):,} tokens")
    return train_data, val_data


def get_batch(data: np.ndarray, batch_size: int, block_size: int):
    """Sample a batch of sequences. Returns (x, y) each shape (B, T)."""
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


def sample_random_neighbors(data: np.ndarray, batch_size: int, K: int, k: int, Ln: int):
    """
    Sample random chunks from the corpus to act as 'retrieved' neighbors.
    Shape: (B, K, k, Ln)
    """
    total = batch_size * K * k
    ix = np.random.randint(0, len(data) - Ln - 1, size=total)
    chunks = np.stack([data[i : i + Ln].astype(np.int64) for i in ix])
    return torch.from_numpy(chunks).view(batch_size, K, k, Ln)


def sample_oracle_neighbors(data: np.ndarray, ix_base: np.ndarray, K: int, k: int, L: int, Ln: int):
    """
    STRONG-SIGNAL ORACLE for diagnostic purposes:
      neighbor for chunk i = the literal TARGETS of chunk i
                           = data[ix_base + i*L + 1 : ix_base + i*L + 1 + Ln]

    Position p of chunk i predicts data[ix_base + i*L + p + 1], which equals
    neighbor[p]. So if the CCA pathway works at all, the model can learn to
    simply copy neighbor[p] to its output at position p of chunk i.

    This is the "is the wiring connected" test, not a realistic retrieval task.

    Shape: (B, K, k, Ln). All k slots get the same chunk (no diversity needed
    for the diagnostic).
    """
    B = len(ix_base)
    out = np.zeros((B, K, k, Ln), dtype=np.int64)
    for b in range(B):
        for i in range(K):
            start = ix_base[b] + i * L + 1   # targets are shifted by 1 from inputs
            if start + Ln > len(data):
                start = len(data) - Ln - 1
            chunk = data[start : start + Ln].astype(np.int64)
            for kk in range(k):
                out[b, i, kk, :] = chunk
    return torch.from_numpy(out)


def get_train_batch(data, batch_size, cfg: RetroConfig, oracle_prob: float):
    """
    Training batch with MIXED neighbors:
      - with probability `oracle_prob`: oracle neighbors (real continuations)
      - otherwise: random neighbors (no signal)
    The mix is decided per-batch. Over many batches the model sees both kinds
    and learns when retrieval is worth attending to.
    """
    # Sample sequences first so we know the base indices (needed for oracle).
    ix = np.random.randint(0, len(data) - cfg.block_size - 1, size=batch_size)
    x = np.stack([data[i : i + cfg.block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + cfg.block_size].astype(np.int64) for i in ix])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    K = cfg.block_size // cfg.chunk_size
    if np.random.rand() < oracle_prob:
        nbrs = sample_oracle_neighbors(
            data, ix, K, cfg.n_neighbors, cfg.chunk_size, cfg.neighbor_len
        )
    else:
        nbrs = sample_random_neighbors(data, batch_size, K, cfg.n_neighbors, cfg.neighbor_len)
    return x.to(DEVICE), y.to(DEVICE), nbrs.to(DEVICE)


# ---------------- Train ----------------
def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_data, val_data = load_data()

    model = RetroGPT(config).to(DEVICE)
    optimizer = model.configure_optimizers(
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        betas=BETAS,
        device_type=DEVICE_TYPE,
    )

    print(f"\ntraining: {MAX_ITERS} iters, batch={BATCH_SIZE}, lr={LEARNING_RATE}, oracle_prob={ORACLE_PROB}")
    print("=" * 70)

    model.train()
    t0 = time.time()
    running = []
    for it in range(MAX_ITERS):
        x, y, nbrs = get_train_batch(train_data, BATCH_SIZE, config, ORACLE_PROB)
        with CTX:
            _, loss = model(x, targets=y, neighbors=nbrs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        running.append(loss.item())
        if (it + 1) % EVAL_INTERVAL == 0:
            avg = sum(running) / len(running)
            running = []
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {it+1:>5d}  loss {avg:.4f}  ({EVAL_INTERVAL} iters in {dt:.1f}s)")

    print("=" * 70)
    print("training done. saving checkpoint.")
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
    }, os.path.join(OUT_DIR, "ckpt.pt"))

    return model, train_data, val_data


# ---------------- The interesting eval: does the model USE retrieval? ----------------
@torch.no_grad()
def eval_three_conditions(model, val_data, n_batches: int = 20):
    """
    Compare validation loss under:
      (a) no retrieval        — neighbors=None
      (b) random retrieval    — same distribution as training
      (c) oracle retrieval    — neighbor = actual continuation of each chunk

    A successful RETRO model should have:    loss(c) << loss(a)
    A model that learned to ignore retrieval: loss(c) ≈ loss(a)
    """
    model.eval()
    K = config.block_size // config.chunk_size

    losses_a, losses_b, losses_c = [], [], []
    print(f"\nevaluating retrieval contribution over {n_batches} batches...")

    for _ in range(n_batches):
        # Sample one set of base indices so all three conditions use the same data.
        ix = np.random.randint(0, len(val_data) - config.block_size - 1, size=BATCH_SIZE)
        x = np.stack([val_data[i : i + config.block_size].astype(np.int64) for i in ix])
        y = np.stack([val_data[i + 1 : i + 1 + config.block_size].astype(np.int64) for i in ix])
        x = torch.from_numpy(x).to(DEVICE)
        y = torch.from_numpy(y).to(DEVICE)

        # (a) no retrieval
        with CTX:
            _, la = model(x, targets=y, neighbors=None)
        # (b) random retrieval
        nbrs_rand = sample_random_neighbors(val_data, BATCH_SIZE, K, config.n_neighbors, config.neighbor_len).to(DEVICE)
        with CTX:
            _, lb = model(x, targets=y, neighbors=nbrs_rand)
        # (c) oracle retrieval — the actual continuation of each chunk
        nbrs_oracle = sample_oracle_neighbors(
            val_data, ix, K, config.n_neighbors, config.chunk_size, config.neighbor_len
        ).to(DEVICE)
        with CTX:
            _, lc = model(x, targets=y, neighbors=nbrs_oracle)

        losses_a.append(la.item())
        losses_b.append(lb.item())
        losses_c.append(lc.item())

    a = sum(losses_a) / len(losses_a)
    b = sum(losses_b) / len(losses_b)
    c = sum(losses_c) / len(losses_c)

    print("\n" + "=" * 70)
    print("VALIDATION LOSS BY RETRIEVAL CONDITION")
    print("=" * 70)
    print(f"  (a) no retrieval          : {a:.4f}")
    print(f"  (b) random retrieval      : {b:.4f}")
    print(f"  (c) oracle retrieval      : {c:.4f}")
    print("-" * 70)
    print(f"  oracle vs no retrieval    : {a - c:+.4f}   (positive = retrieval helps)")
    print(f"  random vs no retrieval    : {a - b:+.4f}   (should be ~0)")
    print("=" * 70)

    if a - c > 0.1:
        print("\n[OK] Model is USING the retrieval pathway. CCA layers learned something.")
    elif a - c > 0.02:
        print("\n[WEAK] Small effect — model partially uses retrieval. More training would help.")
    else:
        print("\n[NULL] Model is essentially ignoring retrieval. Either training was too short,")
        print("       or the random-neighbor signal at training time gave it no reason to learn.")


def main():
    model, train_data, val_data = train()
    eval_three_conditions(model, val_data, n_batches=20)


if __name__ == "__main__":
    main()

"""
================================================================================
eval_bank_scale.py — Does a bigger bank mean better retrieval?
================================================================================

Tests the held-out retrieval benefit at different bank sizes by searching
against subsets of the full bank (first N cells). Re-embeds held-out chunks
and searches against the subsampled bank for each size.

Sizes: 10%, 25%, 50%, 75%, 100% of the 1.8M cell bank.

USAGE:
  cd H:\\MiniLM\\nanogpt
  python eval_bank_scale.py [--n-batches 50] [--batch-size 8]
================================================================================
"""

import argparse
import math
import sqlite3
import struct
import time
from pathlib import Path

# Must import before numpy/torch to avoid pyarrow DLL segfault on Windows
try:
    import datasets  # noqa: F401
except ImportError:
    pass
from transformers import AutoTokenizer, AutoModel  # noqa: E402

import numpy as np
import torch
import tiktoken

from model_retro import RetroConfig, RetroGPT

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data" / "bank"
CKPT_PATH = _SCRIPT_DIR / "out-retro-bank" / "ckpt_best.pt"
DB_PATH = _SCRIPT_DIR.parent / "cc_service" / "bank.db"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
DTYPE = torch.bfloat16 if (DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
CTX = (
    torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
    if DEVICE_TYPE == "cuda"
    else torch.amp.autocast(device_type="cpu", enabled=False)
)
SEED = 42


def stderr(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(var / n)


def load_all_bank_weights(db_path, dim=384):
    """Load ALL weight vectors from bank.db (streaming cursor).
    Uses JOIN source_texts to match cell_tokens.npy alignment."""
    conn = sqlite3.connect(str(db_path))
    n_total = conn.execute(
        "SELECT COUNT(*) FROM cells c "
        "JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single'"
    ).fetchone()[0]
    print(f"  loading {n_total:,} cell weights from bank...")

    w = np.zeros((n_total, dim), dtype=np.float32)
    cur = conn.execute(
        "SELECT c.weight FROM cells c "
        "JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single' ORDER BY c.id"
    )
    for i, (blob,) in enumerate(cur):
        w[i] = np.frombuffer(blob, dtype=np.float32)
        if (i + 1) % 500_000 == 0:
            print(f"    loaded {i+1:>9,}/{n_total:,}")
    conn.close()
    print(f"  bank weights: {w.nbytes / 1e6:.0f} MB")
    return w


def load_whitening(db_path, dim=384):
    """Load whitening params from bank.db."""
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT mu, w_matrix, max_norm FROM whitening LIMIT 1").fetchone()
    conn.close()
    mu = torch.tensor(struct.unpack(f"{dim}f", row[0]))
    W = torch.tensor(struct.unpack(f"{dim*dim}f", row[1])).reshape(dim, dim)
    max_norm = row[2]
    return mu, W, max_norm


def embed_chunks(chunks_text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Embed text chunks with MiniLM, apply whitening, return on GPU.
    Must match prepare_heldout.py exactly: (raw - mu) @ W / max_norm."""

    # Use the same mean-pooling as prepare_heldout.py
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    st_model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    mu, W, max_norm = load_whitening(DB_PATH)
    mu = mu.to(DEVICE)
    W = W.to(DEVICE)

    # Batch encode
    all_embs = []
    BATCH = 256
    for i in range(0, len(chunks_text), BATCH):
        batch = chunks_text[i:i + BATCH]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = st_model(**inputs)
        attn = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (outputs.last_hidden_state * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1e-9)
        raw = (summed / counts).float()  # (B, 384)

        # Apply whitening — same as prepare_heldout.py
        centered = raw - mu[None, :]
        whitened = centered @ W
        q = whitened / (max_norm + 1e-8)
        all_embs.append(q)

    del st_model, tokenizer
    torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default=str(CKPT_PATH))
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model checkpoint
    print(f"loading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = RetroGPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load held-out data
    tokens = np.memmap(DATA_DIR / "heldout.bin", dtype=np.uint16, mode="r")
    cell_tokens = np.load(DATA_DIR / "cell_tokens.npy", mmap_mode="r")
    n_total_cells = len(cell_tokens)

    K = config.block_size // config.chunk_size
    Ln = config.neighbor_len
    max_token_start = len(tokens) - config.block_size - 1
    max_chunk_idx = max_token_start // config.chunk_size

    enc = tiktoken.get_encoding("gpt2")

    # Pre-select chunk indices (same across all bank sizes)
    all_chunk_indices = []
    for _ in range(args.n_batches):
        all_chunk_indices.append(
            np.random.randint(0, max_chunk_idx - K, size=args.batch_size)
        )

    # Collect all unique chunk texts we need to embed
    print("collecting held-out chunk texts for embedding...")
    unique_chunks = set()
    for batch_ci in all_chunk_indices:
        for c in batch_ci:
            for ki in range(K):
                unique_chunks.add(int(c) + ki)
    unique_chunks = sorted(unique_chunks)
    chunk_to_idx = {c: i for i, c in enumerate(unique_chunks)}

    chunk_texts = []
    for c in unique_chunks:
        tstart = c * config.chunk_size
        toks = tokens[tstart:tstart + config.chunk_size].tolist()
        chunk_texts.append(enc.decode(toks))

    print(f"  {len(chunk_texts)} unique chunks to embed")

    # Embed all held-out chunks with MiniLM
    print("embedding held-out chunks with MiniLM...")
    t0 = time.time()
    chunk_embeddings = embed_chunks(chunk_texts)  # (N, 384) on GPU
    print(f"  done in {time.time() - t0:.1f}s")

    # Free MiniLM memory
    torch.cuda.empty_cache()

    # Load ALL bank weights once from SQLite
    print("\nloading full bank weights (once)...")
    t0 = time.time()
    all_bank_w = load_all_bank_weights(DB_PATH)
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Bank sizes to test
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    bank_sizes = [int(n_total_cells * f) for f in fractions]

    results = {}

    for n_cells in bank_sizes:
        pct = n_cells / n_total_cells * 100
        print(f"\n{'=' * 60}")
        print(f"Bank size: {n_cells:,} cells ({pct:.0f}%)")
        print(f"{'=' * 60}")

        # Subsample: take first n_cells rows, move to GPU
        bank_w = torch.from_numpy(all_bank_w[:n_cells]).to(DEVICE)

        # For each held-out chunk, find top-2 neighbors in this bank subset
        print(f"  searching neighbors...")
        t0 = time.time()
        n_emb = chunk_embeddings.shape[0]
        neighbors_map = {}

        SEARCH_BATCH = 2048
        for si in range(0, n_emb, SEARCH_BATCH):
            ei = min(si + SEARCH_BATCH, n_emb)
            q = chunk_embeddings[si:ei]  # (batch, 384)
            sims = q @ bank_w.T  # (batch, n_cells)
            topk_vals, topk_idx = torch.topk(sims, k=config.n_neighbors, dim=1)
            for bi in range(ei - si):
                global_ci = unique_chunks[si + bi]
                neighbors_map[global_ci] = topk_idx[bi].cpu().numpy()

        del bank_w
        torch.cuda.empty_cache()
        print(f"  search done in {time.time() - t0:.1f}s")

        # Evaluate
        losses_real = []
        losses_none = []

        for bi, batch_ci in enumerate(all_chunk_indices):
            B = len(batch_ci)
            x = np.empty((B, config.block_size), dtype=np.int64)
            y = np.empty((B, config.block_size), dtype=np.int64)
            nbrs = np.empty((B, K, config.n_neighbors, Ln), dtype=np.int64)

            for b, c in enumerate(batch_ci):
                tstart = int(c) * config.chunk_size
                x[b] = tokens[tstart:tstart + config.block_size].astype(np.int64)
                y[b] = tokens[tstart + 1:tstart + 1 + config.block_size].astype(np.int64)
                for ki in range(K):
                    gc = int(c) + ki
                    nbr_idx = neighbors_map[gc]
                    # Clamp to valid cell_tokens range
                    nbr_idx = np.clip(nbr_idx, 0, n_total_cells - 1)
                    nbrs[b, ki] = cell_tokens[nbr_idx].astype(np.int64)

            x_t = torch.from_numpy(x).to(DEVICE)
            y_t = torch.from_numpy(y).to(DEVICE)
            nbrs_t = torch.from_numpy(nbrs).to(DEVICE)

            with CTX:
                _, loss_real = model(x_t, targets=y_t, neighbors=nbrs_t)
                _, loss_none = model(x_t, targets=y_t, neighbors=None)

            losses_real.append(loss_real.item())
            losses_none.append(loss_none.item())

        mean_real = sum(losses_real) / len(losses_real)
        mean_none = sum(losses_none) / len(losses_none)
        gap = mean_none - mean_real
        se_real = stderr(losses_real)

        results[n_cells] = {
            "none": mean_none, "real": mean_real,
            "gap": gap, "se": se_real, "pct": pct,
        }
        print(f"  none={mean_none:.4f}  real={mean_real:.4f}  gap={gap:+.4f}")

    # Summary
    print(f"\n{'=' * 72}")
    print("BANK SCALING RESULTS")
    print(f"{'=' * 72}")
    print(f"  {'Bank Size':>12s}  {'%':>5s}  {'None':>7s}  {'Real':>7s}  {'Gap':>7s}  {'±se':>6s}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
    for n_cells in bank_sizes:
        r = results[n_cells]
        print(f"  {n_cells:>12,}  {r['pct']:>4.0f}%  "
              f"{r['none']:.4f}  {r['real']:.4f}  "
              f"{r['gap']:+.4f}  ±{r['se']:.4f}")

    # Scaling analysis
    gaps = [results[n]["gap"] for n in bank_sizes]
    if len(gaps) >= 2:
        improvement = (gaps[-1] - gaps[0]) / gaps[0] * 100
        print(f"\n  Gap improvement from {bank_sizes[0]:,} to {bank_sizes[-1]:,} cells: "
              f"{improvement:+.0f}%")
        print(f"  Gap at 10%: {gaps[0]:.4f}, at 100%: {gaps[-1]:.4f}")

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()

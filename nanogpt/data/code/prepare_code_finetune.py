"""
================================================================================
prepare_code_finetune.py — Prepare code training data for RETRO fine-tuning
================================================================================

Downloads a code instruction dataset, tokenizes it with GPT-2 BPE, splits
into train/val, and precomputes neighbors against the existing wiki bank.

The model will be fine-tuned on CODE tokens while retrieving neighbors from
the WIKI bank — testing whether RETRO can adapt to a new domain while
leveraging its existing knowledge base.

OUTPUTS (data/code/):
  train_code.bin              uint16, code training tokens
  val_code.bin                uint16, code validation tokens
  train_code_neighbors.npy    uint32, (n_train_chunks, 2) wiki neighbor indices
  val_code_neighbors.npy      uint32, (n_val_chunks, 2) wiki neighbor indices
  finetune_meta.json          metadata

PREREQUISITES:
  - data/bank/cell_tokens.npy must exist (wiki bank, from precompute_neighbors.py)
  - H:\\MiniLM\\cc_service\\bank.db must exist (for whitening + bank weights)
  - Stop cc_service before running to free GPU memory

USAGE:
  cd H:\\MiniLM\\nanogpt
  python data/code/prepare_code_finetune.py
================================================================================
"""

import json
import sqlite3
import time
from pathlib import Path

# datasets MUST be imported before numpy/torch to avoid pyarrow DLL segfault
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

import numpy as np
import tiktoken
import torch

# ---- Paths ----
BANK_DB = r"H:\MiniLM\cc_service\bank.db"
SCRIPT_DIR = Path(__file__).resolve().parent
BANK_DIR = SCRIPT_DIR.parent / "bank"
OUT_DIR = SCRIPT_DIR  # data/code/

# ---- Config ----
CHUNK_SIZE = 64
N_NEIGHBORS = 2
ENCODE_BATCH = 256
MIN_CODE_CHARS = 60
VAL_FRACTION = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# MiniLM encoder (same as precompute_neighbors.py / prepare_code_experiment.py)
# ==============================================================================
class MiniLMEncoder:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str):
        self.device = device
        print(f"loading MiniLM ({self.MODEL_NAME}) on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 128) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        attn = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (outputs.last_hidden_state * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1e-9)
        return summed / counts


# ==============================================================================
# Helpers
# ==============================================================================
def load_whitening(conn):
    row = conn.execute(
        "SELECT mu, w_matrix, max_norm FROM whitening WHERE id=1"
    ).fetchone()
    mu = np.frombuffer(row[0], dtype=np.float32).reshape(384).copy()
    W = np.frombuffer(row[1], dtype=np.float32).reshape(384, 384).copy()
    max_norm = float(row[2])
    return mu, W, max_norm


def load_bank_weights(conn):
    """Load whitened wiki cell weights — same JOIN as precompute_neighbors.py."""
    n_total = conn.execute(
        "SELECT COUNT(*) FROM cells c JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single'"
    ).fetchone()[0]
    print(f"loading {n_total:,} wiki cell weights from bank...")

    W_bank = np.zeros((n_total, 384), dtype=np.float32)
    cur = conn.execute(
        "SELECT c.id, c.weight FROM cells c "
        "JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single' ORDER BY c.id"
    )
    for i, (cid, w_blob) in enumerate(cur):
        W_bank[i] = np.frombuffer(w_blob, dtype=np.float32)
        if (i + 1) % 500000 == 0:
            print(f"  loaded {i+1:>9,}/{n_total:,}")

    print(f"  wiki bank weights: {W_bank.nbytes / 1e6:.0f} MB")
    return W_bank


def precompute_neighbors(chunks_tokens, encoder, mu_gpu, W_white_gpu,
                          max_norm, W_bank_gpu, tok_enc, label=""):
    """Embed chunks and find top-K nearest neighbors in wiki bank.

    No self-retrieval filtering needed — code training data is completely
    separate from the wiki bank.
    """
    n_chunks = len(chunks_tokens)
    neighbors = np.zeros((n_chunks, N_NEIGHBORS), dtype=np.uint32)

    print(f"  precomputing neighbors for {n_chunks:,} {label} chunks...")
    t0 = time.time()

    for batch_start in range(0, n_chunks, ENCODE_BATCH):
        batch_end = min(batch_start + ENCODE_BATCH, n_chunks)
        texts = []
        for ci in range(batch_start, batch_end):
            texts.append(tok_enc.decode(chunks_tokens[ci].tolist()))

        raw = encoder.encode(texts).float()
        centered = raw - mu_gpu[None, :]
        whitened = centered @ W_white_gpu
        q = whitened / (max_norm + 1e-8)

        sims = q @ W_bank_gpu.T
        top_idx = sims.topk(N_NEIGHBORS, dim=1).indices
        neighbors[batch_start:batch_end] = top_idx.cpu().numpy()

        if batch_start % (ENCODE_BATCH * 20) == 0 and batch_start > 0:
            elapsed = time.time() - t0
            rate = batch_end / max(elapsed, 0.01)
            eta = (n_chunks - batch_end) / max(rate, 1)
            print(f"    {batch_end:>8,}/{n_chunks:,} ({100*batch_end/n_chunks:5.1f}%)  "
                  f"{rate:.0f} chunks/s  ETA {eta:.0f}s")

    print(f"    done in {time.time() - t0:.0f}s")
    return neighbors


# ==============================================================================
# Main pipeline
# ==============================================================================
def main():
    t_start = time.time()
    tok_enc = tiktoken.get_encoding("gpt2")
    eot = tok_enc.eot_token

    # ==================================================================
    # PART 1: Load code dataset
    # ==================================================================
    print("=" * 72)
    print("PART 1: Loading code training dataset")
    print("=" * 72)

    dataset_name = None
    ds = None

    # Primary: Evol-Instruct-Code-80k (large, diverse code instructions)
    try:
        print("trying nickrosh/Evol-Instruct-Code-80k-v1...")
        ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
        dataset_name = "nickrosh/Evol-Instruct-Code-80k-v1"

        def get_text(item):
            inst = (item.get("instruction") or "").strip()
            out = (item.get("output") or "").strip()
            return f"{inst}\n\n{out}" if inst else out

        print(f"  loaded {len(ds):,} items from {dataset_name}")
    except Exception as e1:
        print(f"  failed: {e1}")

    # Fallback: first 13k from python_code_instructions (known to work)
    if ds is None:
        print("trying iamtarun/python_code_instructions_18k_alpaca (first 13k)...")
        ds_full = load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca", split="train"
        )
        # Last 5000 used for held-out eval — use first 13000 for training.
        n_avail = len(ds_full)
        n_train_items = max(1, n_avail - 5000)
        ds = ds_full.select(range(n_train_items))
        dataset_name = f"iamtarun/python_code_instructions_18k_alpaca (first {n_train_items})"

        def get_text(item):
            prompt = (item.get("prompt") or "").strip()
            out = (item.get("output") or "").strip()
            return f"{prompt}\n\n{out}" if prompt else out

        print(f"  loaded {len(ds):,} items from {dataset_name}")

    # ==================================================================
    # PART 2: Tokenize
    # ==================================================================
    print(f"\n{'=' * 72}")
    print("PART 2: Tokenizing with GPT-2 BPE")
    print("=" * 72)

    all_tokens = []
    n_used = 0
    n_skipped = 0
    for item in ds:
        text = get_text(item)
        if len(text) < MIN_CODE_CHARS:
            n_skipped += 1
            continue
        if len(text) > 3000:
            text = text[:3000]
        tokens = tok_enc.encode_ordinary(text)
        tokens.append(eot)
        all_tokens.extend(tokens)
        n_used += 1
        if n_used % 10000 == 0:
            print(f"  {n_used:,} items, {len(all_tokens):,} tokens")

    arr = np.array(all_tokens, dtype=np.uint16)
    n_total_tokens = len(arr)
    n_total_chunks = n_total_tokens // CHUNK_SIZE
    arr = arr[:n_total_chunks * CHUNK_SIZE]

    print(f"  {n_used:,} items used, {n_skipped:,} skipped")
    print(f"  {n_total_tokens:,} tokens -> {n_total_chunks:,} chunks of {CHUNK_SIZE}")

    # Train/val split
    n_val_chunks = max(1, int(n_total_chunks * VAL_FRACTION))
    n_train_chunks = n_total_chunks - n_val_chunks
    n_train_tokens = n_train_chunks * CHUNK_SIZE
    n_val_tokens = n_val_chunks * CHUNK_SIZE

    train_arr = arr[:n_train_tokens]
    val_arr = arr[n_train_tokens:n_train_tokens + n_val_tokens]

    train_arr.tofile(OUT_DIR / "train_code.bin")
    val_arr.tofile(OUT_DIR / "val_code.bin")

    print(f"  train: {n_train_tokens:,} tokens ({n_train_chunks:,} chunks)")
    print(f"  val:   {n_val_tokens:,} tokens ({n_val_chunks:,} chunks)")

    # Reshape for embedding
    train_chunks = train_arr.reshape(n_train_chunks, CHUNK_SIZE)
    val_chunks = val_arr.reshape(n_val_chunks, CHUNK_SIZE)
    del arr, all_tokens

    # ==================================================================
    # PART 3: Load wiki bank weights + encoder
    # ==================================================================
    print(f"\n{'=' * 72}")
    print("PART 3: Loading wiki bank weights + MiniLM encoder")
    print("=" * 72)

    conn = sqlite3.connect(f"file:{BANK_DB}?mode=ro", uri=True)
    mu, W_white, max_norm = load_whitening(conn)
    print(f"whitening: mu {mu.shape}, W {W_white.shape}, max_norm {max_norm:.4f}")

    W_bank = load_bank_weights(conn)
    n_bank_cells = len(W_bank)
    conn.close()

    # Move to GPU
    mu_gpu = torch.from_numpy(mu).to(DEVICE).float()
    W_white_gpu = torch.from_numpy(W_white).to(DEVICE).float()
    W_bank_gpu = torch.from_numpy(W_bank).to(DEVICE).float()
    del W_bank
    print(f"  bank on GPU: {n_bank_cells:,} cells, "
          f"{W_bank_gpu.nelement() * 4 / 1e6:.0f} MB")

    encoder = MiniLMEncoder(DEVICE)

    # ==================================================================
    # PART 4: Precompute neighbors
    # ==================================================================
    print(f"\n{'=' * 72}")
    print("PART 4: Precomputing wiki neighbors for code chunks")
    print("=" * 72)

    train_nbrs = precompute_neighbors(
        train_chunks, encoder, mu_gpu, W_white_gpu, max_norm,
        W_bank_gpu, tok_enc, label="train",
    )
    np.save(OUT_DIR / "train_code_neighbors.npy", train_nbrs)
    print(f"  saved train_code_neighbors.npy: {train_nbrs.shape}")

    val_nbrs = precompute_neighbors(
        val_chunks, encoder, mu_gpu, W_white_gpu, max_norm,
        W_bank_gpu, tok_enc, label="val",
    )
    np.save(OUT_DIR / "val_code_neighbors.npy", val_nbrs)
    print(f"  saved val_code_neighbors.npy: {val_nbrs.shape}")

    # ==================================================================
    # PART 5: Save metadata
    # ==================================================================
    meta = {
        "dataset": dataset_name,
        "n_items_used": n_used,
        "n_items_skipped": n_skipped,
        "n_train_tokens": int(n_train_tokens),
        "n_val_tokens": int(n_val_tokens),
        "n_train_chunks": int(n_train_chunks),
        "n_val_chunks": int(n_val_chunks),
        "chunk_size": CHUNK_SIZE,
        "n_neighbors": N_NEIGHBORS,
        "bank_source": "wiki-only (data/bank/cell_tokens.npy)",
        "n_bank_cells": int(n_bank_cells),
    }
    with open(OUT_DIR / "finetune_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved finetune_meta.json")

    total = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"DONE in {total:.0f}s")
    print(f"  train: {n_train_tokens:,} tokens, {n_train_chunks:,} chunks")
    print(f"  val:   {n_val_tokens:,} tokens, {n_val_chunks:,} chunks")
    print(f"  bank:  {n_bank_cells:,} wiki cells")
    print(f"\nReady for: python train_retro_code_finetune.py")
    print("=" * 72)


if __name__ == "__main__":
    main()

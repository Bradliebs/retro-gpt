"""
================================================================================
prepare_code_experiment.py — Domain-transfer experiment: code eval data
================================================================================

Tests whether the RETRO model (trained on Wikipedia) can transfer its CCA
skill to code — a domain it has never seen during training.

BANK CELLS (added to existing Wikipedia bank):
  CodeAlpaca-20k instruction/response pairs.  Tokenized with GPT-2 BPE,
  chunked into 64-token cells, embedded with MiniLM + whitening.

HELD-OUT EVAL DATA (zero overlap with bank):
  CodeSearchNet Python split — real Python functions with docstrings.
  Completely separate dataset from CodeAlpaca.

OUTPUTS (this directory, data/code/):
  cell_tokens.npy           uint16, combined wiki + code bank
  heldout_code.bin          uint16, held-out code tokens
  heldout_code_neighbors.npy  uint32, neighbor indices into combined cell_tokens
  heldout_code_meta.json    metadata

USAGE:
  cd H:\\MiniLM\\nanogpt
  python data/code/prepare_code_experiment.py

NOTE: Stop cc_service before running to free GPU memory.
================================================================================
"""

import json
import sqlite3
import time
from pathlib import Path

# NOTE: datasets must be imported BEFORE numpy/torch to avoid pyarrow DLL
# segfault on Windows (0xC0000005).
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

import numpy as np
import tiktoken
import torch

# ---- Paths ----
BANK_DB = r"H:\MiniLM\cc_service\bank.db"
SCRIPT_DIR = Path(__file__).resolve().parent
BANK_DIR = SCRIPT_DIR.parent / "bank"     # existing wiki cell_tokens lives here
OUT_DIR = SCRIPT_DIR                       # data/code/

# ---- Config ----
CHUNK_SIZE = 64
NEIGHBOR_LEN = 64
N_NEIGHBORS = 2
ENCODE_BATCH = 256
MIN_CODE_CHARS = 60         # minimum length for a code item
N_HELDOUT_FUNCS = 5000      # held-out CodeSearchNet functions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# MiniLM encoder (same as prepare_heldout.py)
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
    """Load whitened cell weights — same JOIN as precompute_neighbors.py."""
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


# ==============================================================================
# Main pipeline
# ==============================================================================
def main():
    t_start = time.time()
    tok_enc = tiktoken.get_encoding("gpt2")
    eot = tok_enc.eot_token  # 50256

    # ==================================================================
    # PART 1: Build code bank cells from CodeAlpaca-20k
    # ==================================================================
    print("=" * 72)
    print("PART 1: Building code bank cells from CodeAlpaca-20k")
    print("=" * 72)

    print("downloading CodeAlpaca-20k...")
    code_ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    print(f"  {len(code_ds):,} items")

    # Tokenize each item into chunks.
    code_tokens_list = []
    n_items_used = 0
    n_items_skipped = 0
    for item in code_ds:
        instruction = (item.get("instruction") or "").strip()
        output = (item.get("output") or "").strip()
        if len(output) < MIN_CODE_CHARS:
            n_items_skipped += 1
            continue
        # Combine instruction + output as one cell's text.
        if instruction:
            text = f"{instruction}\n\n{output}"
        else:
            text = output
        # Truncate very long items.
        if len(text) > 3000:
            text = text[:3000]

        tokens = tok_enc.encode_ordinary(text)
        tokens.append(eot)
        code_tokens_list.extend(tokens)
        n_items_used += 1

    code_tokens_arr = np.array(code_tokens_list, dtype=np.uint16)
    n_code_tokens = len(code_tokens_arr)
    n_code_chunks = n_code_tokens // CHUNK_SIZE
    code_tokens_arr = code_tokens_arr[:n_code_chunks * CHUNK_SIZE]
    print(f"  {n_items_used:,} items used, {n_items_skipped:,} skipped")
    print(f"  {n_code_tokens:,} tokens -> {n_code_chunks:,} chunks of {CHUNK_SIZE}")

    # Reshape into (n_code_chunks, CHUNK_SIZE) for cell_tokens.
    code_cell_tokens = code_tokens_arr.reshape(n_code_chunks, CHUNK_SIZE)
    del code_tokens_arr

    # ==================================================================
    # PART 2: Build combined bank (wiki + code cell_tokens)
    # ==================================================================
    print(f"\n{'=' * 72}")
    print("PART 2: Building combined bank (wiki + code)")
    print("=" * 72)

    wiki_cell_tokens = np.load(BANK_DIR / "cell_tokens.npy", mmap_mode="r")
    n_wiki = len(wiki_cell_tokens)
    print(f"  wiki cells:  {n_wiki:,}")
    print(f"  code cells:  {n_code_chunks:,}")

    # Concatenate: wiki cells first (indices 0..n_wiki-1), then code cells.
    combined_cell_tokens = np.vstack([wiki_cell_tokens, code_cell_tokens])
    n_combined = len(combined_cell_tokens)
    print(f"  combined:    {n_combined:,}")

    np.save(OUT_DIR / "cell_tokens.npy", combined_cell_tokens)
    print(f"  saved cell_tokens.npy ({combined_cell_tokens.nbytes / 1e6:.0f} MB)")

    # ==================================================================
    # PART 3: Embed code bank cells with MiniLM
    # ==================================================================
    print(f"\n{'=' * 72}")
    print("PART 3: Embedding code cells with MiniLM")
    print("=" * 72)

    # Load whitening from bank.db.
    conn = sqlite3.connect(f"file:{BANK_DB}?mode=ro", uri=True)
    mu, W_white, max_norm = load_whitening(conn)
    print(f"whitening: mu {mu.shape}, W {W_white.shape}, max_norm {max_norm:.4f}")

    # Load wiki bank weights.
    W_wiki = load_bank_weights(conn)
    conn.close()

    # Move to GPU.
    mu_gpu = torch.from_numpy(mu).to(DEVICE).float()
    W_white_gpu = torch.from_numpy(W_white).to(DEVICE).float()

    # Embed code cells.
    encoder = MiniLMEncoder(DEVICE)

    n_code = len(code_cell_tokens)
    W_code = np.zeros((n_code, 384), dtype=np.float32)
    print(f"\nembedding {n_code:,} code cells...")
    t_emb = time.time()

    for batch_start in range(0, n_code, ENCODE_BATCH):
        batch_end = min(batch_start + ENCODE_BATCH, n_code)
        texts = []
        for ci in range(batch_start, batch_end):
            texts.append(tok_enc.decode(code_cell_tokens[ci].tolist()))
        raw = encoder.encode(texts).float()
        centered = raw - mu_gpu[None, :]
        whitened = centered @ W_white_gpu
        q = whitened / (max_norm + 1e-8)
        W_code[batch_start:batch_end] = q.cpu().numpy()

        if batch_start % (ENCODE_BATCH * 10) == 0 and batch_start > 0:
            elapsed = time.time() - t_emb
            rate = batch_end / max(elapsed, 0.01)
            eta = (n_code - batch_end) / max(rate, 1)
            print(f"  {batch_end:>7,}/{n_code:,} ({100*batch_end/n_code:5.1f}%)  "
                  f"{rate:.0f} cells/s  ETA {eta:.0f}s")

    print(f"  code embedding done in {time.time() - t_emb:.0f}s")

    # Build combined bank weight matrix on GPU.
    W_wiki_gpu = torch.from_numpy(W_wiki).to(DEVICE).float()
    W_code_gpu = torch.from_numpy(W_code).to(DEVICE).float()
    W_combined_gpu = torch.cat([W_wiki_gpu, W_code_gpu], dim=0)
    del W_wiki, W_wiki_gpu, W_code, W_code_gpu
    print(f"  combined bank: {n_combined:,} cells, "
          f"{W_combined_gpu.shape[0] * W_combined_gpu.shape[1] * 4 / 1e6:.0f} MB on GPU")

    # ==================================================================
    # PART 4: Download + tokenize held-out code
    # ==================================================================
    print(f"\n{'=' * 72}")
    print("PART 4: Tokenizing held-out code (Python code instructions)")
    print("=" * 72)

    # Use iamtarun/python_code_instructions_18k_alpaca — standard Parquet
    # dataset with 18k Python instruction/response pairs. Completely separate
    # from CodeAlpaca-20k (different author, different generation).
    print("downloading python_code_instructions_18k_alpaca...")
    code_eval_ds = load_dataset(
        "iamtarun/python_code_instructions_18k_alpaca", split="train",
    )
    total_available = len(code_eval_ds)
    print(f"  {total_available:,} items available")

    # Take the LAST N_HELDOUT_FUNCS items to maximise distance from any
    # ordering similarity with CodeAlpaca (bank used the first items).
    start_idx = max(0, total_available - N_HELDOUT_FUNCS)

    all_code_tokens = []
    n_funcs = 0
    for i in range(start_idx, total_available):
        item = code_eval_ds[i]
        # Combine prompt + output for full context (like the bank cells).
        prompt = (item.get("prompt") or "").strip()
        output = (item.get("output") or "").strip()
        if len(output) < MIN_CODE_CHARS:
            continue
        func_code = f"{prompt}\n\n{output}" if prompt else output
        # Truncate very long items.
        if len(func_code) > 3000:
            func_code = func_code[:3000]
        tokens = tok_enc.encode_ordinary(func_code)
        tokens.append(eot)
        all_code_tokens.extend(tokens)
        n_funcs += 1
        if n_funcs % 1000 == 0:
            print(f"  {n_funcs:,} functions, {len(all_code_tokens):,} tokens")

    heldout_arr = np.array(all_code_tokens, dtype=np.uint16)
    n_heldout_tokens = len(heldout_arr)
    n_heldout_chunks = n_heldout_tokens // CHUNK_SIZE
    heldout_arr = heldout_arr[:n_heldout_chunks * CHUNK_SIZE]
    print(f"\nheld-out code: {n_funcs:,} functions")
    print(f"  {n_heldout_tokens:,} tokens -> {n_heldout_chunks:,} chunks of {CHUNK_SIZE}")

    heldout_arr.tofile(OUT_DIR / "heldout_code.bin")
    print(f"  saved heldout_code.bin ({heldout_arr.nbytes / 1e6:.1f} MB)")

    # ==================================================================
    # PART 5: Precompute neighbors for held-out code chunks
    # ==================================================================
    print(f"\n{'=' * 72}")
    print("PART 5: Precomputing neighbors for held-out code chunks")
    print("=" * 72)

    print(f"precomputing neighbors for {n_heldout_chunks:,} chunks "
          f"against {n_combined:,}-cell combined bank...")
    neighbors = np.zeros((n_heldout_chunks, N_NEIGHBORS), dtype=np.uint32)
    t_enc = time.time()

    for batch_start in range(0, n_heldout_chunks, ENCODE_BATCH):
        batch_end = min(batch_start + ENCODE_BATCH, n_heldout_chunks)
        texts = []
        for ci in range(batch_start, batch_end):
            s = ci * CHUNK_SIZE
            e = s + CHUNK_SIZE
            texts.append(tok_enc.decode(heldout_arr[s:e].tolist()))

        raw = encoder.encode(texts).float()
        centered = raw - mu_gpu[None, :]
        whitened = centered @ W_white_gpu
        q = whitened / (max_norm + 1e-8)

        activations = q @ W_combined_gpu.T
        top_idx = activations.topk(N_NEIGHBORS, dim=1).indices
        neighbors[batch_start:batch_end] = top_idx.cpu().numpy()

        if batch_start % (ENCODE_BATCH * 10) == 0:
            elapsed = time.time() - t_enc
            done = batch_end
            rate = done / max(elapsed, 0.01)
            eta = (n_heldout_chunks - done) / max(rate, 1)
            print(f"  {done:>7,}/{n_heldout_chunks:,} "
                  f"({100*done/n_heldout_chunks:5.1f}%)  "
                  f"{rate:.0f} chunks/s  ETA {eta:.0f}s")

    print(f"  neighbor precompute done in {time.time() - t_enc:.0f}s")

    np.save(OUT_DIR / "heldout_code_neighbors.npy", neighbors)
    print(f"  saved heldout_code_neighbors.npy: shape {neighbors.shape}")

    # ==================================================================
    # PART 6: Save metadata
    # ==================================================================
    meta = {
        "bank_source_wiki": "wikimedia/wikipedia:20231101.en + 20231101.simple",
        "bank_source_code": "sahil2801/CodeAlpaca-20k",
        "heldout_source": "iamtarun/python_code_instructions_18k_alpaca (last 5k)",
        "n_wiki_cells": int(n_wiki),
        "n_code_cells": int(n_code_chunks),
        "n_combined_cells": int(n_combined),
        "n_code_items_used": int(n_items_used),
        "n_heldout_functions": int(n_funcs),
        "n_heldout_tokens_raw": int(n_heldout_tokens),
        "n_heldout_tokens_used": int(len(heldout_arr)),
        "n_heldout_chunks": int(n_heldout_chunks),
        "chunk_size": CHUNK_SIZE,
        "neighbor_len": NEIGHBOR_LEN,
        "n_neighbors": N_NEIGHBORS,
    }
    with open(OUT_DIR / "heldout_code_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved heldout_code_meta.json")

    total_time = time.time() - t_start
    print(f"\ntotal time: {total_time:.0f}s")


if __name__ == "__main__":
    main()

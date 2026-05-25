"""
================================================================================
prepare_heldout.py — Held-out Wikipedia eval data for retrieval leakage test
================================================================================

Downloads 5,000 English Wikipedia articles that were NEVER loaded into the
bank (articles 100,000–104,999 in the 20231101.en dataset; the bank contains
articles 0–99,999). Tokenizes them with GPT-2 BPE and precomputes real
MiniLM-retrieved neighbors against the existing bank.

This data lets us run a 3-condition eval on text the model has never seen
during training, and whose source text is not in the bank at all — a clean
test of whether the retrieval gap is genuine or information leakage.

INPUTS:
  - Wikipedia 20231101.en from HuggingFace (cached after first download)
  - bank.db (read-only) for whitening params + cell embeddings

OUTPUTS (this directory):
  heldout.bin              uint16 token stream
  heldout_neighbors.npy    uint32, shape (n_chunks, N_NEIGHBORS)
  heldout_meta.json        counts and config

USAGE:
  cd H:\\MiniLM\\nanogpt
  python data/bank/prepare_heldout.py

NOTE: Stop cc_service before running to avoid GPU memory contention.
      The bank weight matrix (~2.6 GB) + MiniLM (~80 MB) need GPU headroom.
================================================================================
"""

import json
import sqlite3
import time
from pathlib import Path

# NOTE: datasets must be imported BEFORE numpy/torch to avoid a pyarrow DLL
# load-order segfault on Windows (0xC0000005) in this venv.
from datasets import load_dataset          # noqa: E402 — order matters
from transformers import AutoTokenizer, AutoModel

import numpy as np
import tiktoken
import torch


# ---- Paths ----
BANK_DB = r"H:\MiniLM\cc_service\bank.db"
OUT_DIR = Path(__file__).parent

# ---- Heldout range ----
# The bank was built from articles 0–99,999 of 20231101.en (plus all of
# simple English Wikipedia, which is a separate dataset).
SKIP_ARTICLES = 100_000
N_ARTICLES = 5_000
MIN_CHARS = 80          # same threshold as load_wikipedia_en.py

# ---- Retrieval config (must match precompute_neighbors.py / trainer) ----
CHUNK_SIZE = 64
NEIGHBOR_LEN = 64
N_NEIGHBORS = 2
ENCODE_BATCH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# MiniLM encoder (same as precompute_neighbors.py — avoid sentence_transformers)
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
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        attn = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (outputs.last_hidden_state * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # (B, 384)


# ==============================================================================
# Helpers
# ==============================================================================
def split_paragraphs(text: str, min_chars: int) -> list[str]:
    """Split article text into paragraphs, keeping only substantial ones."""
    chunks = []
    for para in text.split("\n\n"):
        clean = " ".join(para.strip().split())
        if len(clean) >= min_chars:
            chunks.append(clean)
    return chunks


def load_whitening(conn):
    row = conn.execute(
        "SELECT mu, w_matrix, max_norm FROM whitening WHERE id=1"
    ).fetchone()
    mu = np.frombuffer(row[0], dtype=np.float32).reshape(384).copy()
    W = np.frombuffer(row[1], dtype=np.float32).reshape(384, 384).copy()
    max_norm = float(row[2])
    return mu, W, max_norm


def load_bank_weights(conn):
    """Load whitened cell weights from bank (for MIPS). Returns (cell_ids, W_bank).

    MUST use the same JOIN as precompute_neighbors.py so that positional
    indices align with cell_tokens.npy (which only covers cells that have
    both a weight AND source_text).
    """
    n_total = conn.execute(
        "SELECT COUNT(*) FROM cells c JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single'"
    ).fetchone()[0]
    print(f"loading {n_total:,} cell weights from bank...")

    cell_ids = np.zeros(n_total, dtype=np.uint32)
    W_bank = np.zeros((n_total, 384), dtype=np.float32)

    cur = conn.execute(
        "SELECT c.id, c.weight FROM cells c "
        "JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single' ORDER BY c.id"
    )
    for i, (cid, w_blob) in enumerate(cur):
        cell_ids[i] = cid
        W_bank[i] = np.frombuffer(w_blob, dtype=np.float32)
        if (i + 1) % 500000 == 0:
            print(f"  loaded {i+1:>9,}/{n_total:,}")

    print(f"  bank weights: {W_bank.nbytes/1e6:.0f} MB")
    return cell_ids, W_bank


# ==============================================================================
# Main pipeline
# ==============================================================================
def main():
    t_start = time.time()
    tok_enc = tiktoken.get_encoding("gpt2")
    eot = tok_enc.eot_token  # 50256

    # ---- Step 1: Stream held-out articles (avoids 20 GB full download) ----
    print(f"streaming English Wikipedia (20231101.en)...")
    print(f"  skipping first {SKIP_ARTICLES:,} articles, taking next {N_ARTICLES:,}")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True,
    )

    # ---- Step 2: Tokenize ----
    print(f"\ntokenizing held-out articles...")
    all_tokens = []
    n_articles_used = 0
    n_paragraphs = 0
    n_seen = 0
    for article in ds:
        n_seen += 1
        if n_seen <= SKIP_ARTICLES:
            if n_seen % 20000 == 0:
                print(f"  skipping... {n_seen:,}/{SKIP_ARTICLES:,}")
            continue
        if n_articles_used >= N_ARTICLES:
            break

        text = article.get("text", "")
        paragraphs = split_paragraphs(text, MIN_CHARS)
        if not paragraphs:
            continue
        n_articles_used += 1
        for para in paragraphs:
            tokens = tok_enc.encode_ordinary(para)
            tokens.append(eot)
            all_tokens.extend(tokens)
            n_paragraphs += 1
        if n_articles_used % 1000 == 0:
            print(f"  {n_articles_used} articles, {n_paragraphs} paragraphs, "
                  f"{len(all_tokens):,} tokens")

    tokens_arr = np.array(all_tokens, dtype=np.uint16)
    n_tokens = len(tokens_arr)
    n_chunks = n_tokens // CHUNK_SIZE
    # Trim to exact chunk boundary.
    tokens_arr = tokens_arr[:n_chunks * CHUNK_SIZE]
    print(f"\nheldout: {n_articles_used} articles, {n_paragraphs} paragraphs")
    print(f"  {n_tokens:,} tokens -> {n_chunks:,} chunks of {CHUNK_SIZE}")

    # Save token stream.
    tokens_arr.tofile(OUT_DIR / "heldout.bin")
    print(f"  saved heldout.bin ({tokens_arr.nbytes/1e6:.1f} MB)")

    # Free the dataset — we only need the token stream from here.
    del ds

    # ---- Step 3: Open bank for whitening + cell weights ----
    print(f"\nopening bank.db (read-only)...")
    conn = sqlite3.connect(f"file:{BANK_DB}?mode=ro", uri=True)
    mu, W_white, max_norm = load_whitening(conn)
    print(f"whitening: mu {mu.shape}, W {W_white.shape}, max_norm {max_norm:.4f}")

    _, W_bank = load_bank_weights(conn)
    conn.close()

    # ---- Step 4: Move to GPU ----
    print(f"\nmoving to {DEVICE}...")
    W_bank_gpu = torch.from_numpy(W_bank).to(DEVICE).float()
    mu_gpu = torch.from_numpy(mu).to(DEVICE).float()
    W_white_gpu = torch.from_numpy(W_white).to(DEVICE).float()
    del W_bank  # free CPU copy

    # ---- Step 5: MiniLM encoder ----
    encoder = MiniLMEncoder(DEVICE)

    # ---- Step 6: Precompute neighbors for each heldout chunk ----
    print(f"\nprecomputing neighbors for {n_chunks:,} chunks...")
    out = np.zeros((n_chunks, N_NEIGHBORS), dtype=np.uint32)
    t_enc = time.time()

    for batch_start in range(0, n_chunks, ENCODE_BATCH):
        batch_end = min(batch_start + ENCODE_BATCH, n_chunks)

        # Decode each chunk back to text for MiniLM.
        texts_batch = []
        for ci in range(batch_start, batch_end):
            s = ci * CHUNK_SIZE
            e = s + CHUNK_SIZE
            chunk_toks = tokens_arr[s:e].tolist()
            texts_batch.append(tok_enc.decode(chunk_toks))

        # MiniLM encode.
        raw = encoder.encode(texts_batch).float()  # (B, 384)

        # Apply whitening: ((raw - mu) @ W) / max_norm.
        centered = raw - mu_gpu[None, :]
        whitened = centered @ W_white_gpu
        q = whitened / (max_norm + 1e-8)  # (B, 384)

        # Dot product against entire bank.
        activations = q @ W_bank_gpu.T  # (B, N_cells)

        # Top-k (no self-retrieval filtering needed — heldout text is not in bank).
        top_idx = activations.topk(N_NEIGHBORS, dim=1).indices  # (B, N_NEIGHBORS)
        out[batch_start:batch_end] = top_idx.cpu().numpy()

        if batch_start % (ENCODE_BATCH * 10) == 0:
            elapsed = time.time() - t_enc
            done = batch_end
            rate = done / max(elapsed, 0.01)
            eta = (n_chunks - done) / max(rate, 1)
            print(f"  {done:>7,}/{n_chunks:,} "
                  f"({100*done/n_chunks:5.1f}%)  "
                  f"{rate:.0f} chunks/s  ETA {eta:.0f}s")

    elapsed = time.time() - t_enc
    print(f"  neighbor precompute done in {elapsed:.0f}s")

    # ---- Step 7: Save outputs ----
    np.save(OUT_DIR / "heldout_neighbors.npy", out)
    print(f"  saved heldout_neighbors.npy: shape {out.shape}")

    meta = {
        "dataset": "wikimedia/wikipedia:20231101.en",
        "skip_articles": SKIP_ARTICLES,
        "n_articles_requested": N_ARTICLES,
        "n_articles_used": n_articles_used,
        "n_paragraphs": n_paragraphs,
        "n_tokens_raw": n_tokens,
        "n_tokens_used": int(len(tokens_arr)),
        "n_chunks": n_chunks,
        "chunk_size": CHUNK_SIZE,
        "neighbor_len": NEIGHBOR_LEN,
        "n_neighbors": N_NEIGHBORS,
        "min_chars": MIN_CHARS,
    }
    with open(OUT_DIR / "heldout_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved heldout_meta.json")

    print(f"\ntotal time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

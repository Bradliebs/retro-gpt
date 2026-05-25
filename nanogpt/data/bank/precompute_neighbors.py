"""
================================================================================
precompute_neighbors.py — Stage B of Phase 2.2
================================================================================

For every chunk in train.bin and val.bin, find the top-k bank cells whose
whitened embedding has the highest dot-product activation against the chunk's
whitened embedding.

Matches the bank service's /query algorithm exactly:
  raw = MiniLM.encode(text, normalize_embeddings=False)
  q   = ((raw - mu) @ W_matrix) / max_norm
  activations = q @ cells.weight.T
  top_k = argmax(activations)

EXCLUDES self-retrievals: any cell whose cell_id appears in the chunk's
train_cells slice is skipped. Prevents trivial copy.

OUTPUTS (written to this directory):
  cell_ids.npy           uint32, shape (N_cells,)  — bank cell_ids, aligned with...
  cell_tokens.npy        uint16, shape (N_cells, NEIGHBOR_LEN)  — first NEIGHBOR_LEN
                         GPT-2 BPE tokens of each cell's text (pad with EOT)
  train_neighbors.npy    uint32, shape (n_train_chunks, N_NEIGHBORS)
                         indices INTO cell_ids/cell_tokens (not cell_ids directly)
  val_neighbors.npy      uint32, shape (n_val_chunks, N_NEIGHBORS)

At training time, training-script looks up:
  neighbor_token_array = cell_tokens[train_neighbors[chunk_idx, k]]
================================================================================
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import tiktoken
# NOTE: not using sentence_transformers — its __init__ greedily imports
# `datasets` -> `pyarrow.dataset`, which segfaults in this venv due to DLL
# contention with the running cc_service process. MiniLM is just a BERT-style
# transformer + mean pooling, so we load it directly via transformers.
from transformers import AutoTokenizer, AutoModel


BANK_DB = r"H:\MiniLM\cc_service\bank.db"
OUT_DIR = Path(__file__).parent

CHUNK_SIZE = 64          # must match RetroConfig.chunk_size
NEIGHBOR_LEN = 64        # must match RetroConfig.neighbor_len
N_NEIGHBORS = 2          # must match RetroConfig.n_neighbors
ENCODE_BATCH = 256       # batch size for MiniLM encode + retrieval matmul

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# MiniLM encoder — replicates sentence-transformers' encode() without importing
# sentence_transformers (avoiding the pyarrow.dataset DLL crash).
# ==============================================================================
class MiniLMEncoder:
    """all-MiniLM-L6-v2 via raw transformers. Mean-pools last_hidden_state
    with attention mask, returns (B, 384) float tensor. NO L2 normalization
    (matches cc_service which sets normalize_embeddings=False)."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str):
        self.device = device
        print(f"loading MiniLM ({self.MODEL_NAME}) on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 128) -> torch.Tensor:
        # MiniLM-L6 has max position 512 but model card recommends 128 for speed.
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        # Mean pooling: weight token vectors by attention mask, then average.
        attn = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (outputs.last_hidden_state * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1e-9)
        return summed / counts                                  # (B, 384)


def load_whitening(conn):
    """Pull mu, W_matrix, max_norm from the whitening table."""
    row = conn.execute(
        "SELECT mu, w_matrix, max_norm FROM whitening WHERE id=1"
    ).fetchone()
    mu = np.frombuffer(row[0], dtype=np.float32).reshape(384).copy()
    W = np.frombuffer(row[1], dtype=np.float32).reshape(384, 384).copy()
    max_norm = float(row[2])
    return mu, W, max_norm


def load_bank(conn, tok_enc, neighbor_len):
    """
    Stream all single-kind cells that have source_text. Return:
      cell_ids:    (N,) uint32
      W_bank:      (N, 384) float32     post-whitening embeddings
      cell_tokens: (N, neighbor_len) uint16   tokenized text, pad with EOT
    """
    eot = tok_enc.eot_token

    # Count first so we can preallocate.
    n_total = conn.execute(
        "SELECT COUNT(*) FROM cells c JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single'"
    ).fetchone()[0]
    print(f"loading {n_total:,} cells from bank...")

    cell_ids = np.zeros(n_total, dtype=np.uint32)
    W_bank = np.zeros((n_total, 384), dtype=np.float32)
    cell_tokens = np.full((n_total, neighbor_len), eot, dtype=np.uint16)

    cur = conn.execute(
        "SELECT c.id, c.weight, s.text FROM cells c "
        "JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single' ORDER BY c.id"
    )
    t0 = time.time()
    for i, (cid, w_blob, text) in enumerate(cur):
        cell_ids[i] = cid
        W_bank[i] = np.frombuffer(w_blob, dtype=np.float32)
        toks = tok_enc.encode_ordinary(text)[:neighbor_len]
        if toks:
            cell_tokens[i, : len(toks)] = toks
        if (i + 1) % 100000 == 0:
            dt = time.time() - t0
            print(f"  loaded {i+1:>7,}/{n_total:,} cells in {dt:.1f}s")

    print(f"  bank load complete: W_bank {W_bank.nbytes/1e6:.0f} MB, "
          f"cell_tokens {cell_tokens.nbytes/1e6:.0f} MB")
    return cell_ids, W_bank, cell_tokens


@torch.no_grad()
def precompute_split(
    name: str,
    tokens_path: Path,
    cells_path: Path,
    W_bank_gpu: torch.Tensor,
    cell_ids: np.ndarray,
    mu_gpu: torch.Tensor,
    W_white_gpu: torch.Tensor,
    max_norm: float,
    tok_enc: tiktoken.Encoding,
    encoder: MiniLMEncoder,
    n_neighbors: int,
    chunk_size: int,
    encode_batch: int,
):
    """Precompute neighbor indices for one split (train or val)."""
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    cells = np.load(cells_path, mmap_mode="r")
    assert len(tokens) == len(cells), f"{name}: token/cell length mismatch"

    n_chunks = len(tokens) // chunk_size
    print(f"\n[{name}] {len(tokens):,} tokens -> {n_chunks:,} chunks of {chunk_size}")

    # Result: index into cell_ids/cell_tokens, not the cell_id itself.
    out = np.zeros((n_chunks, n_neighbors), dtype=np.uint32)

    # We over-fetch with topk(k_headroom) so that even after dropping
    # self-retrievals we still have N_NEIGHBORS valid candidates.
    k_headroom = max(8, n_neighbors * 4)

    t_start = time.time()
    for batch_start in range(0, n_chunks, encode_batch):
        batch_end = min(batch_start + encode_batch, n_chunks)
        # Decode each chunk's tokens back to text, capture source-cell sets.
        texts_batch = []
        source_sets = []
        for ci in range(batch_start, batch_end):
            s = ci * chunk_size
            e = s + chunk_size
            chunk_toks = tokens[s:e].tolist()
            texts_batch.append(tok_enc.decode(chunk_toks))
            source_sets.append(set(int(x) for x in cells[s:e]))

        # MiniLM encode (returns tensor already on DEVICE).
        raw = encoder.encode(texts_batch).float()              # (B, 384)

        # Apply whitening: ((raw - mu) @ W) / max_norm
        centered = raw - mu_gpu[None, :]
        whitened = centered @ W_white_gpu
        q = whitened / (max_norm + 1e-8)                       # (B, 384)

        # Dot product against bank — matches /query's activation formula.
        activations = q @ W_bank_gpu.T                     # (B, N_cells)

        # Top-k with headroom.
        top_idx = activations.topk(k_headroom, dim=1).indices  # (B, k_headroom)
        top_idx_cpu = top_idx.cpu().numpy()

        # Filter out self-retrievals per chunk.
        for bi, ci in enumerate(range(batch_start, batch_end)):
            sources = source_sets[bi]
            picked = []
            for j in range(k_headroom):
                idx = int(top_idx_cpu[bi, j])
                if int(cell_ids[idx]) not in sources:
                    picked.append(idx)
                    if len(picked) >= n_neighbors:
                        break
            # Edge case: if all top-k_headroom were self-hits (very rare),
            # fall back to the first candidate even if it's a self-hit —
            # it's still a valid token sequence.
            while len(picked) < n_neighbors:
                picked.append(int(top_idx_cpu[bi, 0]))
            out[ci] = picked[:n_neighbors]

        # Progress.
        if batch_start % (encode_batch * 20) == 0:
            elapsed = time.time() - t_start
            done = batch_end
            rate = done / max(elapsed, 0.01)
            eta = (n_chunks - done) / max(rate, 1)
            print(f"  [{name}] {done:>8,}/{n_chunks:,} "
                  f"({100*done/n_chunks:5.1f}%)  "
                  f"{rate:.0f} chunks/s  ETA {eta:.0f}s")

    elapsed = time.time() - t_start
    print(f"  [{name}] done in {elapsed:.0f}s")
    return out


def main():
    t0 = time.time()

    # ---- Open bank (read-only) ----
    conn = sqlite3.connect(f"file:{BANK_DB}?mode=ro", uri=True)

    # ---- Load whitening params ----
    mu, W_white, max_norm = load_whitening(conn)
    print(f"whitening: mu {mu.shape}, W {W_white.shape}, max_norm {max_norm:.4f}")

    # ---- Set up tokenizer + encoder ----
    tok_enc = tiktoken.get_encoding("gpt2")
    encoder = MiniLMEncoder(DEVICE)

    # ---- Load all bank cells ----
    cell_ids, W_bank, cell_tokens = load_bank(conn, tok_enc, NEIGHBOR_LEN)
    conn.close()

    # ---- Save the per-cell artifacts ----
    print("\nsaving cell_ids.npy and cell_tokens.npy ...")
    np.save(OUT_DIR / "cell_ids.npy", cell_ids)
    np.save(OUT_DIR / "cell_tokens.npy", cell_tokens)

    # ---- Move heavy tensors to GPU ----
    print(f"moving W_bank ({W_bank.nbytes/1e6:.0f} MB) to {DEVICE}...")
    W_bank_gpu = torch.from_numpy(W_bank).to(DEVICE).float()
    mu_gpu = torch.from_numpy(mu).to(DEVICE).float()
    W_white_gpu = torch.from_numpy(W_white).to(DEVICE).float()
    del W_bank   # free CPU copy

    # ---- Precompute for train and val ----
    train_neighbors = precompute_split(
        name="train",
        tokens_path=OUT_DIR / "train.bin",
        cells_path=OUT_DIR / "train_cells.npy",
        W_bank_gpu=W_bank_gpu,
        cell_ids=cell_ids,
        mu_gpu=mu_gpu,
        W_white_gpu=W_white_gpu,
        max_norm=max_norm,
        tok_enc=tok_enc,
        encoder=encoder,
        n_neighbors=N_NEIGHBORS,
        chunk_size=CHUNK_SIZE,
        encode_batch=ENCODE_BATCH,
    )
    np.save(OUT_DIR / "train_neighbors.npy", train_neighbors)
    print(f"saved train_neighbors.npy: shape {train_neighbors.shape}")

    val_neighbors = precompute_split(
        name="val",
        tokens_path=OUT_DIR / "val.bin",
        cells_path=OUT_DIR / "val_cells.npy",
        W_bank_gpu=W_bank_gpu,
        cell_ids=cell_ids,
        mu_gpu=mu_gpu,
        W_white_gpu=W_white_gpu,
        max_norm=max_norm,
        tok_enc=tok_enc,
        encoder=encoder,
        n_neighbors=N_NEIGHBORS,
        chunk_size=CHUNK_SIZE,
        encode_batch=ENCODE_BATCH,
    )
    np.save(OUT_DIR / "val_neighbors.npy", val_neighbors)
    print(f"saved val_neighbors.npy: shape {val_neighbors.shape}")

    # ---- Update meta.json ----
    meta_path = OUT_DIR / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["n_bank_cells"] = int(len(cell_ids))
    meta["chunk_size"] = CHUNK_SIZE
    meta["neighbor_len"] = NEIGHBOR_LEN
    meta["n_neighbors"] = N_NEIGHBORS
    meta["n_train_chunks"] = int(train_neighbors.shape[0])
    meta["n_val_chunks"] = int(val_neighbors.shape[0])
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nupdated meta.json")
    print(f"\nTOTAL TIME: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

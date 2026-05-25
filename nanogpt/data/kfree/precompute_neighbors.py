"""
================================================================================
precompute_neighbors.py — Pre-compute bank neighbors for knowledge-free corpus
================================================================================

Same algorithm as data/bank/precompute_neighbors.py but operating on the
kfree corpus (data/kfree/train.bin, val.bin). Uses the SAME bank
(cell_ids, cell_tokens, whitening) from data/bank/.

Since the kfree corpus text isn't from the bank, self-retrieval filtering
is effectively a no-op (sentinel cell IDs never match real bank cells).

PREREQUISITES:
  - data/kfree/train.bin, val.bin, train_cells.npy, val_cells.npy
    (from prepare_kfree.py)
  - data/bank/cell_ids.npy, cell_tokens.npy (from bank precompute)
  - H:\\MiniLM\\cc_service\\bank.db (for whitening params + bank weights)
  - Stop cc_service before running to free GPU memory

USAGE:
  cd H:\\MiniLM\\nanogpt
  H:\\MiniLM\\cc_service\\.venv\\Scripts\\python.exe data/kfree/precompute_neighbors.py
================================================================================
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import tiktoken
from transformers import AutoTokenizer, AutoModel

BANK_DB = r"H:\MiniLM\cc_service\bank.db"
BANK_DATA_DIR = Path(__file__).parent.parent / "bank"
OUT_DIR = Path(__file__).parent

CHUNK_SIZE = 64
NEIGHBOR_LEN = 64
N_NEIGHBORS = 2
ENCODE_BATCH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def load_whitening(conn):
    row = conn.execute(
        "SELECT mu, w_matrix, max_norm FROM whitening WHERE id=1"
    ).fetchone()
    mu = np.frombuffer(row[0], dtype=np.float32).reshape(384).copy()
    W = np.frombuffer(row[1], dtype=np.float32).reshape(384, 384).copy()
    max_norm = float(row[2])
    return mu, W, max_norm


def load_bank_weights(conn):
    """Load bank cell weights (post-whitening embeddings) aligned with cell_ids.npy."""
    cell_ids = np.load(str(BANK_DATA_DIR / "cell_ids.npy"))
    n = len(cell_ids)
    W_bank = np.zeros((n, 384), dtype=np.float32)

    # Build id -> index map
    id2idx = {int(cid): i for i, cid in enumerate(cell_ids)}

    cur = conn.execute(
        "SELECT c.id, c.weight FROM cells c "
        "JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single' ORDER BY c.id"
    )
    loaded = 0
    for cid, w_blob in cur:
        idx = id2idx.get(int(cid))
        if idx is not None:
            W_bank[idx] = np.frombuffer(w_blob, dtype=np.float32)
            loaded += 1

    print(f"  loaded {loaded:,}/{n:,} cell weights")
    return cell_ids, W_bank


@torch.no_grad()
def precompute_split(
    name, tokens_path, cells_path,
    W_bank_gpu, cell_ids, mu_gpu, W_white_gpu, max_norm,
    tok_enc, encoder, n_neighbors, chunk_size, encode_batch,
):
    tokens = np.memmap(str(tokens_path), dtype=np.uint16, mode="r")
    cells = np.load(str(cells_path), mmap_mode="r")

    n_chunks = len(tokens) // chunk_size
    print(f"\n[{name}] {len(tokens):,} tokens -> {n_chunks:,} chunks of {chunk_size}")

    out = np.zeros((n_chunks, n_neighbors), dtype=np.uint32)
    k_headroom = max(8, n_neighbors * 4)

    t_start = time.time()
    for batch_start in range(0, n_chunks, encode_batch):
        batch_end = min(batch_start + encode_batch, n_chunks)

        texts_batch = []
        source_sets = []
        for ci in range(batch_start, batch_end):
            s = ci * chunk_size
            e = s + chunk_size
            chunk_toks = tokens[s:e].tolist()
            texts_batch.append(tok_enc.decode(chunk_toks))
            source_sets.append(set(int(x) for x in cells[s:e]))

        raw = encoder.encode(texts_batch).float()
        centered = raw - mu_gpu[None, :]
        whitened = centered @ W_white_gpu
        q = whitened / (max_norm + 1e-8)
        activations = q @ W_bank_gpu.T
        top_idx = activations.topk(k_headroom, dim=1).indices
        top_idx_cpu = top_idx.cpu().numpy()

        for bi, ci in enumerate(range(batch_start, batch_end)):
            sources = source_sets[bi]
            picked = []
            for j in range(k_headroom):
                idx = int(top_idx_cpu[bi, j])
                if int(cell_ids[idx]) not in sources:
                    picked.append(idx)
                    if len(picked) >= n_neighbors:
                        break
            while len(picked) < n_neighbors:
                picked.append(int(top_idx_cpu[bi, 0]))
            out[ci] = picked[:n_neighbors]

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

    conn = sqlite3.connect(f"file:{BANK_DB}?mode=ro", uri=True)
    mu, W_white, max_norm = load_whitening(conn)
    print(f"whitening: mu {mu.shape}, W {W_white.shape}, max_norm {max_norm:.4f}")

    tok_enc = tiktoken.get_encoding("gpt2")
    encoder = MiniLMEncoder(DEVICE)

    cell_ids, W_bank = load_bank_weights(conn)
    conn.close()

    print(f"\nmoving W_bank ({W_bank.nbytes/1e6:.0f} MB) to {DEVICE}...")
    W_bank_gpu = torch.from_numpy(W_bank).to(DEVICE).float()
    mu_gpu = torch.from_numpy(mu).to(DEVICE).float()
    W_white_gpu = torch.from_numpy(W_white).to(DEVICE).float()
    del W_bank

    for split in ["train", "val"]:
        neighbors = precompute_split(
            name=split,
            tokens_path=OUT_DIR / f"{split}.bin",
            cells_path=OUT_DIR / f"{split}_cells.npy",
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
        np.save(str(OUT_DIR / f"{split}_neighbors.npy"), neighbors)
        print(f"saved {split}_neighbors.npy: shape {neighbors.shape}")

    print(f"\ntotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

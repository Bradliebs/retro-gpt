"""
================================================================================
prepare.py — extract bank corpus, tokenize with GPT-2 BPE, write training files
================================================================================

INPUT:
  H:\\MiniLM\\cc_service\\bank.db  (read-only; sqlite, ~774k single cells)

OUTPUT (this directory):
  train.bin           uint16 token stream (~95% of corpus)
  val.bin             uint16 token stream (~5% of corpus)
  train_cells.npy     uint32 array, same length as train.bin: cell_id origin of each token
  val_cells.npy       uint32 array, same length as val.bin: same for val
  meta.json           config: counts, vocab_size, train/val split seed, EOT id

WHY THE *_cells.npy ARRAYS:
  When we precompute retrieval neighbors for each training chunk (Stage B), we
  must EXCLUDE the chunk's own source cell from the neighbor set — otherwise
  the model learns trivial copy. The cell_id-per-token map lets us figure out,
  for any 64-token chunk, which cell_ids it overlaps with.

STREAMING + MEMORY:
  We collect per-cell numpy arrays in a Python list then concatenate once at
  the end. Peak memory ~500MB. Fine on this machine.

ENCODING NOTES:
  - tiktoken "gpt2" encoding, encode_ordinary (no special-token parsing of
    user text — text is treated as literal).
  - We append the <|endoftext|> token (50256) after each cell as a document
    separator. The model learns this as the "fresh start" signal.
  - vocab_size for the model is 50257 (50256 regular + 1 EOT). RetroConfig
    will use 50304 (rounded up to multiple of 64) for tensor-core efficiency,
    with the extra IDs unused/zero-initialized.
================================================================================
"""

import os
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import tiktoken

BANK_DB = r"H:\MiniLM\cc_service\bank.db"
OUT_DIR = Path(__file__).parent
VAL_FRACTION = 0.05
SEED = 1337

# tiktoken GPT-2 encoding
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256


def main():
    t_start = time.time()

    # Open bank read-only.
    conn = sqlite3.connect(f"file:{BANK_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Get all cell_ids that have source_text. Order by id for determinism.
    rows = conn.execute(
        "SELECT cell_id FROM source_texts ORDER BY cell_id"
    ).fetchall()
    cell_ids_with_text = np.array([r["cell_id"] for r in rows], dtype=np.uint32)
    n_cells = len(cell_ids_with_text)
    print(f"cells with source_text: {n_cells:,}")

    # Train/val split by cell_id.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_cells)
    n_val = int(n_cells * VAL_FRACTION)
    val_idx_in_perm = perm[:n_val]
    val_cell_ids = set(cell_ids_with_text[val_idx_in_perm].tolist())
    print(f"split: {n_cells - n_val:,} train cells, {n_val:,} val cells")

    train_tokens_chunks = []
    train_cells_chunks = []
    val_tokens_chunks = []
    val_cells_chunks = []

    # Stream rows, tokenize.
    cur = conn.execute("SELECT cell_id, text FROM source_texts ORDER BY cell_id")
    n_done = 0
    n_train_tokens = 0
    n_val_tokens = 0
    t_last = time.time()
    for row in cur:
        cell_id = row["cell_id"]
        text = row["text"]
        # encode_ordinary treats the text literally (no special-token parsing).
        tokens = enc.encode_ordinary(text)
        tokens.append(EOT)
        arr = np.array(tokens, dtype=np.uint16)
        cells_arr = np.full(len(tokens), cell_id, dtype=np.uint32)
        if cell_id in val_cell_ids:
            val_tokens_chunks.append(arr)
            val_cells_chunks.append(cells_arr)
            n_val_tokens += len(tokens)
        else:
            train_tokens_chunks.append(arr)
            train_cells_chunks.append(cells_arr)
            n_train_tokens += len(tokens)
        n_done += 1
        if n_done % 50000 == 0:
            dt = time.time() - t_last
            t_last = time.time()
            print(
                f"  cell {n_done:>7,}/{n_cells:,}  "
                f"train_tok={n_train_tokens:>11,}  val_tok={n_val_tokens:>10,}  "
                f"({dt:.1f}s for last 50k)"
            )

    conn.close()

    print(f"\nconcatenating arrays...")
    train_tokens = np.concatenate(train_tokens_chunks)
    train_cells = np.concatenate(train_cells_chunks)
    val_tokens = np.concatenate(val_tokens_chunks)
    val_cells = np.concatenate(val_cells_chunks)
    # Free the chunk lists.
    del train_tokens_chunks, train_cells_chunks, val_tokens_chunks, val_cells_chunks

    assert len(train_tokens) == len(train_cells)
    assert len(val_tokens) == len(val_cells)
    print(f"  train: {len(train_tokens):,} tokens ({train_tokens.nbytes/1e6:.0f} MB)")
    print(f"  val:   {len(val_tokens):,} tokens ({val_tokens.nbytes/1e6:.0f} MB)")
    print(f"  train_cells: {train_cells.nbytes/1e6:.0f} MB")
    print(f"  val_cells:   {val_cells.nbytes/1e6:.0f} MB")

    # Write outputs.
    print(f"\nwriting outputs to {OUT_DIR}...")
    train_tokens.tofile(OUT_DIR / "train.bin")
    val_tokens.tofile(OUT_DIR / "val.bin")
    np.save(OUT_DIR / "train_cells.npy", train_cells)
    np.save(OUT_DIR / "val_cells.npy", val_cells)

    meta = {
        "encoding": "gpt2",
        "vocab_size_raw": enc.n_vocab,            # 50257
        "vocab_size_padded": 50304,               # round up to multiple of 64 for tensor cores
        "eot_token": int(EOT),                    # 50256
        "n_cells_total": int(n_cells),
        "n_val_cells": int(n_val),
        "n_train_cells": int(n_cells - n_val),
        "n_train_tokens": int(len(train_tokens)),
        "n_val_tokens": int(len(val_tokens)),
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "source_db": BANK_DB,
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json written")
    print(f"\ndone in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

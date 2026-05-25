"""
================================================================================
prepare_kfree.py — Build "knowledge-free" mixed corpus for RETRO training
================================================================================

Builds a training corpus from NON-bank sources so the model can't memorize
what's in the Wikipedia bank. All factual knowledge must come from retrieval.

SOURCES:
  1. OpenAssistant (oasst1) English conversations — instruction-following
  2. Wikipedia EN (20k article subset) — linguistic variety only
  3. Existing code tokens (data/code/train_code.bin) — cross-domain

TARGET: 50-100M tokens, deliberately small.

The train_cells.npy uses sentinel cell_id=0xFFFFFFFE for all tokens, so
neighbor precomputation never filters any bank cell as self-retrieval
(the training text isn't from the bank).

OUTPUTS (data/kfree/):
  train.bin         uint16 token stream (~95%)
  val.bin           uint16 token stream (~5%)
  train_cells.npy   uint32, sentinel values (no self-retrieval filtering)
  val_cells.npy     uint32, sentinel values
  meta.json         metadata

PREREQUISITES:
  - data/code/train_code.bin (existing code tokens)
  - Internet access (HuggingFace downloads)

USAGE:
  cd H:\\MiniLM\\nanogpt
  H:\\MiniLM\\cc_service\\.venv\\Scripts\\python.exe data/kfree/prepare_kfree.py
================================================================================
"""

# datasets MUST be imported before numpy/torch (pyarrow DLL segfault on Windows)
from datasets import load_dataset

import json
import time
from pathlib import Path

import numpy as np
import tiktoken

OUT_DIR = Path(__file__).parent
CODE_BIN = OUT_DIR.parent / "code" / "train_code.bin"

VAL_FRACTION = 0.05
SEED = 42
WIKI_ARTICLES = 20000  # small slice for linguistic variety
SENTINEL_CELL_ID = 0xFFFFFFFE  # never matches any real bank cell

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256


def collect_openassistant() -> list[np.ndarray]:
    """Download OpenAssistant oasst1, tokenize English messages as conversations."""
    print("\n[1/3] OpenAssistant (oasst1) — English conversations")
    ds = load_dataset("OpenAssistant/oasst1", split="train+validation")
    print(f"  total messages: {len(ds)}")

    # Group messages into conversation trees
    from collections import defaultdict
    trees = defaultdict(list)
    for row in ds:
        if row["lang"] != "en":
            continue
        trees[row["message_tree_id"]].append(row)

    # Sort each tree by creation date, concatenate text
    all_token_arrays = []
    total_tokens = 0
    for tree_id, messages in trees.items():
        messages.sort(key=lambda m: m["created_date"])
        parts = []
        for msg in messages:
            role = msg["role"]  # "prompter" or "assistant"
            text = msg["text"].strip()
            parts.append(f"<{role}>\n{text}")
        conversation = "\n\n".join(parts)
        toks = enc.encode_ordinary(conversation)
        toks.append(EOT)
        arr = np.array(toks, dtype=np.uint16)
        all_token_arrays.append(arr)
        total_tokens += len(arr)

    print(f"  conversations: {len(trees)}")
    print(f"  tokens: {total_tokens:,}")
    return all_token_arrays


def collect_wikipedia() -> list[np.ndarray]:
    """Stream a small Wikipedia EN subset for linguistic variety."""
    print(f"\n[2/3] Wikipedia EN — {WIKI_ARTICLES:,} articles")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    all_token_arrays = []
    total_tokens = 0
    for i, row in enumerate(ds):
        if i >= WIKI_ARTICLES:
            break
        text = row["text"].strip()
        if len(text) < 200:
            continue
        toks = enc.encode_ordinary(text)
        toks.append(EOT)
        arr = np.array(toks, dtype=np.uint16)
        all_token_arrays.append(arr)
        total_tokens += len(arr)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,} articles, {total_tokens:,} tokens so far")

    print(f"  articles: {len(all_token_arrays)}")
    print(f"  tokens: {total_tokens:,}")
    return all_token_arrays


def collect_code() -> list[np.ndarray]:
    """Load existing code tokens from data/code/train_code.bin."""
    print(f"\n[3/3] Code — from {CODE_BIN}")
    if not CODE_BIN.exists():
        print(f"  WARNING: {CODE_BIN} not found, skipping code data")
        return []

    tokens = np.memmap(str(CODE_BIN), dtype=np.uint16, mode="r")
    total = len(tokens)
    print(f"  tokens: {total:,}")

    # Split into ~1000-token documents at EOT boundaries for shuffling
    all_arrays = []
    eot_positions = np.where(tokens == EOT)[0]
    prev = 0
    for pos in eot_positions:
        chunk = np.array(tokens[prev:pos + 1], dtype=np.uint16)
        if len(chunk) > 10:
            all_arrays.append(chunk)
        prev = pos + 1
    # Tail
    if prev < total:
        tail = np.array(tokens[prev:total], dtype=np.uint16)
        if len(tail) > 10:
            all_arrays.append(tail)

    print(f"  documents: {len(all_arrays)}")
    return all_arrays


def main():
    t_start = time.time()
    np.random.seed(SEED)

    # Collect all sources
    oasst_docs = collect_openassistant()
    wiki_docs = collect_wikipedia()
    code_docs = collect_code()

    all_docs = oasst_docs + wiki_docs + code_docs
    print(f"\n{'=' * 60}")
    print(f"Total documents: {len(all_docs)}")
    total_tokens = sum(len(d) for d in all_docs)
    print(f"Total tokens: {total_tokens:,}")

    # Shuffle all documents together
    np.random.shuffle(all_docs)

    # Split into train/val
    n_val = max(1, int(len(all_docs) * VAL_FRACTION))
    val_docs = all_docs[:n_val]
    train_docs = all_docs[n_val:]

    train_tokens = np.concatenate(train_docs).astype(np.uint16)
    val_tokens = np.concatenate(val_docs).astype(np.uint16)

    print(f"\nTrain: {len(train_tokens):,} tokens ({len(train_docs)} docs)")
    print(f"Val:   {len(val_tokens):,} tokens ({len(val_docs)} docs)")

    # Write token bins
    train_path = OUT_DIR / "train.bin"
    val_path = OUT_DIR / "val.bin"
    train_tokens.tofile(str(train_path))
    val_tokens.tofile(str(val_path))
    print(f"\nWrote {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {val_path} ({val_path.stat().st_size / 1e6:.1f} MB)")

    # Write sentinel cell IDs (no self-retrieval filtering needed)
    train_cells = np.full(len(train_tokens), SENTINEL_CELL_ID, dtype=np.uint32)
    val_cells = np.full(len(val_tokens), SENTINEL_CELL_ID, dtype=np.uint32)
    np.save(str(OUT_DIR / "train_cells.npy"), train_cells)
    np.save(str(OUT_DIR / "val_cells.npy"), val_cells)
    print(f"Wrote train_cells.npy, val_cells.npy (sentinel={SENTINEL_CELL_ID:#x})")

    # Metadata
    meta = {
        "vocab_size": enc.n_vocab,
        "vocab_size_padded": 50304,
        "eot_token": EOT,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "total_tokens": total_tokens,
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "sources": {
            "openassistant": {
                "docs": len(oasst_docs),
                "tokens": sum(len(d) for d in oasst_docs),
            },
            "wikipedia": {
                "docs": len(wiki_docs),
                "tokens": sum(len(d) for d in wiki_docs),
                "n_articles": WIKI_ARTICLES,
            },
            "code": {
                "docs": len(code_docs),
                "tokens": sum(len(d) for d in code_docs),
            },
        },
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote meta.json")

    dt = time.time() - t_start
    print(f"\nDone in {dt:.0f}s")


if __name__ == "__main__":
    main()

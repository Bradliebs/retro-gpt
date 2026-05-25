"""
================================================================================
eval_cross_encoder.py — Exp 9: Cross-encoder re-ranking
================================================================================

Tests whether cross-encoder re-ranking of retrieved neighbors improves
RETRO's loss compared to raw dot-product retrieval.

For each held-out passage:
  1. Chunk into 64-token chunks
  2. Query bank for top-20 candidates (dot-product, current method)
  3. Condition A: use dot-product top-2 as neighbors
  4. Condition B: re-rank top-20 with cross-encoder, use re-ranked top-2
  5. Condition C: no retrieval (baseline)
  6. Compare losses

If re-ranking improves loss, it's a cheap upgrade to the whole pipeline.

REQUIRES:
  - cc_service running at 127.0.0.1:8765 (bank warm)
  - out-retro-bank/ckpt_best.pt
  - cross-encoder/ms-marco-MiniLM-L-6-v2 (auto-downloaded)

USAGE:
  cd H:/MiniLM/nanogpt
  H:/MiniLM/cc_service/.venv/Scripts/python.exe eval_cross_encoder.py
================================================================================
"""

import sys
import time
from pathlib import Path

# datasets must import before numpy/torch (pyarrow DLL segfault on Windows)
import datasets  # noqa: F401

import numpy as np
import torch
import tiktoken
import httpx

from model_retro import RetroConfig, RetroGPT

# ---- Config ----
_SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH = _SCRIPT_DIR / "out-retro-bank" / "ckpt_best.pt"
HELDOUT_BIN = _SCRIPT_DIR / "data" / "bank" / "heldout.bin"
SERVICE_URL = "http://127.0.0.1:8765"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
DTYPE = (torch.bfloat16
         if DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported()
         else torch.float16)

N_PASSAGES = 30       # number of held-out passages to evaluate
TOP_K_RETRIEVE = 20   # retrieve this many candidates
TOP_K_FINAL = 2       # keep this many after re-ranking (matches training)
SEED = 1337


# ---- Bank client ----
_client = httpx.Client(base_url=SERVICE_URL, timeout=120)


def query_bank(text: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    r = _client.post("/query", json={"text": text, "top_k": top_k})
    r.raise_for_status()
    return r.json().get("hits", [])


# ---- Cross-encoder ----
def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    print("Loading cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
    t0 = time.time()
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model


def rerank(cross_enc, query_text: str, hits: list[dict], top_k: int) -> list[dict]:
    """Re-rank hits using cross-encoder. Returns top_k by cross-encoder score."""
    if not hits:
        return []
    pairs = [(query_text, h.get("source_text", "") or "") for h in hits]
    scores = cross_enc.predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    return [h for h, s in ranked[:top_k]]


# ---- Cell token lookup from cell_tokens.npy ----
_CELL_IDS = None
_CELL_TOKENS = None
_ID2IDX = None


def _load_cell_tokens():
    global _CELL_IDS, _CELL_TOKENS, _ID2IDX
    if _CELL_IDS is None:
        data_dir = _SCRIPT_DIR / "data" / "bank"
        _CELL_IDS = np.load(str(data_dir / "cell_ids.npy"))
        _CELL_TOKENS = np.load(str(data_dir / "cell_tokens.npy"), mmap_mode="r")
        _ID2IDX = {int(cid): i for i, cid in enumerate(_CELL_IDS)}
        print(f"  [cell_tokens] {len(_CELL_IDS):,} cells loaded for token lookup")


def hits_to_neighbor_tokens(
    hits: list[dict],
    enc: tiktoken.Encoding,
    n_neighbors: int,
    neighbor_len: int,
) -> np.ndarray:
    """Convert hits to neighbor token array using cell_tokens.npy when available."""
    _load_cell_tokens()
    nbr = np.zeros((n_neighbors, neighbor_len), dtype=np.int64)
    for ni, hit in enumerate(hits[:n_neighbors]):
        cid = hit.get("cell_id")
        idx = _ID2IDX.get(cid) if cid is not None else None
        if idx is not None:
            # Use pre-tokenized tokens (matches what model was trained on)
            nbr[ni] = _CELL_TOKENS[idx].astype(np.int64)
        else:
            # Fallback: tokenize source_text (for cells added after cell_tokens.npy)
            source = hit.get("source_text", "") or ""
            toks = enc.encode(source, allowed_special={"<|endoftext|>"})
            toks = toks[:neighbor_len]
            if len(toks) < neighbor_len:
                toks = toks + [0] * (neighbor_len - len(toks))
            nbr[ni] = toks
    return nbr


# ---- Loss measurement ----
@torch.no_grad()
def measure_loss_with_neighbors(
    model: RetroGPT,
    idx: torch.Tensor,            # (1, block_size)
    targets: torch.Tensor,        # (1, block_size)
    neighbors: torch.Tensor,      # (1, K, n_neighbors, neighbor_len) or None
) -> float:
    ctx = (
        torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
        if DEVICE_TYPE == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )
    with ctx:
        _, loss = model(idx, targets=targets, neighbors=neighbors)
    return loss.item()


# ---- Main ----
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    enc = tiktoken.get_encoding("gpt2")

    # Load model
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = RetroGPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    print(f"  config: block_size={config.block_size}, chunk_size={config.chunk_size}, "
          f"n_neighbors={config.n_neighbors}, neighbor_len={config.neighbor_len}")

    chunk_size = config.chunk_size       # 64
    block_size = config.block_size       # 256
    K = block_size // chunk_size         # 4 chunks per passage
    n_neighbors = config.n_neighbors     # 2
    neighbor_len = config.neighbor_len   # 64

    # Load cross-encoder
    cross_enc = load_cross_encoder()

    # Load held-out data
    print(f"Loading held-out data: {HELDOUT_BIN}")
    heldout_tokens = np.memmap(str(HELDOUT_BIN), dtype=np.uint16, mode="r")
    n_total_tokens = len(heldout_tokens)
    max_start = n_total_tokens - block_size - 1  # -1 for target offset
    print(f"  {n_total_tokens:,} tokens, {max_start // block_size} full passages available")

    # Sample passage start positions (non-overlapping)
    passage_starts = np.arange(0, max_start, block_size)
    np.random.shuffle(passage_starts)
    passage_starts = passage_starts[:N_PASSAGES]
    print(f"  evaluating {len(passage_starts)} passages\n")

    # Results accumulators
    losses_none = []
    losses_dotprod = []
    losses_reranked = []

    rerank_times = []
    query_times = []

    for pi, start in enumerate(passage_starts):
        tokens_x = heldout_tokens[start:start + block_size].astype(np.int64)
        tokens_y = heldout_tokens[start + 1:start + 1 + block_size].astype(np.int64)
        idx = torch.tensor([tokens_x], dtype=torch.long, device=DEVICE)
        tgt = torch.tensor([tokens_y], dtype=torch.long, device=DEVICE)

        # Decode chunks for querying (from input tokens)
        chunks_text = []
        for ci in range(K):
            c_start = ci * chunk_size
            chunk_toks = tokens_x[c_start:c_start + chunk_size].tolist()
            chunks_text.append(enc.decode(chunk_toks))

        # Query bank for all chunks (top-20 each)
        t0 = time.time()
        all_hits = []
        for chunk_text in chunks_text:
            hits = query_bank(chunk_text, top_k=TOP_K_RETRIEVE)
            all_hits.append(hits)
        query_time = time.time() - t0
        query_times.append(query_time)

        # Condition A: dot-product top-2 (take first 2 from each chunk's hits)
        nbrs_dp = np.zeros((1, K, n_neighbors, neighbor_len), dtype=np.int64)
        for ci, hits in enumerate(all_hits):
            nbrs_dp[0, ci] = hits_to_neighbor_tokens(
                hits[:TOP_K_FINAL], enc, n_neighbors, neighbor_len
            )

        # Condition B: cross-encoder re-ranked top-2
        t0 = time.time()
        nbrs_ce = np.zeros((1, K, n_neighbors, neighbor_len), dtype=np.int64)
        for ci, hits in enumerate(all_hits):
            reranked = rerank(cross_enc, chunks_text[ci], hits, TOP_K_FINAL)
            nbrs_ce[0, ci] = hits_to_neighbor_tokens(
                reranked, enc, n_neighbors, neighbor_len
            )
        rerank_time = time.time() - t0
        rerank_times.append(rerank_time)

        # Measure losses
        loss_none = measure_loss_with_neighbors(model, idx, tgt, None)
        loss_dp = measure_loss_with_neighbors(
            model, idx, tgt, torch.from_numpy(nbrs_dp).to(DEVICE)
        )
        loss_ce = measure_loss_with_neighbors(
            model, idx, tgt, torch.from_numpy(nbrs_ce).to(DEVICE)
        )

        losses_none.append(loss_none)
        losses_dotprod.append(loss_dp)
        losses_reranked.append(loss_ce)

        # Check if re-ranking changed the neighbors (did it pick different cells?)
        dp_same_as_ce = np.array_equal(nbrs_dp, nbrs_ce)

        if (pi + 1) % 5 == 0 or pi == 0:
            print(f"  [{pi+1:>3d}/{len(passage_starts)}] "
                  f"none={loss_none:.4f}  "
                  f"dot-prod={loss_dp:.4f}  "
                  f"reranked={loss_ce:.4f}  "
                  f"query={query_time:.2f}s  "
                  f"rerank={rerank_time*1000:.0f}ms  "
                  f"{'SAME' if dp_same_as_ce else 'DIFF'}")

    # ---- Summary ----
    print(f"\n{'=' * 72}")
    print("RESULTS: Cross-Encoder Re-ranking (Exp 9)")
    print(f"{'=' * 72}")

    mean_none = np.mean(losses_none)
    mean_dp = np.mean(losses_dotprod)
    mean_ce = np.mean(losses_reranked)
    std_none = np.std(losses_none) / np.sqrt(len(losses_none))
    std_dp = np.std(losses_dotprod) / np.sqrt(len(losses_dotprod))
    std_ce = np.std(losses_reranked) / np.sqrt(len(losses_reranked))

    retrieval_gap_dp = mean_none - mean_dp
    retrieval_gap_ce = mean_none - mean_ce
    improvement = mean_dp - mean_ce  # positive = re-ranking helps

    print(f"\n  Passages evaluated:  {len(passage_starts)}")
    print(f"  Candidates per chunk: top-{TOP_K_RETRIEVE} → top-{TOP_K_FINAL}")
    print(f"\n  Loss (no retrieval):    {mean_none:.4f} ± {std_none:.4f}")
    print(f"  Loss (dot-product):     {mean_dp:.4f} ± {std_dp:.4f}  "
          f"(gap = {retrieval_gap_dp:.4f})")
    print(f"  Loss (cross-encoder):   {mean_ce:.4f} ± {std_ce:.4f}  "
          f"(gap = {retrieval_gap_ce:.4f})")
    print(f"\n  Re-ranking improvement: {improvement:.4f} nats "
          f"({improvement/retrieval_gap_dp*100:.1f}% of retrieval gap)")

    if improvement > 0:
        print(f"  → Cross-encoder re-ranking HELPS: {improvement:.4f} nats better")
    elif improvement < -0.001:
        print(f"  → Cross-encoder re-ranking HURTS: {-improvement:.4f} nats worse")
    else:
        print(f"  → Cross-encoder re-ranking has NO EFFECT")

    mean_query = np.mean(query_times)
    mean_rerank = np.mean(rerank_times) * 1000
    print(f"\n  Avg query time:   {mean_query:.2f}s per passage ({K} chunks × top-{TOP_K_RETRIEVE})")
    print(f"  Avg rerank time:  {mean_rerank:.0f}ms per passage ({K} chunks × {TOP_K_RETRIEVE} candidates)")

    # Per-passage breakdown: how often did re-ranking pick different neighbors?
    n_diff = sum(1 for dp, ce in zip(losses_dotprod, losses_reranked) if abs(dp - ce) > 0.001)
    print(f"\n  Passages where re-ranking changed loss: {n_diff}/{len(passage_starts)}")

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()

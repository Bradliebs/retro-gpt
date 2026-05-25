"""
================================================================================
generate_retro_p2_bank.py — Phase 2.2 Stage D
================================================================================

Generate from the trained Phase 2.2 RetroGPT (out-retro-bank/ckpt_best.pt).

For each prompt, produces two continuations using the SAME random seed:
  (a) WITH BANK RETRIEVAL: every chunk's neighbors are retrieved live from
      the bank using the same MiniLM+whitening pipeline used at train time.
      The retrieved cells are shown so you can see what's flowing in.
  (b) WITHOUT RETRIEVAL: model.forward(..., neighbors=None). Same weights,
      same prompt, same seed — only difference is CCA layers see no neighbors.

The point: if Phase 2.2 worked (val gap +0.124 nat), retrieval-conditioned
output should be visibly more grounded / less hallucinated.

NOT comparable to retro_generate.py directly:
  - retro_generate.py uses GPT-2 (124M, pretrained on WebText) + RAG by
    pasting bank cells into the prompt. Different model, different mechanism.
  - This script uses our 55M-param RetroGPT (trained from scratch on the
    60M-token bank corpus only) + true RETRO ChunkedCrossAttention. The
    "without retrieval" baseline here is the proper apples-to-apples control.
================================================================================
"""

import sys
import time
import sqlite3
from pathlib import Path

import numpy as np
import torch
import tiktoken
from transformers import AutoTokenizer, AutoModel

from model_retro import RetroConfig, RetroGPT


# ---------------- Paths / config ----------------
# Optional first arg: path to ckpt_best.pt (default: out-retro-bank/ckpt_best.pt).
# Use sys.argv[1] = "out-retro-bank-small/ckpt_best.pt" to sample from the 16M variant.
CKPT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out-retro-bank") / "ckpt_best.pt"
DATA_DIR = Path("data") / "bank"
BANK_DB = r"H:\MiniLM\cc_service\bank.db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 40
SEED = 42

PROMPTS = [
    "Photosynthesis is the process by which plants",
    "The French Revolution began in",
    "Quantum entanglement is a phenomenon where",
    "The Great Wall of China was built",
]


# ==============================================================================
# MiniLM (transformers-based — see precompute_neighbors.py for why)
# ==============================================================================
class MiniLMEncoder:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 128) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(self.device)
        out = self.model(**inputs)
        attn = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * attn).sum(1)
        counts = attn.sum(1).clamp(min=1e-9)
        return summed / counts


# ==============================================================================
# Bank loading (whitening params + cell weights + cell text)
# ==============================================================================
def load_bank_for_retrieval(device: str):
    print(f"opening bank ({BANK_DB})...")
    conn = sqlite3.connect(f"file:{BANK_DB}?mode=ro", uri=True)

    row = conn.execute("SELECT mu, w_matrix, max_norm FROM whitening WHERE id=1").fetchone()
    mu = torch.from_numpy(np.frombuffer(row[0], dtype=np.float32).reshape(384).copy()).to(device)
    W_white = torch.from_numpy(np.frombuffer(row[1], dtype=np.float32).reshape(384, 384).copy()).to(device)
    max_norm = float(row[2])

    n_cells = conn.execute(
        "SELECT COUNT(*) FROM cells c JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single'"
    ).fetchone()[0]
    print(f"  loading {n_cells:,} cells (weights + text)...")

    cell_ids = np.zeros(n_cells, dtype=np.uint32)
    W_bank = np.zeros((n_cells, 384), dtype=np.float32)
    id_to_text: dict[int, str] = {}

    t0 = time.time()
    cur = conn.execute(
        "SELECT c.id, c.weight, s.text FROM cells c "
        "JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single' ORDER BY c.id"
    )
    for i, (cid, w_blob, text) in enumerate(cur):
        cell_ids[i] = cid
        W_bank[i] = np.frombuffer(w_blob, dtype=np.float32)
        id_to_text[int(cid)] = text
        if (i + 1) % 200000 == 0:
            print(f"    loaded {i+1:>7,}/{n_cells:,}")
    conn.close()
    print(f"  bank loaded in {time.time()-t0:.1f}s")

    W_bank_gpu = torch.from_numpy(W_bank).to(device)
    del W_bank
    return cell_ids, W_bank_gpu, id_to_text, mu, W_white, max_norm


# ==============================================================================
# Retrieval at generation time
# ==============================================================================
class BankRetriever:
    """Live retrieval matching Stage B's pipeline exactly:
       q = ((MiniLM(text) - mu) @ W_white) / max_norm
       top_k = argmax(q @ W_bank.T)
    """
    def __init__(self, encoder, cell_ids, W_bank_gpu, cell_tokens_t,
                 mu, W_white, max_norm, n_neighbors, chunk_size, neighbor_len,
                 enc: tiktoken.Encoding, device: str):
        self.encoder = encoder
        self.cell_ids = cell_ids
        self.W_bank_gpu = W_bank_gpu
        self.cell_tokens_t = cell_tokens_t        # (N, neighbor_len) int64 on GPU
        self.mu = mu
        self.W_white = W_white
        self.max_norm = max_norm
        self.n_neighbors = n_neighbors
        self.chunk_size = chunk_size
        self.neighbor_len = neighbor_len
        self.enc = enc
        self.device = device

    @torch.no_grad()
    def retrieve(self, x_tokens: torch.Tensor, fallback_text: str | None = None):
        """x_tokens: (block_size,) int64. Returns:
             nbrs:    (K, n_neighbors, neighbor_len) int64 on DEVICE
             top_idx: (K, n_neighbors) numpy — indices into cell_ids (for display)

        EOT-padded chunks (which decode to '<|endoftext|>'-spam) produce garbage
        MiniLM embeddings that lock onto a few weird cells. When fewer than 8
        non-EOT tokens are present in a chunk, substitute `fallback_text` (the
        original prompt) so the retriever sees something semantically meaningful.
        """
        eot = self.enc.eot_token
        block_size = x_tokens.size(0)
        K = block_size // self.chunk_size
        chunk_texts = []
        for k in range(K):
            toks = x_tokens[k * self.chunk_size : (k + 1) * self.chunk_size].cpu().tolist()
            non_eot = [t for t in toks if t != eot]
            if len(non_eot) < 8 and fallback_text is not None:
                chunk_texts.append(fallback_text)
            elif non_eot:
                # Drop the EOT padding so MiniLM doesn't see literal '<|endoftext|>' tokens.
                chunk_texts.append(self.enc.decode(non_eot))
            else:
                chunk_texts.append(self.enc.decode(toks))
        raw = self.encoder.encode(chunk_texts).float()        # (K, 384)
        q = (raw - self.mu[None, :]) @ self.W_white
        q = q / (self.max_norm + 1e-8)                        # (K, 384)
        activations = q @ self.W_bank_gpu.T                   # (K, N_cells)
        top_idx = activations.topk(self.n_neighbors, dim=1).indices  # (K, n_neighbors)
        nbrs = self.cell_tokens_t[top_idx]                    # (K, n_neighbors, Ln)
        return nbrs, top_idx.cpu().numpy()


# ==============================================================================
# Generation
# ==============================================================================
def sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[-1], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, config, retriever, enc, prompt: str,
             max_new_tokens: int, temperature: float, top_k: int,
             use_retrieval: bool, verbose: bool = False):
    """Generate from prompt. Returns (generated_text, list_of_retrieval_events)."""
    eot = enc.eot_token
    prompt_toks = enc.encode_ordinary(prompt)

    # Pad LEFT with EOT to fill block_size — keeps prompt on the right edge.
    if len(prompt_toks) >= config.block_size:
        ctx = prompt_toks[-config.block_size:]
    else:
        ctx = [eot] * (config.block_size - len(prompt_toks)) + prompt_toks
    ctx = torch.tensor(ctx, dtype=torch.long, device=DEVICE)

    retrieval_events = []
    nbrs_input = None
    if use_retrieval:
        nbrs, top_idx = retriever.retrieve(ctx, fallback_text=prompt)
        nbrs_input = nbrs.unsqueeze(0)                       # (1, K, n_neighbors, Ln)
        retrieval_events.append((0, top_idx))

    generated_ids = []
    for step in range(max_new_tokens):
        x_in = ctx[-config.block_size:].unsqueeze(0)         # (1, block_size)
        logits, _ = model(x_in, neighbors=nbrs_input)
        next_tok = sample_token(logits[0, -1], temperature, top_k)
        ctx = torch.cat([ctx, next_tok], dim=0)
        generated_ids.append(int(next_tok.item()))

        # Re-retrieve at chunk boundaries so neighbors track the new content.
        if use_retrieval and (step + 1) % config.chunk_size == 0 and step + 1 < max_new_tokens:
            nbrs, top_idx = retriever.retrieve(ctx[-config.block_size:], fallback_text=prompt)
            nbrs_input = nbrs.unsqueeze(0)
            retrieval_events.append((step + 1, top_idx))

    return enc.decode(generated_ids), retrieval_events


# ==============================================================================
# Main
# ==============================================================================
def print_retrieval_event(step: int, top_idx: np.ndarray, cell_ids: np.ndarray,
                          id_to_text: dict, label: str):
    K, n_nbr = top_idx.shape
    print(f"  [{label}] retrieval at step {step}: {K} chunks x {n_nbr} neighbors each")
    for k in range(K):
        for j in range(n_nbr):
            idx = int(top_idx[k, j])
            cid = int(cell_ids[idx])
            txt = id_to_text.get(cid, "<missing>").replace("\n", " ")
            print(f"    chunk{k} nbr{j}: cell #{cid:>6}  {txt[:90]!r}")


def main():
    print(f"device={DEVICE}")
    print(f"loading checkpoint {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config: RetroConfig = ckpt["config"]
    print(f"  iter {ckpt.get('iter', '?')}, "
          f"val_with={ckpt.get('val_with', float('nan')):.4f}, "
          f"val_without={ckpt.get('val_without', float('nan')):.4f}")
    print(f"  n_layer={config.n_layer} n_head={config.n_head} n_embd={config.n_embd} "
          f"block_size={config.block_size} chunk_size={config.chunk_size}")

    model = RetroGPT(config).to(DEVICE).eval()
    model.load_state_dict(ckpt["model_state"])

    enc = tiktoken.get_encoding("gpt2")
    print("loading MiniLM encoder...")
    encoder = MiniLMEncoder(DEVICE)
    cell_ids, W_bank_gpu, id_to_text, mu, W_white, max_norm = load_bank_for_retrieval(DEVICE)
    cell_tokens = np.load(DATA_DIR / "cell_tokens.npy")
    cell_tokens_t = torch.from_numpy(cell_tokens.astype(np.int64)).to(DEVICE)
    print(f"  cell_tokens: {cell_tokens.shape}")

    retriever = BankRetriever(
        encoder=encoder, cell_ids=cell_ids, W_bank_gpu=W_bank_gpu,
        cell_tokens_t=cell_tokens_t, mu=mu, W_white=W_white, max_norm=max_norm,
        n_neighbors=config.n_neighbors, chunk_size=config.chunk_size,
        neighbor_len=config.neighbor_len, enc=enc, device=DEVICE,
    )

    for prompt in PROMPTS:
        print("\n" + "=" * 88)
        print(f"PROMPT: {prompt!r}")
        print("=" * 88)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        t0 = time.time()
        with_text, with_events = generate(
            model, config, retriever, enc, prompt,
            MAX_NEW_TOKENS, TEMPERATURE, TOP_K, use_retrieval=True,
        )
        dt_with = time.time() - t0

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        t0 = time.time()
        without_text, _ = generate(
            model, config, retriever, enc, prompt,
            MAX_NEW_TOKENS, TEMPERATURE, TOP_K, use_retrieval=False,
        )
        dt_without = time.time() - t0

        print("\n--- BANK CELLS RETRIEVED (with-retrieval run) ---")
        for step, top_idx in with_events:
            print_retrieval_event(step, top_idx, cell_ids, id_to_text, f"step {step}")

        print(f"\n--- WITH RETRIEVAL ({dt_with:.1f}s) ---")
        print(prompt + with_text)

        print(f"\n--- WITHOUT RETRIEVAL ({dt_without:.1f}s) ---")
        print(prompt + without_text)

    print("\n" + "=" * 88)
    print("done.")


if __name__ == "__main__":
    main()

"""
================================================================================
generate_retro.py — Live retrieval-augmented generation with RetroGPT
================================================================================

Loads the trained RetroGPT checkpoint and generates text with real-time
retrieval from the bank via the cc_service API. Shows side-by-side comparison
of generation with and without retrieval.

Unlike retro_generate.py (which uses vanilla GPT-2 + paste-in-prompt RAG),
this uses the actual RETRO architecture with chunked cross-attention that
was trained to integrate retrieval.

REQUIRES:
  - cc_service running at 127.0.0.1:8765
  - out-retro-bank/ckpt_best.pt

USAGE:
  cd H:\\MiniLM\\nanogpt
  H:\\MiniLM\\cc_service\\.venv\\Scripts\\python.exe generate_retro.py
  H:\\MiniLM\\cc_service\\.venv\\Scripts\\python.exe generate_retro.py --prompt "The French Revolution"
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tiktoken
import httpx

from model_retro import RetroConfig, RetroGPT

# ---- Config ----
_SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH = _SCRIPT_DIR / "out-retro-bank" / "ckpt_best.pt"
SERVICE_URL = "http://127.0.0.1:8765"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
DTYPE = torch.bfloat16 if (DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
SEED = 1337


# ---- Bank client ----
_client = httpx.Client(base_url=SERVICE_URL, timeout=30)


def query_bank(text: str, top_k: int = 2) -> list[dict]:
    """Query the bank for top-k most similar cells."""
    r = _client.post("/query", json={"text": text, "top_k": top_k})
    r.raise_for_status()
    return r.json().get("hits", [])


def retrieve_neighbors_for_chunks(
    chunks_text: list[str],
    enc: tiktoken.Encoding,
    n_neighbors: int,
    neighbor_len: int,
) -> torch.Tensor:
    """
    For each chunk of text, query the bank and return tokenized neighbors.

    Returns: (1, K, n_neighbors, neighbor_len) int64 tensor
    """
    K = len(chunks_text)
    nbrs = np.zeros((1, K, n_neighbors, neighbor_len), dtype=np.int64)

    for ci, chunk_text in enumerate(chunks_text):
        if not chunk_text.strip():
            continue
        hits = query_bank(chunk_text, top_k=n_neighbors)
        for ni, hit in enumerate(hits[:n_neighbors]):
            source = hit.get("source_text", "")
            toks = enc.encode(source, allowed_special={"<|endoftext|>"})
            toks = toks[:neighbor_len]
            # Pad if shorter than neighbor_len
            if len(toks) < neighbor_len:
                toks = toks + [0] * (neighbor_len - len(toks))
            nbrs[0, ci, ni] = toks

    return torch.from_numpy(nbrs).to(DEVICE)


# ---- Generation ----
@torch.no_grad()
def generate_retro(
    model: RetroGPT,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    use_retrieval: bool = True,
    verbose_retrieval: bool = False,
) -> tuple[str, list[list[dict]]]:
    """
    Generate text with the RetroGPT model, optionally with live retrieval.

    Retrieves neighbors once for the prompt, then caches them. Only re-retrieves
    when the token stream crosses into a new chunk boundary.

    Returns: (generated_text, all_retrieved_hits)
    """
    model.eval()
    config = model.config
    chunk_size = config.chunk_size
    block_size = config.block_size
    K = block_size // chunk_size

    # Tokenize prompt
    prompt_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    if len(prompt_ids) > block_size:
        prompt_ids = prompt_ids[-block_size:]

    # Working sequence
    seq = list(prompt_ids)
    generated_ids = []
    all_hits = []
    cached_neighbors = None
    last_retrieval_len = -1  # track which chunk boundary we last retrieved at

    ctx = (
        torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
        if DEVICE_TYPE == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    for _ in range(max_new_tokens):
        # Take the last block_size tokens
        window = seq[-block_size:]
        # Pad to block_size if shorter
        if len(window) < block_size:
            pad_len = block_size - len(window)
            window = [0] * pad_len + window

        idx = torch.tensor([window], dtype=torch.long, device=DEVICE)

        neighbors = None
        if use_retrieval:
            # Only re-retrieve when we cross a chunk boundary
            current_len = len(seq)
            current_chunk = current_len // chunk_size
            if cached_neighbors is None or current_chunk != last_retrieval_len:
                chunks_text = []
                for ci in range(K):
                    start = ci * chunk_size
                    chunk_toks = window[start:start + chunk_size]
                    chunks_text.append(enc.decode(chunk_toks))

                cached_neighbors = retrieve_neighbors_for_chunks(
                    chunks_text, enc, config.n_neighbors, config.neighbor_len
                )
                last_retrieval_len = current_chunk

                if verbose_retrieval and len(generated_ids) == 0:
                    all_hits.append(chunks_text)

            neighbors = cached_neighbors

        with ctx:
            logits, _ = model(idx, neighbors=neighbors)

        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        seq.append(next_id)
        generated_ids.append(next_id)

        # Stop on end-of-text
        if next_id == enc.eot_token:
            break

    return enc.decode(generated_ids), all_hits


# ---- Demo prompts ----
DEMO_PROMPTS = [
    "The theory of general relativity describes",
    "Machine learning is a branch of artificial intelligence that",
    "The French Revolution began in 1789 when",
    "In computer science, a hash table is",
    "The Great Wall of China was built",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (overrides demo prompts)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of samples per prompt")
    parser.add_argument("--ckpt", type=str, default=str(CKPT_PATH))
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Verify bank service is up
    try:
        info = _client.get("/info").json()
        print(f"Bank: {info['n_cells']:,} cells, dim={info['dim']}")
    except Exception as e:
        print(f"ERROR: bank service not reachable at {SERVICE_URL}: {e}",
              file=sys.stderr)
        sys.exit(1)

    # Load model
    print(f"\nloading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = RetroGPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    enc = tiktoken.get_encoding("gpt2")

    prompts = [args.prompt] if args.prompt else DEMO_PROMPTS

    for prompt in prompts:
        print("\n" + "=" * 78)
        print(f"PROMPT: {prompt}")
        print("=" * 78)

        for sample_i in range(args.n_samples):
            if args.n_samples > 1:
                print(f"\n--- Sample {sample_i + 1}/{args.n_samples} ---")

            # Without retrieval
            torch.manual_seed(SEED + sample_i)
            t0 = time.time()
            text_no, _ = generate_retro(
                model, enc, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_retrieval=False,
            )
            dt_no = time.time() - t0

            # With retrieval
            torch.manual_seed(SEED + sample_i)
            t0 = time.time()
            text_ret, hits = generate_retro(
                model, enc, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_retrieval=True,
                verbose_retrieval=True,
            )
            dt_ret = time.time() - t0

            print(f"\n  [WITHOUT retrieval] ({dt_no:.1f}s)")
            print(f"  {prompt}{text_no}")

            print(f"\n  [WITH retrieval] ({dt_ret:.1f}s)")
            print(f"  {prompt}{text_ret}")

    print("\n" + "=" * 78)
    print("Done.")


if __name__ == "__main__":
    main()

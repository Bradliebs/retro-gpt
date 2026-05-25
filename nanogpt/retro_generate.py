"""
================================================================================
retro_generate.py — Retrieval-Augmented Generation with GPT-2 + your bank
================================================================================

WHAT THIS DOES:
  Loads pretrained GPT-2 small (124M params), wires your concept-cells bank in
  as an external memory. Before answering a prompt, it queries the bank for
  relevant snippets and pastes them into the context. Then GPT-2 generates a
  continuation that can leverage that retrieved knowledge.

  This is "RAG" — Retrieval-Augmented Generation. It's the pattern behind
  Perplexity, Bing Chat with search, ChatGPT with browsing, etc. The model
  itself is unchanged; we just give it relevant context at inference time.

WHAT THIS IS NOT:
  - Not "RETRO" proper (DeepMind 2021). True RETRO has chunked cross-attention
    layers that attend to retrieved chunks at every transformer block. We are
    doing the simpler "look stuff up, paste in prompt" version. RAG is what
    99% of production systems use; RETRO requires modifying the architecture
    and training from scratch.
  - Not fine-tuning. The model weights are exactly the pretrained GPT-2.

WHY IT WORKS:
  GPT-2 is a pattern completer. Given "Q: What is X?\nA:" it tries to produce
  text that looks like a plausible answer. If we precede the question with
  "CONTEXT: <relevant facts about X>", the model's next-token distribution
  shifts toward tokens that are consistent with those facts. The bank gives
  us the relevant facts; GPT-2 does the language modelling.

REQUIREMENTS:
  - cc service running at 127.0.0.1:8765 (your bank)
  - Internet on first run (downloads ~500MB of GPT-2 weights, then cached)
  - CUDA GPU (will work on CPU too, just much slower)

USAGE:
  cd h:\\MiniLM\\nanogpt
  h:\\MiniLM\\cc_service\\.venv\\Scripts\\python.exe retro_generate.py
================================================================================
"""

import sys
import time
import torch
import tiktoken
import httpx

from model import GPT

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
SERVICE_URL = "http://127.0.0.1:8765"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"            # smallest GPT-2 — 124M params, fits easily on 8GB
MAX_NEW_TOKENS = 80            # length of each generated continuation
TEMPERATURE = 0.8              # 0.7-0.9 = good balance of coherence vs creativity
TOP_K = 40                     # sample from top-40 tokens each step
N_RETRIEVE = 3                 # how many bank cells to inject as context
MIN_ACTIVATION = 0.30          # don't bother retrieving low-similarity hits
SEED = 1337


# ------------------------------------------------------------------------------
# Bank client
# ------------------------------------------------------------------------------
_client = httpx.Client(base_url=SERVICE_URL, timeout=30)


def query_bank(text: str, top_k: int = N_RETRIEVE) -> list[dict]:
    """Ask the bank for the top-k most similar cells to `text`."""
    r = _client.post("/query", json={"text": text, "top_k": top_k})
    r.raise_for_status()
    return r.json().get("hits", [])


def format_retrieval_context(hits: list[dict]) -> str:
    """
    Turn raw bank hits into a chunk of context the LM can read.

    We filter out weak matches, strip the [code] prefix some cells have,
    and format as numbered "facts". Format matters: the model needs to see
    a clean delimiter between retrieved context and the actual question.
    """
    useful = [h for h in hits if h.get("activation", 0) >= MIN_ACTIVATION]
    if not useful:
        return ""
    lines = []
    for i, h in enumerate(useful, 1):
        # Pull the source text and trim — GPT-2's context is only 1024 tokens.
        text = (h.get("source_text") or "").strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "..."
        lines.append(f"({i}) {text}")
    return "Relevant facts:\n" + "\n".join(lines) + "\n\n"


# ------------------------------------------------------------------------------
# Model + tokenizer setup
# ------------------------------------------------------------------------------
def load_model() -> tuple[GPT, tiktoken.Encoding]:
    print(f"Loading pretrained {MODEL_NAME} (first run downloads ~500MB)...")
    t0 = time.time()
    model = GPT.from_pretrained(MODEL_NAME, dict(dropout=0.0))
    model.eval().to(DEVICE)
    enc = tiktoken.get_encoding("gpt2")
    print(f"  loaded in {time.time() - t0:.1f}s on {DEVICE}")
    return model, enc


# ------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------
@torch.no_grad()
def generate(model: GPT, enc: tiktoken.Encoding, prompt: str,
             retrieve: bool = False) -> tuple[str, list[dict]]:
    """
    Generate a continuation of `prompt`. If retrieve=True, prepend bank context.

    Returns (full_text, hits) where full_text is just the model-generated part
    (not the input prompt) and hits is the list of cells used (empty if not retrieving).
    """
    hits: list[dict] = []
    if retrieve:
        hits = query_bank(prompt)
        context = format_retrieval_context(hits)
        # The structure here ("facts ... Question ... Answer") is a "prompt
        # pattern". GPT-2 wasn't trained on this exact format, but it has
        # seen enough Q&A on the web to recognize it and behave appropriately.
        full_input = f"{context}Question: {prompt}\nAnswer:"
    else:
        full_input = prompt

    ids = enc.encode(full_input, allowed_special={"<|endoftext|>"})

    # GPT-2 has block_size=1024. Leave room for new tokens.
    max_input = model.config.block_size - MAX_NEW_TOKENS
    if len(ids) > max_input:
        ids = ids[-max_input:]  # keep the tail (the actual question)

    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    n_input = idx.size(1)

    # Inference under autocast for ~2x speedup on Ampere (your 3070).
    with torch.amp.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu",
                            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                            enabled=(DEVICE == "cuda")):
        out = model.generate(idx, MAX_NEW_TOKENS,
                             temperature=TEMPERATURE, top_k=TOP_K)

    # Slice off the input to keep only the generated tokens.
    generated_ids = out[0, n_input:].tolist()
    return enc.decode(generated_ids), hits


# ------------------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------------------
DEMO_PROMPTS = [
    # Fact-recall prompts: bank should have good context, generation should
    # noticeably improve with retrieval.
    "What is the largest planet in our solar system?",
    "How do vaccines work?",
    "What causes earthquakes?",

    # Conceptual prompts: bank has explanations, model should use them.
    "Explain what a hash table is.",
    "What is recursion in programming?",

    # An open-ended one: less factual, retrieval helps less.
    "Write a short story about a robot learning to paint.",
]


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Verify service is up before loading 500MB of model weights.
    try:
        info = _client.get("/info").json()
        print(f"Bank: {info['n_cells']:,} cells, dim={info['dim']}\n")
    except Exception as e:
        print(f"ERROR: bank service not reachable at {SERVICE_URL}: {e}",
              file=sys.stderr)
        sys.exit(1)

    model, enc = load_model()

    for prompt in DEMO_PROMPTS:
        print("=" * 78)
        print(f"PROMPT: {prompt}")
        print("=" * 78)

        # Vanilla GPT-2 — no retrieval. This is the baseline.
        torch.manual_seed(SEED)  # reset RNG so the only difference is retrieval
        t0 = time.time()
        text_v, _ = generate(model, enc, prompt, retrieve=False)
        dt_v = time.time() - t0
        print(f"\n--- vanilla GPT-2 ({dt_v:.2f}s) ---")
        print(text_v.strip())

        # With bank retrieval.
        torch.manual_seed(SEED)
        t0 = time.time()
        text_r, hits = generate(model, enc, prompt, retrieve=True)
        dt_r = time.time() - t0
        print(f"\n--- with bank retrieval ({dt_r:.2f}s, {len(hits)} hits) ---")
        for h in hits:
            label = (h.get("label") or "?")[:60]
            print(f"  [act={h['activation']:.3f}] {label}")
        print()
        print(text_r.strip())
        print()


if __name__ == "__main__":
    main()

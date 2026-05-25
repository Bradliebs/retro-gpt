"""
================================================================================
rag_gemma4.py — RAG with Gemma 4 + our RETRO bank
================================================================================

Uses the same cc_service bank (1.8M Wikipedia chunks + MiniLM retrieval)
that our RETRO model uses, but feeds retrieved context into Gemma 4's
prompt instead of through cross-attention layers.

Compares generation WITH vs WITHOUT retrieved context, showing whether
our retrieval pipeline adds value to a capable model.

REQUIRES:
  - Ollama running with gemma4 model:  ollama serve
  - cc_service running:  python -m cc_service.cli.ccmem serve

USAGE:
  python rag_gemma4.py [--model gemma4:latest] [--top-k 2]
================================================================================
"""

import argparse
import json
import re
import time
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
BANK_URL = "http://localhost:8765/query"

PROMPTS = [
    "Explain how photosynthesis works in plants.",
    "What caused the fall of the Roman Empire?",
    "Describe the structure of DNA and its role in heredity.",
]


def query_bank(text: str, top_k: int = 2, timeout: float = 10.0) -> list[dict]:
    """Query our cc_service bank for similar chunks."""
    try:
        r = httpx.post(BANK_URL, json={"text": text, "top_k": top_k},
                       timeout=timeout)
        r.raise_for_status()
        return r.json().get("hits", [])
    except Exception as e:
        print(f"  [bank error: {e}]")
        return []


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from thinking models like Gemma 4."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def generate(model: str, prompt: str, timeout: float = 300.0) -> tuple[str, float]:
    """Generate with Ollama. Returns (text, seconds)."""
    t0 = time.time()
    r = httpx.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1024, "temperature": 0.7},
    }, timeout=timeout)
    r.raise_for_status()
    elapsed = time.time() - t0
    raw = r.json()["response"]
    return strip_thinking(raw), elapsed


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """Build a prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("source_text", "")
        score = chunk.get("activation", 0)
        context_parts.append(f"[Source {i} (similarity: {score:.3f})]\n{text}")

    context_block = "\n\n".join(context_parts)

    return (
        f"Use the following retrieved context to help answer the question. "
        f"If the context is relevant, incorporate it. If not, answer from "
        f"your own knowledge.\n\n"
        f"--- RETRIEVED CONTEXT ---\n{context_block}\n"
        f"--- END CONTEXT ---\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.2:latest")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt instead of the default list")
    args = parser.parse_args()

    prompts = [args.prompt] if args.prompt else PROMPTS

    print(f"Model: {args.model}")
    print(f"Bank: {BANK_URL} (top_k={args.top_k})")
    print(f"Prompts: {len(prompts)}")

    # Warm up Ollama
    print("\nwarming up Ollama...")
    try:
        generate(args.model, "Hi", timeout=60.0)
    except Exception as e:
        print(f"ERROR: Ollama not available ({e})")
        print("Start it with: ollama serve")
        return

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'=' * 72}")
        print(f"PROMPT {i}: {prompt}")
        print(f"{'=' * 72}")

        # Retrieve from bank
        print(f"\n  Retrieving from bank (top_k={args.top_k})...")
        chunks = query_bank(prompt, top_k=args.top_k)
        if chunks:
            for j, c in enumerate(chunks, 1):
                text = (c.get("source_text") or "")[:120]
                score = c.get("activation", 0)
                print(f"    [{j}] score={score:.3f}: {text}...")
        else:
            print("    (no results from bank)")

        # Generate WITHOUT context
        print(f"\n  --- WITHOUT RETRIEVAL ---")
        t0 = time.time()
        text_bare, dur_bare = generate(args.model, f"Question: {prompt}\n\nAnswer:")
        print(f"  ({dur_bare:.1f}s)")
        # Trim to ~300 chars for display
        print(f"  {text_bare[:500]}")

        # Generate WITH context
        if chunks:
            rag_prompt = build_rag_prompt(prompt, chunks)
            print(f"\n  --- WITH RETRIEVAL (RAG) ---")
            text_rag, dur_rag = generate(args.model, rag_prompt)
            print(f"  ({dur_rag:.1f}s)")
            print(f"  {text_rag[:500]}")
        else:
            print("\n  --- WITH RETRIEVAL (RAG) ---")
            print("  (skipped — no bank results)")

    print(f"\n{'=' * 72}")
    print("Done.")


if __name__ == "__main__":
    main()

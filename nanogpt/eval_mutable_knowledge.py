"""
================================================================================
eval_mutable_knowledge.py - Exp 8: Mutable Knowledge Demo
================================================================================

Proves that RETRO's knowledge lives in the bank, not the weights.
Measures cross-entropy loss on topic-specific passages under three conditions:

  1. BASELINE: Loss with full bank retrieval
  2. ABLATED:  Loss after removing ALL relevant cells for this topic
  3. RESTORED: Loss after re-adding cells - should match baseline

Uses loss measurement (not generation) because the 55M model's text
generation is too noisy to show effects from retrieval changes. Loss is
deterministic and sensitive to even small changes in retrieval quality.

REQUIRES:
  - cc_service running at 127.0.0.1:8765 (bank warm)
  - out-retro-bank/ckpt_best.pt

USAGE:
  cd H:/MiniLM/nanogpt
  H:/MiniLM/cc_service/.venv/Scripts/python.exe eval_mutable_knowledge.py
================================================================================
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass

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
DTYPE = (
    torch.bfloat16
    if (DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported())
    else torch.float16
)

# First query loads 1.8M weights into memory - needs longer timeout
_client = httpx.Client(base_url=SERVICE_URL, timeout=120)


# ---- Bank operations ----
@dataclass
class SavedCell:
    cell_id: int
    text: str
    label: str | None
    activation: float


def query_bank(text: str, top_k: int = 2) -> list[dict]:
    r = _client.post("/query", json={"text": text, "top_k": top_k})
    r.raise_for_status()
    return r.json().get("hits", [])


def query_bank_silent(text: str, top_k: int = 50) -> list[dict]:
    r = _client.post(
        "/query", json={"text": text, "top_k": top_k, "include_silent": True}
    )
    r.raise_for_status()
    return r.json().get("hits", [])


def delete_cell(cell_id: int) -> dict:
    r = _client.delete(f"/cells/{cell_id}")
    r.raise_for_status()
    return r.json()


def write_cell(text: str, label: str | None = None) -> dict:
    r = _client.post("/write", json={"text": text, "label": label})
    r.raise_for_status()
    return r.json()


def get_bank_info() -> dict:
    r = _client.get("/info")
    r.raise_for_status()
    return r.json()


# ---- Retrieval for RETRO ----
def retrieve_neighbors_for_chunks(
    chunks_text: list[str],
    enc: tiktoken.Encoding,
    n_neighbors: int,
    neighbor_len: int,
) -> torch.Tensor:
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
            if len(toks) < neighbor_len:
                toks = toks + [0] * (neighbor_len - len(toks))
            nbrs[0, ci, ni] = toks
    return torch.from_numpy(nbrs).to(DEVICE)


# ---- Loss measurement ----
@torch.no_grad()
def measure_loss(
    model: RetroGPT,
    enc: tiktoken.Encoding,
    passage: str,
    use_retrieval: bool = True,
) -> float:
    """Compute cross-entropy loss on a passage with optional retrieval."""
    model.eval()
    config = model.config
    chunk_size = config.chunk_size
    block_size = config.block_size
    K = block_size // chunk_size

    tokens = enc.encode(passage, allowed_special={"<|endoftext|>"})
    # Trim to block_size (we need exactly one forward pass)
    tokens = tokens[:block_size]
    if len(tokens) < block_size:
        tokens = tokens + [0] * (block_size - len(tokens))

    idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    neighbors = None
    if use_retrieval:
        chunks_text = []
        for ci in range(K):
            start = ci * chunk_size
            chunk_toks = tokens[start : start + chunk_size]
            chunks_text.append(enc.decode(chunk_toks))
        neighbors = retrieve_neighbors_for_chunks(
            chunks_text, enc, config.n_neighbors, config.neighbor_len
        )

    ctx = (
        torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
        if DEVICE_TYPE == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    with ctx:
        logits, loss = model(idx, targets=idx, neighbors=neighbors)

    return loss.item()


# ---- Topic search: find ALL related cells ----
def find_topic_cells(
    search_queries: list[str],
    min_activation: float = 0.25,
    max_total: int = 50,
) -> list[SavedCell]:
    """Search with multiple queries and collect unique cells above threshold."""
    seen = {}
    for q in search_queries:
        hits = query_bank_silent(q, top_k=50)
        for h in hits:
            cid = h["cell_id"]
            act = h["activation"]
            if act >= min_activation and cid not in seen:
                seen[cid] = SavedCell(
                    cell_id=cid,
                    text=h.get("source_text", ""),
                    label=h.get("label"),
                    activation=act,
                )
    # Sort by activation descending
    cells = sorted(seen.values(), key=lambda c: c.activation, reverse=True)
    return cells[:max_total]


# ---- Topics ----
TOPICS = [
    {
        "name": "Photosynthesis",
        "passage": (
            "Photosynthesis is a process used by plants and other organisms to "
            "convert light energy into chemical energy that can be stored and "
            "later released to fuel the organism's activities. In most cases, "
            "oxygen is released as a waste product. Most plants, algae, and "
            "cyanobacteria perform photosynthesis. Such organisms are called "
            "photoautotrophs. Photosynthesis is largely responsible for "
            "producing and maintaining the oxygen content of the Earth's "
            "atmosphere. Although photosynthesis is performed differently by "
            "different species, the process always begins when energy from "
            "light is absorbed by proteins called reaction centres that "
            "contain green chlorophyll pigments."
        ),
        "search_queries": [
            "photosynthesis plants convert sunlight energy",
            "chlorophyll pigment green plants",
            "photosynthesis chloroplast oxygen carbon dioxide",
            "photoautotroph organism light energy",
        ],
        "min_activation": 0.28,
    },
    {
        "name": "DNA Structure",
        "passage": (
            "Deoxyribonucleic acid is a molecule composed of two polynucleotide "
            "chains that coil around each other to form a double helix. The "
            "structure carries genetic instructions for the development, "
            "functioning, growth and reproduction of all known organisms and "
            "many viruses. DNA and ribonucleic acid are nucleic acids. "
            "Alongside proteins, lipids and complex carbohydrates, nucleic "
            "acids are one of the four major types of macromolecules that are "
            "essential for all known forms of life. The two DNA strands are "
            "known as polynucleotides as they are composed of simpler "
            "monomeric units called nucleotides."
        ),
        "search_queries": [
            "DNA double helix structure Watson Crick",
            "deoxyribonucleic acid nucleotide base pair",
            "genetic information DNA molecule",
            "DNA replication strand polynucleotide",
        ],
        "min_activation": 0.28,
    },
    {
        "name": "French Revolution",
        "passage": (
            "The French Revolution was a period of radical political and "
            "societal change in France that began with the Estates General of "
            "1789 and ended with the formation of the French Consulate in "
            "November 1799. Many of its ideas are considered fundamental "
            "principles of liberal democracy, while phrases like liberty, "
            "equality, fraternity reappeared in other revolts such as the "
            "1917 Russian Revolution. The values and institutions of the "
            "Revolution dominate French politics to this day. The Revolution "
            "resulted in the suppression of the feudal system, the "
            "emancipation of the individual, and a greater division of "
            "landed property."
        ),
        "search_queries": [
            "French Revolution 1789 Bastille",
            "Estates General France revolution",
            "liberty equality fraternity revolution",
            "feudal system France abolition",
        ],
        "min_activation": 0.25,
    },
]


def run_topic(model, enc, topic):
    name = topic["name"]
    passage = topic["passage"]
    queries = topic["search_queries"]
    min_act = topic["min_activation"]

    print(f"\n{'=' * 78}")
    print(f"  TOPIC: {name}")
    print(f"{'=' * 78}")

    # --- Find ALL related cells ---
    print(f"  Searching for related cells (min_activation={min_act})...")
    cells = find_topic_cells(queries, min_activation=min_act, max_total=50)
    print(f"  Found {len(cells)} cells above threshold")

    if not cells:
        print("  WARNING: No cells found. Skipping topic.")
        return None

    for c in cells[:5]:
        snippet = c.text[:70].replace("\n", " ")
        print(f"    cell {c.cell_id:>8d}  act={c.activation:.4f}  \"{snippet}...\"")
    if len(cells) > 5:
        print(f"    ... and {len(cells) - 5} more")

    # --- Phase 1: BASELINE loss ---
    print(f"\n  Phase 1: BASELINE (full bank)")
    loss_baseline = measure_loss(model, enc, passage, use_retrieval=True)
    loss_no_retrieval = measure_loss(model, enc, passage, use_retrieval=False)
    print(f"    Loss with retrieval:    {loss_baseline:.4f}")
    print(f"    Loss without retrieval: {loss_no_retrieval:.4f}")
    print(f"    Retrieval benefit:      {loss_no_retrieval - loss_baseline:+.4f}")

    # --- Phase 2: DELETE cells ---
    print(f"\n  Phase 2: DELETING {len(cells)} topic cells...")
    deleted = []
    for c in cells:
        result = delete_cell(c.cell_id)
        if result.get("deleted"):
            deleted.append(c)
        else:
            print(f"    SKIP cell {c.cell_id}: {result.get('reason')}")

    info = get_bank_info()
    print(f"    Deleted {len(deleted)} cells. Bank: {info['n_cells']:,}")

    loss_ablated = measure_loss(model, enc, passage, use_retrieval=True)
    print(f"    Loss after deletion:    {loss_ablated:.4f}")
    print(f"    Loss delta vs baseline: {loss_ablated - loss_baseline:+.4f}")

    check_hits = query_bank_silent(queries[0], top_k=3)
    if check_hits:
        print(f"    Fallback retrievals:")
        for h in check_hits:
            snippet = (h.get("source_text") or "")[:60].replace("\n", " ")
            print(f"      cell {h['cell_id']:>8d}  act={h['activation']:.4f}"
                  f"  \"{snippet}...\"")

    # --- Phase 3: RESTORE cells ---
    print(f"\n  Phase 3: RESTORING {len(deleted)} cells...")
    restored_ids = []
    for c in deleted:
        result = write_cell(c.text, c.label)
        restored_ids.append(result["cell_id"])

    info = get_bank_info()
    print(f"    Restored {len(restored_ids)} cells. Bank: {info['n_cells']:,}")

    loss_restored = measure_loss(model, enc, passage, use_retrieval=True)
    print(f"    Loss after restore:     {loss_restored:.4f}")
    print(f"    Loss delta vs baseline: {loss_restored - loss_baseline:+.4f}")

    # --- Summary ---
    gap_baseline = loss_no_retrieval - loss_baseline
    gap_ablated = loss_no_retrieval - loss_ablated
    gap_restored = loss_no_retrieval - loss_restored
    degradation = loss_ablated - loss_baseline

    print(f"\n  {'~' * 70}")
    print(f"  RESULTS for {name}: ({len(deleted)} cells deleted/restored)")
    print(f"  {'~' * 70}")
    print(f"    No retrieval:  {loss_no_retrieval:.4f}")
    print(f"    Baseline:      {loss_baseline:.4f}  (gap = {gap_baseline:+.4f})")
    print(f"    Ablated:       {loss_ablated:.4f}  (gap = {gap_ablated:+.4f})")
    print(f"    Restored:      {loss_restored:.4f}  (gap = {gap_restored:+.4f})")
    print(f"    Degradation:   {degradation:+.4f} nats "
          f"({degradation / gap_baseline * 100:+.1f}% of retrieval benefit lost)"
          if gap_baseline > 0 else "")

    return {
        "name": name,
        "n_deleted": len(deleted),
        "loss_none": loss_no_retrieval,
        "loss_baseline": loss_baseline,
        "loss_ablated": loss_ablated,
        "loss_restored": loss_restored,
        "gap_baseline": gap_baseline,
        "gap_ablated": gap_ablated,
        "degradation": degradation,
        "pct_lost": degradation / gap_baseline * 100 if gap_baseline > 0 else 0,
    }


def main():
    print("=" * 78)
    print("  Exp 8: Mutable Knowledge - RETRO Bank as Editable Memory")
    print("=" * 78)

    try:
        info = get_bank_info()
        print(f"\nBank: {info['n_cells']:,} cells, dim={info['dim']}")
    except Exception as e:
        print(f"\nERROR: bank service not reachable at {SERVICE_URL}: {e}",
              file=sys.stderr)
        print("Start cc_service first.", file=sys.stderr)
        sys.exit(1)

    initial_count = info["n_cells"]

    print(f"Loading checkpoint from {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = RetroGPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"Chunk size: {config.chunk_size}, Neighbors: {config.n_neighbors}, "
          f"Block size: {config.block_size}")

    results = []
    for topic in TOPICS:
        result = run_topic(model, enc, topic)
        if result:
            results.append(result)

    # Final report
    final_info = get_bank_info()
    final_count = final_info["n_cells"]
    print(f"\n{'=' * 78}")
    print(f"  FINAL REPORT - Exp 8: Mutable Knowledge")
    print(f"{'=' * 78}")
    print(f"\n  Bank cells: {initial_count:,} -> {final_count:,} "
          f"(delta: {final_count - initial_count:+d})")

    print(f"\n  {'Topic':<20s} {'Cells':>5s} {'Baseline':>8s} {'Ablated':>8s} "
          f"{'Restored':>8s} {'Degradation':>12s}")
    print(f"  {'~' * 70}")
    for r in results:
        print(f"  {r['name']:<20s} {r['n_deleted']:>5d} {r['loss_baseline']:>8.4f} "
              f"{r['loss_ablated']:>8.4f} {r['loss_restored']:>8.4f} "
              f"{r['degradation']:>+8.4f} ({r['pct_lost']:>+5.1f}%)")

    any_degraded = any(r["degradation"] > 0.001 for r in results)
    any_restored = any(
        abs(r["loss_restored"] - r["loss_baseline"]) < 0.05 for r in results
    )

    if any_degraded and any_restored:
        print(f"\n  VERDICT: PASS")
        print("  Removing topic cells degrades loss; restoring them recovers it.")
        print("  The bank is a mutable knowledge store with immediate effect.")
    elif any_degraded:
        print(f"\n  VERDICT: PARTIAL - Deletion degrades loss but "
              "restoration imperfect.")
    else:
        print(f"\n  VERDICT: FAIL - No measurable degradation from cell deletion.")

    print(f"\n{'=' * 78}")


if __name__ == "__main__":
    main()

"""
================================================================================
eval_retro_heldout.py — 3-condition held-out eval for retrieval leakage test
================================================================================

Loads the best Phase 2.3 checkpoint and evaluates it on held-out Wikipedia
text (articles never in the bank or train/val splits) under three conditions:

  1. val_without    — neighbors=None  (CCA off)
  2. val_with_random — random bank cells as neighbors (fresh per chunk)
  3. val_with_real   — MiniLM-retrieved neighbors from heldout_neighbors.npy

DECISION MATRIX:
  real < random < without  →  retrieval does real semantic work. Gap is honest.
  real ≈ random < without  →  "having tokens in CCA" helps, but semantic
                               retrieval adds nothing over random.
  real ≈ without           →  gap was leakage; collapses on held-out data.

USAGE:
  cd H:\\MiniLM\\nanogpt
  python eval_retro_heldout.py [--n-batches 100] [--batch-size 8]

REQUIRES:
  data/bank/heldout.bin
  data/bank/heldout_neighbors.npy
  data/bank/cell_tokens.npy
  out-retro-bank/ckpt_best.pt
================================================================================
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from model_retro import RetroConfig, RetroGPT


# ---- Paths (defaults, overridable via CLI) ----
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data" / "bank"
CKPT_PATH = _SCRIPT_DIR / "out-retro-bank" / "ckpt_best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
DTYPE = torch.bfloat16 if (DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
CTX = (
    torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
    if DEVICE_TYPE == "cuda"
    else torch.amp.autocast(device_type="cpu", enabled=False)
)

SEED = 42  # different from training seed for extra independence


# ==============================================================================
# Held-out dataset
# ==============================================================================
class HeldoutDataset:
    """Wraps heldout.bin + precomputed neighbors + random neighbor generation."""

    def __init__(self, config: RetroConfig, data_dir: Path = None,
                 heldout_bin: str = "heldout.bin",
                 heldout_neighbors: str = "heldout_neighbors.npy",
                 cell_tokens_path: str = "cell_tokens.npy"):
        d = data_dir or DATA_DIR
        self.tokens = np.memmap(d / heldout_bin, dtype=np.uint16, mode="r")
        self.neighbors = np.load(d / heldout_neighbors, mmap_mode="r")
        self.cell_tokens = np.load(d / cell_tokens_path, mmap_mode="r")

        self.config = config
        self.K = config.block_size // config.chunk_size  # chunks per sample
        self.Ln = config.neighbor_len

        # Last valid chunk_idx so x+1 still fits inside tokens.
        max_token_start = len(self.tokens) - config.block_size - 1
        self.max_chunk_idx = max_token_start // config.chunk_size
        self.max_chunk_idx = min(self.max_chunk_idx, len(self.neighbors) - self.K)

        self.n_bank_cells = len(self.cell_tokens)

        print(f"  [heldout] {len(self.tokens):,} tokens, "
              f"{len(self.neighbors):,} chunks-with-neighbors, "
              f"bank has {self.n_bank_cells:,} cells")
        print(f"  sampling range chunk_idx [0, {self.max_chunk_idx})")

    def get_batch(self, batch_size: int, chunk_indices: np.ndarray,
                  mode: str = "real"):
        """Build a batch from pre-selected chunk indices.

        mode:
          "none"   — returns neighbors=None
          "real"   — uses precomputed MiniLM-retrieved neighbors
          "random" — draws fresh random bank cells as neighbors
          "real1"  — slot 0 = real neighbor, slot 1 = random (marginal-value test)

        Using the SAME chunk_indices across modes ensures fair comparison.
        """
        K = self.K
        Ln = self.Ln
        B = len(chunk_indices)

        x = np.empty((B, self.config.block_size), dtype=np.int64)
        y = np.empty((B, self.config.block_size), dtype=np.int64)

        for b, c in enumerate(chunk_indices):
            tstart = int(c) * self.config.chunk_size
            x[b] = self.tokens[tstart:tstart + self.config.block_size].astype(np.int64)
            y[b] = self.tokens[tstart + 1:tstart + 1 + self.config.block_size].astype(np.int64)

        x_t = torch.from_numpy(x).to(DEVICE)
        y_t = torch.from_numpy(y).to(DEVICE)

        if mode == "none":
            return x_t, y_t, None

        nbrs = np.empty((B, K, self.config.n_neighbors, Ln), dtype=np.int64)

        if mode == "real":
            for b, c in enumerate(chunk_indices):
                nbr_idx = self.neighbors[int(c):int(c) + K]  # (K, n_neighbors)
                nbrs[b] = self.cell_tokens[nbr_idx].astype(np.int64)

        elif mode == "real1":
            # Slot 0: real MiniLM-retrieved neighbor. Slot 1: random bank cell.
            for b, c in enumerate(chunk_indices):
                nbr_idx = self.neighbors[int(c):int(c) + K]  # (K, n_neighbors)
                real_cells = self.cell_tokens[nbr_idx].astype(np.int64)
                nbrs[b, :, 0, :] = real_cells[:, 0]
                rand_idx = np.random.randint(0, self.n_bank_cells, size=(K,))
                nbrs[b, :, 1, :] = self.cell_tokens[rand_idx].astype(np.int64)

        elif mode == "random":
            for b in range(B):
                # Fresh random bank cells for each chunk — no correlation.
                rand_idx = np.random.randint(
                    0, self.n_bank_cells,
                    size=(K, self.config.n_neighbors),
                )
                nbrs[b] = self.cell_tokens[rand_idx].astype(np.int64)

        return x_t, y_t, torch.from_numpy(nbrs).to(DEVICE)


# ==============================================================================
# Evaluation
# ==============================================================================
@torch.no_grad()
def evaluate(model, ds: HeldoutDataset, n_batches: int, batch_size: int):
    """Run 3-condition eval on the SAME random batches. Returns dict of mean losses."""
    model.eval()

    losses = {"none": [], "random": [], "real": []}

    for bi in range(n_batches):
        # Same chunk indices for all three conditions.
        chunk_indices = np.random.randint(0, ds.max_chunk_idx, size=batch_size)

        for mode in ["none", "random", "real"]:
            x, y, nbrs = ds.get_batch(batch_size, chunk_indices, mode=mode)
            with CTX:
                _, loss = model(x, targets=y, neighbors=nbrs)
            losses[mode].append(loss.item())

        if (bi + 1) % 20 == 0:
            means = {k: sum(v) / len(v) for k, v in losses.items()}
            print(f"  batch {bi+1:>4d}/{n_batches}  "
                  f"none={means['none']:.4f}  "
                  f"random={means['random']:.4f}  "
                  f"real={means['real']:.4f}")

    return {k: sum(v) / len(v) for k, v in losses.items()}


# ==============================================================================
# Standard error
# ==============================================================================
def stderr(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(var / n)


@torch.no_grad()
def evaluate_with_stats(model, ds: HeldoutDataset, n_batches: int, batch_size: int):
    """Same as evaluate but also computes standard error per condition."""
    model.eval()

    modes = ["none", "random", "real1", "real"]
    losses = {m: [] for m in modes}

    for bi in range(n_batches):
        chunk_indices = np.random.randint(0, ds.max_chunk_idx, size=batch_size)

        for mode in modes:
            x, y, nbrs = ds.get_batch(batch_size, chunk_indices, mode=mode)
            with CTX:
                _, loss = model(x, targets=y, neighbors=nbrs)
            losses[mode].append(loss.item())

        if (bi + 1) % 20 == 0:
            means = {k: sum(v) / len(v) for k, v in losses.items()}
            print(f"  batch {bi+1:>4d}/{n_batches}  "
                  f"none={means['none']:.4f}  "
                  f"random={means['random']:.4f}  "
                  f"real1={means['real1']:.4f}  "
                  f"real={means['real']:.4f}")

    results = {}
    for mode in modes:
        vals = losses[mode]
        results[mode] = {
            "mean": sum(vals) / len(vals),
            "stderr": stderr(vals),
            "n": len(vals),
        }
    return results


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=100,
                        help="Number of eval batches (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Samples per batch (default: 8)")
    parser.add_argument("--ckpt", type=str, default=str(CKPT_PATH),
                        help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with heldout data + cell_tokens ")
    parser.add_argument("--heldout-bin", type=str, default="heldout.bin",
                        help="Filename of held-out token stream")
    parser.add_argument("--heldout-neighbors", type=str,
                        default="heldout_neighbors.npy",
                        help="Filename of precomputed neighbor indices")
    parser.add_argument("--cell-tokens", type=str, default="cell_tokens.npy",
                        help="Filename of bank cell tokens")
    parser.add_argument("--inspect", type=int, default=0,
                        help="Print N random chunks with their retrieved neighbors")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- Load checkpoint ----
    print(f"loading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    print(f"  config: n_layer={config.n_layer}, n_head={config.n_head}, "
          f"n_embd={config.n_embd}, vocab={config.vocab_size}")
    print(f"  trained to iter {ckpt.get('iter', '?')}, "
          f"val_with={ckpt.get('val_with', '?'):.4f}, "
          f"val_without={ckpt.get('val_without', '?'):.4f}")

    model = RetroGPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.2f}M")

    # ---- Load heldout data ----
    data_dir = Path(args.data_dir) if args.data_dir else None
    print(f"\nloading held-out data from {data_dir or DATA_DIR}...")
    ds = HeldoutDataset(config, data_dir=data_dir,
                        heldout_bin=args.heldout_bin,
                        heldout_neighbors=args.heldout_neighbors,
                        cell_tokens_path=args.cell_tokens)

    # ---- Neighbor inspection ----
    if args.inspect > 0:
        import tiktoken
        tok_enc = tiktoken.get_encoding("gpt2")
        inspect_indices = np.random.randint(0, ds.max_chunk_idx, size=args.inspect)
        print(f"\n{'=' * 72}")
        print(f"NEIGHBOR INSPECTION ({args.inspect} random chunks)")
        print(f"{'=' * 72}")
        for ii, ci in enumerate(inspect_indices):
            tstart = int(ci) * config.chunk_size
            chunk_toks = ds.tokens[tstart:tstart + config.chunk_size].tolist()
            chunk_text = tok_enc.decode(chunk_toks)
            nbr_indices = ds.neighbors[int(ci)]  # (n_neighbors,)
            print(f"\n--- Chunk {ii+1} (idx={ci}) ---")
            print(f"  TEXT: {chunk_text[:200]}{'...' if len(chunk_text) > 200 else ''}")
            for ni, nbr_idx in enumerate(nbr_indices):
                nbr_toks = ds.cell_tokens[int(nbr_idx)].tolist()
                nbr_text = tok_enc.decode(nbr_toks)
                # Check if this is a wiki cell or code cell.
                origin = "wiki" if int(nbr_idx) < 1817204 else "code"
                print(f"  NEIGHBOR {ni+1} (cell={nbr_idx}, {origin}): "
                      f"{nbr_text[:200]}{'...' if len(nbr_text) > 200 else ''}")
        print(f"{'=' * 72}\n")

    # ---- Run eval ----
    print(f"\nrunning 4-condition eval: {args.n_batches} batches x {args.batch_size} samples")
    print(f"  total chunk evaluations: {args.n_batches * args.batch_size * (config.block_size // config.chunk_size):,}")
    print("=" * 72)
    t0 = time.time()
    results = evaluate_with_stats(model, ds, args.n_batches, args.batch_size)
    elapsed = time.time() - t0

    # ---- Report ----
    print("\n" + "=" * 72)
    print("HELD-OUT EVALUATION RESULTS")
    print("=" * 72)
    print(f"  checkpoint:     {args.ckpt}")
    print(f"  held-out tokens: {len(ds.tokens):,}")
    print(f"  eval batches:   {args.n_batches} x {args.batch_size}")
    print(f"  eval time:      {elapsed:.1f}s")
    print()

    for mode in ["none", "random", "real1", "real"]:
        r = results[mode]
        print(f"  {mode:>8s}:  {r['mean']:.4f}  (±{r['stderr']:.4f}, n={r['n']})")
    print()

    none_mean = results["none"]["mean"]
    random_mean = results["random"]["mean"]
    real1_mean = results["real1"]["mean"]
    real_mean = results["real"]["mean"]

    gap_real = none_mean - real_mean
    gap_random = none_mean - random_mean
    gap_semantic = random_mean - real_mean
    gap_nbr1 = random_mean - real1_mean
    gap_nbr2 = real1_mean - real_mean

    print(f"  gap (none - real):     {gap_real:+.4f}  [total retrieval benefit]")
    print(f"  gap (none - random):   {gap_random:+.4f}  [CCA-as-regularizer benefit]")
    print(f"  gap (random - real):   {gap_semantic:+.4f}  [pure semantic retrieval benefit]")
    print()
    print(f"  --- neighbor count ablation ---")
    print(f"  gap (random - real1):  {gap_nbr1:+.4f}  [1st real neighbor value]")
    print(f"  gap (real1 - real):    {gap_nbr2:+.4f}  [2nd real neighbor marginal value]")
    if gap_nbr1 > 0.001:
        pct = gap_nbr2 / gap_nbr1 * 100 if gap_nbr1 else 0
        print(f"  2nd neighbor adds {pct:.0f}% of what the 1st provides")
    print()

    # ---- Decision ----
    print("-" * 72)
    if gap_real > 0.05 and gap_semantic > 0.02:
        print("[PASS] Retrieval does real semantic work on held-out data.")
        print(f"  Real neighbors beat no-neighbors by {gap_real:+.4f}")
        print(f"  Real neighbors beat random neighbors by {gap_semantic:+.4f}")
        print("  The gap is NOT explained by information leakage.")
    elif gap_real > 0.02 and gap_semantic > 0.005:
        print("[WEAK PASS] Retrieval shows some semantic benefit on held-out data,")
        print("  but the effect is small.")
        print(f"  Real neighbors beat no-neighbors by {gap_real:+.4f}")
        print(f"  Real neighbors beat random neighbors by {gap_semantic:+.4f}")
    elif gap_random > 0.02 and gap_semantic < 0.005:
        print("[PARTIAL] Having tokens in CCA helps, but semantic retrieval adds")
        print("  nothing over random neighbors.")
        print(f"  Any-neighbors beat no-neighbors by {gap_random:+.4f}")
        print(f"  Real vs random: {gap_semantic:+.4f} (negligible)")
    else:
        print("[FAIL] Retrieval gap collapses on held-out data.")
        print(f"  Gap (none - real): {gap_real:+.4f}")
        print("  The training gap was likely information leakage.")
    print("=" * 72)


if __name__ == "__main__":
    main()

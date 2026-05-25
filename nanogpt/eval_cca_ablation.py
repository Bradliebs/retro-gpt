"""
================================================================================
eval_cca_ablation.py — Which CCA layers matter most?
================================================================================

The trained model has CCA in layers [1, 3, 5, 7]. This script evaluates
retrieval benefit with individual CCA layers disabled at eval time.

This is an inference-time ablation: we temporarily turn off specific CCA
layers and measure the impact on loss. It tests "which layers contribute
most to the retrieval benefit" from the already-trained model.

Conditions:
  all_cca    — all 4 CCA layers active (baseline = "real" eval)
  drop_L1    — disable CCA in layer 1 only
  drop_L3    — disable CCA in layer 3 only
  drop_L5    — disable CCA in layer 5 only
  drop_L7    — disable CCA in layer 7 only
  no_cca     — all CCA layers disabled (= "none" eval)

USAGE:
  cd H:\\MiniLM\\nanogpt
  python eval_cca_ablation.py [--n-batches 100] [--batch-size 8]
================================================================================
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from model_retro import RetroConfig, RetroGPT

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
SEED = 42


def stderr(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(var / n)


class HeldoutDataset:
    def __init__(self, config, data_dir=None, heldout_bin="heldout.bin",
                 heldout_neighbors="heldout_neighbors.npy",
                 cell_tokens_path="cell_tokens.npy"):
        d = data_dir or DATA_DIR
        self.tokens = np.memmap(d / heldout_bin, dtype=np.uint16, mode="r")
        self.neighbors = np.load(d / heldout_neighbors, mmap_mode="r")
        self.cell_tokens = np.load(d / cell_tokens_path, mmap_mode="r")
        self.config = config
        self.K = config.block_size // config.chunk_size
        self.Ln = config.neighbor_len
        max_token_start = len(self.tokens) - config.block_size - 1
        self.max_chunk_idx = max_token_start // config.chunk_size
        self.max_chunk_idx = min(self.max_chunk_idx, len(self.neighbors) - self.K)
        self.n_bank_cells = len(self.cell_tokens)

    def get_batch(self, batch_size, chunk_indices, use_real=True):
        K, Ln = self.K, self.Ln
        B = len(chunk_indices)
        x = np.empty((B, self.config.block_size), dtype=np.int64)
        y = np.empty((B, self.config.block_size), dtype=np.int64)
        for b, c in enumerate(chunk_indices):
            tstart = int(c) * self.config.chunk_size
            x[b] = self.tokens[tstart:tstart + self.config.block_size].astype(np.int64)
            y[b] = self.tokens[tstart + 1:tstart + 1 + self.config.block_size].astype(np.int64)
        x_t = torch.from_numpy(x).to(DEVICE)
        y_t = torch.from_numpy(y).to(DEVICE)
        if not use_real:
            return x_t, y_t, None
        nbrs = np.empty((B, K, self.config.n_neighbors, Ln), dtype=np.int64)
        for b, c in enumerate(chunk_indices):
            nbr_idx = self.neighbors[int(c):int(c) + K]
            nbrs[b] = self.cell_tokens[nbr_idx].astype(np.int64)
        return x_t, y_t, torch.from_numpy(nbrs).to(DEVICE)


@torch.no_grad()
def eval_with_mask(model, ds, n_batches, batch_size, disabled_layers):
    """Eval with specific CCA layers disabled. disabled_layers is a set of layer indices."""
    model.eval()
    # Temporarily disable CCA layers
    original_flags = {}
    for i, block in enumerate(model.transformer.h):
        if i in disabled_layers and block.use_cca:
            original_flags[i] = True
            block.use_cca = False

    losses = []
    for bi in range(n_batches):
        chunk_indices = np.random.randint(0, ds.max_chunk_idx, size=batch_size)
        x, y, nbrs = ds.get_batch(batch_size, chunk_indices, use_real=True)
        with CTX:
            _, loss = model(x, targets=y, neighbors=nbrs)
        losses.append(loss.item())

    # Restore
    for i, flag in original_flags.items():
        model.transformer.h[i].use_cca = flag

    mean = sum(losses) / len(losses)
    se = stderr(losses)
    return mean, se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default=str(CKPT_PATH))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--heldout-bin", type=str, default="heldout.bin")
    parser.add_argument("--heldout-neighbors", type=str, default="heldout_neighbors.npy")
    parser.add_argument("--cell-tokens", type=str, default="cell_tokens.npy")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"loading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = RetroGPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    data_dir = Path(args.data_dir) if args.data_dir else None
    ds = HeldoutDataset(config, data_dir=data_dir,
                        heldout_bin=args.heldout_bin,
                        heldout_neighbors=args.heldout_neighbors,
                        cell_tokens_path=args.cell_tokens)

    cca_layers = model.cca_layers  # [1, 3, 5, 7]
    print(f"CCA layers: {cca_layers}")
    print(f"Evaluating {args.n_batches} batches x {args.batch_size} samples each condition\n")

    # Define conditions
    conditions = [
        ("all_cca", set()),
        ("no_cca", set(cca_layers)),
    ]
    for layer in cca_layers:
        conditions.append((f"drop_L{layer}", {layer}))
    # Also test keeping only one layer
    for layer in cca_layers:
        others = set(cca_layers) - {layer}
        conditions.append((f"only_L{layer}", others))

    results = {}
    t0 = time.time()
    for name, disabled in conditions:
        # Reset RNG for each condition so same batches are drawn
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        mean, se = eval_with_mask(model, ds, args.n_batches, args.batch_size, disabled)
        results[name] = (mean, se)
        print(f"  {name:>12s}:  {mean:.4f}  (±{se:.4f})  disabled={sorted(disabled) or 'none'}")

    elapsed = time.time() - t0

    print(f"\n{'=' * 72}")
    print("CCA LAYER ABLATION RESULTS")
    print(f"{'=' * 72}")
    print(f"  eval time: {elapsed:.1f}s\n")

    all_loss = results["all_cca"][0]
    no_loss = results["no_cca"][0]
    total_gap = no_loss - all_loss

    print(f"  all_cca (baseline):   {all_loss:.4f}")
    print(f"  no_cca (no retrieval): {no_loss:.4f}")
    print(f"  total retrieval gap:  {total_gap:+.4f}\n")

    print("  --- drop one layer (how much does each contribute?) ---")
    for layer in cca_layers:
        name = f"drop_L{layer}"
        loss, se = results[name]
        degradation = loss - all_loss
        pct = degradation / total_gap * 100 if total_gap > 0 else 0
        print(f"  {name:>12s}:  {loss:.4f}  (±{se:.4f})  "
              f"degradation={degradation:+.4f}  ({pct:.0f}% of gap)")

    print("\n  --- keep only one layer (how much does each provide alone?) ---")
    for layer in cca_layers:
        name = f"only_L{layer}"
        loss, se = results[name]
        benefit = no_loss - loss
        pct = benefit / total_gap * 100 if total_gap > 0 else 0
        print(f"  {name:>12s}:  {loss:.4f}  (±{se:.4f})  "
              f"benefit={benefit:+.4f}  ({pct:.0f}% of gap)")

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()

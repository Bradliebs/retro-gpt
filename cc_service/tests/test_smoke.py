"""Smoke test for the memory bank, without using the real encoder.

We swap in a deterministic fake encoder that maps text to a hash-derived
embedding. This lets us test the persistence and bank logic end-to-end
without downloading MiniLM.

Run:
    pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from service.memory import MemoryBank, apply_whitening, fit_whitening


class FakeEncoder:
    """Deterministic, network-free encoder for testing.

    Produces vectors from a single shared distribution so that fit-time
    (reference corpus) and apply-time (queries, writes) embeddings have
    matched statistics — which is what real MiniLM achieves naturally
    and what whitening assumes.

    Each unique text gets a unique vector via a deterministic hash-derived
    seed. Vectors are sampled from a Gaussian with a fixed shared mean and
    moderate variance, then unit-normalized — approximating MiniLM's output
    shape.
    """
    model_name = "fake-encoder"

    # Shared per-process mean direction, so all fake vectors live in the
    # same anisotropic cone (like real encoder output)
    _SHARED_MEAN: np.ndarray = None  # set on first use
    _DIM: int = 384

    def __init__(self):
        self.dim = self._DIM
        if FakeEncoder._SHARED_MEAN is None:
            # Deterministic shared mean across all FakeEncoder instances
            mean_rng = np.random.default_rng(seed=12345)
            mean = mean_rng.standard_normal(self._DIM).astype(np.float32) * 0.5
            FakeEncoder._SHARED_MEAN = mean

    def _hash_to_vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        # Item-specific Gaussian perturbation around the shared mean
        noise = rng.standard_normal(self._DIM).astype(np.float32) * 0.3
        v = self._SHARED_MEAN + noise
        # Normalize to unit length — matches MiniLM output scale
        return v / np.linalg.norm(v)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return np.stack([self._hash_to_vec(t) for t in texts])

    def encode_one(self, text: str) -> np.ndarray:
        return self._hash_to_vec(text)


@pytest.fixture
def bank():
    """A MemoryBank using a temporary SQLite db and the fake encoder."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_bank.db"
        b = MemoryBank(db_path=str(db_path))
        # Swap the real encoder for our fake
        b.encoder = FakeEncoder()
        yield b
        b.store.close()


def _init_bank(bank: MemoryBank, n: int = 1200) -> None:
    """Initialize with n random reference texts.

    Note: we use 1200 rather than 300 because the fake encoder produces
    high-dimensional vectors (384d) and stable ZCA whitening requires
    more samples than dimensions. Real MiniLM has the same requirement;
    the architecture validation used 2000.
    """
    ref_texts = [f"reference text number {i} about various topics" for i in range(n)]
    bank.init_from_corpus(ref_texts)


def test_init_then_write_then_query(bank: MemoryBank):
    _init_bank(bank)
    assert bank.is_initialized
    assert bank.count() == 0

    # Write two distinct notes
    cid_a = bank.write("the cat sat on the mat")
    cid_b = bank.write("python is a programming language")
    assert cid_a != cid_b
    assert bank.count() == 2

    # Query with the exact text of A — should fire cell A
    hits = bank.query("the cat sat on the mat", include_silent=True, top_k=5)
    assert len(hits) > 0
    # The first hit should be cell A
    assert hits[0].cell_id == cid_a, (
        f"expected first hit to be cell {cid_a}, got {hits[0].cell_id}"
    )


def test_query_with_unrelated_returns_no_fires(bank: MemoryBank):
    _init_bank(bank)
    bank.write("text about apples")
    bank.write("text about oranges")
    # Use include_silent=False so only firing cells are returned
    hits = bank.query("totally different unrelated content", include_silent=False)
    # With fake (Gaussian) embeddings and the validated theta, unrelated content
    # should not fire any cell.
    assert len(hits) == 0


def test_bind_two_cells_then_query_either(bank: MemoryBank):
    _init_bank(bank)
    cid_a = bank.write("apples are red and grow on trees")
    cid_b = bank.write("bananas are yellow and grow on plants")
    result = bank.bind([cid_a, cid_b], label="fruits")
    # All bound items should fire after binding
    assert all(result.items_fire_after_binding), (
        f"bind did not produce firing for all items: "
        f"{result.items_fire_after_binding}"
    )
    # Alignment should be ~1.0
    assert result.alignment_to_mean > 0.95


def test_delete_unblocks_when_no_dependents(bank: MemoryBank):
    _init_bank(bank)
    cid = bank.write("standalone note")
    ok = bank.delete_cell(cid)
    assert ok
    assert bank.get_cell(cid) is None


def test_delete_blocked_when_bound(bank: MemoryBank):
    _init_bank(bank)
    cid_a = bank.write("a")
    cid_b = bank.write("b")
    bank.bind([cid_a, cid_b])
    # Cannot delete cid_a because it's a source of a bound cell
    ok = bank.delete_cell(cid_a)
    assert not ok


def test_persistence_across_reopen(bank: MemoryBank):
    _init_bank(bank)
    cid = bank.write("a persistent note", label="test")
    db_path = bank.store.db_path
    bank.store.close()

    # Reopen
    b2 = MemoryBank(db_path=str(db_path))
    b2.encoder = FakeEncoder()
    assert b2.is_initialized
    assert b2.count() == 1
    row = b2.get_cell(cid)
    assert row is not None
    assert row.label == "test"
    assert row.source_text == "a persistent note"

    # Query still works
    hits = b2.query("a persistent note", include_silent=True, top_k=5)
    assert hits[0].cell_id == cid
    b2.store.close()


def test_whitening_fit_and_apply_shapes(bank: MemoryBank):
    # Standalone unit test for the whitening functions
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((300, 384)).astype(np.float32)
    params = fit_whitening(raw, reference_n=300)
    assert params.mu.shape == (384,)
    assert params.w_matrix.shape == (384, 384)
    assert params.max_norm > 0

    # Apply to one sample
    sample = rng.standard_normal(384).astype(np.float32)
    out = apply_whitening(sample, params)
    assert out.shape == (384,)
    # The max-norm vector from the reference set should have ||x|| = 1 after scaling
    # (Approximately; verify on the full reference)
    transformed = apply_whitening(raw, params)
    norms = np.linalg.norm(transformed, axis=1)
    assert norms.max() <= 1.0 + 1e-4

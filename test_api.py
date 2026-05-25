"""End-to-end API test using FastAPI's TestClient.

Verifies the full HTTP stack: schema validation, route handlers, persistence,
and the bank logic together. Uses the fake encoder so no network/GPU needed.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from tests.test_smoke import FakeEncoder


@pytest.fixture
def client(monkeypatch):
    """Spin up the FastAPI app with a temp db and the fake encoder."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "api_test_bank.db"
        monkeypatch.setenv("CCMEM_DB_PATH", str(db_path))

        # Import after env is set so lifespan picks it up
        from service.main import app
        # Patch the encoder class globally before the lifespan creates a bank
        import service.memory as mem_mod
        original_singleton = mem_mod.EncoderSingleton
        mem_mod.EncoderSingleton = lambda *a, **kw: FakeEncoder()

        try:
            with TestClient(app) as c:
                yield c
        finally:
            mem_mod.EncoderSingleton = original_singleton


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"]


def test_info_before_init(client):
    r = client.get("/info")
    assert r.status_code == 200
    data = r.json()
    assert data["initialized"] is False
    assert data["n_cells"] == 0


def test_init_then_write_then_query(client):
    ref_texts = [f"reference text {i} about various topics" for i in range(1200)]
    r = client.post("/init", json={"reference_texts": ref_texts})
    assert r.status_code == 200, r.text
    assert r.json()["initialized"]

    # Write a note
    r = client.post("/write", json={"text": "the cat sat on the mat",
                                      "label": "cat"})
    assert r.status_code == 200
    cell_id = r.json()["cell_id"]

    # Query for it
    r = client.post("/query", json={"text": "the cat sat on the mat",
                                      "top_k": 5, "include_silent": True})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) > 0
    assert hits[0]["cell_id"] == cell_id


def test_bind_via_api(client):
    ref_texts = [f"reference text {i}" for i in range(1200)]
    client.post("/init", json={"reference_texts": ref_texts})

    a = client.post("/write", json={"text": "apples are red"}).json()["cell_id"]
    b = client.post("/write", json={"text": "bananas are yellow"}).json()["cell_id"]

    r = client.post("/bind", json={"source_cell_ids": [a, b], "label": "fruit"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert all(data["items_fire_after_binding"])
    assert data["alignment_to_mean"] > 0.95


def test_cannot_init_twice(client):
    ref_texts = [f"reference text {i}" for i in range(1200)]
    r1 = client.post("/init", json={"reference_texts": ref_texts})
    assert r1.status_code == 200
    r2 = client.post("/init", json={"reference_texts": ref_texts})
    assert r2.status_code == 409  # conflict


def test_write_before_init_is_400(client):
    r = client.post("/write", json={"text": "should fail"})
    assert r.status_code == 400


def test_list_and_show_and_delete(client):
    ref_texts = [f"reference text {i}" for i in range(1200)]
    client.post("/init", json={"reference_texts": ref_texts})

    cid = client.post("/write", json={"text": "deletable note"}).json()["cell_id"]

    r = client.get("/cells")
    assert r.status_code == 200
    assert r.json()["total"] == 1

    r = client.get(f"/cells/{cid}")
    assert r.status_code == 200
    assert r.json()["source_text"] == "deletable note"

    r = client.delete(f"/cells/{cid}")
    assert r.status_code == 200
    assert r.json()["deleted"]

    r = client.get(f"/cells/{cid}")
    assert r.status_code == 404

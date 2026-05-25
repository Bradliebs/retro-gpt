# Concept Cells Service (ccmem)

A research-notebook memory service built on the Concept Cells architecture
(see the parent `concept_cells/ARCHITECTURE.md` for the underlying mechanism).

## What this does

- **Add notes** (paragraphs of text) — each becomes a concept cell.
- **Query** with a question or topic — returns matching cells with confidence.
- **Bind** related notes into a single composite concept cell. The bound cell
  fires for any of the source notes.
- **Inspect, edit, and delete** cells. Memory is transparent and modifiable.

The architecture validated in `concept_cells/` supports M=2000+ cells and
binding up to m=8 items per bound cell on real text embeddings, with
near-perfect recall and near-zero false positives.

## Setup

```bash
pip install -r requirements.txt

# First-time init: fits whitening parameters from a reference corpus.
# This is a one-time step. Frozen parameters are stored in the bank file.
ccmem init --reference-corpus wikitext --reference-n 2000

# Add a note (becomes one concept cell)
ccmem add "The double-slit experiment shows that electrons exhibit wave-particle duality..."

# Add notes from a file (one paragraph per blank-line-separated section)
ccmem add --file my_notes.txt

# Query
ccmem query "what is wave-particle duality"

# Bind multiple cells into one concept
ccmem bind 3 7 12 --label "wave-particle duality fundamentals"

# List cells, inspect one
ccmem list
ccmem show 7
ccmem delete 7
```

## Architecture

```
┌──────────────┐    HTTP    ┌────────────────────────┐    ┌──────────┐
│  ccmem CLI   │ ─────────► │  FastAPI service       │ ─► │ SQLite   │
└──────────────┘            │  (memory.py, encoder)  │    │  bank.db │
                            └────────────────────────┘    └──────────┘
                                       │
                                       └─► MiniLM encoder (loaded once)
```

The service holds the encoder in memory; the SQLite file holds:
- Whitening parameters (frozen at init time)
- Per-cell `(weight_vector, threshold, label, metadata)` rows
- Per-bind history (which cells were composed into which)

## Files

| Path | Purpose |
|---|---|
| `service/main.py` | FastAPI entry point |
| `service/memory.py` | `MemoryBank`: ties encoder, preprocessing, bank, and SQLite |
| `service/encoder.py` | MiniLM wrapper, loaded once per process |
| `service/persistence.py` | SQLite schema + read/write |
| `service/schema.py` | Pydantic request/response models |
| `cli/ccmem.py` | Command-line client |
| `tests/test_api.py` | Smoke tests |

## Status

v0.1 — single-machine, single-bank, no auth, no concurrency control beyond
SQLite's. Intended for personal use against your own notes corpus.

# Session Log — May 23, 2026

End-to-end build of the cc_service memory bank: scaffolding → tests → web UI →
bulk loaders → GPU → data load → diagnostic tests → bug fix.

## What got built

### Package structure (refactored from flat files)

```
cc_service/
├── service/          # FastAPI app
│   ├── main.py       # routes
│   ├── memory.py     # MemoryBank (encoder + whitening + persistence)
│   ├── encoder.py    # sentence-transformers wrapper
│   ├── persistence.py # SQLite store
│   ├── schema.py     # Pydantic request/response models
│   └── static/
│       └── index.html # web dashboard
├── cli/
│   ├── ccmem.py      # Typer CLI client
│   └── bulk_load.py  # NEW — load a folder of text/markdown files
├── tests/
│   ├── test_smoke.py
│   └── test_api.py
├── load_wikipedia.py # NEW — Simple English Wikipedia loader
├── load_coding.py    # NEW — CodeAlpaca-20k loader
├── test_coding.py    # NEW — coding-content sanity check
├── test_full.py      # NEW — 8-group diagnostic suite
├── README.md
├── requirements.txt
└── bank.db           # 0.62 GB
```

### New endpoints / surfaces

- `GET /` redirects to `/ui/index.html`
- `GET /ui/*` serves the dashboard (single-page app, dark theme)
- Dashboard tabs: Query, Add, Bind, Detail; sidebar with cell list + filter;
  bind checkboxes when on the Bind tab; toast notifications; auto-refresh
  health every 30s; `Enter` to query, `Ctrl+Enter` to add.

### CUDA acceleration

- Installed `torch 2.6.0+cu124` for RTX 3070 (8 GB VRAM):
  `pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps`
- Service uses GPU when launched with `$env:CCMEM_DEVICE = 'cuda'`.

### PowerShell profile

Aliases added: `ccmem`, `note`.

## Data loaded

| Source | Cells | Notes |
|---|---:|---|
| `wikitext` reference corpus | 2,000 | Used once to fit ZCA whitening (frozen) |
| Simple English Wikipedia (`wikimedia/wikipedia` `20231101.simple`) | ~253,000 | Paragraphs, min 80 chars |
| CodeAlpaca-20k (`sahil2801/CodeAlpaca-20k`) | 16,100 | `[code]` label prefix |
| **Total** | **269,526** | dim=384, db=0.62 GB |

Wikipedia load was interrupted twice (SQLite concurrency crash, then by the
deliberate service restart for the bug fix). All loaded data persists. Can
resume with `load_wikipedia.py --skip-articles N`.

## Tests

### Unit tests
- 14/14 passing in `tests/test_smoke.py` and `tests/test_api.py`.

### Diagnostic suite (`test_full.py`, 8 groups)

| Test | Result |
|---|---|
| 1. Cross-domain knowledge retrieval (10 queries) | 10/10 relevant |
| 2. Programming / CS retrieval (10 queries) | 9/10 perfect (CAP theorem not loaded) |
| 3. Disambiguation (Python/Apple/Mercury/etc.) | 8/10 correct |
| 4. Top-5 diversity on open query | All on-topic |
| 5. Activation profile across queries | Healthy spread |
| 6. Hebbian binding (fuse + query + cleanup) | PASS — bound cell ranked #3 |
| 7. Edge cases (gibberish, empty, single char) | Surfaced empty-string bug |
| 8. Latency (20 queries) | median 94 ms, p95 6.3 s (under load) |

Latency p95 is elevated because the Wikipedia loader was hammering SQLite at
the time. Without the loader, queries are sub-100 ms.

## Bugs found and fixed

### Empty-query 422 (fixed)

- **Symptom**: `POST /query` with `{"text": ""}` returned 422 instead of empty hits.
- **Root cause**: `QueryRequest.text: str = Field(..., min_length=1)` in schema.
- **Fix**:
  - [service/schema.py](cc_service/service/schema.py#L51-L54) — `text: str = ""`
  - [service/main.py](cc_service/service/main.py#L141-L150) — early return when
    `req.text.strip()` is empty
- **Verified**: `POST /query {"text":""}` now returns `200 {"n_hits": 0, "hits": []}`.

## Known issues / not yet fixed

### SQLite concurrency

- Running two loaders in parallel (or a loader plus heavy dashboard use) causes
  `sqlite3.OperationalError: cannot commit - no transaction is active` and the
  service crashes.
- **Workaround**: one writer at a time.
- **Real fix would be**: per-thread connections, `PRAGMA journal_mode=WAL`,
  retry-on-busy. Not done yet.

### Wikipedia loader incomplete

- ~253k of an estimated ~600k paragraphs loaded.
- Resume with: `python load_wikipedia.py --skip-articles N` where N is the
  number already processed. Do not run other loaders concurrently.

### No git commit yet

- `.git` initialized, `.gitignore` in place, but nothing committed.

## How to run things

### Start service (GPU)
```powershell
$env:CCMEM_DB_PATH = 'h:\MiniLM\cc_service\bank.db'
$env:CCMEM_DEVICE  = 'cuda'
h:\MiniLM\cc_service\.venv\Scripts\python.exe -m uvicorn service.main:app `
    --host 127.0.0.1 --port 8765 --app-dir h:\MiniLM\cc_service
```

### Use the dashboard
Open <http://127.0.0.1:8765> in a browser.

### Resume Wikipedia load
```powershell
h:\MiniLM\cc_service\.venv\Scripts\python.exe h:\MiniLM\cc_service\load_wikipedia.py --skip-articles <N>
```

### Run unit tests
```powershell
cd h:\MiniLM\cc_service
.\.venv\Scripts\python.exe -m pytest
```

### Run diagnostic suite
```powershell
h:\MiniLM\cc_service\.venv\Scripts\python.exe h:\MiniLM\cc_service\test_full.py
```

## Files created this session

- [cc_service/service/static/index.html](cc_service/service/static/index.html) — dashboard
- [cc_service/cli/bulk_load.py](cc_service/cli/bulk_load.py) — folder loader
- [cc_service/load_wikipedia.py](cc_service/load_wikipedia.py)
- [cc_service/load_coding.py](cc_service/load_coding.py)
- [cc_service/test_loaded.py](cc_service/test_loaded.py)
- [cc_service/test_coding.py](cc_service/test_coding.py)
- [cc_service/test_full.py](cc_service/test_full.py)
- [cc_service/_stat.py](cc_service/_stat.py) — quick stats helper
- [cc_service/SESSION_LOG.md](cc_service/SESSION_LOG.md) — this file

## Files modified this session

- [cc_service/service/main.py](cc_service/service/main.py) — `/` redirect, `/ui` mount, empty-query early return
- [cc_service/service/schema.py](cc_service/service/schema.py) — relaxed `QueryRequest.text` validation
- [cc_service/cli/ccmem.py](cc_service/cli/ccmem.py) — configurable HTTP timeout

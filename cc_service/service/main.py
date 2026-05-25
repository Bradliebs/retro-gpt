"""FastAPI service exposing the concept-cell memory bank.

Run with:
    uvicorn service.main:app --host 0.0.0.0 --port 8765

Environment variables:
    CCMEM_DB_PATH      path to the SQLite bank file (default: ./bank.db)
    CCMEM_DEVICE       cuda | cpu | mps (default: auto)
    CCMEM_ENCODER      sentence-transformers model name (default: all-MiniLM-L6-v2)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .memory import MemoryBank
from .schema import (
    BindRequest, BindResponse,
    CellOut, DeleteResponse,
    InfoResponse,
    InitRequest, InitResponse,
    ListCellsResponse,
    QueryHitOut, QueryRequest, QueryResponse,
    WriteManyRequest, WriteManyResponse,
    WriteRequest, WriteResponse,
)


# ---------- Lifespan ----------

_bank: Optional[MemoryBank] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bank
    db_path = os.environ.get("CCMEM_DB_PATH", "./bank.db")
    device = os.environ.get("CCMEM_DEVICE")
    encoder = os.environ.get("CCMEM_ENCODER", "all-MiniLM-L6-v2")
    _bank = MemoryBank(db_path=db_path, encoder_model=encoder, device=device)
    yield
    _bank.store.close()


app = FastAPI(
    title="Concept Cells Memory Service",
    version="0.1.0",
    description="One-shot, high-dimensional concept-cell memory with Hebbian binding.",
    lifespan=lifespan,
)


def get_bank() -> MemoryBank:
    if _bank is None:
        raise HTTPException(status_code=503, detail="Bank not loaded.")
    return _bank


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/index.html")


app.mount("/ui", StaticFiles(
    directory=str(Path(__file__).parent / "static"),
), name="ui")


# ---------- Health & info ----------

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(**get_bank().info())


# ---------- Init ----------

@app.post("/init", response_model=InitResponse)
def init_bank(req: InitRequest):
    bank = get_bank()
    if bank.is_initialized:
        raise HTTPException(
            status_code=409,
            detail="Bank already initialized. Delete the db file to re-init.",
        )
    try:
        params = bank.init_from_corpus(req.reference_texts)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return InitResponse(
        initialized=True,
        reference_n=params.reference_n,
        fitted_at=params.fitted_at,
        dim=bank.encoder.dim,
    )


# ---------- Write ----------

@app.post("/write", response_model=WriteResponse)
def write(req: WriteRequest):
    bank = get_bank()
    if not bank.is_initialized:
        raise HTTPException(status_code=400,
                              detail="Bank not initialized. POST /init first.")
    try:
        cell_id = bank.write(req.text, label=req.label, theta=req.theta)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    row = bank.get_cell(cell_id)
    return WriteResponse(cell_id=cell_id, label=row.label, theta=row.theta)


@app.post("/write_many", response_model=WriteManyResponse)
def write_many(req: WriteManyRequest):
    bank = get_bank()
    if not bank.is_initialized:
        raise HTTPException(status_code=400,
                              detail="Bank not initialized. POST /init first.")
    try:
        cell_ids = bank.write_many(req.texts, labels=req.labels, theta=req.theta)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return WriteManyResponse(cell_ids=cell_ids)


# ---------- Query ----------

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    bank = get_bank()
    if not bank.is_initialized:
        raise HTTPException(status_code=400,
                              detail="Bank not initialized. POST /init first.")
    if not req.text.strip():
        return QueryResponse(query=req.text, n_hits=0, hits=[])
    hits = bank.query(req.text, top_k=req.top_k,
                       include_silent=req.include_silent)
    hits_out = [QueryHitOut(
        cell_id=h.cell_id, label=h.label, kind=h.kind,
        activation=h.activation, margin=h.margin,
        source_text=h.source_text, source_cell_ids=h.source_cell_ids,
    ) for h in hits]
    return QueryResponse(query=req.text, n_hits=len(hits_out), hits=hits_out)


# ---------- Bind ----------

@app.post("/bind", response_model=BindResponse)
def bind(req: BindRequest):
    bank = get_bank()
    if not bank.is_initialized:
        raise HTTPException(status_code=400,
                              detail="Bank not initialized. POST /init first.")
    try:
        result = bank.bind(
            source_cell_ids=req.source_cell_ids,
            label=req.label,
            n_steps=req.n_steps,
            safety_margin=req.safety_margin,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return BindResponse(
        bound_cell_id=result.bound_cell_id,
        source_cell_ids=result.source_cell_ids,
        theta_readout=result.theta_readout,
        alignment_to_mean=result.alignment_to_mean,
        items_fire_after_binding=result.items_fire_after_binding,
    )


# ---------- Cells ----------

@app.get("/cells", response_model=ListCellsResponse)
def list_cells(limit: int = 100, offset: int = 0, kind: Optional[str] = None):
    bank = get_bank()
    cells = bank.list_cells(limit=limit, offset=offset, kind=kind)
    out = [CellOut(
        id=c.id, label=c.label, theta=c.theta, kind=c.kind,
        created_at=c.created_at, source_text=c.source_text,
        source_cell_ids=c.source_cell_ids,
    ) for c in cells]
    return ListCellsResponse(total=bank.count(), cells=out)


@app.get("/cells/{cell_id}", response_model=CellOut)
def get_cell(cell_id: int):
    bank = get_bank()
    row = bank.get_cell(cell_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found")
    return CellOut(
        id=row.id, label=row.label, theta=row.theta, kind=row.kind,
        created_at=row.created_at, source_text=row.source_text,
        source_cell_ids=row.source_cell_ids,
    )


@app.delete("/cells/{cell_id}", response_model=DeleteResponse)
def delete_cell(cell_id: int):
    bank = get_bank()
    if bank.get_cell(cell_id) is None:
        raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found")
    ok = bank.delete_cell(cell_id)
    if not ok:
        return DeleteResponse(
            deleted=False, cell_id=cell_id,
            reason="Cell is a source of one or more bound cells; "
                    "delete the bound cells first.",
        )
    return DeleteResponse(deleted=True, cell_id=cell_id)

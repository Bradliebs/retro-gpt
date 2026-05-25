"""Pydantic models for the API request/response shapes."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------- Init ----------

class InitRequest(BaseModel):
    reference_texts: List[str] = Field(
        ..., min_length=200,
        description="Texts used to fit the whitening parameters. Minimum 200.",
    )


class InitResponse(BaseModel):
    initialized: bool
    reference_n: int
    fitted_at: float
    dim: int


# ---------- Write ----------

class WriteRequest(BaseModel):
    text: str = Field(..., min_length=1)
    label: Optional[str] = None
    theta: Optional[float] = None


class WriteResponse(BaseModel):
    cell_id: int
    label: Optional[str]
    theta: float


class WriteManyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    labels: Optional[List[Optional[str]]] = None
    theta: Optional[float] = None


class WriteManyResponse(BaseModel):
    cell_ids: List[int]


# ---------- Query ----------

class QueryRequest(BaseModel):
    text: str = ""
    top_k: int = Field(default=10, ge=1, le=100)
    include_silent: bool = False


class QueryHitOut(BaseModel):
    cell_id: int
    label: Optional[str]
    kind: str
    activation: float
    margin: float
    source_text: Optional[str]
    source_cell_ids: Optional[List[int]]


class QueryResponse(BaseModel):
    query: str
    n_hits: int
    hits: List[QueryHitOut]


# ---------- Bind ----------

class BindRequest(BaseModel):
    source_cell_ids: List[int] = Field(..., min_length=2)
    label: Optional[str] = None
    n_steps: Optional[int] = None
    safety_margin: Optional[float] = None


class BindResponse(BaseModel):
    bound_cell_id: int
    source_cell_ids: List[int]
    theta_readout: float
    alignment_to_mean: float
    items_fire_after_binding: List[bool]


# ---------- Cell inspection ----------

class CellOut(BaseModel):
    id: int
    label: Optional[str]
    theta: float
    kind: str
    created_at: float
    source_text: Optional[str] = None
    source_cell_ids: Optional[List[int]] = None


class ListCellsResponse(BaseModel):
    total: int
    cells: List[CellOut]


class DeleteResponse(BaseModel):
    deleted: bool
    cell_id: int
    reason: Optional[str] = None


# ---------- Info ----------

class InfoResponse(BaseModel):
    db_path: str
    encoder_model: str
    dim: int
    initialized: bool
    n_cells: int
    whitening_fitted_at: Optional[float]
    whitening_reference_n: Optional[int]

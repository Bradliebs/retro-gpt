"""SQLite persistence for the concept-cell bank.

Schema (intentionally minimal):

  meta              key/value table for bank-wide settings (dim, encoder, etc.)
  whitening         the fitted whitening parameters (mu, W_matrix, max_norm)
                    stored as one row with BLOB-packed float32 arrays
  cells             one row per concept cell:
                      id, label, weight (BLOB), theta, kind, created_at
                    kind ∈ {"single", "bound"}
  bound_sources     for kind="bound" cells, the source cell IDs they were bound from
  source_texts      original text that was written, indexed by cell_id (for "single"
                    cells; "bound" cells inherit through bound_sources)

Embeddings (the float32 weight vectors) live as BLOBs because:
 - they're fixed-length per bank (whitening fixes the dim)
 - SQLite handles ~1KB blobs efficiently for our N
 - this avoids an extra sidecar file (.npz + .db) that could desynchronise

The whole bank is a single .db file. Backups are `cp bank.db backup.db`.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------- Schema ----------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS whitening (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    mu          BLOB NOT NULL,
    w_matrix    BLOB NOT NULL,
    max_norm    REAL NOT NULL,
    fitted_at   REAL NOT NULL,
    reference_n INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cells (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT,
    weight      BLOB NOT NULL,
    theta       REAL NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('single', 'bound')),
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS bound_sources (
    bound_cell_id  INTEGER NOT NULL,
    source_cell_id INTEGER NOT NULL,
    PRIMARY KEY (bound_cell_id, source_cell_id),
    FOREIGN KEY (bound_cell_id) REFERENCES cells(id) ON DELETE CASCADE,
    FOREIGN KEY (source_cell_id) REFERENCES cells(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS source_texts (
    cell_id INTEGER PRIMARY KEY,
    text    TEXT NOT NULL,
    FOREIGN KEY (cell_id) REFERENCES cells(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cells_kind ON cells(kind);
CREATE INDEX IF NOT EXISTS idx_bound_sources_source ON bound_sources(source_cell_id);
"""


# ---------- Data classes ----------

@dataclass
class WhiteningParams:
    """Frozen whitening + ball-scaling parameters.

    Apply via: x_new = ((x_raw - mu) @ w_matrix) / max_norm
    """
    mu: np.ndarray          # (D,)
    w_matrix: np.ndarray    # (D, D)
    max_norm: float
    reference_n: int
    fitted_at: float


@dataclass
class CellRow:
    """One row from the cells table."""
    id: int
    label: Optional[str]
    weight: np.ndarray      # (D,)
    theta: float
    kind: str               # "single" or "bound"
    created_at: float
    source_text: Optional[str] = None
    source_cell_ids: Optional[List[int]] = None  # populated for "bound" cells


# ---------- Bank store ----------

class BankStore:
    """SQLite-backed store for the concept-cell bank.

    Single connection per process. SQLite is fine for that — we don't expect
    concurrent writers, and reads are serialised by the GIL anyway.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- Meta ---

    def set_meta(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_meta(self, key: str, default=None):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else default

    # --- Whitening ---

    def save_whitening(self, params: WhiteningParams) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO whitening
               (id, mu, w_matrix, max_norm, fitted_at, reference_n)
               VALUES (1, ?, ?, ?, ?, ?)""",
            (
                params.mu.astype(np.float32).tobytes(),
                params.w_matrix.astype(np.float32).tobytes(),
                float(params.max_norm),
                float(params.fitted_at),
                int(params.reference_n),
            ),
        )
        self.conn.commit()

    def load_whitening(self) -> Optional[WhiteningParams]:
        row = self.conn.execute(
            "SELECT mu, w_matrix, max_norm, fitted_at, reference_n FROM whitening WHERE id = 1"
        ).fetchone()
        if not row:
            return None
        mu_bytes, w_bytes, max_norm, fitted_at, reference_n = row
        # Recover shape from the stored dim
        dim = int(self.get_meta("dim", 0))
        if dim == 0:
            raise RuntimeError("Bank has whitening but no recorded dim — db corrupt.")
        mu = np.frombuffer(mu_bytes, dtype=np.float32).reshape((dim,)).copy()
        w_matrix = np.frombuffer(w_bytes, dtype=np.float32).reshape((dim, dim)).copy()
        return WhiteningParams(
            mu=mu, w_matrix=w_matrix, max_norm=float(max_norm),
            reference_n=int(reference_n), fitted_at=float(fitted_at),
        )

    # --- Cells ---

    def insert_cell(self, label: Optional[str], weight: np.ndarray,
                    theta: float, kind: str,
                    source_text: Optional[str] = None,
                    source_cell_ids: Optional[List[int]] = None) -> int:
        cursor = self.conn.execute(
            """INSERT INTO cells (label, weight, theta, kind, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                label,
                weight.astype(np.float32).tobytes(),
                float(theta),
                kind,
                time.time(),
            ),
        )
        cell_id = cursor.lastrowid
        if source_text is not None:
            self.conn.execute(
                "INSERT INTO source_texts (cell_id, text) VALUES (?, ?)",
                (cell_id, source_text),
            )
        if source_cell_ids:
            self.conn.executemany(
                "INSERT INTO bound_sources (bound_cell_id, source_cell_id) VALUES (?, ?)",
                [(cell_id, sid) for sid in source_cell_ids],
            )
        self.conn.commit()
        return cell_id

    def get_cell(self, cell_id: int) -> Optional[CellRow]:
        row = self.conn.execute(
            """SELECT c.id, c.label, c.weight, c.theta, c.kind, c.created_at,
                      s.text
                 FROM cells c
                 LEFT JOIN source_texts s ON s.cell_id = c.id
                 WHERE c.id = ?""",
            (cell_id,),
        ).fetchone()
        if not row:
            return None
        cid, label, weight_bytes, theta, kind, created_at, text = row
        dim = int(self.get_meta("dim"))
        weight = np.frombuffer(weight_bytes, dtype=np.float32).reshape((dim,)).copy()

        source_cell_ids = None
        if kind == "bound":
            source_cell_ids = [r[0] for r in self.conn.execute(
                "SELECT source_cell_id FROM bound_sources WHERE bound_cell_id = ?",
                (cell_id,),
            ).fetchall()]

        return CellRow(
            id=cid, label=label, weight=weight, theta=theta, kind=kind,
            created_at=created_at, source_text=text,
            source_cell_ids=source_cell_ids,
        )

    def list_cells(self, limit: int = 1000, offset: int = 0,
                    kind: Optional[str] = None) -> List[CellRow]:
        sql = "SELECT id FROM cells"
        params: tuple = ()
        if kind is not None:
            sql += " WHERE kind = ?"
            params = (kind,)
        sql += " ORDER BY id ASC LIMIT ? OFFSET ?"
        params = params + (limit, offset)
        ids = [r[0] for r in self.conn.execute(sql, params).fetchall()]
        return [self.get_cell(i) for i in ids]

    def all_weights_and_thresholds(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Return stacked (W, theta, cell_ids) for vectorised query.

        Used by MemoryBank to compute activations in a single matmul.
        """
        rows = self.conn.execute(
            "SELECT id, weight, theta FROM cells ORDER BY id ASC"
        ).fetchall()
        if not rows:
            dim = int(self.get_meta("dim", 0))
            return (np.zeros((0, dim), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    [])
        dim = int(self.get_meta("dim"))
        weights = np.stack([
            np.frombuffer(r[1], dtype=np.float32).reshape((dim,)).copy()
            for r in rows
        ])
        thetas = np.array([r[2] for r in rows], dtype=np.float32)
        cell_ids = [r[0] for r in rows]
        return weights, thetas, cell_ids

    def delete_cell(self, cell_id: int) -> bool:
        """Delete a cell. Returns False if the cell is the source of any bound cell.

        We refuse to delete a cell that has been bound into another cell, because
        that would break the audit trail. The user has to delete the bound cell
        first, then the source.
        """
        bound_into = self.conn.execute(
            "SELECT COUNT(*) FROM bound_sources WHERE source_cell_id = ?",
            (cell_id,),
        ).fetchone()[0]
        if bound_into > 0:
            return False
        cur = self.conn.execute("DELETE FROM cells WHERE id = ?", (cell_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0]

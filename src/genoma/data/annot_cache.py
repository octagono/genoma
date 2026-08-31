"""Persistent local SQLite cache for REST annotations (VEP, Phen2Gene).

Variants are static genomic coordinates — their VEP/Phen2Gene annotations
never change. A local cache makes repeated test runs, debug sessions, and
self-consistency loops free: zero external network budget, milliseconds per
lookup. SQLite (stdlib ``sqlite3``) — no new dependency, crash-safe via
WAL, and trivially purged by ``compliance.purge_caches`` (it lives inside
``cfg.DATA_DIR`` and is genome-carrying material).

Design:
- Keyed by the canonical variant coordinate (chr:pos:ref>alt), value is the
  full annotation JSON blob.
- Never stale within a dataset version: the cache is keyed to
  ``cfg.DATA_VERSION`` so a new dataset version never reads old entries.
- Fail-safe: a corrupted/locked DB degrades to a cache miss (re-fetch), never
  a crash — annotation correctness is more important than cache hits.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from genoma import config as cfg

_TABLE = "annotations"
_KEY_COL = "coord"
_VAL_COL = "payload"
_VER_COL = "version"


def cache_path() -> Path:
    """Single source of truth for the annotation cache location."""
    return cfg.DATA_DIR / "annotation_cache.db"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    db = db_path or cache_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
            {_KEY_COL} TEXT PRIMARY KEY,
            {_VER_COL} TEXT NOT NULL,
            {_VAL_COL} TEXT NOT NULL
        )"""
    )
    return conn


def cache_get(coord: str, db_path: Path | None = None) -> dict | None:
    """Return the cached annotation dict for ``coord``, or None on any issue.

    A cache miss (or DB problem) must never block the pipeline — callers
    re-fetch from the network in that case.
    """
    try:
        conn = _connect(db_path)
        try:
            row = conn.execute(
                f"SELECT {_VAL_COL} FROM {_TABLE} WHERE {_KEY_COL}=? AND {_VER_COL}=?",
                (coord, cfg.DATA_VERSION),
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else None
    except sqlite3.Error, json.JSONDecodeError, OSError:
        return None


def cache_set(coord: str, payload: dict, db_path: Path | None = None) -> None:
    """Upsert the annotation dict for ``coord``. Best-effort: a write failure
    only costs future cache hits, never correctness."""
    try:
        conn = _connect(db_path)
        try:
            conn.execute(
                f"INSERT OR REPLACE INTO {_TABLE} ({_KEY_COL}, {_VER_COL}, {_VAL_COL}) "
                "VALUES (?, ?, ?)",
                (coord, cfg.DATA_VERSION, json.dumps(payload)),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error, OSError:
        pass


def cache_size(db_path: Path | None = None) -> int:
    """Number of cached annotations (0 if the DB is absent/broken)."""
    db = db_path or cache_path()
    if not db.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db), timeout=5.0)
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0

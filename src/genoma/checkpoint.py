"""Chunked parallel scoring with JSONL checkpointing + opt-in resilience.

Policy (matches the documented fail-fast contract):

- **Default = fail-fast.** A per-candidate error aborts the run loudly. This
  also FIXES a silent-decay bug in the old parallel path (``dspy.Parallel``
  returns exceptions as results, and the old caller dropped them with a bare
  ``continue`` — a failed candidate silently vanished).

- **``resilient=True`` is the production-mode exception, opt-in per call.**
  A failing candidate logs a LOUD warning and yields the caller-provided
  conservative default record (never a silent placeholder) while the rest of
  the batch continues.

- **Checkpoints:** after every chunk, per-candidate records are appended to a
  JSONL file. A restarted run skips candidates already checkpointed (resume),
  so an interrupted run continues from the last chunk, not from candidate #1.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import dspy

logger = logging.getLogger("genoma.checkpoint")


def _coord_of(candidate) -> str:
    coord = getattr(candidate, "coord", None)
    if coord:
        return str(coord)
    chrom = getattr(candidate, "chrom", None)
    if chrom:
        return f"{chrom}:{getattr(candidate, 'pos', '')}:{getattr(candidate, 'ref', '')}>{getattr(candidate, 'alt', '')}"
    return str(candidate)  # opaque fallback (e.g. test doubles)


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """coord -> record map from a JSONL checkpoint."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        done[rec["coord"]] = rec
    return done


def _append_checkpoint(path: Path, coord: str, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"coord": coord, **record}) + "\n")


def score_chunked(
    candidates: list,
    hpo_terms: list[str],
    score_one,  # (candidate, hpo_terms) -> dict (serialized result record)
    default_record,  # (candidate, error_msg) -> dict (conservative fallback)
    *,
    checkpoint_path: str | Path | None = None,
    chunk_size: int = 50,
    num_threads: int = 8,
    resilient: bool = False,
) -> list[tuple]:
    """Score candidates in parallel chunks with checkpointing + resilience.

    Returns ``[(candidate, record), ...]`` in original candidate order.
    ``record`` is the dict returned by ``score_one`` (or ``default_record``
    on a resilient-caught failure).
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    done = _load_checkpoint(ckpt) if ckpt else {}
    pending = [c for c in candidates if _coord_of(c) not in done]

    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        chunk_records = _run_chunk(
            chunk, hpo_terms, score_one, default_record, resilient, num_threads
        )
        for c, rec in chunk_records:
            done[_coord_of(c)] = rec
            if ckpt:
                _append_checkpoint(ckpt, _coord_of(c), rec)

    # Assemble in original order: checkpointed (resumed) + freshly scored.
    ordered: list[tuple] = []
    for c in candidates:
        rec = done.get(_coord_of(c))
        if rec is not None:
            ordered.append((c, rec))
    return ordered


def _run_chunk(
    chunk: list,
    hpo_terms: list[str],
    score_one,
    default_record,
    resilient: bool,
    num_threads: int,
) -> list[tuple]:
    # dspy.Parallel returns None (and logs) for a plain-callable that raises —
    # it does NOT surface the exception as a result the way it does for
    # dspy.Module. Wrap so the real error survives to the fail-fast raise.
    def _call(candidate, hpo_terms):
        try:
            return score_one(candidate, hpo_terms)
        except BaseException as e:  # noqa: BLE001 -- re-raised or logged below
            return ("__error__", e)

    examples = [
        dspy.Example(candidate=c, hpo_terms=hpo_terms).with_inputs("candidate", "hpo_terms")
        for c in chunk
    ]
    runner = dspy.Parallel(num_threads=num_threads, access_examples=True)
    results = runner([(_call, ex) for ex in examples])

    out: list[tuple] = []
    for c, r in zip(chunk, results, strict=True):
        if isinstance(r, tuple) and r and r[0] == "__error__":
            err = r[1]
            if not resilient:
                # Fail-fast: a broken prediction must not silently vanish.
                raise err
            msg = (
                f"candidate {_coord_of(c)} failed ({type(err).__name__}: {err}); using safe default"
            )
            logger.error(msg)
            print(f"⚠️  {msg}")
            out.append((c, default_record(c, str(err))))
            continue
        if r is None:
            # dspy.Parallel logged a failure but returned None; treat as
            # opaque failure (exact error unavailable).
            if not resilient:
                raise RuntimeError(f"candidate {_coord_of(c)} failed in parallel scoring")
            msg = f"candidate {_coord_of(c)} failed (unknown parallel error); using safe default"
            logger.error(msg)
            print(f"⚠️  {msg}")
            out.append((c, default_record(c, "unknown parallel error")))
            continue
        out.append((c, r))
    return out

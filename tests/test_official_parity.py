"""Differential tests: our metrics.py must match the OFFICIAL challenge scorer.

Loads the official evaluation.py from the local challenge clone and compares
rank_points + f_max on identical inputs. This is the alignment guarantee
for pre-submission validation matching what the official scorer computes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from genoma.data.submission import (
    pair_row,
    single_variant_row,
)
from genoma.track1.metrics import AnswerKey, f_max, rank_points

# ── Official scorer import (local challenge clone) ──────────────────────────
SPACE_ROOT = Path(__file__).resolve().parent.parent / "mva-hackathon-2026"
sys.path.insert(0, str(SPACE_ROOT))

pytest.importorskip("evaluation", reason="Space clone not present")
import evaluation  # noqa: E402  # deliberate: requires sys.path setup above

T1 = ("chr2", 47908000, "G", "A")
T2 = ("chr2", 47912000, "C", "T")
D = ("chr9", 1, "A", "T")


def _coord(t) -> str:
    return f"{t[0]}:{t[1]}:{t[2]}>{t[3]}"


ANSWER = AnswerKey(causal_variants={_coord(T1), _coord(T2)}, compound_het=True)


def _official(rows):
    """Run the official scorer; return (rank_points, f_max).

    Mirrors the official CSV parser: ref/alt are normalized via strip+upper
    (evaluation._parse_variant) before becoming Variant tuples.
    """
    official_rows = []
    for i, r in enumerate(rows, start=1):
        v1 = (r.chrom_1, r.pos_1, r.ref_1.strip().upper(), r.alt_1.strip().upper())
        variants = frozenset([v1])
        if r.chrom_2:
            v2 = (r.chrom_2, r.pos_2, r.ref_2.strip().upper(), r.alt_2.strip().upper())
            variants = frozenset([v1, v2])
        official_rows.append(
            evaluation.SubmissionRow(
                variants=variants, epcr=r.epcr, rank=i, finding_type=r.finding_type
            )
        )
    res = evaluation.score_proband("PROBAND01", official_rows, frozenset([T1, T2]))
    return res.rank_points, res.f_max


def _assert_match(rows):
    mine = (rank_points(rows, ANSWER), f_max(rows, ANSWER))
    official = _official(rows)
    assert mine == pytest.approx(official, abs=1e-9), f"mine={mine} official={official}"


def test_differential_full_pair_rank1():
    rows = [pair_row(*T1, *T2, 0.95)]
    _assert_match(rows)


def test_differential_full_pair_rank3():
    rows = [single_variant_row("chr9", i, "A", "T", 0.99 - i * 0.01) for i in range(1, 3)] + [
        pair_row(*T1, *T2, 0.9)
    ]
    _assert_match(rows)


def test_differential_separate_rows_no_full():
    """Two true variants in separate rows = partial (50) NOT full (100)."""
    rows = [
        single_variant_row(*T1, 0.95),
        single_variant_row(*T2, 0.9),
    ]
    _assert_match(rows)


def test_differential_partial_with_decoy_pair():
    """One true + decoy in a pair row = partial; decoy hurts F-max."""
    rows = [pair_row(*T1, *D, 0.9)]
    _assert_match(rows)


def test_differential_fmax_dedup_same_variant_two_rows():
    """Duplicate variant across rows: F-max must use deduped set-union."""
    rows = [
        single_variant_row(*T1, 0.95),
        pair_row(*T1, *T2, 0.9),  # T1 again in a second row
    ]
    _assert_match(rows)


def test_differential_no_match():
    rows = [single_variant_row("chr9", 1, "A", "T", 0.9)]
    _assert_match(rows)


def test_differential_decoy_higher_epcr():
    rows = [
        single_variant_row(*D, 0.99),
        single_variant_row(*T1, 0.98),
        single_variant_row(*T2, 0.97),
    ]
    _assert_match(rows)


def test_differential_lowercase_alleles_normalized():
    """Lowercase ref/alt (spec-permitted VCF) must still match the official scorer."""
    rows = [
        pair_row("chr2", 47908000, "g", "a", "chr2", 47912000, "c", "t", 0.95),
    ]
    _assert_match(rows)

"""Track 1 — local scoring metric (rank points + F-max).

Reimplementation of the official scorer (mva-hackathon-2026/evaluation.py,
`score_proband`) for local pre-submission validation.

Key rules:
- **full match**: a row whose variant set EQUALS the true set (a compound-het
  pair must be in ONE row) -> full tier points by that row's rank.
- **partial match** (compound-het only): a row containing EXACTLY ONE of the
  two true variants (alone OR paired with a wrong variant) -> half tier points.
- **F-max**: sweep EPCR thresholds; at each, take the DEDUPED set-union of all
  variant coords in rows with epcr >= t, then tp/fp/fn against the true set.
"""

from __future__ import annotations

from dataclasses import dataclass

from genoma.data.submission import SubmissionRow


@dataclass
class AnswerKey:
    """Known (or synthetic) causal answer for local scoring."""

    causal_variants: set[str]  # "chrN:pos:ref>alt" causal coordinates
    compound_het: bool = False  # is the causal mechanism a compound-het pair?
    note: str = ""


RANK_POINT_TIERS = [(1, 100), (3, 50), (5, 25), (10, 10)]


def _row_variant_coords(row: SubmissionRow) -> list[str]:
    """The variant coordinate(s) a submission row carries (single or pair).

    Ref/alt are normalized (strip + upper) to match the official scorer's
    ``_parse_variant``, so local validation stays byte-identical even when a
    VCF emits lowercase alleles (spec-permitted).
    """
    coords = [f"{row.chrom_1}:{row.pos_1}:{row.ref_1.strip().upper()}>{row.alt_1.strip().upper()}"]
    if row.chrom_2:
        coords.append(
            f"{row.chrom_2}:{row.pos_2}:{row.ref_2.strip().upper()}>{row.alt_2.strip().upper()}"
        )
    return coords


def _row_variant_set(row: SubmissionRow) -> frozenset[str]:
    return frozenset(_row_variant_coords(row))


def _rank_to_points(rank: int) -> float:
    """Official tier mapping: rank<=1 ->100, <=3 ->50, <=5 ->25, <=10 ->10."""
    for max_rank, points in RANK_POINT_TIERS:
        if rank <= max_rank:
            return float(points)
    return 0.0


def rank_points(rows: list[SubmissionRow], answer: AnswerKey) -> float:
    """Rank points, mirroring official score_proband.

    - full match: a row's variant set == true set -> tier(rank of that row).
    - compound-het partial: a row sharing exactly one true variant ->
      0.5 * tier(rank of that row).
    - otherwise 0.
    """
    ordered = sorted(rows, key=lambda r: r.epcr, reverse=True)
    true_set = answer.causal_variants

    # Full match: exact variant-set equality (compound-het pair in ONE row).
    for rank, row in enumerate(ordered, start=1):
        if set(_row_variant_coords(row)) == true_set:
            return _rank_to_points(rank)

    # Partial match (compound-het only): exactly one true variant in a row.
    if answer.compound_het:
        for rank, row in enumerate(ordered, start=1):
            if _row_variant_set(row) & true_set:
                return 0.5 * _rank_to_points(rank)

    return 0.0


def f_max(rows: list[SubmissionRow], answer: AnswerKey) -> float:
    """F-max over EPCR thresholds — official algorithm (deduped set-union)."""
    true_set = answer.causal_variants
    if not rows or not true_set:
        return 0.0

    thresholds = sorted({r.epcr for r in rows}, reverse=True)
    best_f = 0.0
    for t in thresholds:
        predicted: set[str] = set()
        for row in rows:
            if row.epcr >= t:
                predicted.update(_row_variant_coords(row))  # deduped union
        tp = len(predicted & true_set)
        fp = len(predicted - true_set)
        fn = len(true_set - predicted)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        best_f = max(best_f, f)
    return best_f


def score_submission(rows: list[SubmissionRow], answer: AnswerKey) -> dict:
    """Full local score: rank points + F-max + a simple combined view."""
    rp = rank_points(rows, answer)
    fm = f_max(rows, answer)
    return {
        "rank_points": rp,
        "f_max": fm,
        "combined": 0.5 * (rp / 100.0) + 0.5 * fm,
        "rows": len(rows),
        "compound_het": answer.compound_het,
    }

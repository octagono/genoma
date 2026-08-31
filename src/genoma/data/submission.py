"""Track 1 submission CSV writer + validator.

Matches the official `track1_submission_template.csv` schema exactly:

    proband_id,chrom_1,pos_1,ref_1,alt_1,chrom_2,pos_2,ref_2,alt_2,epcr,finding_type,notes

- one row per proposed causal variant (or compound-het pair)
- up to 10 rows; EPCR in (0, 1]
- finding_type: primary | secondary
- coordinates GRCh38
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from genoma import config as cfg

CSV_HEADER = [
    "proband_id",
    "chrom_1",
    "pos_1",
    "ref_1",
    "alt_1",
    "chrom_2",
    "pos_2",
    "ref_2",
    "alt_2",
    "epcr",
    "finding_type",
    "notes",
]


@dataclass
class SubmissionRow:
    """One candidate row: a single variant or a compound-het pair."""

    chrom_1: str
    pos_1: int
    ref_1: str
    alt_1: str
    epcr: float
    finding_type: str = "primary"
    notes: str = ""
    chrom_2: str = ""
    pos_2: int | str = ""
    ref_2: str = ""
    alt_2: str = ""

    def validate(self) -> None:
        if not (0.0 < self.epcr <= 1.0):
            raise ValueError(f"EPCR must be in (0,1], got {self.epcr}")
        if self.finding_type not in ("primary", "secondary"):
            raise ValueError(f"finding_type must be primary|secondary, got {self.finding_type}")
        if (self.chrom_2 or self.pos_2) and not all(
            [self.chrom_2, str(self.pos_2), self.ref_2, self.alt_2]
        ):
            raise ValueError("Compound-het rows need all chrom_2..alt_2 fields")
        for c in (self.chrom_1, self.chrom_2):
            if c and not str(c).startswith("chr"):
                raise ValueError(f"Coordinates must be GRCh38 chrN form, got {c!r}")

    def to_row(self) -> list:
        self.validate()
        return [
            cfg.PROBAND_ID,
            self.chrom_1,
            self.pos_1,
            self.ref_1,
            self.alt_1,
            self.chrom_2,
            self.pos_2,
            self.ref_2,
            self.alt_2,
            f"{self.epcr:.3f}",
            self.finding_type,
            self.notes,
        ]


def write_submission(
    rows: list[SubmissionRow],
    out_path: Path | str,
    *,
    sort_by_epcr_desc: bool = True,
) -> Path:
    """Write a validated Track 1 submission CSV. Returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) > cfg.MAX_SUBMISSION_ROWS:
        raise ValueError(
            f"{len(rows)} rows exceeds the {cfg.MAX_SUBMISSION_ROWS}-row limit. "
            "Rank your top candidates only."
        )
    if sort_by_epcr_desc:
        rows = sorted(rows, key=lambda r: r.epcr, reverse=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow(row.to_row())
    return out_path


def single_variant_row(
    chrom: str, pos: int, ref: str, alt: str, epcr: float, *, note: str = ""
) -> SubmissionRow:
    """A single-variant submission row."""
    return SubmissionRow(
        chrom_1=chrom,
        pos_1=pos,
        ref_1=ref,
        alt_1=alt,
        epcr=epcr,
        finding_type="primary",
        notes=note,
    )


def pair_row(
    chrom_1: str,
    pos_1: int,
    ref_1: str,
    alt_1: str,
    chrom_2: str,
    pos_2: int,
    ref_2: str,
    alt_2: str,
    epcr: float,
    *,
    note: str = "",
) -> SubmissionRow:
    """A compound-heterozygous pair submission row (both variants in one row)."""
    return SubmissionRow(
        chrom_1=chrom_1,
        pos_1=pos_1,
        ref_1=ref_1,
        alt_1=alt_1,
        chrom_2=chrom_2,
        pos_2=pos_2,
        ref_2=ref_2,
        alt_2=alt_2,
        epcr=epcr,
        finding_type="primary",
        notes=note,
    )


def build_submission_rows(
    singles: list,
    pairs: list,
    *,
    max_rows: int | None = None,
    strategy: str = "mixed",
) -> list[SubmissionRow]:
    """Build the submission CSV rows from scored singles + compound-het pairs.

    ``singles``: list of objects with .chrom/.pos/.ref/.alt/.epcr/.rationale
    ``pairs``: list of ScoredPair objects (.variant_1/.variant_2/.epcr)

    Strategy:
    - "pairs": compound-het pair rows only (the expected causal mechanism).
    - "singles": single-variant rows only.
    - "mixed" (default): pairs first, then fill remaining rows with singles
      whose variants aren't already covered by a pair.

    The 10-row budget is the binding constraint; pair rows carry both causal
    variants in a single row.
    """
    max_rows = max_rows or cfg.MAX_SUBMISSION_ROWS
    rows: list[SubmissionRow] = []
    used: set[str] = set()

    def _add_single(v) -> bool:
        coord = f"{v.chrom}:{v.pos}:{v.ref}>{v.alt}"
        if coord in used or len(rows) >= max_rows:
            return False
        rows.append(
            single_variant_row(
                v.chrom,
                v.pos,
                v.ref,
                v.alt,
                v.epcr,
                note=getattr(v, "rationale", "")[:80],
            )
        )
        used.add(coord)
        return True

    if strategy in ("pairs", "mixed"):
        for p in pairs:
            if len(rows) >= max_rows:
                break
            c1, c2 = p.variant_1, p.variant_2
            k1 = f"{c1.chrom}:{c1.pos}:{c1.ref}>{c1.alt}"
            k2 = f"{c2.chrom}:{c2.pos}:{c2.ref}>{c2.alt}"
            rows.append(
                pair_row(
                    c1.chrom,
                    c1.pos,
                    c1.ref,
                    c1.alt,
                    c2.chrom,
                    c2.pos,
                    c2.ref,
                    c2.alt,
                    p.epcr,
                    note=getattr(p, "rationale", "")[:80],
                )
            )
            used.update({k1, k2})
    if strategy in ("singles", "mixed"):
        for v in sorted(singles, key=lambda x: x.epcr, reverse=True):
            if len(rows) >= max_rows:
                break
            _add_single(v)

    return rows[:max_rows]


def load_submission(path: Path | str) -> list[SubmissionRow]:
    """Round-trip a submission CSV for inspection/testing."""
    rows: list[SubmissionRow] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                SubmissionRow(
                    chrom_1=r["chrom_1"],
                    pos_1=int(r["pos_1"]),
                    ref_1=r["ref_1"],
                    alt_1=r["alt_1"],
                    chrom_2=r.get("chrom_2") or "",
                    pos_2=int(r["pos_2"]) if r.get("pos_2") else "",
                    ref_2=r.get("ref_2") or "",
                    alt_2=r.get("alt_2") or "",
                    epcr=float(r["epcr"]),
                    finding_type=r.get("finding_type", "primary"),
                    notes=r.get("notes", ""),
                )
            )
    return rows

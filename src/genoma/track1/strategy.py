"""Track 1 — submission strategy planner (active learning).

Each evaluation returns match/no-match feedback; this planner treats that
feedback as a weak oracle:

- record what was submitted and which variants matched.
- re-weight candidate EPCRs: matched variants (and their gene partners, the
  compound-het hypothesis) get boosted; submitted-but-unmatched variants get
  penalized so they don't waste future rows.
- suggest the next submission: pairs-first when a single causal variant is
  confirmed, since the ground truth is compound-heterozygous.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from genoma.data.submission import SubmissionRow


@dataclass
class SubmissionRecord:
    rows: list[SubmissionRow]
    matched: list[str]  # variant coords confirmed by feedback
    note: str = ""


@dataclass
class SubmissionPlanner:
    """Tracks submission history and re-weights candidates after feedback."""

    history: list[SubmissionRecord] = field(default_factory=list)
    epcr: dict[str, float] = field(default_factory=dict)  # coord -> current epcr
    gene: dict[str, str] = field(default_factory=dict)  # coord -> gene
    boost: float = 0.25
    penalty: float = 0.35

    def set_candidates(self, candidates: list, epcr: dict[str, float] | None = None) -> None:
        """Seed candidate gene map (and EPCRs if provided).

        ``candidates`` need .coord and .gene. ``epcr`` is an optional
        coord->score map (from the scoring step); defaults to 0.5.
        """
        for c in candidates:
            self.gene[c.coord] = c.gene
            if epcr is not None:
                self.epcr[c.coord] = epcr.get(c.coord, 0.5)
            else:
                self.epcr.setdefault(c.coord, 0.5)

    def record(self, rows: list[SubmissionRow], matched: list[str], note: str = "") -> None:
        """Record a submission + its match feedback, then re-weight."""
        self.history.append(SubmissionRecord(rows=rows, matched=matched, note=note))
        submitted = set()
        for row in rows:
            submitted.add(f"{row.chrom_1}:{row.pos_1}:{row.ref_1}>{row.alt_1}")
            if row.chrom_2:
                submitted.add(f"{row.chrom_2}:{row.pos_2}:{row.ref_2}>{row.alt_2}")

        matched_set = set(matched)
        for coord in submitted:
            if coord in matched_set:
                self.epcr[coord] = min(1.0, self.epcr.get(coord, 0.5) + self.boost)
            elif coord in self.epcr:
                # Unmatched submitted variant: reduce so it doesn't recur.
                self.epcr[coord] = max(0.0, self.epcr[coord] - self.penalty)

        # Boost gene partners of matched variants (compound-het hypothesis).
        matched_genes = {self.gene.get(m, "") for m in matched if m in self.gene}
        for coord, gene in self.gene.items():
            if gene in matched_genes and coord not in matched_set:
                self.epcr[coord] = min(1.0, self.epcr.get(coord, 0.5) + self.boost)

    def suggest_pairs(self, candidates: list, n: int = 5) -> list[tuple]:
        """Top same-gene pairs by updated EPCR (compound-het candidates)."""
        by_gene: dict[str, list] = {}
        for c in candidates:
            by_gene.setdefault(c.gene or c.chrom, []).append(c)
        pairs: list[tuple] = []
        for cs in by_gene.values():
            cs = sorted(cs, key=lambda c: self.epcr.get(c.coord, 0.0), reverse=True)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    pairs.append((cs[i], cs[j]))
        pairs.sort(
            key=lambda p: self.epcr.get(p[0].coord, 0.0) + self.epcr.get(p[1].coord, 0.0),
            reverse=True,
        )
        return pairs[:n]

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def to_dict(self) -> dict:
        return {
            "epcr": {k: round(v, 4) for k, v in self.epcr.items()},
            "history": [
                {
                    "rows": [r.to_row() for r in rec.rows],
                    "matched": rec.matched,
                    "note": rec.note,
                }
                for rec in self.history
            ],
        }

    @classmethod
    def load(cls, path: Path | str) -> SubmissionPlanner:
        data = json.loads(Path(path).read_text())
        p = cls()
        p.epcr = {k: float(v) for k, v in data.get("epcr", {}).items()}
        p.history = [
            SubmissionRecord(rows=[], matched=rec["matched"], note=rec.get("note", ""))
            for rec in data.get("history", [])
        ]
        return p

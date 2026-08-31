"""Demo/synthetic data: lets the full DSPy pipeline run without the gated dataset.

Synthetic candidate variants are engineered so the causal signal is
recoverable in principle (compound-heterozygous pair in a known MVA gene,
plus decoys), which makes the demo useful for developing the scoring
pipeline and the GEPA judge before real data arrives.
"""

from __future__ import annotations

from genoma.data.annotate import VariantCandidate
from genoma.data.submission import SubmissionRow

# Demo compound-het causal pair (synthetic).
DEMO_CAUSAL_PAIR: tuple[SubmissionRow, SubmissionRow] = (
    SubmissionRow(
        chrom_1="chr2",
        pos_1=47_908_000,
        ref_1="G",
        alt_1="A",
        epcr=0.90,
        finding_type="primary",
        notes="demo causal candidate 1",
    ),
    SubmissionRow(
        chrom_1="chr2",
        pos_1=47_912_000,
        ref_1="C",
        alt_1="T",
        epcr=0.85,
        finding_type="primary",
        notes="demo causal candidate 2",
    ),
)

DEMO_HPO = [
    "HP:0001510 Growth delay",
    "HP:0000252 Microcephaly",
    "HP:0004322 Short stature",
]


def demo_candidates(n_decoys: int = 8) -> list[VariantCandidate]:
    """A small candidate table: the causal pair + decoy variants."""
    candidates = [
        VariantCandidate(
            chrom="chr2",
            pos=47_908_000,
            ref="G",
            alt="A",
            gene="BUB1B",
            consequence="missense_variant",
            af=0.0002,
            cadd=28.5,
            alphamissense=0.72,
            evo2=1.4,
        ),
        VariantCandidate(
            chrom="chr2",
            pos=47_912_000,
            ref="C",
            alt="T",
            gene="BUB1B",
            consequence="missense_variant",
            af=0.0001,
            cadd=31.0,
            alphamissense=0.85,
            evo2=1.9,
        ),
        VariantCandidate(
            chrom="chr3",
            pos=12_345_678,
            ref="A",
            alt="G",
            gene="CACNA1D",
            consequence="missense_variant",
            af=0.002,
            cadd=18.0,
            alphamissense=0.30,
            evo2=-0.2,
        ),
    ]
    # Decoys: common / noncoding / benign-looking rows.
    for i in range(max(0, n_decoys - len(candidates))):
        candidates.append(
            VariantCandidate(
                chrom="chr1",
                pos=100_000_000 + i * 1_000,
                ref="T",
                alt="C",
                gene=f"DECOY{i}",
                consequence="intron_variant",
                af=0.02 if i % 2 else 0.0005,
                cadd=6.0 + i,
            )
        )
    return candidates

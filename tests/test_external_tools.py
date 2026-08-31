"""Tests for the external-tool optimizations: phasing, phen2gene, exomiser, calibration."""

from __future__ import annotations

import pytest

from genoma.data.annotate import VariantCandidate
from genoma.data.exomiser import exomiser_available, write_analysis_yaml
from genoma.data.phasing import (
    PairPhase,
    parse_phase_sets,
    phase_pairs,
    phase_status,
)
from genoma.data.phen2gene import _fallback_ranking, rank_mva_genes
from genoma.track1.calibrate import calibrate_submission_epcrs
from genoma.track1.scoring import ScoredPair, ScoredVariant

# ── Phasing ─────────────────────────────────────────────────────────────────


def _rec(chrom, pos, ps=None, hp=None):
    return {
        "chrom": chrom,
        "pos": pos,
        "ref": "G",
        "alt": "A",
        "format": ({"PS": ps, "HP": hp} if ps is not None else {}),
    }


def test_phase_trans():
    a = _rec("chr2", 100, ps="500", hp="1")
    b = _rec("chr2", 200, ps="500", hp="2")
    assert phase_status(a, b).status == "trans"


def test_phase_cis():
    a = _rec("chr2", 100, ps="500", hp="1")
    b = _rec("chr2", 200, ps="500", hp="1")
    assert phase_status(a, b).status == "cis"


def test_phase_unknown_without_tags():
    a = _rec("chr2", 100)
    b = _rec("chr2", 200)
    assert phase_status(a, b).status == "unknown"


def test_parse_phase_sets_pairs():
    records = [
        _rec("chr2", 100, ps="500", hp="1"),
        _rec("chr2", 200, ps="500", hp="2"),
        _rec("chr3", 300),  # unphased
    ]
    out = parse_phase_sets(records)
    assert len(out) == 1
    key = (("chr2", 100), ("chr2", 200))
    assert out[key].status == "trans"


def test_phase_pairs_over_candidates():
    records = [_rec("chr2", 100, ps="500", hp="1"), _rec("chr2", 200, ps="500", hp="2")]
    cands = [
        VariantCandidate(chrom="chr2", pos=100, ref="G", alt="A", gene="BUB1B"),
        VariantCandidate(chrom="chr2", pos=200, ref="G", alt="A", gene="BUB1B"),
    ]
    statuses = phase_pairs(cands, records)
    assert len(statuses) == 1
    assert next(iter(statuses.values())).status == "trans"


# ── Phen2Gene ───────────────────────────────────────────────────────────────


def test_phen2gene_fallback_ranks_all_mva_genes():
    ranked = _fallback_ranking()
    assert set(ranked) == {"BUB1B", "CEP57", "TRIP13", "MAD1L1", "MAD2L1BP", "CEP192"}
    assert max(ranked.values()) == pytest.approx(1.0)
    # BUB1B (archetypal MVA1) ranks first in the curated prior.
    assert max(ranked, key=lambda g: ranked[g]) == "BUB1B"


def test_rank_mva_genes_fails_fast_offline():
    """Fail-fast: no silent fallback; API errors propagate."""
    # A dead endpoint makes _run_api raise deterministically.
    import urllib.error

    import genoma.data.phen2gene as p2g

    orig = p2g._API_URL
    p2g._API_URL = "http://127.0.0.1:9/api"  # nothing listens here
    try:
        with pytest.raises(urllib.error.URLError):
            rank_mva_genes(["HP:0000252 Microcephaly"])
    finally:
        p2g._API_URL = orig


def test_run_phen2gene_requires_hpo_ids():
    """No valid HPO ids -> fail fast (no fallback)."""
    from genoma.data.phen2gene import run_phen2gene

    with pytest.raises(ValueError):
        run_phen2gene(["not-an-hpo"])


# ── Exomiser wrapper ────────────────────────────────────────────────────────


def test_exomiser_unavailable_raises_clear_error():
    import os

    from genoma.data.exomiser import run_exomiser

    had_jar = os.environ.pop("EXOMISER_JAR", None)
    try:
        if exomiser_available():
            pytest.skip("Exomiser installed on this machine")
        with pytest.raises(RuntimeError, match="Exomiser not configured"):
            run_exomiser("nope.vcf", ["HP:1"], "/tmp/opencode/exo_test")
    finally:
        if had_jar:
            os.environ["EXOMISER_JAR"] = had_jar


def test_exomiser_yaml_generation(tmp_path):
    yaml_path = write_analysis_yaml(
        "fake.vcf",
        ["HP:0000252 Microcephaly"],
        tmp_path,
        candidate_genes=["BUB1B"],
    )
    text = yaml_path.read_text()
    # Phenopacket schema (Exomiser 15.1.x current format).
    assert "phenopacket:" in text
    assert "htsFiles:" in text and "genomeAssembly: hg38" in text
    assert "AUTOSOMAL_RECESSIVE_COMP_HET: 2.0" in text
    assert "HP:0000252" in text
    assert "hiPhivePrioritiser" in text


def test_parse_exomiser_output_current_format(tmp_path):
    """genes.tsv + variants.tsv (15.1.x): MOI=AR + CONTRIBUTING_VARIANT -> comp-het."""
    from genoma.data.exomiser import parse_exomiser_output

    genes = tmp_path / "exomiser_result.genes.tsv"
    genes.write_text(
        "#RANK\tID\tGENE_SYMBOL\tMOI\tP-VALUE\tEXOMISER_GENE_COMBINED_SCORE\tEXOMISER_GENE_PHENO_SCORE\tEXOMISER_VARIANT_SCORE\n"
        "1\t701\tBUB1B\tAR\t0.001\t0.88\t0.85\t0.92\n"
    )
    variants = tmp_path / "exomiser_result.variants.tsv"
    variants.write_text(
        "#GENE_SYMBOL\tMOI\tCONTRIBUTING_VARIANT\tCONTIG\tSTART\tREF\tALT\n"
        "BUB1B\tAR\t1\tchr2\t47908000\tG\tA\n"
        "BUB1B\tAR\t1\tchr2\t47912000\tC\tT\n"
    )
    parsed = parse_exomiser_output(tmp_path)
    assert "BUB1B" in parsed["genes"]
    assert parsed["genes"]["BUB1B"]["exomiser_score"] == pytest.approx(0.88)
    assert parsed["genes"]["BUB1B"]["comp_het"] is True
    assert len(parsed["genes"]["BUB1B"]["variants"]) == 2


# ── EPCR submission calibration ─────────────────────────────────────────────


def _sv(pos, epcr):
    return ScoredVariant(chrom="chr2", pos=pos, ref="G", alt="A", epcr=epcr, rationale="r")


def _pair(p1, p2, epcr):
    return ScoredPair(variant_1=_sv(p1, 0.0), variant_2=_sv(p2, 0.0), epcr=epcr, rationale="r")


def test_calibration_top_pair_gets_max():
    pairs = [_pair(100, 200, 0.9), _pair(300, 400, 0.8)]
    singles = [_sv(500, 0.7), _sv(600, 0.6)]
    cp, cs = calibrate_submission_epcrs(pairs, singles)
    # Top-ranked non-cis pair reserves the unique max (1.0); next drops by decay.
    assert cp[0].epcr == pytest.approx(1.0, abs=0.01)
    assert cp[1].epcr == pytest.approx(0.95, abs=0.01)
    assert cp[0].epcr > cp[1].epcr > cs[0].epcr > cs[1].epcr  # well-separated monotone
    assert cs[0].epcr < 0.95  # singles below pairs


def test_calibration_no_ties():
    """Strictly unique EPCRs — a tie at the top collapses F-max."""
    pairs = [_pair(i * 100, i * 100 + 1, 0.5) for i in range(1, 6)]
    cp, cs = calibrate_submission_epcrs(pairs, [])
    all_epcrs = [p.epcr for p in cp] + [s.epcr for s in cs]
    assert len(all_epcrs) == len(set(all_epcrs))  # no ties


def test_calibration_cis_demoted():
    pairs = [_pair(100, 200, 0.9)]
    singles = []
    phase = {"chr2:100:G>A|chr2:200:G>A": PairPhase("cis", "test")}
    cp, _ = calibrate_submission_epcrs(pairs, singles, phase_info=phase)
    assert cp[0].epcr < 0.15  # cis pair demoted hard


def test_calibration_trans_promoted():
    pairs = [_pair(100, 200, 0.55)]
    phase = {"chr2:100:G>A|chr2:200:G>A": PairPhase("trans", "test")}
    cp, _ = calibrate_submission_epcrs(pairs, [], phase_info=phase)
    assert cp[0].epcr == pytest.approx(1.0, abs=0.01)  # trans top pair reserves 1.0


# ── WhatsHap GT-style phasing (default --tag=PS output format) ──────────────


def _gt_rec(chrom, pos, ps=None, gt=None):
    return {
        "chrom": chrom,
        "pos": pos,
        "ref": "G",
        "alt": "A",
        "format": ({"PS": ps, "GT": gt} if ps is not None else {}),
    }


def test_phase_trans_gt_style():
    """WhatsHap default output: same PS, GT 0|1 and 1|0 -> TRANS."""
    a = _gt_rec("chr2", 100, ps="100", gt="0|1")
    b = _gt_rec("chr2", 200, ps="100", gt="1|0")
    assert phase_status(a, b).status == "trans"


def test_phase_cis_gt_style():
    """Same PS, GT 0|1 and 0|1 -> CIS (both ALTs on the same haplotype)."""
    a = _gt_rec("chr2", 100, ps="100", gt="0|1")
    b = _gt_rec("chr2", 200, ps="100", gt="0|1")
    assert phase_status(a, b).status == "cis"


def test_phase_unphased_gt_is_unknown():
    """Unphased GT (0/1) even with PS -> unknown."""
    a = _gt_rec("chr2", 100, ps="100", gt="0/1")
    b = _gt_rec("chr2", 200, ps="100", gt="0/1")
    assert phase_status(a, b).status == "unknown"


def test_phase_different_ps_unknown():
    a = _gt_rec("chr2", 100, ps="100", gt="0|1")
    b = _gt_rec("chr2", 200, ps="200", gt="1|0")
    assert phase_status(a, b).status == "unknown"

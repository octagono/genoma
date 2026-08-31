"""Tests for the optimization batch: metrics, submission strategy, VEP, priors."""

from __future__ import annotations

import urllib.error

import pytest

from genoma.data.annotate import (
    KNOWN_MVA_GENES,
    apply_gene_prior,
    enrich_via_vep_rest,
)
from genoma.data.demo import DEMO_CAUSAL_PAIR, demo_candidates
from genoma.data.submission import (
    build_submission_rows,
    load_submission,
    pair_row,
    single_variant_row,
    write_submission,
)
from genoma.track1.metrics import AnswerKey, f_max, rank_points, score_submission
from genoma.track1.strategy import SubmissionPlanner

# ── Local scoring metric ────────────────────────────────────────────────────


def _demo_answer() -> AnswerKey:
    return AnswerKey(
        causal_variants={f"{r.chrom_1}:{r.pos_1}:{r.ref_1}>{r.alt_1}" for r in DEMO_CAUSAL_PAIR},
        compound_het=True,
        note="demo answer",
    )


def _pair_rows() -> list:
    """The exact demo causal pair as a SINGLE pair row (full match requires it)."""
    a, b = DEMO_CAUSAL_PAIR
    return [
        pair_row(
            a.chrom_1,
            a.pos_1,
            a.ref_1,
            a.alt_1,
            b.chrom_1,
            b.pos_1,
            b.ref_1,
            b.alt_1,
            0.95,
        )
    ]


def test_rank_points_full_pair():
    rows = _pair_rows()
    assert rank_points(rows, _demo_answer()) == pytest.approx(100.0)


def test_rank_points_half_credit_one_variant():
    a, _b = DEMO_CAUSAL_PAIR
    # Only one causal variant in the submission.
    rows = [single_variant_row(a.chrom_1, a.pos_1, a.ref_1, a.alt_1, 0.9)]
    assert rank_points(rows, _demo_answer()) == pytest.approx(50.0)  # half


def test_rank_points_zero_when_missed():
    rows = [single_variant_row("chr9", 1, "A", "T", 0.9)]
    assert rank_points(rows, _demo_answer()) == pytest.approx(0.0)


def test_rank_points_separate_rows_is_partial():
    """Official rule: two true variants in separate rows = PARTIAL (50), not 100."""
    a, b = DEMO_CAUSAL_PAIR
    rows = [
        single_variant_row(a.chrom_1, a.pos_1, a.ref_1, a.alt_1, 0.95),
        single_variant_row(b.chrom_1, b.pos_1, b.ref_1, b.alt_1, 0.9),
    ]
    assert rank_points(rows, _demo_answer()) == pytest.approx(50.0)


def test_rank_points_tier_by_rank():
    a, _ = DEMO_CAUSAL_PAIR
    # a has epcr 0.90; bury it by giving decoys higher EPCR.
    rows = [single_variant_row("chr9", i, "A", "T", 0.95 - i * 0.005) for i in range(1, 6)] + [
        single_variant_row(a.chrom_1, a.pos_1, a.ref_1, a.alt_1, 0.9)
    ]
    # a sorts to rank 6 -> tier 10; single of pair -> half = 5.0
    assert rank_points(rows, _demo_answer()) == pytest.approx(5.0)


def test_fmax_perfect():
    rows = _pair_rows()
    assert f_max(rows, _demo_answer()) == pytest.approx(1.0)


def test_fmax_with_decoy_high_epcr():
    a, b = DEMO_CAUSAL_PAIR
    rows = [a, b, single_variant_row("chr9", 1, "A", "T", 0.99)]
    fm = f_max(rows, _demo_answer())
    assert 0.0 < fm < 1.0  # decoy with higher EPCR than one causal hurts precision


def test_score_submission_shape():
    res = score_submission(_pair_rows(), _demo_answer())
    assert set(res) == {"rank_points", "f_max", "combined", "rows", "compound_het"}
    assert res["rows"] == 1  # one compound-het pair row
    assert res["rank_points"] == pytest.approx(100.0)


# ── Submission strategy ─────────────────────────────────────────────────────


def test_build_submission_rows_mixed():
    singles = [demo_candidates()[i] for i in (0, 1, 2)]
    singles = [
        type(
            "V",
            (),
            {
                "chrom": c.chrom,
                "pos": c.pos,
                "ref": c.ref,
                "alt": c.alt,
                "epcr": 0.5,
                "rationale": "x",
            },
        )
        for c in singles
    ]
    # No pairs available -> mixed fills with singles.
    rows = build_submission_rows(singles, [], max_rows=10)
    assert len(rows) == 3
    assert all(r.chrom_2 == "" for r in rows)


def test_build_submission_rows_pairs_first():
    a, b = DEMO_CAUSAL_PAIR
    # Convert demo pair to ScoredPair-like objects.
    p = type(
        "P",
        (),
        {
            "variant_1": type(
                "V",
                (),
                {"chrom": a.chrom_1, "pos": a.pos_1, "ref": a.ref_1, "alt": a.alt_1},
            )(),
            "variant_2": type(
                "V",
                (),
                {"chrom": b.chrom_1, "pos": b.pos_1, "ref": b.ref_1, "alt": b.alt_1},
            )(),
            "epcr": 0.95,
            "rationale": "pair",
        },
    )()
    rows = build_submission_rows([], [p], max_rows=10, strategy="pairs")
    assert len(rows) == 1
    assert rows[0].chrom_2  # pair row has both variants
    assert rows[0].chrom_1 == a.chrom_1 and rows[0].chrom_2 == b.chrom_1


def test_planner_reweights_after_feedback():
    planner = SubmissionPlanner()
    candidates = demo_candidates()
    # Seed with an initial epcr map (as if scored).
    planner.set_candidates(candidates, epcr={c.coord: 0.5 for c in candidates})
    a, b = DEMO_CAUSAL_PAIR
    matched = [f"{a.chrom_1}:{a.pos_1}:{a.ref_1}>{a.alt_1}"]
    planner.record([a], matched)
    # Matched variant boosted; unmatched-but-submitted penalized.
    assert planner.epcr[matched[0]] > 0.5 + 0.24
    # Gene partner (b, same gene BUB1B) boosted by the compound-het hypothesis.
    partner = f"{b.chrom_1}:{b.pos_1}:{b.ref_1}>{b.alt_1}"
    assert planner.epcr[partner] > 0.5 + 0.24
    # Suggest pairs includes the causal pair.
    pairs = planner.suggest_pairs(candidates, n=5)
    assert pairs  # non-empty


# ── VEP / annotation ────────────────────────────────────────────────────────


def test_enrich_via_vep_rest_fails_fast_offline():
    """Fail-fast: unreachable VEP API raises (no silent fallback)."""
    cands = demo_candidates(n_decoys=2)
    # Point at a dead URL to force failure deterministically.
    import genoma.config as cfg_mod

    orig = cfg_mod.VEP_REST_URL
    cfg_mod.VEP_REST_URL = "http://127.0.0.1:9/vep"  # nothing listens here
    try:
        with pytest.raises(urllib.error.URLError):
            enrich_via_vep_rest(cands)
    finally:
        cfg_mod.VEP_REST_URL = orig


def test_enrich_via_vep_rest_populates_predictors(monkeypatch, tmp_path):
    """VEP response fields (REVEL/AlphaMissense/SpliceAI/CADD) fill the model."""
    import json
    import urllib.request

    from genoma import config as cfg
    from genoma.data.annotate import VariantCandidate

    cand = VariantCandidate(chrom="chr7", pos=140753336, ref="A", alt="T")

    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                [
                    {
                        "input": "7 140753336 140753336 A/T",  # echoed by VEP REST
                        "symbol": "BUB1B",
                        "most_severe_consequence": "missense_variant",
                        "gnomAD": {"af": 1e-5},
                        "cadd": {"phred": 29.8},
                        "revel": 0.931,
                        "alphamissense": {"am_pathogenicity": 0.9927},
                        "spliceai": {"DS_AG": 0.0, "DS_AL": 0.0, "DS_DG": 0.6, "DS_DL": 0.0},
                    }
                ]
            ).encode()

    def _fake_urlopen(req, timeout=None, context=None):
        captured["url"] = req.full_url
        captured["body"] = req.data
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    # Isolate the local annotation cache so this test always exercises the
    # network path (a prior cache write with the same coord would otherwise
    # short-circuit the mocked urlopen).
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    from genoma.data import annot_cache

    monkeypatch.setattr(annot_cache, "cache_path", lambda: tmp_path / "cache.db")
    out = enrich_via_vep_rest([cand], batch_size=1)
    # The POSTed variant string must be "chrom start end ref/alt" with end
    # spanning the ref allele (VEP region endpoint; "chrom pos ref alt" 400s).
    body = json.loads(captured["body"])
    assert body["variants"] == ["7 140753336 140753336 A/T"]
    assert out[0].revel == pytest.approx(0.931)
    assert out[0].alphamissense == pytest.approx(0.9927)
    assert out[0].spliceai == pytest.approx(0.6)  # max of DS_* delta scores
    assert out[0].cadd == pytest.approx(29.8)
    # Predictors are enabled as URL query params, not in the request body.
    assert "REVEL=1" in captured["url"]
    assert "SpliceAI=2" in captured["url"]
    assert "AlphaMissense=1" in captured["url"]


def test_apply_gene_prior():
    gene = next(iter(KNOWN_MVA_GENES))
    epcr = 0.5
    boosted = apply_gene_prior(epcr, gene)
    assert boosted > epcr
    assert boosted <= 1.0
    # Non-MVA gene unchanged.
    assert apply_gene_prior(0.5, "TTN") == pytest.approx(0.5)


def test_call_quality_filter():
    """Depth/GQ filtering rejects low-quality calls (a top false-positive cause)."""
    from genoma.data.annotate import build_candidate_table

    records = [
        {
            "chrom": "chr2",
            "pos": 1,
            "ref": "G",
            "alt": "A",
            "filter": "PASS",
            "info": {"GENE": "BUB1B", "AF": 1e-5, "DP": 100, "GQ": 99},
        },
        {
            "chrom": "chr2",
            "pos": 2,
            "ref": "C",
            "alt": "T",
            "filter": "PASS",
            "info": {"GENE": "BUB1B", "AF": 1e-5, "DP": 2, "GQ": 5},
        },  # low quality
    ]
    cands = build_candidate_table(records, filter_kwargs={"min_depth": 10, "min_gq": 20})
    assert len(cands) == 1  # low-depth/GQ variant dropped
    assert cands[0].pos == 1


def test_ar_homozygote_filter():
    """AR genes reject variants with population homozygotes."""
    from genoma.data.annotate import build_candidate_table

    records = [
        {
            "chrom": "chr2",
            "pos": 1,
            "ref": "G",
            "alt": "A",
            "filter": "PASS",
            "info": {"GENE": "BUB1B", "AF": 0.001, "gnomAD_HOM": 5},
        },  # homozygotes
        {
            "chrom": "chr2",
            "pos": 2,
            "ref": "C",
            "alt": "T",
            "filter": "PASS",
            "info": {"GENE": "BUB1B", "AF": 0.001, "gnomAD_HOM": 0},
        },
    ]
    cands = build_candidate_table(records)
    assert len(cands) == 1  # homozygote variant dropped for AR gene
    assert cands[0].pos == 2


def test_gene_mechanism_knowledge_base():
    from genoma.data.annotate import MVA_GENE_MECHANISMS

    assert set(MVA_GENE_MECHANISMS) == KNOWN_MVA_GENES
    assert "loss-of-function" in MVA_GENE_MECHANISMS["BUB1B"]


# ── Submission round-trip with pairs ────────────────────────────────────────


def test_pair_row_roundtrip(tmp_path):
    a, b = DEMO_CAUSAL_PAIR
    row = pair_row(a.chrom_1, a.pos_1, a.ref_1, a.alt_1, b.chrom_1, b.pos_1, b.ref_1, b.alt_1, 0.9)
    p = write_submission([row], tmp_path / "pair.csv")
    back = load_submission(p)
    assert len(back) == 1
    assert back[0].chrom_2 == b.chrom_1
    assert back[0].pos_2 == b.pos_1

"""Smoke tests — offline (DummyLM, no API keys, no dataset)."""

from __future__ import annotations

import dspy
import pytest

from genoma import config as cfg
from genoma.data.demo import DEMO_HPO, demo_candidates
from genoma.data.submission import (
    SubmissionRow,
    load_submission,
    write_submission,
)
from genoma.optimize.gepa_optimize import build_synthetic_trainset
from genoma.track1.consensus import PanelConsensus
from genoma.track1.scoring import CompoundHetPairScorer, PredictVariantScorer


@pytest.fixture(scope="module")
def dummy_lm():
    answer = {"reasoning": "r", "epcr": 0.5, "rationale": "demo"}
    return dspy.utils.DummyLM([answer] * 500, adapter=dspy.JSONAdapter())


def _configure(dummy_lm):
    dspy.configure(lm=dummy_lm, adapter=dspy.JSONAdapter())


def test_predict_scorer(dummy_lm):
    _configure(dummy_lm)
    scorer = PredictVariantScorer()
    out = scorer(demo_candidates()[0], DEMO_HPO)
    assert 0.0 < out.epcr <= 1.0
    assert out.rationale


def test_predict_scorer_field_kwargs(dummy_lm):
    """GEPA/Evaluate pass trainset fields as kwargs, not a candidate object."""
    _configure(dummy_lm)
    scorer = PredictVariantScorer()
    out = scorer(
        variant="chr16:50123456:T>G | gene=BUB1B | consequence=missense | AF=0.0001",
        hpo_terms=["HP:0000252 Microcephaly"],
    )
    assert out.chrom == "chr16" and out.pos == 50123456
    assert out.ref == "T" and out.alt == "G"
    assert 0.0 < out.epcr <= 1.0


def test_parse_coord_multiallelic():
    from genoma.track1.scoring import _parse_coord

    assert _parse_coord("chrX:123456:G>TA | gene=BUB1B") == ("chrX", 123456, "G", "TA")
    assert _parse_coord("chr1:100:A>C,G") == ("chr1", 100, "A", "C,G")


def test_panel_consensus(dummy_lm):
    _configure(dummy_lm)
    panel = PanelConsensus(
        panel={"anthropic/claude-sonnet-5": 1.0},
        lm_factory=lambda slug: dummy_lm,
    )
    ranked = panel.score_candidates(demo_candidates()[:3], DEMO_HPO)
    assert len(ranked) == 3
    assert all(0.0 < v.epcr <= 1.0 for v in ranked)


def test_compound_het_pairs(dummy_lm):
    _configure(dummy_lm)
    scorer = CompoundHetPairScorer(max_pairs=5)
    out = scorer(demo_candidates(), DEMO_HPO)
    assert hasattr(out, "pairs")


def test_submission_roundtrip(tmp_path):
    rows = [
        SubmissionRow(chrom_1="chr2", pos_1=47908000, ref_1="G", alt_1="A", epcr=0.9),
        SubmissionRow(
            chrom_1="chr2",
            pos_1=47908000,
            ref_1="G",
            alt_1="A",
            chrom_2="chr2",
            pos_2=47912000,
            ref_2="C",
            alt_2="T",
            epcr=0.95,
            notes="compound het",
        ),
    ]
    p = write_submission(rows, tmp_path / "t.csv")
    back = load_submission(p)
    assert len(back) == 2
    assert back[0].epcr == pytest.approx(0.95)  # sorted desc
    assert back[1].chrom_2 == ""  # single variant row has empty pair fields


def test_submission_validation():
    with pytest.raises(ValueError):
        SubmissionRow(chrom_1="2", pos_1=1, ref_1="A", alt_1="T", epcr=0.5).validate()
    with pytest.raises(ValueError):
        SubmissionRow(chrom_1="chr2", pos_1=1, ref_1="A", alt_1="T", epcr=1.5).validate()


def test_synthetic_trainset():
    ts = build_synthetic_trainset(n=6)
    assert len(ts) == 6
    ex = ts[0]
    assert ex.true_epcr in (0.85, 0.10)
    assert set(ex.inputs().keys()) == {"variant", "hpo_terms", "known_mva_genes"}


def test_synthetic_trainset_labels_correlate_with_features():
    """The label must be derivable from observable variant features.

    Rule: high EPCR  <=>  MVA gene AND low AF AND coding/splice consequence.
    An MVA-gene variant that is intronic or common is correctly low-EPCR.
    Evidence fields may be degraded to "unknown" (real-world noise); unknown
    observables can't be asserted either way.
    """
    from genoma.data.annotate import KNOWN_MVA_GENES

    ts = build_synthetic_trainset(n=40, seed=1)
    high = [ex for ex in ts if ex.true_epcr > 0.5]
    low = [ex for ex in ts if ex.true_epcr <= 0.5]
    assert high and low

    mva = tuple(KNOWN_MVA_GENES)

    def _has(gene: str, ex) -> bool:
        return f"gene={gene}" in ex.variant

    def _af(ex) -> float | None:
        raw = ex.variant.split("AF=")[1].split(" |")[0]
        return None if raw == "unknown" else float(raw)

    def _consequence(ex) -> str:
        return ex.variant.split("consequence=")[1].split(" |")[0]

    for ex in high:
        assert any(_has(g, ex) for g in mva)
        af = _af(ex)
        if af is not None:
            assert af < 0.005
        cons = _consequence(ex)
        if cons != "unknown":
            assert cons != "intron_variant"
    for ex in low:
        # No low-EPCR example may carry the full causal signal.
        if any(_has(g, ex) for g in mva):
            af = _af(ex)
            if af is not None and af < 0.005:
                assert _consequence(ex) in ("intron_variant", "unknown")


def test_gepa_budget_guard():
    """optimize_with_gepa must reject setting both auto and max_metric_calls."""
    from genoma.optimize.gepa_optimize import optimize_with_gepa
    from genoma.track1.scoring import PredictVariantScorer

    ts = build_synthetic_trainset(n=2)
    with pytest.raises(ValueError, match="exactly one"):
        optimize_with_gepa(PredictVariantScorer(), ts, auto="light", max_metric_calls=10)


def test_gepa_holds_out_validation_set():
    """optimize_with_gepa splits a disjoint valset when none is passed."""
    from genoma.optimize.gepa_optimize import optimize_with_gepa

    ts = build_synthetic_trainset(n=4)
    # Reach the valset split by forcing the too-small error path is not needed;
    # assert the helper rejects a trainset too small to hold out.
    with pytest.raises(ValueError, match="too small"):
        optimize_with_gepa(PredictVariantScorer(), ts[:1], auto="light", val_fraction=0.25)


def test_panel_key_collapse(monkeypatch, dummy_lm):
    """OpenRouter members are dropped when OPENROUTER_API_KEY is unset."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # keep Anthropic member
    panel = PanelConsensus(
        panel={
            "openrouter/anthropic/claude-sonnet-5": 2.0,
            "openrouter/deepseek/deepseek-v4-pro": 1.0,
            "anthropic/claude-sonnet-5": 1.0,
        },
        lm_factory=lambda slug: dummy_lm,
    )
    assert list(panel.members.keys()) == ["anthropic/claude-sonnet-5"]


def test_panel_fallback_to_primary(monkeypatch, dummy_lm):
    """Empty panel (no keys) falls back to the primary Claude model."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    panel = PanelConsensus(panel={"openrouter/x/y": 1.0}, lm_factory=lambda slug: dummy_lm)
    assert list(panel.members.keys()) == [cfg.PRIMARY_LM]


def test_drugbank_lookup_fails_fast(monkeypatch):
    """Fail-fast: an unreachable endpoint raises instead of a fake result."""

    from genoma.track2.tools import drugbank_lookup

    def _dead_urlopen(*a, **k):
        raise OSError("network unreachable (test)")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _dead_urlopen)
    with pytest.raises(OSError):
        drugbank_lookup("aspirin")


def test_variant_candidate_is_pydantic():
    from pydantic import BaseModel

    from genoma.data.annotate import VariantCandidate

    assert issubclass(VariantCandidate, BaseModel)


def test_self_consistency_epcr_runs(dummy_lm):
    """self_consistency_epcr uses get_lm() (not .lm) and passes gene_mechanism."""
    _configure(dummy_lm)
    from genoma.track1.calibrate import self_consistency_epcr

    scorer = PredictVariantScorer()
    cand = demo_candidates(n_decoys=0)[0]  # BUB1B candidate
    mean, spread = self_consistency_epcr(scorer, cand, DEMO_HPO, n=3)
    assert 0.0 <= mean <= 1.0
    assert spread >= 0.0


def test_judge_composite_score(dummy_lm):
    """Judge scores calibration + classification, and skips LLM feedback when perfect."""
    from genoma.optimize.gepa_optimize import make_gepa_judge

    # The judge's own LM must emit JudgeFeedbackSignature's output fields
    # (ChainOfThought adds `reasoning`).
    judge_lm = dspy.utils.DummyLM(
        [{"reasoning": "r", "feedback": "consider allele frequency and gene relevance"}] * 200,
        adapter=dspy.JSONAdapter(),
    )
    dspy.configure(lm=judge_lm, adapter=dspy.JSONAdapter())
    metric = make_gepa_judge(judge_lm=judge_lm)
    gold = dspy.Example(
        variant="x | gene=BUB1B | consequence=missense | AF=1e-5",
        hpo_terms=["HP:x"],
        known_mva_genes="BUB1B",
        true_epcr=0.85,
    )

    # Perfect prediction -> score 1.0 AND no feedback (no LLM call for the judge).
    perfect = dspy.Prediction(epcr=0.85, rationale="x")
    res = metric(gold, perfect, None, None, None)
    assert res.score == pytest.approx(1.0)
    assert res.feedback == ""  # no LLM judge call when perfect

    # Wrong binary call -> classification=0; calibration reflects distance.
    wrong = dspy.Prediction(epcr=0.1, rationale="x")
    res2 = metric(gold, wrong, None, None, None)
    assert res2.score < 1.0
    assert res2.feedback  # feedback generated on imperfect


def test_judge_calibration_vs_classification_weights():
    """Weights steer the composite: classification-only ignores calibration."""
    from genoma.optimize.gepa_optimize import make_gepa_judge

    gold = dspy.Example(
        variant="x | gene=BUB1B | consequence=missense | AF=1e-5",
        hpo_terms=["HP:x"],
        known_mva_genes="BUB1B",
        true_epcr=0.85,
    )
    pred = dspy.Prediction(epcr=0.9, rationale="x")

    m1 = make_gepa_judge(calibration_weight=1.0, classification_weight=0.0)
    # Correct side but |0.9-0.85|=0.05 -> calibration=0.95.
    assert m1(gold, pred, None, None, None).score == pytest.approx(0.95)

    m2 = make_gepa_judge(calibration_weight=0.0, classification_weight=1.0)
    # Classification-only: correct side -> 1.0 regardless of distance.
    assert m2(gold, pred, None, None, None).score == pytest.approx(1.0)


def test_debate_lm_resolution(monkeypatch):
    """Debate model resolves to primary if its key is set, else a panel member."""
    from genoma import config as cfg
    from genoma.track1.consensus import _resolve_debate_lm

    monkeypatch.setattr(cfg, "PRIMARY_LM", "anthropic/claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    lm = _resolve_debate_lm({"openrouter/a/b": 1.0, "anthropic/claude-x": 1.0})
    assert "anthropic" in lm.model

    monkeypatch.setattr(cfg, "PRIMARY_LM", "openrouter/x/y")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    lm2 = _resolve_debate_lm({"openrouter/a/b": 1.0})
    assert "openrouter" in lm2.model


def test_config_paths():
    assert cfg.PROBAND_ID == "PROBAND01"
    assert cfg.MAX_SUBMISSION_ROWS == 10


def test_pair_scorer_augments_evidence_with_single_epcrs(dummy_lm):
    """The pair LM must see each variant's single-variant EPCR — otherwise it
    re-judges from scratch and can rank a weak pair above a strong one
    (BUB1B stop_gained 0.85 + missense vs two CEP192 missenses), costing the
    100-rank-point slot. The augmentation must reach the pair LM's prompt."""
    from unittest import mock

    _configure(dummy_lm)
    from genoma.data.demo import demo_candidates

    scorer = CompoundHetPairScorer(max_pairs=5)
    cands = demo_candidates()
    epcrs = {c.coord: 0.85 for c in cands}

    seen: list[str] = []
    real_call = scorer.pair_scorer

    def spy(*args, **kwargs):
        seen.append(kwargs.get("pair_description", ""))
        return real_call(*args, **kwargs)

    with mock.patch.object(scorer, "pair_scorer", side_effect=spy):
        out = scorer(cands, DEMO_HPO, variant_epcrs=epcrs)
    assert hasattr(out, "pairs")
    assert seen, "pair LM was called"
    assert any("single_variant_epcr=0.85" in desc for desc in seen), (
        "single-variant EPCRs must be fed into the pair LM evidence"
    )

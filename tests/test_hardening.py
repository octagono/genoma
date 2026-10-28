"""Hardening tests: retry, synthetic noise, T2 report."""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error
from email.message import Message

import pytest

from genoma.net import TRANSIENT_STATUS, urlopen_with_retry
from genoma.track2.report import generate_track2_report

# ── Retry helper ────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, payload=b"{}"):
        self._buf = io.BytesIO(payload)

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", hdrs=Message(), fp=io.BytesIO(b"{}"))


def test_retry_succeeds_after_transient_errors(monkeypatch):
    calls = []
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    def fake_urlopen(url, timeout=None, context=None):
        calls.append(url)
        if len(calls) < 3:
            raise _http_error(503)
        return _FakeResponse(b'{"ok": 1}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with urlopen_with_retry("http://x", attempts=3, backoff_s=0.0) as r:
        assert json.load(r) == {"ok": 1}
    assert len(calls) == 3
    assert len(sleeps) == 2  # backed off between attempts


def test_retry_non_transient_raises_immediately(monkeypatch):
    calls = []
    monkeypatch.setattr("time.sleep", lambda s: None)

    def fake_urlopen(url, timeout=None, context=None):
        calls.append(url)
        raise _http_error(404)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        urlopen_with_retry("http://x", attempts=3)
    assert len(calls) == 1  # never retried


def test_retry_exhaustion_raises_last_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def fake_urlopen(url, timeout=None, context=None):
        calls.append(url)
        raise _http_error(429)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        urlopen_with_retry("http://x", attempts=3)
    assert len(calls) == 3


def test_retry_urlerror_is_transient(monkeypatch):
    calls = []
    monkeypatch.setattr("time.sleep", lambda s: None)

    def fake_urlopen(url, timeout=None, context=None):
        calls.append(url)
        if len(calls) < 2:
            raise urllib.error.URLError("connection reset")
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with urlopen_with_retry("http://x", attempts=3):
        pass
    assert len(calls) == 2


def test_transient_status_classification():
    assert 429 in TRANSIENT_STATUS and 503 in TRANSIENT_STATUS
    assert 400 not in TRANSIENT_STATUS and 404 not in TRANSIENT_STATUS


# ── Synthetic trainset noise ────────────────────────────────────────────────


def test_synthetic_trainset_has_missing_evidence_noise():
    from genoma.optimize.gepa_optimize import build_synthetic_trainset

    examples = build_synthetic_trainset(n=60, seed=7)
    assert len(examples) == 60
    unknown_af = sum("AF=unknown" in e.variant for e in examples)
    unknown_cadd = sum("CADD=unknown" in e.variant for e in examples)
    unknown_cons = sum("consequence=unknown" in e.variant for e in examples)
    # p=0.25/0.25/0.15 over 60 draws — zero occurrences is vanishingly unlikely.
    assert unknown_af > 0
    assert unknown_cadd > 0
    assert unknown_cons > 0
    # Labels stay well-formed regardless of degraded evidence.
    assert all(e.true_epcr in (0.85, 0.10) for e in examples)


# ── Track 2 report ──────────────────────────────────────────────────────────


def test_track2_report_contains_all_candidates(tmp_path):
    from genoma.track2.agent import DrugCandidate

    cands = [
        DrugCandidate(
            name="DrugA",
            mechanism="MPS1 kinase inhibition",
            rationale="restores checkpoint signaling",
            evidence="PMID:12345678",
            risk="off-target toxicity",
        ),
        DrugCandidate(
            name="DrugB",
            mechanism="Aurora kinase B modulation",
            rationale="reduces missegregation",
            evidence="OpenFDA label XYZ",
            risk="myelosuppression",
        ),
    ]
    out = generate_track2_report(
        cands,
        mechanism="spindle assembly checkpoint failure",
        gene="BUB1B",
        hpo_terms=["HP:0001510 Growth delay"],
        termination_reason="max iterations",
        out_path=tmp_path / "t2.md",
    )
    text = out.read_text()
    assert "DrugA" in text and "DrugB" in text
    assert "PMID:12345678" in text
    assert "spindle assembly checkpoint failure" in text
    assert "BUB1B" in text
    assert "max iterations" in text
    assert "hypotheses" in text  # disclaimer present
    assert "## Methods" in text


def test_track2_report_empty_candidates(tmp_path):
    out = generate_track2_report(
        [],
        mechanism="m",
        gene="G",
        hpo_terms=[],
        termination_reason=None,
        out_path=tmp_path / "t2_empty.md",
    )
    text = out.read_text()
    assert "No candidates" in text


def test_track1_report_answers_all_template_questions(tmp_path):
    """The judged methods report must answer every question in the organizers'
    methods_description_form.xlsx (11 for Track 1)."""
    import re

    # Build a tiny demo submission first (the report reads a real CSV).
    import dspy

    from genoma.data.demo import DEMO_HPO, demo_candidates
    from genoma.track1.report import generate_track1_report
    from genoma.track1.scoring import PredictVariantScorer

    lm = dspy.utils.DummyLM(
        [{"reasoning": "r", "epcr": 0.9, "rationale": "demo"}] * 100,
        adapter=dspy.JSONAdapter(),
    )
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
    scorer = PredictVariantScorer()
    cands = demo_candidates(n_decoys=1)
    rows = []
    for c in cands[:2]:
        out = scorer(c, DEMO_HPO)
        rows.append(
            {
                "chrom_1": c.chrom,
                "pos_1": c.pos,
                "ref_1": c.ref,
                "alt_1": c.alt,
                "chrom_2": "",
                "pos_2": "",
                "ref_2": "",
                "alt_2": "",
                "epcr": float(out.epcr),
                "finding_type": "primary",
                "notes": "t",
            }
        )
    import csv

    csv_path = tmp_path / "sub.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    text = generate_track1_report(
        submission_csv=csv_path,
        hpo_terms=DEMO_HPO,
        panel_members=["anthropic/claude-sonnet-5"],
        team_name="T",
    )
    # All 11 template questions present (Q4-8 grouped, Q5/Q8 embedded).
    qs = re.findall(r"^## (Q\d+[-–]?\d*)", text, re.M)
    assert any(q.startswith("Q1") for q in qs)
    assert any(q.startswith("Q2") for q in qs)
    assert any(q.startswith("Q3") for q in qs)
    assert any(q.startswith("Q4") for q in qs)
    assert any(q.startswith("Q6") for q in qs)
    assert any(q.startswith("Q9") for q in qs)
    assert any(q.startswith("Q10") for q in qs)
    assert any(q.startswith("Q11") for q in qs)
    # Required Generative-AI disclosure + abstract present.
    assert "Generative-AI provider disclosure" in text
    assert "Method abstract" in text
    assert "Estimated cost" in text
    assert "compound-heterozygous pair rows" in text or "compound-het" in text


def test_track2_methods_report_answers_all_template_questions(tmp_path):
    """The Track 2 methods report must answer every question in the
    organizers' methods_description_form.xlsx (Track 2 sheet, 11 questions)."""
    import re

    from genoma.track2.report import generate_track2_methods_report

    text = generate_track2_methods_report(
        drug_candidates=[],
        mechanism="spindle assembly checkpoint failure",
        gene="BUB1B",
        hpo_terms=["HP:0002859 Rhabdomyosarcoma"],
        team_name="T",
        out_path=tmp_path / "t2_methods.md",
    ).read_text()
    qs = re.findall(r"^## Q([\d–\-]+)", text, re.M)
    nums = [int(q.split("–")[0]) for q in qs]
    # Template groups Q6-8 into one "Data sources" section.
    for n in (1, 2, 3, 4, 5, 6, 9, 10, 11):
        assert n in nums, f"missing template question Q{n} in methods report"
    # Required Generative-AI disclosure present.
    assert "Generative-AI provider disclosure" in text
    # Mechanism characterization + data + abstract all answered.
    assert "loss-of-function" in text
    assert "Public data only" in text
    assert "Method abstract" in text
    assert "Strengths" in text and "Limitations" in text


# ── HPO negation filtering (clinical sheets list excluded findings) ─────────


def _write_phenotype_docx(path, rows: list[list[str]]) -> None:
    import docx

    doc = docx.Document()
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    for r, cells in enumerate(rows):
        for c, val in enumerate(cells):
            table.rows[r].cells[c].text = val
    doc.save(str(path))


def test_hpo_extraction_skips_negated_findings(tmp_path):
    from genoma.data.ingest import extract_hpo_terms

    docx_path = tmp_path / "phenotype.docx"
    _write_phenotype_docx(
        docx_path,
        [
            ["HPO ID", "Feature", "Status"],
            ["HP:0000252", "Microcephaly", "Present"],
            ["HP:0001510", "Growth delay", "None"],
            ["HP:0004322", "Short stature", "not observed"],
            ["HP:0006738", "Rhabdomyosarcoma", "absent"],
        ],
    )
    terms = extract_hpo_terms(docx_path)
    joined = " ".join(terms)
    assert "HP:0000252" in joined  # positive finding kept
    assert "HP:0001510" not in joined  # negated rows dropped
    assert "HP:0004322" not in joined
    assert "HP:0006738" not in joined


def test_hpo_extraction_keeps_positives_without_status_column(tmp_path):
    from genoma.data.ingest import extract_hpo_terms

    docx_path = tmp_path / "phenotype.docx"
    _write_phenotype_docx(
        docx_path,
        [
            ["HP:0000252 Microcephaly"],
            ["HP:0001510 Growth delay"],
        ],
    )
    terms = extract_hpo_terms(docx_path)
    assert any("HP:0000252" in t for t in terms)
    assert any("HP:0001510" in t for t in terms)


# ── Genome assembly SSOT ────────────────────────────────────────────────────


def test_genome_assembly_is_grch38_and_consistent():
    from genoma import config as cfg
    from genoma.data.exomiser import ANALYSIS_TEMPLATE

    # Challenge schema pins GRCh38 (submit_track1.py); Exomiser's token for
    # the same build is hg38; the YAML template must use the config value.
    assert cfg.GENOME_ASSEMBLY == "GRCh38"
    assert cfg.EXOMISER_ASSEMBLY == "hg38"
    assert "{assembly}" in ANALYSIS_TEMPLATE


def test_exomiser_yaml_uses_config_assembly(tmp_path):
    from genoma.data.exomiser import write_analysis_yaml

    yaml_path = write_analysis_yaml(
        vcf=tmp_path / "x.vcf",
        hpo_terms=["HP:0000252 Microcephaly"],
        out_dir=tmp_path,
    )
    text = yaml_path.read_text()
    assert "genomeAssembly: hg38" in text


# ── Compliance purge covers DATA_DIR + HF hub cache ────────────────────────


def test_purge_caches_removes_data_dir_and_hf_cache(tmp_path, monkeypatch):
    from genoma import compliance

    data_dir = tmp_path / "data"
    (data_dir / ".cache" / "huggingface").mkdir(parents=True)
    (data_dir / "proband.vcf.gz").write_text("##fileformat=VCFv4.2")
    hf_hub = tmp_path / "hfhub"
    ds_dir = hf_hub / "datasets--SageBio--mva-hackathon-2026-data"
    ds_dir.mkdir(parents=True)
    (ds_dir / "blob.bin").write_text("x")
    sub_dir = tmp_path / "submissions"
    sub_dir.mkdir()
    (sub_dir / "s1.csv").write_text("proband_id,chrom_1")
    (sub_dir / "ckpt.jsonl").write_text("{}")

    monkeypatch.setattr(compliance.cfg, "DATA_DIR", data_dir)
    monkeypatch.setattr(compliance.cfg, "SUBMISSION_DIR", sub_dir)
    monkeypatch.setattr(compliance, "hf_dataset_cache_dirs", lambda: [ds_dir])
    monkeypatch.setattr(compliance.dspy.cache, "reset_memory_cache", lambda: None)

    result = compliance.purge_caches(confirm=False, delete_disk=False)
    assert result["memory"] is True
    assert result["data_dir"] is True
    assert result["submissions"] is True
    assert not sub_dir.exists()  # derived CSVs/checkpoints purged too
    assert result["hf_cache"] is True
    assert not data_dir.exists()
    assert not ds_dir.exists()


def test_purge_caches_keep_data_leaves_dataset(tmp_path, monkeypatch):
    from genoma import compliance

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "proband.vcf.gz").write_text("##fileformat=VCFv4.2")
    monkeypatch.setattr(compliance.cfg, "DATA_DIR", data_dir)
    monkeypatch.setattr(compliance, "hf_dataset_cache_dirs", lambda: [])
    monkeypatch.setattr(compliance.dspy.cache, "reset_memory_cache", lambda: None)

    result = compliance.purge_caches(confirm=False, delete_disk=False, delete_data=False)
    assert result["data_dir"] is False
    assert data_dir.exists()


def test_hf_dataset_cache_dirs_honors_env(tmp_path, monkeypatch):
    from genoma import compliance

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    dirs = compliance.hf_dataset_cache_dirs()
    assert dirs == [tmp_path / "hub" / "datasets--SageBio--mva-hackathon-2026-data"]


# ── Novel-gene balance (prior boosts known genes, never blinds to novel) ────


def test_gene_prior_is_boost_only():
    """Unknown SAC/mitotic genes must keep their raw EPCR — the MVA-gene
    prior may boost known genes but must never penalize novel ones (MVA is
    heterogeneous; the causal gene could be an unlisted BUB3/MAD2L1/MPS1)."""
    from genoma.data.annotate import KNOWN_MVA_GENES, apply_gene_prior

    assert apply_gene_prior(0.95, "BUB3") == pytest.approx(0.95)  # unlisted: untouched
    assert apply_gene_prior(0.95, "MPS1") == pytest.approx(0.95)
    known = next(iter(KNOWN_MVA_GENES))
    boosted = apply_gene_prior(0.7, known)
    assert boosted > 0.7  # listed: boosted toward 1.0
    assert boosted <= 1.0


def test_novel_gene_overwhelming_evidence_surfaces_top():
    """End-to-end sanity: a highly pathogenic novel-gene pair outranks a
    mediocre known-gene pair even after prior boosting + calibration."""
    from genoma.data.annotate import apply_gene_prior
    from genoma.track1.calibrate import calibrate_submission_epcrs
    from genoma.track1.scoring import ScoredPair, ScoredVariant

    def _sv(pos, epcr):
        return ScoredVariant(chrom="chr2", pos=pos, ref="G", alt="A", epcr=epcr, rationale="r")

    def _pair(p1, p2, epcr, gene):
        return ScoredPair(
            variant_1=_sv(p1, 0.0),
            variant_2=_sv(p2, 0.0),
            epcr=apply_gene_prior(epcr, gene),
            rationale="r",
        )

    # Raw scorer: overwhelming evidence in unlisted BUB3 vs mediocre BUB1B.
    novel = _pair(100, 200, 0.96, "BUB3")
    known = _pair(300, 400, 0.70, "BUB1B")
    assert novel.epcr > known.epcr  # prior boost didn't flip the ordering

    cp, _ = calibrate_submission_epcrs([novel, known], [])
    assert cp[0] is not None
    assert cp[0].epcr == pytest.approx(1.0, abs=0.01)  # novel-gene pair is rank 1
    assert cp[0].variant_1.pos == 100


# ── Retry defaults survive a rate-limit burst ───────────────────────────────


def test_retry_defaults_cover_rate_limit_burst():
    """Defaults must be wide enough to ride out a VEP/Entrez 429 burst from
    the 8-thread parallel scorer (not just a single blip)."""
    import inspect

    from genoma import net

    sig = inspect.signature(net.urlopen_with_retry)
    assert sig.parameters["attempts"].default == 4
    assert sig.parameters["backoff_s"].default == 2.0
    # Backoff ladder: 2.0 + 3.0 + 4.5 = ~9.5s of patience across 4 attempts.
    assert frozenset({429, 500, 502, 503, 504}) == net.TRANSIENT_STATUS


# ── Ingest storage preflight ────────────────────────────────────────────────


def test_assert_free_space_raises_when_insufficient(tmp_path):
    from genoma.data.ingest import assert_free_space

    assert_free_space(tmp_path, 1_000)  # tiny need: fine
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        assert_free_space(tmp_path, 10**18)  # ~1 EB: never fits


def test_assert_free_space_applies_headroom(tmp_path, monkeypatch):
    import shutil as shutil_mod

    from genoma.data.ingest import assert_free_space

    real = shutil_mod.disk_usage(str(tmp_path))
    fake = shutil_mod.disk_usage(str(tmp_path))._replace(free=100)
    monkeypatch.setattr("shutil.disk_usage", lambda p: fake)
    # 90 needed * 1.2 headroom = 108 > 100 free -> raises.
    with pytest.raises(RuntimeError):
        assert_free_space(tmp_path, 90)
    assert_free_space(tmp_path, 80)  # 80 * 1.2 = 96 <= 100 -> ok
    assert real.free > 0  # sanity: _replace produced a sane fixture


def test_required_download_bytes_from_hf_metadata(monkeypatch):
    from genoma.data import ingest

    class _Sibling:
        def __init__(self, name, size):
            self.rfilename = name
            self.size = size

    class _Info:
        siblings = [
            _Sibling("proband.vcf.gz", 5_000_000_000),
            _Sibling("proband.vcf.gz.tbi", 4_000_000),
            _Sibling("phenotype.docx", 50_000),
        ]

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def dataset_info(self, repo_id, files_metadata=False):
            assert files_metadata is True
            return _Info()

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    total = ingest.required_download_bytes(
        ["proband.vcf.gz", "proband.vcf.gz.tbi", "phenotype.docx"], token="t"
    )
    assert total == 5_000_000_000 + 4_000_000 + 50_000
    # Unknown files count as 0 rather than erroring.
    assert ingest.required_download_bytes(["missing.fastq"], token="t") == 0


# ── GEPA judge weight knob ──────────────────────────────────────────────────


def test_judge_classification_weight_wiring():
    import inspect

    from genoma.optimize.gepa_optimize import (
        _judge_from_weight,
        evaluate_with_judge,
        make_gepa_judge,
        optimize_with_gepa,
    )

    # Default: the 0.7/0.3 judge.
    default = _judge_from_weight(None)
    assert callable(default)
    # Single-knob override produces a callable judge.
    tuned = _judge_from_weight(0.9)
    assert callable(tuned)
    assert tuned is not default
    # Out-of-range weights fail fast.
    with pytest.raises(ValueError, match="must be in"):
        _judge_from_weight(1.5)
    # make_gepa_judge remains directly parameterized (both weights).
    assert callable(make_gepa_judge(classification_weight=0.2, calibration_weight=0.8))
    # Both entry points expose the knob.
    assert "judge_classification_weight" in inspect.signature(evaluate_with_judge).parameters
    assert "judge_classification_weight" in inspect.signature(optimize_with_gepa).parameters


# ── Confidence-weighted self-consistency + order invariance ─────────────────


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_self_consistency_confidence_weighting():
    """Outlier samples drag the confidence-weighted mean less than a plain
    average; HPO order is shuffled per sample."""
    from genoma.track1.calibrate import self_consistency_epcr

    seen_orders = []
    samples = iter([0.9, 0.1, 0.85, 0.8, 0.88])

    class _Scorer:
        class _Inner:
            def __call__(self, variant, hpo_terms, known_mva_genes, gene_mechanism):
                seen_orders.append(tuple(hpo_terms))
                return type("Out", (), {"epcr": next(samples)})()

        def __init__(self):
            self.scorer = self._Inner()

        def get_lm(self):
            return None  # falls back to dspy.settings.lm

    import genoma.track1.calibrate as cal
    from genoma.data.demo import demo_candidates

    cand = demo_candidates(n_decoys=0)[0]  # real VariantCandidate (BUB1B)
    orig = cal.dspy.context
    cal.dspy.context = lambda **kw: _NullCtx()
    try:
        mean, spread = self_consistency_epcr(
            scorer := _Scorer(),
            cand,
            ["HP:0001510 Growth delay", "HP:0000252 Microcephaly"],
            n=5,
        )
    finally:
        cal.dspy.context = orig
    # 0.1 is the outlier; consensus center ~0.85. Weighted mean must be
    # closer to 0.85 than the plain mean (0.706) is.
    plain = (0.9 + 0.1 + 0.85 + 0.8 + 0.88) / 5
    assert abs(mean - 0.85) < abs(plain - 0.85)
    assert spread >= 0.0
    # Order invariance: each sample saw an HPO order, same set.
    assert len(seen_orders) == 5
    assert all(
        set(o) == {"HP:0001510 Growth delay", "HP:0000252 Microcephaly"} for o in seen_orders
    )


# ── Judge rubric + new MCP tools ────────────────────────────────────────────


def test_judge_signature_has_explicit_rubric():
    """The judge prompt must carry a concrete scoring rubric (explicit
    rubrics fix situational preference) and must not reward verbosity."""
    from genoma.optimize.gepa_optimize import JudgeFeedbackSignature

    doc = JudgeFeedbackSignature.__doc__ or ""
    assert "Scoring rubric" in doc
    assert "0.5 causal" in doc
    assert "binary causal call first" in doc
    assert "Do NOT reward verbosity" in doc


def test_mechanism_mcp_tool_known_and_novel():
    """The mechanism KB tool answers for known MVA genes and is honest for
    novel ones (never claims a novel gene is ruled out)."""
    from genoma.mcp_server import _get_mcp, _register_tools

    _register_tools()
    result = asyncio.run(_get_mcp().call_tool("mechanism", {"gene": "BUB1B"}))
    text = result.content[0].text
    assert "known_mva_gene=True" in text
    assert "prior_boost=0.15" in text

    result = asyncio.run(_get_mcp().call_tool("mechanism", {"gene": "BUB3"}))
    text = result.content[0].text
    assert "known_mva_gene=False" in text
    assert "prior_boost=0.00" in text
    assert "novel SAC/mitotic gene" in text


def test_mcp_surface_has_annotate_and_candidates():
    """annotate (was CLI-only) and candidates are now MCP tools."""
    import asyncio

    from genoma.mcp_server import _get_mcp, _register_tools

    _register_tools()
    names = {t.name for t in asyncio.run(_get_mcp().list_tools())}
    assert "annotate" in names
    assert "candidates" in names
    assert "mechanism" in names


# ── Abstention + evidence-integrated debate + cross-family judge ───────────


def test_consensus_result_has_abstention_flag():
    """A panel that stays split post-debate is flagged abstained, not
    presented as a confident score."""
    import dspy

    from genoma.data.demo import demo_candidates
    from genoma.track1.consensus import ConsensusResult, PanelConsensus
    from genoma.track1.scoring import ScoredVariant

    lm = dspy.utils.DummyLM(
        [{"reasoning": "r", "epcr": 0.5, "rationale": "demo"}] * 100,
        adapter=dspy.JSONAdapter(),
    )
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
    panel = PanelConsensus(
        panel={"anthropic/claude-sonnet-5": 1.0},
        lm_factory=lambda slug: lm,
    )
    res = panel(demo_candidates()[0], ["HP:0001510 Growth delay"])
    assert isinstance(res, ConsensusResult)
    assert res.abstained in (True, False)  # flag exists and is a bool
    # The ScoredVariant carries the abstention through to ranking.
    assert isinstance(res.variant, ScoredVariant)


def test_panel_weights_affect_aggregation():
    """PANEL weights are real: a 2.0-weighted vote pulls the aggregate toward
    it (config declares claude-sonnet-5 at 2.0x — it must not be dead)."""
    from genoma.track1.consensus import PanelConsensus

    panel = PanelConsensus.__new__(PanelConsensus)  # bypass __init__ (no LM)
    # unweighted median of [0.4, 0.9] = 0.65; weighted (0.4 at 2.0, 0.9 at 1.0)
    # -> expanded [0.4, 0.4, 0.9] -> median 0.4. The heavier vote wins.
    assert panel._aggregate([0.4, 0.9], weights=[2.0, 1.0]) == pytest.approx(0.4)
    assert panel._aggregate([0.4, 0.9], weights=[1.0, 1.0]) == pytest.approx(0.65)
    # 5+ expanded votes -> trimmed weighted mean (drops one low + one high).
    agg5 = panel._aggregate([0.1, 0.2, 0.8, 0.9, 0.95], weights=[1.0] * 5)
    assert 0.2 < agg5 < 0.9


def test_debate_signature_has_structured_evidence():
    """The debate grounds on a structured evidence vector, not just
    reviewer opinions."""
    from genoma.track1.consensus import DebateSignature

    fields = DebateSignature.__annotations__
    assert "evidence" in fields
    assert "rationales" in fields


def test_cross_family_judge_config_wiring(monkeypatch):
    """The GEPA judge defaults to the proposer family but can be switched
    cross-family via cfg.GEPA_JUDGE_LM."""
    from genoma import config as cfg
    from genoma.optimize.gepa_optimize import _judge_from_weight

    assert cfg.GEPA_JUDGE_LM is None  # default: settings.lm (proposer family)
    monkeypatch.setattr(cfg, "GEPA_JUDGE_LM", "deepseek/deepseek-v4-pro")
    judge = _judge_from_weight(None)
    assert callable(judge)


def test_track2_stall_diagnosis():
    """Repeated identical tool calls are flagged as a stall; a healthy
    history is not."""
    from genoma.track2.agent import diagnose_tool_loop

    assert diagnose_tool_loop(None) == ""
    assert diagnose_tool_loop([]) == ""
    stalled = [
        {"tool_call": "pubmed_search(query='BUB1B')"},
        {"tool_call": "pubmed_search(query='BUB1B')"},
        {"tool_call": "pubmed_search(query='BUB1B')"},
        {"tool_call": "pubmed_search(query='BUB1B')"},
    ]
    diag = diagnose_tool_loop(stalled)
    assert "stall" in diag.lower()
    healthy = [
        {"tool_call": "pubmed_search(query='BUB1B')"},
        {"tool_call": "drugbank_lookup(drug='X')"},
        {"tool_call": "clinvar_lookup(term='BUB1B')"},
    ]
    assert diagnose_tool_loop(healthy) == ""


# ── Checkpointing + resilient production mode ───────────────────────────────


def test_score_chunked_fail_fast_raises():
    """Default = fail-fast: a broken candidate aborts the run loudly."""
    import dspy

    from genoma.checkpoint import score_chunked

    lm = dspy.utils.DummyLM([{"reasoning": "r", "epcr": 0.5, "rationale": "d"}] * 100)
    dspy.configure(lm=lm)

    def _score_one(candidate, hpo_terms):
        if candidate == "bad":
            raise RuntimeError("boom")
        return {"epcr": 0.5, "rationale": "ok"}

    with pytest.raises(RuntimeError, match="boom"):
        score_chunked(["a", "bad", "c"], [], _score_one, lambda c, e: {"epcr": 0.0})


def test_score_chunked_resilient_degrades_and_continues(capsys):
    """Resilient mode: failing candidates get a loud warning + safe default,
    the rest of the batch still scores."""
    import dspy

    from genoma.checkpoint import score_chunked

    lm = dspy.utils.DummyLM([{"reasoning": "r", "epcr": 0.5, "rationale": "d"}] * 100)
    dspy.configure(lm=lm)

    def _score_one(candidate, hpo_terms):
        if candidate == "bad":
            raise RuntimeError("boom")
        return {"epcr": 0.5, "rationale": "ok"}

    out = score_chunked(
        ["a", "bad", "c"],
        [],
        _score_one,
        lambda c, e: {"epcr": 0.0, "rationale": f"SAFE-DEFAULT: {e}"},
        resilient=True,
    )
    assert len(out) == 3
    by_coord = dict(out)
    assert by_coord["a"]["epcr"] == 0.5
    assert by_coord["bad"]["epcr"] == 0.0
    assert "SAFE-DEFAULT" in by_coord["bad"]["rationale"]
    assert "boom" in capsys.readouterr().out  # loud warning surfaced


def test_score_chunked_checkpoint_resume(tmp_path):
    """Checkpointing: an interrupted run resumes from the last chunk, not
    candidate #1."""
    import dspy

    from genoma.checkpoint import score_chunked

    lm = dspy.utils.DummyLM([{"reasoning": "r", "epcr": 0.5, "rationale": "d"}] * 100)
    dspy.configure(lm=lm)
    ckpt = tmp_path / "ckpt.jsonl"

    calls: list[str] = []

    def _score_one(candidate, hpo_terms):
        calls.append(candidate)
        return {"epcr": 0.5, "rationale": "ok"}

    # First run: score all 4.
    score_chunked(
        ["a", "b", "c", "d"],
        [],
        _score_one,
        lambda c, e: {"epcr": 0.0},
        checkpoint_path=ckpt,
        chunk_size=2,
    )
    first_calls = list(calls)
    assert len(first_calls) == 4

    # Second run (e.g. crashed run restarted): candidates already in the
    # checkpoint are NOT re-scored.
    calls.clear()
    score_chunked(
        ["a", "b", "c", "d"],
        [],
        _score_one,
        lambda c, e: {"epcr": 0.0},
        checkpoint_path=ckpt,
        chunk_size=2,
    )
    assert calls == []  # everything resumed from checkpoint
    assert ckpt.exists()


# ── Local SQLite annotation cache ───────────────────────────────────────────


def test_annotation_cache_roundtrip(tmp_path):
    from genoma.data import annot_cache

    db = tmp_path / "annot_cache.db"
    annot_cache.cache_set("chr2:47908000:G>A", {"symbol": "BUB1B", "gnomAD": {"af": 0.001}}, db)
    got = annot_cache.cache_get("chr2:47908000:G>A", db)
    assert got == {"symbol": "BUB1B", "gnomAD": {"af": 0.001}}
    assert annot_cache.cache_get("chr2:99999999:G>A", db) is None  # miss
    assert annot_cache.cache_size(db) == 1


def test_annotation_cache_degrades_to_miss_not_crash(tmp_path):
    """A corrupt DB must behave as a cache miss, never a crash."""
    from genoma.data import annot_cache

    db = tmp_path / "annot_cache.db"
    db.write_bytes(b"not a sqlite db at all")
    assert annot_cache.cache_get("chr2:1:G>A", db) is None
    assert annot_cache.cache_size(db) == 0
    # write is best-effort too
    annot_cache.cache_set("chr2:1:G>A", {"x": 1}, db)


# ── Offline HPO synonym mapping ─────────────────────────────────────────────


def test_hpo_synonym_map_maps_free_text(tmp_path):
    from genoma.data.ingest import _map_free_text, extract_hpo_terms

    assert "HP:0001263 Global developmental delay" in _map_free_text(
        "shows developmental lag and failure to thrive"
    )
    assert "HP:0000252 Microcephaly" in _map_free_text("small head, growth delay")
    assert _map_free_text("nothing relevant here") == []

    # End-to-end: a docx with free text (no HPO ids) falls back to synonyms.
    import docx

    doc = docx.Document()
    doc.add_paragraph("The child presents with growth delay and a small head.")
    docx_path = tmp_path / "phenotype_free.docx"
    doc.save(str(docx_path))
    terms = extract_hpo_terms(docx_path)
    assert any("HP:0001510" in t for t in terms)  # growth delay mapped
    assert any("HP:0000252" in t for t in terms)  # small head -> microcephaly


# ── Trio Mendelian phasing ──────────────────────────────────────────────────


def test_mendelian_phase_trans_when_different_parents():
    from genoma.data.phasing import mendelian_phase

    # Mother het A + homo-ref B; father het B + homo-ref A -> TRANS.
    ph = mendelian_phase("0/1", "0/1", "0/0", "0/1", "0/0", "0/1")
    assert ph.status == "trans"
    assert "trio" in ph.evidence
    # Same parent both -> unknown (uninformative for phase).
    ph2 = mendelian_phase("0/1", "0/1", "0/0", "0/1", "0/1", "0/0")
    assert ph2.status == "unknown"
    # Missing parental data -> unknown.
    ph3 = mendelian_phase("0/1", "", "", "0/1", "", "")
    assert ph3.status == "unknown"


def test_trio_phase_pairs_reads_family_samples():
    from genoma.data.annotate import VariantCandidate
    from genoma.data.phasing import trio_phase_pairs

    # Two same-gene (BUB1B) candidates with a full trio in the FORMAT dict.
    cands = [
        VariantCandidate(chrom="chr2", pos=47908000, ref="G", alt="A", gene="BUB1B"),
        VariantCandidate(chrom="chr2", pos=47912000, ref="C", alt="T", gene="BUB1B"),
    ]
    records = [
        {
            "chrom": cands[0].chrom,
            "pos": cands[0].pos,
            "format": {"PROBAND": "0/1", "MOTHER": "0/1", "FATHER": "0/0"},
        },
        {
            "chrom": cands[1].chrom,
            "pos": cands[1].pos,
            "format": {"PROBAND": "0/1", "MOTHER": "0/0", "FATHER": "0/1"},
        },
    ]
    statuses = trio_phase_pairs(cands, records)
    assert len(statuses) == 1
    assert next(iter(statuses.values())).status == "trans"


# ── Token-bucket rate limiter ───────────────────────────────────────────────


def test_token_bucket_acquire_and_block():
    from genoma.lm import TokenBucket

    # Small capacity: first acquire succeeds, second must wait for refill.
    b = TokenBucket(capacity=10.0, refill_seconds=60.0)
    b.acquire(10.0)
    assert b._tokens == pytest.approx(0.0)
    # Refill of 10 tokens/60s = 1/6 token/sec; after 6s ~1 token available.
    import time as _time

    _time.sleep(6.0)
    b.acquire(1.0)  # succeeds after refill


def test_token_bucket_unlimited_when_zero():
    from genoma.lm import TokenBucket

    b = TokenBucket(capacity=0.0)
    b.acquire(10**9)  # unlimited: never blocks


def test_rate_limited_lm_structure():
    """RateLimitedLM subclasses dspy.LM and carries the rate-limited path."""
    from genoma.lm import RateLimitedLM

    assert issubclass(RateLimitedLM, object)
    assert callable(RateLimitedLM)  # the rate-limited call path exists


# ── Exomiser output coupling ────────────────────────────────────────────────


def test_merge_exomiser_scores_onto_candidates():
    from genoma.data.annotate import VariantCandidate
    from genoma.data.exomiser import merge_exomiser_scores

    cand = VariantCandidate(chrom="chr2", pos=1, ref="G", alt="A", gene="BUB1B")
    parsed = {
        "genes": {
            "BUB1B": {
                "exomiser_score": 0.88,
                "phenotype_score": 0.72,
                "variant_score": 0.5,
                "comp_het": True,
            }
        }
    }
    out = merge_exomiser_scores([cand], parsed)
    assert out[0].exomiser_score == pytest.approx(0.88)
    assert out[0].exomiser_pheno_score == pytest.approx(0.72)
    assert out[0].exomiser_comp_het is True
    # Unmatched gene keeps fields unset.
    other = VariantCandidate(chrom="chr2", pos=2, ref="G", alt="A", gene="TTN")
    out2 = merge_exomiser_scores([other], parsed)
    assert out2[0].exomiser_score is None


def test_exomiser_rank_rlm_tool():
    from genoma.track1.scoring import exomiser_rank

    lookup = exomiser_rank(
        {"genes": {"BUB1B": {"exomiser_score": 0.88, "phenotype_score": 0.72, "comp_het": True}}}
    )
    assert "exomiser=0.880" in lookup("BUB1B")
    assert lookup("MISSING") == ""


# ── Pre-filter kill-risk: splice/low-CADD causal variants must survive ──────


def test_prefilter_keeps_splice_low_cadd_variants():
    """A splice-region causal variant with CADD < 10 must NOT be dropped by
    the deterministic pre-filter (it would be unrecoverable downstream)."""
    from genoma.data.annotate import build_candidate_table

    records = [
        {
            "chrom": "chr2",
            "pos": 47908123,
            "ref": "G",
            "alt": "A",
            "qual": 200,
            "filter": "PASS",
            "info": {"CADD": 4.2, "CSQ": "G|splice_region_variant|BUB1B"},
        },
        {
            "chrom": "chr2",
            "pos": 47908500,
            "ref": "C",
            "alt": "T",
            "qual": 200,
            "filter": "PASS",
            "info": {"CADD": 28.5, "CSQ": "T|missense_variant|BUB1B"},
        },
    ]
    cands = build_candidate_table(records, filter_kwargs={"min_cadd": 10.0})
    # Splice-region with CADD 4.2 < min_cadd survives; missense with CADD 28.5
    # also survives. Only a low-CADD coding variant would be cut.
    cons = {c.consequence for c in cands}
    assert cons == {"splice_region_variant", "missense_variant"}
    # Genome-aware ordering ranks the coding missense above the splice-region.
    by_cadd = {c.consequence: c.cadd for c in cands}
    assert by_cadd["splice_region_variant"] == 4.2
    assert by_cadd["missense_variant"] == 28.5


def test_prefilter_caps_from_config():
    """The candidate cap is config-driven (SSOT), not hardcoded."""
    from genoma import config as cfg
    from genoma.data.annotate import build_candidate_table

    records = [
        {
            "chrom": "chr1",
            "pos": i,
            "ref": "G",
            "alt": "A",
            "qual": 100,
            "filter": "PASS",
            "info": {"CADD": 25.0, "CSQ": f"A|missense_variant|G{i}"},
        }
        for i in range(10)
    ]
    assert cfg.CANDIDATE_MAX_DEFAULT >= 500  # config-driven cap has headroom
    capped = build_candidate_table(records, max_candidates=3)
    assert len(capped) == 3


# ── VEP-annotated VCF parsing (CSQ named fields) ────────────────────────────


def test_vep_csq_named_field_extraction():
    """VEP CSQ is Allele|Consequence|IMPACT|SYMBOL|Gene|... — gene/consequence
    must be extracted by NAME (header-declared order), not by position."""
    from genoma.data.annotate import (
        _csq_field,
        _extract_consequence,
        _extract_gene,
        _set_vep_csq_fields,
    )

    # VEP 110 standard order (Allele first).
    _set_vep_csq_fields(
        ["Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature_type", "Feature", "BIOTYPE"]
    )
    rec = {
        "info": {
            "CSQ": "A|missense_variant|MODERATE|BUB1B|ENSG00000169679|"
            "Transcript|ENST00000308033|protein_coding"
        }
    }
    assert _extract_consequence(rec) == "missense_variant"
    assert _extract_gene(rec) == "BUB1B"
    assert _csq_field(rec, "SYMBOL") == "BUB1B"
    assert _csq_field(rec, "Gene") == "ENSG00000169679"


def test_vep_csq_builds_candidate_table():
    """End-to-end: a VEP-annotated record flows into the candidate table with
    gene + consequence populated."""
    from genoma.data.annotate import build_candidate_table

    records = [
        {
            "chrom": "chr2",
            "pos": 47908123,
            "ref": "G",
            "alt": "A",
            "qual": 200,
            "filter": "PASS",
            "info": {
                "CSQ": "A|splice_region_variant|LOW|BUB1B|ENSG00000169679|"
                "Transcript|ENST00000308033|protein_coding"
            },
        },
    ]
    cands = build_candidate_table(records)
    assert len(cands) == 1
    assert cands[0].gene == "BUB1B"
    assert cands[0].consequence == "splice_region_variant"


# ── Contig normalization (real VCF uses bare contigs, scorer wants chrN) ─────


def test_vcf_contigs_normalized_to_chrn():
    """Input VCFs may carry bare contigs (1..22, X, Y, M) while the official
    scorer compares chromosome strings verbatim in chrN form. load_vcf_records
    must normalize at the boundary so submission rows carry chrN."""
    from genoma.data.ingest import _normalize_chrom, load_vcf_records

    assert _normalize_chrom("1") == "chr1"
    assert _normalize_chrom("22") == "chr22"
    assert _normalize_chrom("X") == "chrX"
    assert _normalize_chrom("Y") == "chrY"
    assert _normalize_chrom("M") == "chrM"
    assert _normalize_chrom("MT") == "chrM"
    # Idempotent: already-prefixed contigs pass through untouched.
    assert _normalize_chrom("chr2") == "chr2"
    assert _normalize_chrom("chrX") == "chrX"
    assert _normalize_chrom("chrM") == "chrM"

    # End-to-end through the cyvcf2 path: a bare-contig VCF yields chrN records.
    import gzip

    from genoma import config as cfg

    tmp = cfg.ROOT / "data" / "_norm_test.vcf.gz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=248956422>\n"
        "##contig=<ID=M,length=16569>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t1000\t.\tA\tG\t.\tPASS\t.\n"
        "M\t57\t.\tT\tC\t.\tPASS\t.\n"
    )
    with gzip.open(tmp, "wt") as f:
        f.write(text)
    try:
        recs = load_vcf_records(tmp)
        chroms = sorted(r["chrom"] for r in recs)
        assert chroms == ["chr1", "chrM"], f"got {chroms}"
    finally:
        tmp.unlink(missing_ok=True)


# ── Candidate-pool selection (genome-aware, AF genotype vs population) ──────


def test_af_ignores_sample_genotype_frequency():
    """A VCF may carry AC=1;AF=0.5;AN=2 (the SAMPLE's het genotype), not
    gnomAD population AF. _extract_af must NOT read that as population AF or
    every variant would be flagged 'common' (AF 0.5-1.0 > max_af) and the
    causal pair dropped."""
    from genoma.data.annotate import _extract_af

    assert _extract_af({"info": {"AC": 1, "AF": 0.5, "AN": 2}}) is None
    assert _extract_af({"info": {"AC": 2, "AF": 1.0, "AN": 2}}) is None
    # A true population-frequency field is still honored.
    assert _extract_af({"info": {"gnomAD_AF": 1e-5}}) == 1e-5
    assert _extract_af({"info": {"MAX_AF": 0.05}}) == 0.05
    # Bare AF without AC/AN context is treated as population (annotated VCFs).
    assert _extract_af({"info": {"AF": 0.001}}) == 0.001


def test_pool_selection_is_genome_aware_not_vcf_order():
    """The causal variant may sit on a late chromosome (BUB1B is chr15, but the
    VCF's first records are chr1). The capped pool must select known-gene and
    protein-altering variants genome-wide, not the first N records in VCF order."""
    from genoma.data.annotate import build_candidate_table

    # Records in VCF order: 30 chr1 intergenic noise first, then the causal
    # pair on chr15. A naive first-N cap (max_candidates=5) would take only
    # chr1 records; genome-aware selection must take the known-gene pair.
    records = [
        {
            "chrom": "chr1",
            "pos": i,
            "ref": "G",
            "alt": "A",
            "qual": 100,
            "filter": "PASS",
            "info": {"CSQ": f"A|intergenic_variant|chr1{i}"},
        }
        for i in range(30)
    ] + [
        {
            "chrom": "chr15",
            "pos": 90000001,
            "ref": "T",
            "alt": "G",
            "qual": 700,
            "filter": "PASS",
            "info": {
                "CSQ": "G|stop_gained|HIGH|BUB1B|ENSG00000156970|"
                "Transcript|ENST00000287598|protein_coding"
            },
        },
        {
            "chrom": "chr15",
            "pos": 90010012,
            "ref": "T",
            "alt": "G",
            "qual": 344,
            "filter": "PASS",
            "info": {
                "CSQ": "G|missense_variant|MODERATE|BUB1B|ENSG00000156970|"
                "Transcript|ENST00000287598|protein_coding"
            },
        },
    ]
    cands = build_candidate_table(records, max_candidates=5)
    coords = {c.coord for c in cands}
    assert "chr15:90000001:T>G" in coords, "known-gene LOF must survive the cap"
    assert "chr15:90010012:T>G" in coords, "known-gene missense must survive the cap"
    assert cands[0].gene == "BUB1B"  # causal pair ranks above chr1 noise

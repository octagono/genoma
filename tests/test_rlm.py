"""RLM execution test — verifies the sandbox loop with real runnable code.

RLM (dspy 3.3.1) lets the LM write Python that runs in a Deno-sandboxed
interpreter. We drive it with a DummyLM that emits *actual working code*
so the full chain is verified offline:

  action prompt -> code generation -> sandbox execution -> SUBMIT -> typed parse

The emitted code reads the injected `variants` list (JSON dicts), computes
a heuristic score per variant, and SUBMITs a ranked list of ScoredVariant.
"""

from __future__ import annotations

import dspy

from genoma.data.demo import DEMO_HPO, demo_candidates
from genoma.track1.scoring import (
    RLMVariantScorer,
    gene_prior_fraction,
    known_mva_gene,
    mechanism_lookup,
)

# Real Python that runs inside the sandbox: rank by a CADD-like field.
_SANDBOX_CODE = """\
def score(v):
    s = 0.0
    if "BUB1B" in str(v.get("gene", "")):
        s += 0.4
    af = v.get("af")
    if af is not None and af < 0.005:
        s += 0.3
    cadd = v.get("cadd")
    if cadd is not None and cadd > 20:
        s += 0.2
    return min(s, 0.95)

ranked = sorted(variants, key=score, reverse=True)
out = []
for v in ranked:
    out.append({
        "chrom": v["chrom"],
        "pos": v["pos"],
        "ref": v["ref"],
        "alt": v["alt"],
        "epcr": score(v),
        "rationale": "heuristic demo score",
    })
SUBMIT(ranked=out)
"""


def _make_action_lm():
    """DummyLM that always emits the runnable scoring code (JSONAdapter)."""
    answer = {
        "reasoning": "compute heuristic scores over the variants table",
        "code": _SANDBOX_CODE,
    }
    return dspy.utils.DummyLM([answer] * 200, adapter=dspy.JSONAdapter())


def test_rlm_runs_code_and_submits():
    """Full RLM loop: code gen -> sandbox exec -> SUBMIT -> typed parse."""
    dspy.configure(lm=_make_action_lm(), adapter=dspy.JSONAdapter())
    scorer = RLMVariantScorer(max_llm_calls=5)

    candidates = demo_candidates(n_decoys=4)
    pred = scorer(candidates, DEMO_HPO)
    ranked = pred.ranked

    assert ranked, "RLM returned an empty ranked list"
    for v in ranked:
        assert 0.0 < v.epcr <= 1.0
        assert v.rationale
    # The heuristic gives BUB1B variants the highest scores -> they rank first.
    top_coords = {f"{v.chrom}:{v.pos}" for v in ranked[:2]}
    assert any(f"{c.chrom}:{c.pos}" in top_coords for c in candidates if c.gene == "BUB1B")


# ── RLM sandbox domain tools (dspy 3.3.1 `RLM(tools=...)`) ──────────────────


def test_rlm_domain_tools_work():
    """Domain tools are callable host-side (mechanism KB, known-gene, prior)."""
    assert "spindle-assembly" in mechanism_lookup("BUB1B")
    assert mechanism_lookup("TTN") == ""  # not an MVA gene
    assert known_mva_gene("CEP57") is True
    assert known_mva_gene("TTN") is False
    assert gene_prior_fraction("BUB1B") > 0.0
    assert gene_prior_fraction("TTN") == 0.0


def test_rlm_injects_tools_into_sandbox():
    """The sandboxed code can call a domain tool by name (host-side exec)."""
    code = """
out = []
for v in variants:
    m = mechanism_lookup(v["gene"])
    out.append({
        "chrom": v["chrom"],
        "pos": v["pos"],
        "ref": v["ref"],
        "alt": v["alt"],
        "epcr": 0.9 if "spindle" in m or "centrosome" in m else 0.1,
        "rationale": "mechanism tool result: " + (m or "none"),
    })
SUBMIT(ranked=out)
"""
    answer = {"reasoning": "use the mechanism tool per variant", "code": code}
    dspy.configure(lm=dspy.utils.DummyLM([answer] * 200, adapter=dspy.JSONAdapter()))
    scorer = RLMVariantScorer(max_llm_calls=5)  # default = domain tools on

    candidates = demo_candidates(n_decoys=4)
    pred = scorer(candidates, DEMO_HPO)
    ranked = pred.ranked

    assert ranked
    # MVA-gene variants got the mechanism-tool-driven high EPCR.
    high = {f"{v.chrom}:{v.pos}" for v in ranked if v.epcr > 0.5}
    assert any(f"{c.chrom}:{c.pos}" in high for c in candidates if c.gene == "BUB1B")


def test_rlm_tools_can_be_disabled():
    """tools=[] disables domain tools (RLM still runs, pure sandbox heuristics)."""
    code = """
out = []
for v in variants:
    out.append({
        "chrom": v["chrom"],
        "pos": v["pos"],
        "ref": v["ref"],
        "alt": v["alt"],
        "epcr": 0.5,
        "rationale": "no tools",
    })
SUBMIT(ranked=out)
"""
    answer = {"reasoning": "no tools", "code": code}
    dspy.configure(lm=dspy.utils.DummyLM([answer] * 200, adapter=dspy.JSONAdapter()))
    scorer = RLMVariantScorer(max_llm_calls=5, tools=[])
    pred = scorer(demo_candidates(n_decoys=2), DEMO_HPO)
    assert pred.ranked and all(v.epcr == 0.5 for v in pred.ranked)

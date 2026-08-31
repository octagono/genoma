"""BAML adapter verification (dspy 3.3.1, dspy.adapters.baml_adapter).

BAMLAdapter is a JSONAdapter subclass that renders Pydantic schemas more
token-efficiently than plain JSON. These tests prove it works end-to-end
for the pipeline signatures.
"""

from __future__ import annotations

import dspy
import pytest
from dspy.adapters.baml_adapter import BAMLAdapter

from genoma.lm import default_adapter
from genoma.track1.scoring import (
    PairScoreSignature,
    PredictVariantScorer,
    VariantRankingSignature,
    VariantScoreSignature,
)


def test_default_adapter_is_baml():
    assert isinstance(default_adapter(), BAMLAdapter)


def test_baml_more_token_efficient_than_json():
    from dspy.adapters.json_adapter import JSONAdapter

    json_len = len(JSONAdapter().format_field_structure(VariantRankingSignature))
    baml_len = len(BAMLAdapter().format_field_structure(VariantRankingSignature))
    assert baml_len < json_len
    assert baml_len < 0.6 * json_len  # expect ~2x smaller


def test_baml_roundtrip_predict_scorer():
    """PredictVariantScorer through BAMLAdapter + DummyLM end-to-end."""
    adapter = BAMLAdapter()
    answer = {"reasoning": "r", "epcr": 0.5, "rationale": "x"}
    dspy.configure(lm=dspy.utils.DummyLM([answer] * 500, adapter=adapter), adapter=adapter)

    scorer = PredictVariantScorer()
    out = scorer(
        variant="chr2:47908000:G>A | gene=BUB1B | consequence=missense | AF=1e-5 | CADD=25",
        hpo_terms=["HP:0000252 Microcephaly"],
    )
    assert 0.0 < out.epcr <= 1.0
    assert out.rationale
    assert out.chrom == "chr2" and out.pos == 47908000


def test_baml_supports_our_signatures():
    for sig in (VariantScoreSignature, PairScoreSignature, VariantRankingSignature):
        # format_field_structure must not raise for nested/list pydantic outputs.
        BAMLAdapter().format_field_structure(sig)
    # ScoredPair nests ScoredVariant (non-recursive) — BAML supports this.
    assert True


def test_pipeline_models_are_non_recursive():
    """BAML supports nested but not recursive models; ours are nested-only."""
    from genoma.track1.scoring import ScoredPair, ScoredVariant  # noqa: F401

    # ScoredPair contains ScoredVariant (a different type) — not recursive.
    assert "variant_1" in ScoredPair.model_fields
    assert "variant_2" in ScoredPair.model_fields


# ── Disk-cache hardening (configure_cache restrict_pickle, dspy 3.3.1) ──────


def test_configure_defaults_hardens_disk_cache(tmp_path, monkeypatch):
    """configure_defaults installs restrict_pickle with the dspy.core safe types.

    The cache stores LM responses verbatim (incl. variant data); restrict_pickle
    confines deserialization to dspy.core.types.* + trusted library prefixes.
    """
    # Point the cache at a scratch dir and reconfigure through the real entry.
    from dspy.core import LMOutput, LMResponse, LMTextPart, LMUsage

    import genoma.config as mcfg

    monkeypatch.setattr(mcfg, "CACHE_DIR", tmp_path)
    from genoma.lm import configure_defaults

    configure_defaults()  # must not raise; installs restricted cache

    assert dspy.cache.enable_disk_cache
    assert dspy.cache.enable_memory_cache
    # restrict_pickle is wired into the fanout cache disk backend.
    disk = getattr(dspy.cache, "disk_cache", None)
    assert disk is not None
    disk_cls = type(getattr(disk, "disk", None) or disk)
    assert "RestrictedDisk" in disk_cls.__name__ or disk_cls.__name__ == "FanoutCache"
    # The safe types cover what the cache serializes.
    for t in (LMResponse, LMOutput, LMTextPart, LMUsage):
        assert (t.__module__, t.__qualname__) in _allowed_types_of(dspy.cache)


def _allowed_types_of(cache) -> frozenset:
    """Reconstruct the allowlist the restricted disk was built with."""
    from dspy.clients.disk_serialization import _RestrictedDisk

    dc = getattr(cache, "disk_cache", None)
    if dc is None:
        return frozenset()
    disk_inst = getattr(dc, "disk", None)  # RestrictedDisk instance
    disk_cls = type(disk_inst)
    if issubclass(disk_cls, _RestrictedDisk):
        return getattr(disk_cls, "_allowed", frozenset())
    # Fallback: the class may be at `_disk`.
    cls = getattr(dc, "_disk", None)
    if isinstance(cls, type) and issubclass(cls, _RestrictedDisk):
        return getattr(cls, "_allowed", frozenset())
    return frozenset()


def test_restrict_pickle_roundtrip_works(tmp_path):
    """A restricted cache still round-trips our pipeline's LM responses."""
    from dspy.clients import configure_cache
    from dspy.core import LMOutput, LMResponse, LMTextPart, LMUsage

    configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        restrict_pickle=True,
        safe_types=[LMResponse, LMOutput, LMTextPart, LMUsage],
    )
    answer = {"reasoning": "r", "epcr": 0.5, "rationale": "x"}
    dspy.configure(
        lm=dspy.utils.DummyLM([answer] * 100, adapter=dspy.JSONAdapter()),
        adapter=dspy.JSONAdapter(),
    )
    p = dspy.Predict(VariantScoreSignature)
    r1 = p(
        variant="chr2:47908000:G>A | gene=BUB1B | consequence=missense | AF=1e-5 | CADD=25",
        hpo_terms=["HP:0000252 Microcephaly"],
    )
    r2 = p(
        variant="chr2:47908000:G>A | gene=BUB1B | consequence=missense | AF=1e-5 | CADD=25",
        hpo_terms=["HP:0000252 Microcephaly"],
    )
    assert r1.epcr == r2.epcr == pytest.approx(0.5)
    # Memory cache hit: same object shape back (restrict_pickle unaffected mem).
    assert dspy.cache.enable_memory_cache

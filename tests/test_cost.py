"""Tests for the cost dry-run estimator (no API calls made)."""

from __future__ import annotations

from genoma import config as cfg
from genoma.cost import (
    plan_evaluate,
    plan_full_runs,
    plan_optimize,
    plan_panel,
    plan_score,
    plan_track2,
)


def test_plan_score_counts_calls():
    lines = plan_score(n_candidates=100, n_pairs=5)
    # 1 Predict per candidate + 1 CoT per pair.
    assert sum(line["calls"] for line in lines) == 105
    assert all(line["low"] <= line["high"] for line in lines)
    assert all(line["low"] > 0 for line in lines)


def test_plan_score_with_rlm():
    lines = plan_score(n_candidates=100, n_pairs=5, rlm=True)
    assert len(lines) == 3  # candidates + pairs + RLM
    assert any(line["calls"] == 40 for line in lines)  # RLM max_llm_calls budget


def test_plan_panel_members():
    """panel cost uses the availability-resolved panel, not raw cfg.PANEL."""
    from genoma.track1.consensus import _resolve_panel

    lines = plan_panel(n_candidates=50, n_pairs=0, debate=False)
    resolved = _resolve_panel(cfg.PANEL)
    # 2 lines per resolved member (candidates + pairs).
    assert len(lines) == len(resolved) * 2


def test_plan_optimize_respects_auto():
    light = plan_optimize("light")
    heavy = plan_optimize("heavy")
    assert sum(line["calls"] for line in heavy) > sum(line["calls"] for line in light)


def test_plan_track2():
    lines = plan_track2(n_iter=10)
    assert sum(line["calls"] for line in lines) == 15  # ~1.5 calls/iteration


def test_plan_full_runs_matches_components():
    full = plan_full_runs(n_score_runs=1, n_panel_runs=0)
    # Full run = 1× score + optimize + track2 (no panel at 0 runs).
    expected = sum(line["low"] for line in plan_score())
    expected += sum(line["low"] for line in plan_optimize())
    expected += sum(line["low"] for line in plan_track2())
    assert sum(line["low"] for line in full) == expected


def test_evaluate_plan_small():
    lines = plan_evaluate(n_examples=3)
    assert sum(line["calls"] for line in lines) == 3


def test_prices_are_current():
    """The configured models must have a price entry (or env override)."""
    for model in [cfg.PRIMARY_LM, cfg.SUB_LM, cfg.REFLECTION_LM, *cfg.PANEL]:
        assert model in (__import__("genoma.cost", fromlist=["MODEL_COST"]).MODEL_COST), (
            f"{model} missing from MODEL_COST"
        )

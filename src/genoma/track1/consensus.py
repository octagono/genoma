"""Track 1 — heterogeneous multi-model consensus panel.

- No ``dspy.Ensemble`` module exists in 3.3.1; ``dspy.majority`` is wrong
  for float EPCR. Use a custom panel.
- One ``PredictVariantScorer`` instance per panel model + ``set_lm``.
- Aggregate with a robust statistic (median / trimmed mean / weighted),
  NOT majority voting.
- High-spread (disagreement) items trigger ONE debate round under the
  primary model that folds all rationales + votes into a final verdict.

Key availability:
- OpenRouter members are skipped when ``OPENROUTER_API_KEY`` is unset.
- If the resulting panel is empty (no keys at all), fall back to the
  primary Anthropic model — a single-model panel still works.
"""

from __future__ import annotations

import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import dspy

from genoma import config as cfg
from genoma.data.annotate import KNOWN_MVA_GENES, VariantCandidate
from genoma.lm import panel_lm, primary_lm
from genoma.track1.scoring import PredictVariantScorer, ScoredVariant


class DebateSignature(dspy.Signature):
    """You are the chair of a panel of independent AI reviewers. Each reviewer
    independently scored a candidate variant for causality in a rare-disease
    proband. Their rationales disagree. Produce one calibrated final score,
    weighing evidence quality over confidence.

    Ground your verdict in the structured evidence vector (deterministic
    pre-filter scores) rather than trusting any single reviewer's rationale —
    the scores are computed, the rationales are opinions."""

    variant: str = dspy.InputField(desc="the candidate variant and its annotations")
    evidence: str = dspy.InputField(
        desc="structured deterministic evidence: AF/CADD/REVEL/AlphaMissense/SpliceAI/gene prior"
    )
    hpo_terms: list[str] = dspy.InputField(desc="proband phenotype terms")
    votes: list[float] = dspy.InputField(desc="each reviewer's EPCR (0,1]")
    rationales: list[str] = dspy.InputField(desc="each reviewer's reasoning")

    final_epcr: float = dspy.OutputField(ge=0.0, le=1.0)
    verdict: str = dspy.OutputField(desc="which evidence won and why")


@dataclass
class ConsensusResult:
    variant: ScoredVariant
    votes: dict[str, float]  # model slug -> epcr
    median_epcr: float
    spread: float
    debated: bool
    abstained: bool = False  # post-debate abstention: panel failed to converge
    verdict: str = ""


def _model_available(slug: str) -> bool:
    """True if the model's provider API key is present (or it's a dummy)."""
    if "openrouter/" in slug:
        return bool(os.environ.get("OPENROUTER_API_KEY"))
    if "anthropic/" in slug:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return True  # assume other providers have their own key handling


def _resolve_panel(panel: dict[str, float] | None) -> dict[str, float]:
    """Drop members whose provider key is missing; fall back to primary if empty."""
    panel = dict(panel or cfg.PANEL)
    available = {s: w for s, w in panel.items() if _model_available(s)}
    if not available:
        available = {cfg.PRIMARY_LM: 1.0}  # single-model fallback
    return available


def _resolve_debate_lm(panel: dict[str, float]) -> dspy.LM:
    """Choose the debate (chair) model: primary if usable, else strongest member.

    The debate folds all panel rationales into a final verdict, so it should
    use the strongest *available* model — not necessarily Claude if the panel
    is all-OpenRouter and no Anthropic key is set.
    """
    if _model_available(cfg.PRIMARY_LM):
        return primary_lm()
    for slug in panel:
        if _model_available(slug):
            return panel_lm(slug)
    return primary_lm()  # last resort; will surface a clear auth error


def _spread_of(values, center: float) -> float:
    """Max distance from center across the panel votes."""
    return max((abs(float(v) - center) for v in values), default=0.0)


def _shrink_to_center(epcr: float, spread: float) -> float:
    """Pull EPCR toward 0.5 proportionally to disagreement (spread-as-confidence).

    High spread = low confidence -> shrink toward 0.5 (neutral). This improves
    F-max by keeping uncertain calls off the extreme, well-separated region.
    """
    shrink = min(1.0, cfg.SPREAD_SHRINK * (spread / max(cfg.PANEL_DEBATE_THRESHOLD, 1e-9)))
    return 0.5 + (epcr - 0.5) * (1.0 - shrink)


def _evidence_vector(candidate: VariantCandidate) -> str:
    """The deterministic pre-filter scores as an explicit evidence vector the
    debate grounds on, so the chair reasons over computed facts, not just
    reviewer opinions."""

    def _fmt(v) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return "NA" if v is None else str(v)

    return (
        f"AF={_fmt(candidate.af)} CADD={_fmt(candidate.cadd)} REVEL={_fmt(candidate.revel)} "
        f"AlphaMissense={_fmt(candidate.alphamissense)} SpliceAI={_fmt(candidate.spliceai)} "
        f"gene_prior={'KNOWN_MVA' if candidate.gene in KNOWN_MVA_GENES else 'none'}"
    )


class PanelConsensus(dspy.Module):
    """Heterogeneous multi-model EPCR consensus with debate escalation.

    ``lm_factory``: callable(slug) -> dspy.LM. Defaults to ``panel_lm``
    (real providers). Pass a DummyLM factory for offline testing.
    """

    def __init__(
        self,
        panel: dict[str, float] | None = None,
        lm_factory: Callable[[str], dspy.LM] | None = None,
    ) -> None:
        super().__init__()
        lm_factory = lm_factory or panel_lm
        self.panel = _resolve_panel(panel)
        self.members: dict[str, PredictVariantScorer] = {}
        for slug in self.panel:
            scorer = PredictVariantScorer()
            scorer.set_lm(lm_factory(slug))
            self.members[slug] = scorer
        self.debate = dspy.ChainOfThought(DebateSignature)

    def _aggregate(self, eps: list[float], weights: list[float] | None = None) -> float:
        """Robust weighted aggregate of panel votes.

        Uses the declared per-model weights (median of the weighted
        distribution when few members; trimmed weighted mean for 5+).
        Weights default to equal (1.0 each) so a single-model panel is a
        plain median.
        """
        if weights is None:
            weights = [1.0] * len(eps)
        # Replicate each vote by its integer weight (a 2.0x-weighted model
        # counts twice) then aggregate robustly.
        expanded = [e for e, w in zip(eps, weights, strict=True) for _ in range(int(round(w)))]
        if len(expanded) >= 5:
            trimmed = sorted(expanded)[1:-1]  # trimmed mean resists outliers
            return statistics.mean(trimmed)
        return statistics.median(expanded)

    def forward(self, candidate: VariantCandidate, hpo_terms: list[str]) -> ConsensusResult:
        votes: dict[str, float] = {}
        rationales: dict[str, str] = {}
        weights: list[float] = []
        for slug, member in self.members.items():
            pred = member(candidate, hpo_terms)
            votes[slug] = float(pred.epcr)
            rationales[slug] = str(pred.rationale)
            weights.append(self.panel.get(slug, 1.0))

        eps = list(votes.values())
        median = self._aggregate(eps, weights)
        spread = max(eps) - min(eps)
        debated = spread > cfg.PANEL_DEBATE_THRESHOLD
        abstained = False
        verdict = ""
        if debated:
            with dspy.context(lm=_resolve_debate_lm(self.panel)):
                out = self.debate(
                    variant=candidate.to_evidence_str(),
                    evidence=_evidence_vector(candidate),
                    hpo_terms=hpo_terms,
                    votes=eps,
                    rationales=[rationales[s] for s in self.members],
                )
            median, verdict = float(out.final_epcr), str(out.verdict)
            spread = max(spread, _spread_of(votes.values(), median))
            # Post-debate abstention: if the debate could not converge the
            # panel, abstain — flag for review instead of forcing a score.
            abstained = spread > cfg.PANEL_ABSTAIN_THRESHOLD

        # Spread-as-confidence: high disagreement -> pull EPCR toward 0.5.
        median = _shrink_to_center(median, spread)

        return ConsensusResult(
            variant=ScoredVariant(
                chrom=candidate.chrom,
                pos=candidate.pos,
                ref=candidate.ref,
                alt=candidate.alt,
                epcr=median,
                rationale=verdict or "; ".join(rationales.values()),
                debated=debated,
            ),
            votes=votes,
            median_epcr=median,
            spread=spread,
            debated=debated,
            abstained=abstained,
            verdict=verdict,
        )

    def score_candidates(
        self,
        candidates: list[VariantCandidate],
        hpo_terms: list[str],
        *,
        checkpoint_path: str | Path = "",
        resilient: bool = False,
    ) -> list[ScoredVariant]:
        """Score and rank a candidate table through the panel, in parallel.

        Fail-fast by default (a broken candidate aborts loudly). With
        ``resilient=True`` (production mode), per-candidate failures degrade
        to a loud warning + conservative default and the run continues;
        ``checkpoint_path`` enables JSONL resume across interrupted runs.
        """
        n = len(candidates)
        if n == 0:
            return []

        from genoma.checkpoint import score_chunked

        def _score_one(candidate, hpo_terms):
            res = self.forward(candidate, hpo_terms)
            return {
                "chrom": res.variant.chrom,
                "pos": res.variant.pos,
                "ref": res.variant.ref,
                "alt": res.variant.alt,
                "epcr": float(res.variant.epcr),
                "rationale": str(res.variant.rationale),
                "debated": res.variant.debated,
                "abstained": res.abstained,
            }

        def _default_record(candidate, err: str) -> dict:
            return {
                "chrom": candidate.chrom,
                "pos": candidate.pos,
                "ref": candidate.ref,
                "alt": candidate.alt,
                "epcr": 0.0,
                "rationale": f"SAFE-DEFAULT: {err}",
                "debated": False,
                "abstained": True,
            }

        records = score_chunked(
            candidates,
            hpo_terms,
            _score_one,
            _default_record,
            checkpoint_path=checkpoint_path or None,
            num_threads=dspy.settings.num_threads or 8,
            resilient=resilient,
        )

        scored: list[ScoredVariant] = []
        for _c, res in records:
            # res is always a dict (fresh score, resilient default, or resumed).
            scored.append(
                ScoredVariant(
                    chrom=res["chrom"],
                    pos=res["pos"],
                    ref=res["ref"],
                    alt=res["alt"],
                    epcr=float(res["epcr"]),
                    rationale=str(res["rationale"]),
                    debated=bool(res.get("debated", False)),
                )
            )
        scored.sort(key=lambda v: v.epcr, reverse=True)
        return scored

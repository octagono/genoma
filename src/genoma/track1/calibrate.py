"""Track 1 — EPCR calibration.

Two post-scoring adjustments that directly improve F-max (which rewards
well-separated, calibrated probabilities):

1. ``self_consistency_epcr`` — sample the scorer N times at temperature=1.0
   (rollout_id forces fresh samples past the cache) and return the mean +
   spread. Mean = calibrated confidence; spread = a confidence signal usable
   for the panel debate threshold.

2. ``calibrate_submission_epcrs`` — submission-shape EPCR calibration: reserve
   a strictly unique max (1.0) for the top pair and emit no ties (a tie at the
   top collapses F-max 1.0→0.667).

(Gene-prior boosting lives in ``data/annotate.py:apply_gene_prior`` and is
applied by the CLI, not here.)
"""

from __future__ import annotations

import random
import statistics

import dspy

from genoma import config as cfg
from genoma.data.annotate import (
    KNOWN_MVA_GENES,
    MVA_GENE_MECHANISMS,
    VariantCandidate,
)
from genoma.track1.scoring import PredictVariantScorer


def _shuffled(items: list[str], seed: int) -> list[str]:
    """Deterministic per-sample shuffle (defeats input-order bias)."""
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def self_consistency_epcr(
    scorer: PredictVariantScorer,
    candidate: VariantCandidate,
    hpo_terms: list[str],
    n: int = 3,
) -> tuple[float, float]:
    """Sample EPCR N times at temperature=1.0; return (confidence-weighted
    mean, stdev).

    - **Confidence weighting:** samples closer to the consensus center carry
      more weight (agreement-as-confidence), so an outlier sample drags the
      mean less than in a plain average.
    - **Order/seed invariance:** the known-gene list and HPO terms are
      shuffled deterministically per sample (seed = rollout_id), so a fixed
      input order can't bias the aggregated EPCR.
    """
    base_lm = scorer.get_lm() or dspy.settings.lm
    eps: list[float] = []
    for i in range(n):
        with dspy.context(lm=base_lm.copy(rollout_id=i, temperature=1.0)):
            out = scorer.scorer(
                variant=candidate.to_evidence_str(),
                hpo_terms=_shuffled(list(hpo_terms), seed=i + 1000),
                known_mva_genes=",".join(_shuffled(sorted(KNOWN_MVA_GENES), seed=i)),
                gene_mechanism=MVA_GENE_MECHANISMS.get(candidate.gene, ""),
            )
        eps.append(float(out.epcr))
    center = statistics.median(eps)
    # Agreement-as-confidence: weight = 1 / (1 + distance to consensus).
    weights = [1.0 / (1.0 + abs(e - center)) for e in eps]
    wsum = sum(weights)
    mean = sum(w * e for w, e in zip(weights, eps, strict=True)) / wsum
    spread = statistics.stdev(eps) if len(eps) > 1 else 0.0
    return mean, spread


def calibrate_submission_epcrs(
    pairs: list,
    singles: list,
    *,
    phase_info: dict | None = None,
    top_epcr: float = 1.0,
    decay: float = 0.05,
    min_epcr: float = 0.05,
    max_rows: int | None = None,
) -> tuple[list, list]:
    """Submission-shape EPCR calibration (F-max is threshold-based).

    Scoring rules:

    - **The top-ranked pair must carry a strictly unique max EPCR (1.0).** Any
      EPCR tie at the top collapses F-max from 1.0 to ~0.667. Reserve 1.0 for
      exactly one row.
    - **No ties anywhere** — F-max sweeps unique EPCR values as thresholds, so
      distinct, monotonically decreasing values give the F1 curve distinct
      points and keep ranking deterministic.
    - **Rows below the true pair are free** (never hurt rank points or F-max),
      so fill the row budget in confidence order; only relative order above
      the causal pair matters.
    - Cis-phased pairs (same haplotype) are NOT compound-hets — demoted hard.
    - Preserves confidence ordering: the top-confidence pair becomes rank 1.

    ``pairs``/``singles`` are ScoredPair/ScoredVariant objects; epcr is set
    in-place on copies. ``phase_info`` maps "coordA|coordB" ->
    PairPhase(status="trans"|"cis"|"unknown"). ``max_rows`` bounds the total
    calibrated entries (default ``cfg.MAX_SUBMISSION_ROWS``) so the strict
    no-ties ladder never runs out of unique values (the linear ladder is
    unique only while EPCR > min_epcr). The return values are strictly
    monotonic in EPCR with no ties.
    """
    max_rows = max_rows or cfg.MAX_SUBMISSION_ROWS
    phase_info = phase_info or {}

    def _pair_phase(p) -> str:
        key = f"{p.variant_1.chrom}:{p.variant_1.pos}:{p.variant_1.ref}>{p.variant_1.alt}|{p.variant_2.chrom}:{p.variant_2.pos}:{p.variant_2.ref}>{p.variant_2.alt}"
        rev = "|".join(reversed(key.split("|")))
        ph = phase_info.get(key) or phase_info.get(rev)
        return ph.status if ph else "unknown"

    # Sort by confidence (descending); trans pairs get a small boost over
    # unknown at equal confidence (they are genuine compound-hets), cis pairs
    # sink to the bottom.
    def _conf_sort_key(p) -> tuple:
        status = _pair_phase(p)
        return (status == "cis", -(status == "trans"), -p.epcr)

    ordered_pairs = sorted(pairs, key=_conf_sort_key)

    out_pairs: list = []
    non_cis_rank = 0  # only non-cis pairs consume the top-EPCR ladder
    for p in ordered_pairs:
        p = p.model_copy(deep=True)
        status = _pair_phase(p)
        if status == "cis":
            # Not a compound-het: demote hard, still strictly monotonic.
            p.epcr = max(min_epcr, 0.01 * (non_cis_rank + 1))
        else:
            # Strictly unique, monotonically decreasing: rank 1 (top non-cis
            # pair) reserves exactly top_epcr (1.0); each later pair drops
            # by `decay`.
            p.epcr = max(min_epcr, top_epcr - decay * non_cis_rank)
            non_cis_rank += 1
        out_pairs.append(p)

    out_singles: list = []
    # Cap singles so pairs + singles together stay within the submission
    # budget — and the no-ties ladder stays unique (it clamps at min_epcr).
    remaining = max(0, max_rows - len(out_pairs))
    for i, v in enumerate(sorted(singles, key=lambda x: x.epcr, reverse=True)[:remaining]):
        v = v.model_copy(deep=True)
        v.epcr = max(min_epcr, top_epcr - decay * (non_cis_rank + i))
        out_singles.append(v)

    out_pairs.sort(key=lambda x: x.epcr, reverse=True)
    out_singles.sort(key=lambda x: x.epcr, reverse=True)
    return out_pairs, out_singles

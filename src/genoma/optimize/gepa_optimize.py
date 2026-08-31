"""Optimization: GEPA with an LLM-as-judge metric.

GEPA's metric contract is ``(gold, pred, trace, pred_name, pred_trace) ->
Prediction(score, feedback)`` and the *feedback string is threaded into
reflection*.

Strategy:
1. Build a small synthetic trainset from known MVA-gene examples.
2. The judge metric = numeric correctness (rank-order + calibration vs the
   synthetic answer) PLUS LLM-written feedback about what went wrong.
3. GEPA(metric=judge, reflection_lm=...) compiles the scorer.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import dspy

from genoma import config as cfg
from genoma.lm import reflection_lm

# ── Synthetic trainset ──────────────────────────────────────────────────────


def build_synthetic_trainset(n: int = 12, seed: int = 0) -> list[dspy.Example]:
    """Synthetic labeled examples for prompt optimization.

    Each example is a candidate variant + HPO terms with a known ``true_epcr``
    (synthetic ground truth). Labels are derived from *observable variant
    features* so an optimizer can actually learn a rule from the data:

      high EPCR  <=  gene in KNOWN_MVA_GENES AND AF < 0.005 AND coding/splice
      low EPCR   <=  otherwise

    This gives GEPA a learnable signal (as opposed to random labels).
    """
    import random

    from genoma.data.annotate import KNOWN_MVA_GENES
    from genoma.data.demo import DEMO_HPO

    rng = random.Random(seed)
    mva_genes = sorted(KNOWN_MVA_GENES)
    benign_genes = ["TTN", "RYR2", "MUC16", "KCNQ1", "PCSK9"]
    consequences = ["missense_variant", "splice_donor_variant", "intron_variant"]
    hpo = DEMO_HPO

    def make_variant(i: int) -> tuple[str, str, float, str, float]:
        """Return (variant_str, gene, af, consequence, true_epcr)."""
        # Mix of causal-like and benign-like examples with noisy AF/CADD.
        gene = rng.choice(mva_genes) if i % 2 == 0 else rng.choice(benign_genes)
        consequence = rng.choice(consequences)
        af = rng.choice([1e-5, 2e-4, 8e-4]) if gene in mva_genes else rng.choice([0.02, 0.05, 0.12])
        cadd = rng.uniform(22, 32) if gene in mva_genes else rng.uniform(1, 8)
        high = gene in mva_genes and af < 0.005 and consequence != "intron_variant"
        variant = (
            f"chr{rng.randint(1, 22)}:{50_000_000 + i * 123_456}:{rng.choice(['G', 'A', 'C', 'T'])}>"
            f"{rng.choice(['T', 'C', 'G', 'A'])}"
        )
        # Real-world evidence is often incomplete (no gnomAD AF for rare
        # variants, VUS with unknown consequence, missing CADD). Degrade the
        # *evidence string* only — the label stays derived from the true
        # features, so the optimizer learns to handle missing fields.
        af_str = "unknown" if rng.random() < 0.25 else f"{af:.6g}"
        cadd_str = "unknown" if rng.random() < 0.25 else f"{cadd:.1f}"
        cons_str = "unknown" if rng.random() < 0.15 else consequence
        evidence = (
            f"{variant} | gene={gene} | consequence={cons_str} | AF={af_str} | CADD={cadd_str}"
        )
        return evidence, gene, af, consequence, (0.85 if high else 0.10)

    examples: list[dspy.Example] = []
    for i in range(n):
        evidence, _gene, _af, _consequence, true_epcr = make_variant(i)
        examples.append(
            dspy.Example(
                variant=evidence,
                hpo_terms=hpo,
                known_mva_genes=",".join(mva_genes),
                true_epcr=true_epcr,
            ).with_inputs("variant", "hpo_terms", "known_mva_genes")
        )
    return examples


# ── Judge metric (GEPA 5-arg contract) ──────────────────────────────────────


def make_gepa_judge(
    judge_lm: dspy.LM | None = None,
    *,
    calibration_weight: float = 0.3,
    classification_weight: float = 0.7,
    feedback_on_perfect: bool = False,
) -> Callable[..., dspy.Prediction]:
    """Return a GEPA-compatible judge metric (composite, tunable).

    Contract: the metric must accept 5 positional args ``(gold, pred, trace,
    pred_name, pred_trace)`` — GEPA binds the signature at construction and
    raises TypeError otherwise — and return ``dspy.Prediction(score=...,
    feedback=...)``. The feedback string is threaded verbatim into GEPA's
    reflection prompt.

    Score = calibration_weight * calibration_score
          + classification_weight * classification_score

    - calibration_score: 1 - |pred_epcr - true_epcr| (how close the number is)
    - classification_score: 1 if (pred > 0.5) == (true > 0.5) else 0
      (whether the binary causal call is right)

    Defaults weight classification over calibration (0.7 / 0.3): the binary
    causal call drives the rank of the true pair, while F-max is
    threshold-robust to absolute EPCR scale — so getting the causal call
    right matters more than calibrating the exact number.

    LLM feedback is only generated when the score is imperfect (or
    ``feedback_on_perfect=True``), so optimization doesn't burn an LLM call
    on every example.
    """

    def metric(
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace=None,
        pred_name=None,
        pred_trace=None,
        program_trace=None,
    ):
        # Fail fast: a candidate without a numeric epcr is a broken prediction.
        pred_epcr = float(pred.epcr)
        true_epcr = float(gold.true_epcr)

        calibration = max(0.0, 1.0 - abs(pred_epcr - true_epcr))
        classification = 1.0 if (pred_epcr > 0.5) == (true_epcr > 0.5) else 0.0
        score = calibration_weight * calibration + classification_weight * classification

        feedback = ""
        if score < 1.0 or feedback_on_perfect:
            feedback_lm = judge_lm or dspy.settings.lm
            judge = dspy.ChainOfThought(JudgeFeedbackSignature)
            with dspy.context(lm=feedback_lm):
                fb = judge(
                    variant=gold.variant,
                    hpo_terms=gold.hpo_terms,
                    pred_epcr=pred_epcr,
                    true_epcr=true_epcr,
                    rationale=getattr(pred, "rationale", ""),
                )
            feedback = fb.feedback
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


class JudgeFeedbackSignature(dspy.Signature):
    """You are an expert clinical-genomics reviewer. Given a candidate
    variant, the proband phenotype, a predicted EPCR and rationale, and the
    true EPCR, write 1-2 sentences of constructive feedback that would help
    the predictor improve (over-confidence? missing gene knowledge? ignoring
    allele frequency?).

    Scoring rubric (apply it when judging the prediction):
    - Pathogenicity evidence: rare AF (gnomAD < 0.5%), coding/splice
      consequence, high CADD/REVEL/AlphaMissense, known MVA-gene membership.
    - An EPCR near the true value on the right side of the 0.5 causal
      boundary is better than a well-calibrated number on the wrong side —
      the submission is scored on the binary causal call first, then F-max.
    - Flag over-confidence (EPCR close to 1.0 on thin evidence) and
      under-confidence (EPCR near 0.5 despite strong MVA-gene + rare-AF
      signal) explicitly.
    - Do NOT reward verbosity or prompt length — only reasoning quality."""

    variant: str = dspy.InputField()
    hpo_terms: list[str] = dspy.InputField()
    pred_epcr: float = dspy.InputField()
    true_epcr: float = dspy.InputField()
    rationale: str = dspy.InputField()

    feedback: str = dspy.OutputField()


def _judge_from_weight(judge_classification_weight: float | None):
    """Judge metric from a single classification weight (calibration = 1-w).

    None -> the default 0.7/0.3 judge. Exposed as one knob so operators can
    shift toward robust reasoning (higher classification) when noisy-data
    optimization degrades exact EPCR calibration.

    Cross-family judging: if cfg.GEPA_JUDGE_LM is set (a model from a
    DIFFERENT family than the proposer), it is used as the judge LM — a
    same-family judge overrates its own outputs and inflates the optimization
    signal. Falls back to settings.lm otherwise.
    """
    judge_lm = None
    if cfg.GEPA_JUDGE_LM:
        from genoma.lm import make_lm

        judge_lm = make_lm(cfg.GEPA_JUDGE_LM)
    if judge_classification_weight is None:
        return make_gepa_judge(judge_lm=judge_lm)
    if not 0.0 <= judge_classification_weight <= 1.0:
        raise ValueError(
            f"judge_classification_weight must be in [0, 1], got {judge_classification_weight}"
        )
    return make_gepa_judge(
        judge_lm=judge_lm,
        classification_weight=judge_classification_weight,
        calibration_weight=1.0 - judge_classification_weight,
    )


def evaluate_with_judge(
    program: dspy.Module,
    devset: list[dspy.Example],
    metric=None,
    num_threads: int = 4,
    *,
    judge_classification_weight: float | None = None,
) -> float:
    """Score a program on a devset with the judge metric (dspy.Evaluate).

    Returns the mean judge score in 0–1 (``dspy.Evaluate`` reports 0–100
    percent; we divide back to match the judge/GEPA 0–1 contract).
    """
    if metric is None:
        metric = _judge_from_weight(judge_classification_weight)
    evaluator = dspy.Evaluate(devset=devset, metric=metric, num_threads=num_threads)
    result = evaluator(program)
    return float(result.score) / 100.0


# ── GEPA wrapper ────────────────────────────────────────────────────────────


def optimize_with_gepa(
    program: dspy.Module,
    trainset: list[dspy.Example],
    valset: list[dspy.Example] | None = None,
    *,
    metric=None,
    judge_classification_weight: float | None = None,
    auto: str | None = None,
    max_metric_calls: int | None = None,
    track_stats: bool | None = None,
    val_fraction: float = 0.25,
    out_path: Path | str | None = None,
) -> dspy.Module:
    """Compile a program with GEPA. Returns the optimized module.

    Exactly one of ``auto`` / ``max_metric_calls`` may be set (GEPA asserts
    this). If neither is set, ``cfg.GEPA_AUTO`` is used; if that is ``None``
    then ``cfg.GEPA_MAX_METRIC_CALLS`` applies. Setting both raises a clear
    ValueError instead of tripping GEPA's internal assert.

    ``valset`` must be disjoint from ``trainset`` (GEPA's Pareto/validation
    score is otherwise meaningless — it would just re-score the training
    examples). If not provided, a stratified hold-out of ``val_fraction`` is
    split off the END of the trainset (deterministic, seed-free) so the
    reported score reflects held-out generalization, not memorization.
    """
    if auto is not None and max_metric_calls is not None:
        raise ValueError("Set exactly one of auto / max_metric_calls, not both.")
    if auto is None and max_metric_calls is None:
        # Config fallback: use GEPA_AUTO if set; else GEPA_MAX_METRIC_CALLS.
        auto = cfg.GEPA_AUTO
        if auto is None:
            max_metric_calls = cfg.GEPA_MAX_METRIC_CALLS

    if metric is None:
        metric = _judge_from_weight(judge_classification_weight)

    if valset is None:
        n_train = len(trainset)
        n_val = max(1, int(n_train * val_fraction))
        n_train = n_train - n_val
        if n_train < 1:
            raise ValueError(
                f"trainset too small ({len(trainset)}) to hold out a validation set; "
                "pass an explicit `valset` instead."
            )
        trainset = trainset[:n_train]
        valset = trainset[n_train:]

    optim = dspy.GEPA(
        metric=metric,
        auto=auto,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm(),
        track_stats=cfg.GEPA_TRACK_STATS if track_stats is None else track_stats,
        seed=0,
    )
    optimized = optim.compile(program, trainset=trainset, valset=valset)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        optimized.save(str(out_path))
    return optimized

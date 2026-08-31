"""Track 1 — variant scoring.

Two complementary scorers:

1. ``PredictVariantScorer`` — a plain ``dspy.Predict`` over a structured
   signature (BAMLAdapter + pydantic ``epcr`` field). This is the workhorse
   used by the consensus panel (one instance per model + set_lm per model).

2. ``RLMVariantScorer`` — a ``dspy.RLM`` (Recursive Language Model): the LM
   writes Python in a sandbox, explores the candidate table programmatically,
   can call a cheap sub-LM (``llm_query``) per candidate, and emits a ranked
   list. ProgramOfThought/CodeAct are deprecated in dspy 3.4; RLM is their
   replacement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import dspy
from pydantic import BaseModel, Field

from genoma import config as cfg
from genoma.data.annotate import (
    KNOWN_MVA_GENES,
    MVA_GENE_MECHANISMS,
    VariantCandidate,
)

# ── Structured outputs (pydantic, BAMLAdapter-enforced) ────────────────────


class ScoredVariant(BaseModel):
    chrom: str
    pos: int
    ref: str
    alt: str
    epcr: float = Field(ge=0.0, le=1.0, description="estimated probability of causal relationship")
    rationale: str = Field(description="one-two sentence evidence-based reasoning")
    debated: bool = Field(
        default=False, description="whether panel disagreement triggered a debate round"
    )


class ScoredPair(BaseModel):
    """A compound-heterozygous candidate pair."""

    variant_1: ScoredVariant
    variant_2: ScoredVariant
    epcr: float = Field(ge=0.0, le=1.0)
    rationale: str


# ── Signatures ─────────────────────────────────────────────────────────────


class VariantScoreSignature(dspy.Signature):
    """Given one candidate variant, the proband's phenotype, and the known
    disease mechanism of the affected gene, estimate the probability that
    this variant is causal.

    The disease is Mosaic Variegated Aneuploidy (MVA) — an autosomal-recessive
    spindle-assembly-checkpoint / centrosome disorder. The causal mechanism is
    compound-heterozygous biallelic loss-of-function in one of the six known
    genes. Use the gene_mechanism field (retrieved from the disease knowledge
    base) to judge whether the variant's predicted effect is consistent with
    that mechanism. Consider: allele frequency (AR-causal alleles are near
    absent in population databases), conservation/effect scores (CADD,
    AlphaMissense, Evo2), and call quality."""

    variant: str = dspy.InputField(desc="chrom:pos:ref>alt | gene | consequence | AF | scores")
    hpo_terms: list[str] = dspy.InputField(desc="proband phenotype terms")
    known_mva_genes: str = dspy.InputField(
        desc="comma-separated genes with established MVA causality"
    )
    gene_mechanism: str = dspy.InputField(
        desc="known disease mechanism of the variant's gene (from the MVA "
        "knowledge base; empty if the gene is not a known MVA gene)"
    )

    epcr: Annotated[float, Field(ge=0.0, le=1.0)] = dspy.OutputField(
        desc="estimated probability of causal relationship (0,1]"
    )
    rationale: Annotated[str, Field(default="")] = dspy.OutputField(
        desc="evidence-based reasoning (report-only; the model may omit it)"
    )


class VariantRankingSignature(dspy.Signature):
    """Rank the given candidate variants by likelihood of causing the
    proband's phenotype. Return the full ranked list, best first.

    Use the evidence strings provided. Note that the causal mechanism in
    this family is expected to be compound heterozygous: two variants in
    the same gene, one on each chromosome. You may write Python to help
    yourself reason over the table."""

    variants: list[VariantCandidate] = dspy.InputField(desc="candidate variant evidence strings")
    hpo_terms: list[str] = dspy.InputField(desc="proband phenotype terms")

    ranked: list[ScoredVariant] = dspy.OutputField(
        desc="all candidates, ranked best-first with EPCR + rationale"
    )


class PairScoreSignature(dspy.Signature):
    """Score a compound-heterozygous candidate pair for causality.

    Compound-heterozygous means the two variants are in the same gene and
    are presumed to be on opposite homologous chromosomes, together
    abolishing or severely impairing gene function."""

    pair_description: str = dspy.InputField(desc="the two variants and their annotations")
    hpo_terms: list[str] = dspy.InputField(desc="proband phenotype terms")

    epcr: Annotated[float, Field(ge=0.0, le=1.0)] = dspy.OutputField()
    rationale: Annotated[str, Field(default="")] = dspy.OutputField()


# ── Scorers ────────────────────────────────────────────────────────────────


class PredictVariantScorer(dspy.Module):
    """Structured per-variant scorer (Predict + BAMLAdapter).

    ``forward`` accepts the signature fields as keyword args
    (``variant``, ``hpo_terms``, ``known_mva_genes``) — this is what
    GEPA/MIPROv2/Evaluate pass from trainset examples — OR a
    ``VariantCandidate`` object plus ``hpo_terms`` (used by the panel
    and CLI). Both paths produce the same structured output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scorer = dspy.ChainOfThought(VariantScoreSignature)

    def forward(
        self,
        candidate: VariantCandidate | None = None,
        hpo_terms: list[str] | None = None,
        variant: str | None = None,
        known_mva_genes: str | None = None,
        gene_mechanism: str | None = None,
    ) -> dspy.Prediction:
        if candidate is not None:
            variant = candidate.to_evidence_str()
            hpo_terms = hpo_terms or []
            known_mva_genes = ",".join(sorted(KNOWN_MVA_GENES))
            gene_mechanism = MVA_GENE_MECHANISMS.get(candidate.gene, "")
            chrom, pos, ref, alt = (
                candidate.chrom,
                candidate.pos,
                candidate.ref,
                candidate.alt,
            )
        else:
            if variant is None or hpo_terms is None:
                raise ValueError(
                    "PredictVariantScorer needs either a VariantCandidate or "
                    "(variant, hpo_terms[, known_mva_genes])"
                )
            # Recover coordinates from the evidence string for the output row.
            chrom, pos, ref, alt = _parse_coord(variant)
            # Infer the gene name from the evidence string for mechanism lookup.
            gene = _parse_gene(variant)
            if gene_mechanism is None:
                gene_mechanism = MVA_GENE_MECHANISMS.get(gene, "")

        out = self.scorer(
            variant=variant,
            hpo_terms=hpo_terms,
            known_mva_genes=known_mva_genes or ",".join(sorted(KNOWN_MVA_GENES)),
            gene_mechanism=gene_mechanism or "",
        )
        return dspy.Prediction(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
            epcr=out.epcr,
            rationale=out.rationale,
        )


def _parse_gene(evidence: str) -> str:
    """Extract the gene symbol from an evidence string ('... | gene=XXX | ...')."""
    for part in evidence.split(" | "):
        if part.startswith("gene="):
            return part[5:].strip()
    return ""


def _parse_coord(evidence: str) -> tuple[str, int, str, str]:
    """Extract chrom/pos/ref/alt from a 'chrN:pos:ref>alt | ...' evidence string."""
    coord_part = evidence.split(" | ")[0]  # drop annotation suffix
    chrom, pos, ref_alt = coord_part.split(":")
    ref, alt = ref_alt.split(">", 1)
    return chrom, int(pos), ref, alt


# ── RLM sandbox tools ───────────────────────────────────────────────────────
# RLM injects these as callables into the sandboxed interpreter. They give the
# sandboxed LM deterministic domain lookups instead of relying on the LM to
# know the MVA literature. Pure + host-side: they work offline and don't
# consume the `max_llm_calls` budget (only llm_query/llm_query_batched do).


def mechanism_lookup(gene: str) -> str:
    """Return the known MVA disease mechanism + clinical clue for a gene.

    Empty string if the gene is not one of the six established MVA genes.
    """
    return MVA_GENE_MECHANISMS.get(gene, "")


def known_mva_gene(gene: str) -> bool:
    """True if the gene is one of the six established MVA genes."""
    return gene in KNOWN_MVA_GENES


def gene_prior_fraction(gene: str) -> float:
    """Return the EPCR boost fraction applied to variants in known MVA genes."""
    return cfg.GENE_PRIOR_BOOST if gene in KNOWN_MVA_GENES else 0.0


def exomiser_rank(exomiser_genes: dict) -> Callable[[str], str]:
    """Build an Exomiser gene-rank lookup tool for the RLM sandbox.

    ``exomiser_genes`` is the ``{"genes": {...}}`` dict from
    ``parse_exomiser_output``. Returns a closure ``(gene) -> str`` giving the
    gene's Exomiser combined/phenotype scores + comp-het flag, or "" if the
    gene is absent. Pure + host-side: works offline, costs no LLM budget.
    """
    genes = exomiser_genes.get("genes", {})

    def _lookup(gene: str) -> str:
        g = genes.get(gene)
        if not g:
            return ""
        return (
            f"{gene}: exomiser={g.get('exomiser_score'):.3f} "
            f"pheno={g.get('phenotype_score'):.3f} "
            f"comp_het={g.get('comp_het', False)}"
        )

    return _lookup


RLM_DOMAIN_TOOLS = [mechanism_lookup, known_mva_gene, gene_prior_fraction]


class RLMVariantScorer(dspy.Module):
    """RLM-based scorer over the full candidate table.

    The LM writes Python inside the sandboxed interpreter to explore the
    candidate list, optionally queries a cheap sub-LM per candidate
    (llm_query / llm_query_batched), and SUBMITs a ranked list.

    ``tools`` are injected into the sandbox (default: the MVA domain tools —
    mechanism KB lookup, known-gene check, gene-prior fraction). Pass
    ``tools=[]`` to disable. ``sub_lm`` defaults to the configured cheap model
    (``cfg.SUB_LM``).
    """

    def __init__(
        self,
        *,
        sub_lm: dspy.LM | None = None,
        max_llm_calls: int = 40,
        tools: list | None = None,
    ) -> None:
        super().__init__()
        if sub_lm is None:
            from genoma.lm import sub_lm as _sub_lm

            sub_lm = _sub_lm()
        self.analyzer = dspy.RLM(
            VariantRankingSignature,
            max_iters=15,
            max_llm_calls=max_llm_calls,
            max_output_chars=12000,
            sub_lm=sub_lm,
            tools=RLM_DOMAIN_TOOLS if tools is None else tools,
        )

    def forward(self, candidates: list[VariantCandidate], hpo_terms: list[str]) -> dspy.Prediction:
        out = self.analyzer(variants=candidates, hpo_terms=hpo_terms)
        return dspy.Prediction(ranked=out.ranked, trajectory=getattr(out, "trajectory", None))


class CompoundHetPairScorer(dspy.Module):
    """Rank compound-heterozygous candidate pairs.

    Grouping logic (same gene, plausible biallelic combination) is a
    deterministic pre-pass; the LLM scores the surviving pairs. The causal
    mechanism is a compound-het pair, so recovering the *pair* (not just one
    variant) is what earns full rank points.
    """

    def __init__(self, *, max_pairs: int | None = None) -> None:
        super().__init__()
        self.max_pairs = max_pairs or cfg.COMPHET_MAX_PAIRS
        self.pair_scorer = dspy.ChainOfThought(PairScoreSignature)

    def _make_pairs(
        self, candidates: list[VariantCandidate]
    ) -> list[tuple[VariantCandidate, VariantCandidate]]:
        """Same-gene candidate pairs (the compound-het prior)."""
        by_gene: dict[str, list[VariantCandidate]] = {}
        for c in candidates:
            by_gene.setdefault(c.gene or c.chrom, []).append(c)
        pairs: list[tuple[VariantCandidate, VariantCandidate]] = []
        for cs in by_gene.values():
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    pairs.append((cs[i], cs[j]))
        return pairs[: self.max_pairs]

    def forward(
        self,
        candidates: list[VariantCandidate],
        hpo_terms: list[str],
        variant_epcrs: dict[str, float] | None = None,
    ) -> dspy.Prediction:
        """Score same-gene pairs.

        ``variant_epcrs`` maps ``coord -> single-variant EPCR`` (from the
        per-variant scorer). When provided, each variant's evidence string is
        augmented with its own EPCR so the pair LM can reason with the
        per-variant signal — without it, the pair LM independently re-judges
        both variants from scratch and can rank a weak pair (two low-scoring
        missenses) above a strong pair (stop_gained + missense in a known
        gene), costing rank points.
        """
        pairs = self._make_pairs(candidates)
        scored: list[ScoredPair] = []
        for a, b in pairs:
            a_ev = a.to_evidence_str()
            b_ev = b.to_evidence_str()
            member_epcrs: list[float] = []
            if variant_epcrs:
                a_e = variant_epcrs.get(a.coord)
                b_e = variant_epcrs.get(b.coord)
                if a_e is not None:
                    member_epcrs.append(float(a_e))
                    a_ev += f" | single_variant_epcr={a_e:.3f}"
                if b_e is not None:
                    member_epcrs.append(float(b_e))
                    b_ev += f" | single_variant_epcr={b_e:.3f}"
            out = self.pair_scorer(
                pair_description=f"A: {a_ev}\nB: {b_ev}",
                hpo_terms=hpo_terms,
            )
            pair_epcr = float(out.epcr)
            if member_epcrs:
                # Blend the pair LM's judgment with the members' own single-
                # variant EPCRs. The pair LM independently re-judges both
                # variants from scratch and can over-rank a pair of weak
                # missenses (e.g. CEP192) above a strong pair (known-gene
                # stop_gained + missense). A pair whose members scored high
                # individually is more likely causal — anchor the pair score
                # to the strongest member so that signal is never discarded.
                pair_epcr = 0.6 * pair_epcr + 0.4 * max(member_epcrs)
            scored.append(
                ScoredPair(
                    variant_1=ScoredVariant(
                        chrom=a.chrom,
                        pos=a.pos,
                        ref=a.ref,
                        alt=a.alt,
                        epcr=0.0,
                        rationale="",
                    ),
                    variant_2=ScoredVariant(
                        chrom=b.chrom,
                        pos=b.pos,
                        ref=b.ref,
                        alt=b.alt,
                        epcr=0.0,
                        rationale="",
                    ),
                    epcr=pair_epcr,
                    rationale=out.rationale,
                )
            )
        scored.sort(key=lambda p: p.epcr, reverse=True)
        return dspy.Prediction(pairs=scored)

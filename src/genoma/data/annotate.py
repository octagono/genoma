"""Variant annotation and candidate construction.

Deterministic pre-filter (Python, no LLM): quality / allele-frequency /
consequence filtering collapses millions of raw VCF rows to a bounded
candidate table.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from genoma import config as cfg
from genoma.data.ingest import load_vcf_records

# Known MVA genes; used for candidate highlighting.
KNOWN_MVA_GENES = {
    "BUB1B",
    "CEP57",
    "TRIP13",
    "MAD1L1",
    "MAD2L1BP",
    "CEP192",
}

# Gene -> disease mechanism + key clinical clue. Used for mechanism-aware
# prompting and as a deterministic hard-constraint layer (domain logic
# encoded around the LLM rather than relying on it to know it).
MVA_GENE_MECHANISMS: dict[str, str] = {
    "BUB1B": "biallelic loss-of-function (MVA1, MIM 257300); mitotic spindle-assembly "
    "checkpoint kinase; cancer predisposition (rhabdomyosarcoma, Wilms tumor, leukemia)",
    "CEP57": "biallelic (MVA2, MIM 614114); centrosome/centriole protein; growth retardation, "
    "dysmorphism; malignancies NOT reported",
    "TRIP13": "biallelic loss-of-function (MVA3, MIM 617598); AAA+ ATPase regulating the mitotic "
    "checkpoint; Wilms tumor predisposition",
    "MAD1L1": "biallelic loss-of-function; mitotic checkpoint component; high tumor susceptibility",
    "MAD2L1BP": "biallelic loss-of-function; MAD2-binding regulator (p31comet); juvenile "
    "granulosa cell tumors",
    "CEP192": "biallelic; centrosomal scaffold; MVA with tetraploidy and male infertility",
}

# All six genes are autosomal recessive with biallelic loss-of-function as the
# disease mechanism. For an AR gene, a truly causal allele should be absent or
# near-absent in homozygotes in population databases (gnomAD).
MVA_GENES_AUTOSOMAL_RECESSIVE = frozenset(KNOWN_MVA_GENES)


class VariantCandidate(BaseModel):
    """One candidate variant ready for LLM scoring (pydantic — DSPy typed-I/O)."""

    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str = ""
    consequence: str = "unknown"
    af: float | None = None  # population allele frequency (gnomAD)
    cadd: float | None = None  # CADD PHRED-like score
    revel: float | None = None  # REVEL missense pathogenicity (0-1)
    alphamissense: float | None = None  # AlphaMissense pathogenicity score (0-1)
    spliceai: float | None = None  # SpliceAI max delta score (0-1)
    evo2: float | None = None  # Evo 2 zero-shot variant-effect log-likelihood ratio
    borzoi: float | None = None  # Borzoi alt-minus-ref expression effect
    exomiser_score: float | None = None  # Exomiser gene combined score (0-1)
    exomiser_pheno_score: float | None = None  # Exomiser gene phenotype score (0-1)
    exomiser_comp_het: bool = False  # Exomiser flagged biallelic AR pair
    extra: dict = Field(default_factory=dict)

    @property
    def coord(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"

    def to_evidence_str(self) -> str:
        """Compact evidence string fed to the LLM scorer."""
        parts = [
            f"{self.coord}",
            f"gene={self.gene or 'NA'}",
            f"consequence={self.consequence}",
            f"AF={self.af if self.af is not None else 'NA'}",
            f"CADD={self.cadd if self.cadd is not None else 'NA'}",
        ]
        if self.revel is not None:
            parts.append(f"REVEL={self.revel:.3f}")
        if self.alphamissense is not None:
            parts.append(f"AlphaMissense={self.alphamissense:.3f}")
        if self.spliceai is not None:
            parts.append(f"SpliceAI={self.spliceai:.3f}")
        if self.evo2 is not None:
            parts.append(f"Evo2={self.evo2:.3f}")
        if self.borzoi is not None:
            parts.append(f"Borzoi={self.borzoi:.3f}")
        return " | ".join(parts)


def _selection_priority(consequence: str, gene: str) -> int:
    """Priority score for candidate-pool selection (higher = selected first).

    The pool must be genome-aware, not VCF-order: the first N records of a
    chromosome-sorted VCF are all chr1, so a causal gene on chr15 would never
    enter a naively-capped pool. Priority tiers (highest first):

    1. protein-altering variants in known MVA genes (LOF/missense/splice) —
       the causal mechanism is a known gene, so these are guaranteed in-pool
    2. protein-altering variants in ANY gene (novel causal candidates —
       never truncated out; the gene prior only boosts, never blinds)
    3. non-coding variants in known MVA genes (intronic/UTR — could be
       deep-intronic splice drivers)
    4. everything else (fillers to the cap)

    ``consequence`` may be ``&``-joined (VEP multi-term) — the worst term
    determines the tier.
    """
    sev = 0
    for term in str(consequence).split("&"):
        t = term.strip().lower()
        if any(s in t for s in ("stop_gained", "frameshift", "splice_acceptor", "splice_donor")):
            sev = max(sev, 300)
        elif t == "missense_variant":
            sev = max(sev, 200)
        elif "splice_region" in t:
            sev = max(sev, 150)
        elif any(s in t for s in ("exon_variant", "synonymous", "utr")):
            sev = max(sev, 80)
        elif any(s in t for s in ("intron", "upstream", "downstream")):
            sev = max(sev, 30)

    if gene in KNOWN_MVA_GENES:
        if sev >= 150:
            return 1000 + sev  # coding known-gene: guaranteed top tier
        if sev >= 80:
            return 200 + sev  # UTR/synonymous known-gene: moderate boost
        return 40 + sev  # intronic known-gene: mild boost, below novel coding
    return sev


def build_candidate_table(
    records: list[dict] | None = None,
    *,
    filter_kwargs: dict | None = None,
    max_candidates: int | None = None,
) -> list[VariantCandidate]:
    """Deterministic pre-filter: raw VCF records -> bounded candidate table.

    Extracts AF/CADD/gene/consequence from the VCF INFO fields (gnomAD AF,
    CADD, ANN/CSQ styles) and drops common variants (AF > max_af). This is
    the step that keeps the LLM layer cheap and focused: pre-filter outside
    the LM, score inside.

    Pool selection is genome-aware (``_selection_priority``): known MVA genes
    and protein-altering variants are selected first regardless of genomic
    position, then the cap is filled with the remaining passing variants —
    so a causal variant on a late chromosome can never be truncated out.
    """
    if records is None:
        records = load_vcf_records()
    max_candidates = max_candidates or cfg.CANDIDATE_MAX_DEFAULT
    fk = {**cfg.CANDIDATE_PRE_FILTER, **(filter_kwargs or {})}
    max_af = fk["max_af"]
    min_depth = fk.get("min_depth", 0.0)
    min_gq = fk.get("min_gq", 0.0)
    min_cadd = fk.get("min_cadd", 0.0)

    candidates: list[VariantCandidate] = []
    high_priority: list[VariantCandidate] = []  # known genes + protein-altering
    deferred: list[VariantCandidate] = []  # everything else (fills the cap)
    for r in records:
        af = _extract_af(r)
        if af is not None and af > max_af:
            continue
        # Call-quality filters: depth + GQ (low-quality calls are the dominant
        # false-positive source).
        depth = _extract_depth(r)
        gq = _extract_gq(r)
        if depth is not None and depth < min_depth:
            continue
        if gq is not None and gq < min_gq:
            continue
        # AR genes: reject variants present in homozygotes in population DBs.
        if _has_homozygotes(r) and _extract_gene(r) in MVA_GENES_AUTOSOMAL_RECESSIVE:
            continue
        cadd = _extract_cadd(r)
        consequence = _extract_consequence(r)
        # CADD cut is a quality heuristic, NOT a hard gate. Splice-region and
        # consequence-unknown variants can be causal with CADD < min_cadd
        # (e.g. deep-intronic indels) — dropping them here would make them
        # unrecoverable by any downstream scoring. Only apply the cut when we
        # have a confident coding missense/lof signal to rank.
        if cadd is not None and cadd < min_cadd and _cadd_applies(consequence):
            continue
        gene = _extract_gene(r)
        cand = VariantCandidate(
            chrom=r["chrom"],
            pos=int(r["pos"]),
            ref=r["ref"],
            alt=r["alt"],
            gene=gene,
            consequence=consequence,
            af=af,
            cadd=cadd,
            extra={"filter": r.get("filter", "PASS"), "depth": depth, "gq": gq},
        )
        if _selection_priority(consequence, gene) >= 150:
            high_priority.append(cand)
        elif len(deferred) < max_candidates:
            deferred.append(cand)

    # Genome-aware selection: sort high-priority candidates by priority (known
    # genes + protein-altering) so a causal variant on a late chromosome is
    # never truncated out by VCF-order cap — then fill with the rest.
    high_priority.sort(key=lambda c: _selection_priority(c.consequence, c.gene), reverse=True)
    candidates = high_priority[:max_candidates]
    if len(candidates) < max_candidates and deferred:
        deferred.sort(key=lambda c: _selection_priority(c.consequence, c.gene), reverse=True)
        candidates.extend(deferred[: max_candidates - len(candidates)])
    return candidates[:max_candidates]


def _cadd_applies(consequence: str) -> bool:
    """True when the CADD score is a meaningful ranking signal for this class.

    CADD is calibrated on coding/deleterious variants. For splice-region or
    unannotated consequences it is often NA/low even for causal variants, so
    the CADD cut must not apply to them (deep-intronic and splice-driven
    causal variants would otherwise be dropped).
    """
    low = consequence.lower()
    if not low or low == "unknown":
        return False
    return "splice" not in low and "intron" not in low and "utr" not in low


def _extract_af(r: dict) -> float | None:
    """Population allele frequency, or None when unknown.

    Only gnomAD-sourced / explicitly-population fields count as population
    AF. The bare ``AF`` field in a single-sample VCF is the SAMPLE's
    genotype frequency (``AC=1;AF=0.5;AN=2`` for a heterozygous call) — reading
    it as population AF would flag every variant as "common" (AF 0.5-1.0 >
    max_af) and drop the causal pair. Bare ``AF`` is only trusted when the
    record carries no AC/AN genotype context (e.g. a population-annotated VCF).
    """
    info = r.get("info", {})
    for key in ("gnomAD_AF", "MAX_AF", "INFO_AF", "AF_popmax"):
        v = info.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    if "AC" not in info and "AN" not in info:  # no genotype context
        v = info.get("AF")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _extract_cadd(r: dict) -> float | None:
    info = r.get("info", {})
    for key in ("CADD", "CADD_PHRED", "PHRED"):
        v = info.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _extract_gene(r: dict) -> str:
    info = r.get("info", {})
    for key in ("GENE", "SYMBOL", "Gene", "Gene.refGene"):
        g = info.get(key)
        if g:
            return str(g).split(",")[0]
    # VEP CSQ: field order from the header "Format: ..." description.
    csq = info.get("CSQ")
    if csq:
        gene = _csq_field(r, "SYMBOL") or _csq_field(r, "Gene")
        return gene.split(",")[0] if gene else ""
    return ""


def _extract_consequence(r: dict) -> str:
    info = r.get("info", {})
    for key in ("Consequence", "ANN", "Func.refGene"):
        v = info.get(key)
        if v:
            s = str(v)
            if "|" in s:  # ANN/CSQ style: consequence is the second field
                return s.split("|")[1]
            return s.split(",")[0]
    # VEP CSQ: consequence is the named field, not position 1 (Allele is).
    if info.get("CSQ"):
        return _csq_field(r, "Consequence") or "unknown"
    return "unknown"


def _csq_field(r: dict, name: str) -> str:
    """Extract a named field from a VEP CSQ string using the INFO header.

    VEP's CSQ is ``Allele|Consequence|IMPACT|SYMBOL|Gene|...`` with the exact
    order declared in the ``##INFO=<ID=CSQ,Description="Format: ...">`` header
    of the annotated VCF. We parse the header once per VCF and index by name,
    so gene/consequence positions never hardcode an assumption about VEP
    version.
    """
    info = r.get("info", {})
    csq = info.get("CSQ")
    if not csq:
        return ""
    fields = _vep_csq_fields()
    if not fields:
        # No header seen (defensive): assume standard order.
        fields = [
            "Allele",
            "Consequence",
            "IMPACT",
            "SYMBOL",
            "Gene",
            "Feature_type",
            "Feature",
            "BIOTYPE",
            "EXON",
            "INTRON",
        ]
    try:
        idx = fields.index(name)
    except ValueError:
        return ""
    parts = str(csq).split("|")
    return parts[idx] if idx < len(parts) else ""


_vep_csq_header: list[str] | None = None


def _vep_csq_fields() -> list[str]:
    """The VEP CSQ field order, parsed from the annotated VCF's header.

    Set by ``build_candidate_table`` when it loads the VCF (the header is
    available from cyvcf2). Cached per process; falls back to [] before any
    annotated VCF has been seen.
    """
    return _vep_csq_header or []


def _set_vep_csq_fields(fields: list[str]) -> None:
    global _vep_csq_header
    _vep_csq_header = fields


def _extract_depth(r: dict) -> float | None:
    info = r.get("info", {})
    for key in ("DP", "DP_FILTERED", "TOTAL_DP"):
        v = info.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _extract_gq(r: dict) -> float | None:
    info = r.get("info", {})
    for key in ("GQ", "QUAL_GQ"):
        v = info.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _has_homozygotes(r: dict) -> bool:
    """True if population data shows homozygotes for this variant (AR red flag)."""
    info = r.get("info", {})
    for key in ("gnomAD_HOM", "HOM_COUNT", "hom_count", "NHOMALT"):
        v = info.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return True
        if isinstance(v, str) and v.isdigit() and int(v) > 0:
            return True
    return False


# ── VEP REST enrichment (key-free) ─────────────────────────────────────────


def enrich_via_vep_rest(
    candidates: list[VariantCandidate],
    *,
    batch_size: int | None = None,
    timeout: int = 60,
) -> list[VariantCandidate]:
    """Annotate candidates with Ensembl VEP REST (no API key).

    Fills gene / consequence / gnomAD AF / CADD / REVEL / AlphaMissense /
    SpliceAI for candidates that the raw VCF didn't carry. Fails fast: any
    network/HTTP error propagates (annotation is a hard dependency — no
    silent fallback).

    Ensembl REST: https://rest.ensembl.org/vep/human/region
    Predictors are enabled as URL QUERY PARAMS (not JSON body — the body
    silently ignores them): `REVEL=1;AlphaMissense=1;CADD=snv_indels;
    SpliceAI=2` returns revel / alphamissense / cadd_phred / spliceai in the
    same response as the annotation.
    """
    import json
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    from genoma.net import urlopen_with_retry

    if not candidates:
        return candidates
    batch_size = batch_size or cfg.VEP_BATCH_SIZE

    # Predictors as query-string params (REVEL/AlphaMissense/SpliceAI/CADD).
    params = urllib.parse.urlencode(
        {"REVEL": 1, "AlphaMissense": 1, "CADD": "snv_indels", "SpliceAI": 2}
    )
    url = f"{cfg.VEP_REST_URL}?{params}"

    def _post(batch: list[VariantCandidate]) -> dict[str, dict]:
        # VEP region POST expects "chrom start end ref/alt" strings. end must
        # span the reference allele: end = pos + len(ref) - 1 — single-base
        # variants collapse to "pos pos", indels widen to the ref span,
        # otherwise VEP 400s with "start > end+1".
        variants = [
            f"{c.chrom.replace('chr', '')} {c.pos} {c.pos + len(str(c.ref)) - 1} {c.ref}/{c.alt}"
            for c in batch
        ]
        body = json.dumps({"variants": variants, "symbol": 1, "gnomad": 1}).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urlopen_with_retry(req, timeout=timeout) as r:
            resp = json.load(r)
        # VEP echoes the exact input string back in each result's "input"
        # field and silently DROPS variants it cannot annotate (e.g. complex
        # multi-allelic indels) — match by input, never by position.
        return {str(info.get("input")): info for info in resp if info.get("input")}

    def _to_float(v) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip():
            try:
                return float(v)
            except ValueError:
                return None
        return None

    def _spliceai_max(info: dict) -> float | None:
        sa = info.get("spliceai") or {}
        if isinstance(sa, dict):
            vals = [_to_float(sa.get(k)) for k in ("DS_AG", "DS_AL", "DS_DG", "DS_DL")]
            vals = [v for v in vals if v is not None]
            if vals:
                return max(vals)
        return None

    from genoma.data.annot_cache import cache_get, cache_set

    enriched: list[VariantCandidate] = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]

        # Cache split: already-annotated coords resolve locally (zero network);
        # only uncached coords hit the REST endpoint.
        cached = [(c, cache_get(c.coord)) for c in batch]
        uncached = [c for c, info in cached if info is None]

        resp_map: dict[str, dict] = {}
        if uncached:
            resp_map = _post(uncached)
            for c in uncached:
                key = (
                    f"{c.chrom.replace('chr', '')} {c.pos} "
                    f"{c.pos + len(str(c.ref)) - 1} {c.ref}/{c.alt}"
                )
                info = resp_map.get(key)
                if info is not None:
                    cache_set(c.coord, info)

        for cand, info in cached:
            key = (
                f"{cand.chrom.replace('chr', '')} {cand.pos} "
                f"{cand.pos + len(str(cand.ref)) - 1} {cand.ref}/{cand.alt}"
            )
            info = info if info is not None else resp_map.get(key)
            if info is None:
                # VEP REST silently drops variants it cannot annotate (complex
                # multi-allelic indels) — keep the local VEP annotation intact
                # and carry the candidate through without REST enrichment.
                enriched.append(cand)
                continue
            gene = (info.get("symbol") or "").strip() or cand.gene
            consequence = info.get("most_severe_consequence") or "unknown"
            af = cand.af
            gnomad = info.get("gnomAD") or {}
            if isinstance(gnomad, dict):
                for pop in ("af", "AF", "gnomade", "AF_popmax"):
                    if pop in gnomad and isinstance(gnomad[pop], (int, float)):
                        af = float(gnomad[pop])
                        break
            cadd = cand.cadd
            if isinstance(info.get("cadd"), dict):
                phred = info["cadd"].get("phred") or info["cadd"].get("PHRED")
                if isinstance(phred, (int, float)):
                    cadd = float(phred)
            revel = _to_float(info.get("revel"))
            if revel is None:
                revel = cand.revel
            alphamissense = _to_float((info.get("alphamissense") or {}).get("am_pathogenicity"))
            if alphamissense is None:
                alphamissense = cand.alphamissense
            spliceai = _spliceai_max(info)
            if spliceai is None:
                spliceai = cand.spliceai
            enriched.append(
                cand.model_copy(
                    update={
                        "gene": gene,
                        "consequence": consequence,
                        "af": af,
                        "cadd": cadd,
                        "revel": revel,
                        "alphamissense": alphamissense,
                        "spliceai": spliceai,
                    }
                )
            )
        if i + batch_size < len(candidates):
            time.sleep(cfg.VEP_RATE_LIMIT_S)
    return enriched


def apply_gene_prior(epcr: float, gene: str, *, boost: float | None = None) -> float:
    """Bayesian-ish prior: pull EPCR up for variants in known MVA genes.

    For an ultra-rare disease (<50 cases), the causal gene is very likely
    one of the 6 SAC/centrosome genes. Boost = (1 - epcr) * boost fraction.
    """
    boost = boost if boost is not None else cfg.GENE_PRIOR_BOOST
    if gene in KNOWN_MVA_GENES:
        return min(1.0, epcr + (1.0 - epcr) * boost)
    return epcr

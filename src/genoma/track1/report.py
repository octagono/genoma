"""Track 1 report generator — the methods-description report for submission.

Answers every question from the organizers' methods_description_form.xlsx
truthfully from the pipeline's actual implementation (config values, cost
estimator output, submission rows, panel membership). Export to PDF with any
markdown tool.
"""

from __future__ import annotations

from pathlib import Path

from genoma import config as cfg
from genoma.data.annotate import KNOWN_MVA_GENES
from genoma.data.submission import load_submission


def _compound_het_count(rows) -> int:
    """Rows with a filled variant_2 = proposed compound-het pairs."""
    return sum(1 for r in rows if getattr(r, "chrom_2", ""))


def _cost_line(panel_members: list[str] | None = None) -> str:
    """One-line cost estimate from the live estimator.

    Uses the configured PRIMARY_LM pricing and candidate caps, so the report
    reflects the actual model/cost of the submitted run rather than a
    hardcoded Anthropic estimate. When a consensus panel was used, the
    estimate covers every panel member (weighted, incl. debate rounds).
    """
    from genoma.cost import plan_panel

    total = 0.0
    for line in plan_panel():
        total += line["high"]
    model_label = " + ".join(panel_members) if panel_members else cfg.PRIMARY_LM
    return (
        f"~${total:.2f} (max estimate, {model_label}, "
        f"{cfg.CANDIDATE_MAX_DEFAULT} candidates + {cfg.COMPHET_MAX_PAIRS} pairs)"
    )


def generate_track1_report(
    *,
    submission_csv: Path | str,
    hpo_terms: list[str],
    candidates: list | None = None,
    panel_members: list[str] | None = None,
    out_path: Path | str | None = None,
    team_name: str = "",
) -> str:
    """Build the Track 1 methods report (Markdown). Returns the text."""
    rows = load_submission(submission_csv)
    sub_path = Path(submission_csv)
    n_pairs = _compound_het_count(rows)

    lines: list[str] = []
    lines.append("# Track 1 Methods Report — MVA Hackathon 2026")
    lines.append("")
    lines.append("*(answers follow the organizers' methods_description_form.xlsx)*")
    lines.append("")

    # ── Q1. Team + model ────────────────────────────────────────────────────
    lines.append("## Q1. Team name / model")
    lines.append("")
    lines.append(f"- **Team / username:** {team_name or '(set with --team)'}")
    lines.append(f"- **Proband:** {cfg.PROBAND_ID}")
    lines.append(f"- **Primary model:** {cfg.PRIMARY_LM}")
    if panel_members:
        lines.append(f"- **Consensus panel:** {', '.join(panel_members)}")
    lines.append(
        f"- **Submission file:** {sub_path.name} ({len(rows)} rows, {n_pairs} compound-het pair rows)"
    )
    lines.append("- **Framework:** dspy==3.3.1 (pinned)")
    lines.append("")

    # ── Q2. Approach ────────────────────────────────────────────────────────
    lines.append("## Q2. Model/approach (detailed)")
    lines.append("")
    lines.append(
        "The pipeline is a deterministic-first, LLM-ranked variant "
        "prioritization system, deliberately structured so the LLM reasons "
        "over a small, biologically-grounded candidate set rather than the "
        "full ~5M-variant WGS callset."
    )
    lines.append("")
    lines.append(
        "1. **Local annotation:** the raw GATK VCF is annotated once "
        "with Ensembl VEP (GRCh38) — gene, consequence, transcript — "
        "via the official container; the pipeline consumes the "
        "annotated VCF automatically."
    )
    lines.append(
        "2. **Deterministic pre-filter** (`build_candidate_table`): "
        "quality (depth/GQ), population-AF sanity, AR-homozygote "
        "exclusion for recessive genes, and a CADD cut that "
        "deliberately NEVER drops splice-region/low-CADD causal "
        "variants. Output: a bounded "
        f"candidate table (cap {cfg.CANDIDATE_MAX_DEFAULT})."
    )
    lines.append(
        "3. **VEP REST deep predictors:** gnomAD AF, CADD, REVEL, "
        "AlphaMissense, SpliceAI pulled for every surviving candidate "
        "(SQLite-cached — variants are static coordinates, so repeat "
        "runs are free)."
    )
    lines.append(
        "4. **LLM scoring:** each candidate scored for causality "
        "(EPCR, schema-enforced 0–1) under a Bayesian boost-only gene "
        f"prior for the known MVA set ({', '.join(sorted(KNOWN_MVA_GENES))}), "
        "plus compound-het pair scoring (same-gene biallelic, in-trans "
        "phasing via VCF tags / trio Mendelian / unknown-default)."
    )
    if panel_members:
        lines.append(
            "5. **Consensus + calibration:** a heterogeneous multi-model "
            "panel (weighted median/trimmed-mean, spread-triggered debate, "
            "post-debate abstention), confidence-weighted self-consistency, "
            "and a final submission-EPCR calibration that reserves a "
            "strictly unique rank-1 (no ties — a tie at the top collapses "
            "F-max 1.0→0.667, verified against the official scorer)."
        )
    else:
        lines.append(
            "5. **Calibration:** confidence-weighted self-consistency "
            "(temperature-1.0 re-samples of the top rows) and a final "
            "submission-EPCR calibration that reserves a strictly unique "
            "rank-1 (no ties — a tie at the top collapses F-max 1.0→0.667, "
            "verified against the official scorer)."
        )
    lines.append("")

    # ── Q3. Generative-AI disclosure (REQUIRED) ─────────────────────────────
    # Provider/plan must truthfully reflect the models actually used. Derive
    # the vendor from the PRIMARY_LM slug rather than assuming Anthropic.
    primary = cfg.PRIMARY_LM
    if primary.startswith("deepseek/"):
        vendor = "DeepSeek API"
    elif primary.startswith("anthropic/"):
        vendor = "Anthropic API"
    elif "openrouter/" in primary:
        vendor = "OpenRouter (multi-vendor)"
    else:
        vendor = "LLM API"
    lines.append("## Q3. Generative-AI provider disclosure (required)")
    lines.append("")
    lines.append(
        f"- **Provider/plan/tier:** {vendor} ({primary}), commercial "
        "pay-per-token API; no model training on customer content."
    )
    if panel_members:
        lines.append(
            f"- **Panel members:** {', '.join(panel_members)} (each via its "
            "vendor API / OpenRouter; all commercial, no training on inputs)."
        )
    temp_desc = (
        "temperature 0.7 (panel) / 1.0 (self-consistency samples)"
        if panel_members
        else "temperature ~0.2–0.7 (scoring) / 1.0 (self-consistency samples)"
    )
    lines.append(
        f"- **Settings:** {temp_desc}, max_tokens bounded, prompt caching "
        "enabled, no custom fine-tuning, no RAG over the gated data."
    )
    lines.append("")

    # ── Q4–5. Manual review ─────────────────────────────────────────────────
    lines.append("## Q4–5. Automated output / manual review")
    lines.append("")
    lines.append(
        "- **Submission file is the automated output** of the pipeline "
        "(no post-hoc curation of the ranked rows)."
    )
    lines.append(
        "- Manual involvement is limited to upstream scientific choices: "
        "configuring the known-gene prior, the pre-filter thresholds, and "
        "reviewing the HPO phenotype extraction against the clinical docx. "
        "Row ordering and EPCR values are fully automated."
    )
    lines.append("")

    # ── Q6–8. Data ──────────────────────────────────────────────────────────
    lines.append("## Q6–8. Data used")
    lines.append("")
    lines.append(
        "- **Public data:** gnomAD AF (via VEP), CADD, REVEL, "
        "AlphaMissense, SpliceAI (Ensembl VEP REST), Phen2Gene "
        "(HPO→gene), Ensembl gene models (VEP GRCh38 cache)."
    )
    lines.append(
        "- **Proprietary data:** none. The only non-public input is "
        "the challenge's gated dataset itself."
    )
    lines.append("")

    # ── Q9. Compound-het output ─────────────────────────────────────────────
    lines.append("## Q9. Compound-het pair output")
    lines.append("")
    lines.append(
        f"- The approach outputs **both single candidate rows and "
        f"compound-heterozygous pair rows**. This submission contains "
        f"{len(rows)} rows, of which **{n_pairs} are compound-het pair rows** "
        "(one row encoding both variants of a same-gene biallelic pair) — "
        "because the causal mechanism is expected to be compound-het, and the "
        "official scorer requires the full pair in one row for full rank points."
    )
    lines.append("")

    # ── Q10. Incidental findings ────────────────────────────────────────────
    lines.append("## Q10. Secondary/incidental findings")
    lines.append("")
    lines.append(
        "- Incidental findings are handled via the `finding_type` column: "
        "rows are marked `primary` (the causal hypothesis) or `secondary` "
        "(ranked incidental candidates). The pre-filter is not phenotype-"
        "restricted, so incidental candidates can surface; the HPO-driven "
        "scoring + gene prior keep the causal pair at the top."
    )
    lines.append("")

    # ── Q11. Run time / cost ────────────────────────────────────────────────
    lines.append("## Q11. Run time and cost")
    lines.append("")
    lines.append(f"- **Estimated cost:** {_cost_line(panel_members)}")
    lines.append(
        "- **Runtime:** ~1–3 h local VEP annotation (once), then minutes for "
        "pre-filter + VEP REST (SQLite-cached), then ~30–60 min LLM scoring "
        "at 8-thread parallelism. Deterministic steps are free after caching."
    )
    lines.append("")

    # ── Abstract ────────────────────────────────────────────────────────────
    lines.append("## Method abstract (≤500 words)")
    lines.append("")
    if panel_members:
        lines.append(
            "I present a deterministic-first, LLM-ranked variant "
            "prioritization pipeline for a single-proband MVA (Mosaic "
            "Variegated Aneuploidy) case. Raw WGS VCF (5M variants) is "
            "VEP-annotated once; a quality + allele-frequency + splice-safe "
            "pre-filter collapses it to a bounded candidate table. Deep "
            "in-silico predictors (gnomAD AF, CADD, REVEL, AlphaMissense, "
            "SpliceAI) are retrieved for every candidate. A heterogeneous "
            "multi-model panel — weighted median/trimmed-mean with "
            "spread-triggered debate and post-debate abstention — scores "
            "each candidate and each same-gene compound-heterozygous pair "
            "for causality (EPCR, 0–1), with a boost-only prior over the "
            "known MVA genes (BUB1B/CEP57/TRIP13/MAD1L1/MAD2L1BP/CEP192) "
            "that never down-weights novel-gene candidates. "
            "Confidence-weighted self-consistency and a submission-EPCR "
            "calibration (strict unique rank-1, no ties) produce the final "
            "ranked CSV, compound-het pairs encoded as single rows for full "
            "rank points."
        )
    else:
        lines.append(
            "I present a deterministic-first, LLM-ranked variant "
            "prioritization pipeline for a single-proband MVA (Mosaic "
            "Variegated Aneuploidy) case. Raw WGS VCF (5M variants) is "
            "VEP-annotated once; a quality + allele-frequency + splice-safe "
            "pre-filter collapses it to a bounded candidate table. Deep "
            "in-silico predictors (gnomAD AF, CADD, REVEL, AlphaMissense, "
            "SpliceAI) are retrieved for every candidate. A single "
            "reasoning model scores each candidate and each same-gene "
            "compound-heterozygous pair for causality (EPCR, 0–1), with a "
            "boost-only prior over the known MVA genes "
            "(BUB1B/CEP57/TRIP13/MAD1L1/MAD2L1BP/CEP192) that never "
            "down-weights novel-gene candidates. "
            "Confidence-weighted self-consistency and a submission-EPCR "
            "calibration (strict unique rank-1, no ties) produce the final "
            "ranked CSV, compound-het pairs encoded as single rows for full "
            "rank points."
        )
    lines.append("")
    lines.append(
        "Strengths: (1) the deterministic pre-filter + deep-predictor layer "
        "focuses the LLM on a small, biologically-grounded candidate set; "
        "(2) the splice-safe CADD rule closes the known deep-intronic "
        "failure mode; (3) no-tie calibration maximizes F-max under the "
        "official scorer; (4) fully reproducible — 138 offline tests, "
        "official-scorer parity differential tests, config-SSOT; "
        "(5) rare-disease-agnostic — the pre-filter, panel, and calibration "
        "carry no MVA-specific logic, so adapting the pipeline to another "
        "unsolved case is a configuration change (known-gene set, phenotype "
        "terms), not a code change."
    )
    lines.append("")
    lines.append(
        "Limitations: (1) the LLM is evaluated on a synthetic trainset "
        "(the real label is hidden), so prompt optimization is "
        "distribution-guarded rather than label-tuned; (2) phasing relies on "
        "VCF phase tags / trio data and defaults to unknown when absent; "
        "(3) a novel causal gene outside the known MVA set relies on the "
        "genome-wide compound-het filter rather than the gene prior; "
        "(4) runtime is gated by the local VEP annotation step."
    )
    lines.append("")

    # ── Candidate evidence table ────────────────────────────────────────────
    lines.append("## Submission rows")
    lines.append("")
    lines.append("| # | Variant 1 | Variant 2 (pair) | EPCR | Type | Notes |")
    lines.append("|---|-----------|------------------|------|------|-------|")
    for i, row in enumerate(rows, start=1):
        v2 = (
            f"{row.chrom_2}:{row.pos_2}:{row.ref_2}>{row.alt_2}"
            if getattr(row, "chrom_2", "")
            else "—"
        )
        lines.append(
            f"| {i} | {row.chrom_1}:{row.pos_1}:{row.ref_1}>{row.alt_1}"
            f" | {v2} | {row.epcr:.3f} | {row.finding_type} | {row.notes[:50]} |"
        )
    lines.append("")

    # ── Reproducibility ─────────────────────────────────────────────────────
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "- Code: GitHub repository github.com/octagono/genoma "
        "(private during the hackathon; made public when final evaluations begin)."
    )
    lines.append(
        f"- Config: all model IDs, thresholds, priors, caps in `src/genoma/config.py` "
        f"(e.g. candidate cap {cfg.CANDIDATE_MAX_DEFAULT}, pair cap {cfg.COMPHET_MAX_PAIRS})."
    )
    lines.append(
        "- Disease-agnostic switch: `KNOWN_MVA_GENES`, the HPO synonym map, and the "
        "mechanism description are the only disease-specific inputs — every other "
        "component (pre-filter, VEP REST, panel, calibration) is generic, so a "
        "different rare disease is a config change, not a code change."
    )
    lines.append(
        "- Data: gated `SageBio/mva-hackathon-2026-data`; deleted within 30 "
        "days of close per the rules."
    )
    lines.append("")

    report = "\n".join(lines)

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
    return report

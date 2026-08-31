"""Track 2 report writer: the expert-reviewed Markdown artifact.

A clean written report (drug → target → pathway link → evidence) is the
submission artifact, not console output.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from genoma import config as cfg


def _vendor_label() -> str:
    """Truthful provider label derived from the configured PRIMARY_LM slug."""
    primary = cfg.PRIMARY_LM
    if primary.startswith("deepseek/"):
        return f"DeepSeek API ({primary})"
    if primary.startswith("anthropic/"):
        return f"Anthropic API ({primary})"
    if "openrouter/" in primary:
        return f"OpenRouter (multi-vendor) ({primary})"
    return f"LLM API ({primary})"


def generate_track2_report(
    drug_candidates: list,
    *,
    mechanism: str,
    gene: str,
    hpo_terms: list[str],
    termination_reason: str | None = None,
    stall_diagnosis: str = "",
    out_path: str | Path | None = None,
) -> Path:
    """Write the Track 2 drug-repurposing report (Markdown) and return its path.

    ``drug_candidates`` items follow the ``DrugCandidate`` model
    (name / mechanism / rationale / evidence / risk).
    """
    out = Path(out_path) if out_path else cfg.SUBMISSION_DIR / "track2_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Track 2 — Drug Repurposing Report",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Disease context",
        "",
        "- **Disorder**: Mosaic Variegated Aneuploidy (MVA)",
        f"- **Gene of interest**: {gene}",
        f"- **Variant-driven mechanism**: {mechanism}",
        f"- **Phenotype (HPO)**: {', '.join(hpo_terms) if hpo_terms else 'n/a'}",
        "",
        "## Proposed repurposing candidates",
        "",
    ]

    if not drug_candidates:
        lines.append("_No candidates returned by the agent._")
    for i, cand in enumerate(drug_candidates, start=1):
        lines += [
            f"### {i}. {cand.name}",
            "",
            f"- **Proposed mechanism of action**: {cand.mechanism}",
            f"- **Rationale (pathway link)**: {cand.rationale}",
            f"- **Evidence**: {cand.evidence}",
            f"- **Risks / unknowns**: {cand.risk}",
            "",
        ]

    lines += [
        "## Methods",
        "",
        "- ReActV2 tool-using agent (DSPy 3.3.1, native function calling) over",
        "  the primary model; tools: PubMed (E-utilities), OpenFDA drug labels,",
        "  ClinVar (E-utilities).",
        f"- Agent termination: {termination_reason or 'max iterations reached'}",
    ]
    if stall_diagnosis:
        lines += ["", f"- **Agent loop note**: {stall_diagnosis}"]
    lines += [
        "",
        "## Disclaimer",
        "",
        "These are computational hypotheses for follow-up investigation —",
        "not claims of clinical efficacy or treatment recommendations.",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_track2_methods_report(
    *,
    drug_candidates: list,
    mechanism: str,
    gene: str,
    hpo_terms: list[str],
    team_name: str = "",
    out_path: str | Path | None = None,
) -> Path:
    """Write the Track 2 methods-description report answering the organizers'
    methods_description_form.xlsx questions (Track 2 sheet, rows 7-17).

    Answers every question truthfully from the actual Track 2 agent
    implementation: ReActV2 (DSPy 3.3.1) over a primary LLM with
    evidence-gathering tools (PubMed E-utilities, OpenFDA drug labels,
    ClinVar E-utilities), therapeutic-window guidance in the signature,
    and a Markdown drug report as the review artifact.
    """
    out = Path(out_path) if out_path else cfg.SUBMISSION_DIR / "track2_methods_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Track 2 Methods Report — MVA Hackathon 2026",
        "",
        "*(answers follow the organizers' methods_description_form.xlsx, Track 2 sheet)*",
        "",
        "## Q1. Team name",
        "",
        f"- **Team / username:** {team_name or '(set with --team)'}",
        "",
        "## Q2. Approach (variant/mechanism → candidate medications)",
        "",
        "The agent receives the proband's variant-driven mechanism "
        f"(here: **{mechanism}**, gene {gene}) and the HPO phenotype "
        f"({', '.join(hpo_terms) or 'n/a'}). It is a **ReActV2 tool-using "
        "agent** (DSPy 3.3.1, native function calling) that iteratively:",
        "",
        "1. **Reason over the mechanism** — the signature frames MVA as a "
        "spindle-assembly-checkpoint / chromosome-missegregation / "
        "centrosome disorder and steers candidate drugs toward "
        "aneuploid-cell-selective vulnerabilities (proteotoxic-stress "
        "response, HSP90, autophagy) while **explicitly discouraging broadly "
        "cytotoxic spindle poisons** (taxanes/vinca alkaloids) that would "
        "worsen systemic aneuploidy in a child.",
        "2. **Gather evidence with tools** — PubMed E-utilities "
        "(esearch/esummary), OpenFDA drug-label API, and ClinVar "
        "E-utilities return verifiable citations, approvals, and "
        "variant-level context per candidate.",
        "3. **Emit structured candidates** — each `DrugCandidate` carries "
        "name / proposed mechanism of action / pathway rationale / evidence "
        "with citations / risks-and-unknowns (schema-enforced).",
        "4. **Write the review artifact** — the expert-judged "
        "`track2_report.md` (drug → mechanism → rationale → evidence → "
        "risks + methods + disclaimer).",
        "",
        "## Q3. Generative-AI provider disclosure (required)",
        "",
        f"- **Provider/plan/tier:** {_vendor_label()}, commercial "
        "pay-per-token API; no model training on customer content.",
        "- **Settings:** ReActV2 loop (bounded `max_iters`), native function "
        "calling, temperature default, prompt caching enabled; no custom "
        "fine-tuning; tool results are fetched from public APIs at run time.",
        "",
        "## Q4. Automated vs manual candidate identification",
        "",
        "- **Fully automated.** Candidate medications are identified by the "
        "ReActV2 agent via tool-grounded search (PubMed / OpenFDA / ClinVar) "
        "driven by the mechanism prompt — not by manual literature review.",
        "",
        "## Q5. Manual literature review / curation",
        "",
        "- None performed on candidate identification. Human involvement is "
        "limited to configuring the mechanism description and reviewing the "
        "generated report before submission.",
        "",
        "## Q6–8. Data sources",
        "",
        "- **Public data only:** PubMed (NCBI E-utilities), FDA drug labels "
        "(OpenFDA), ClinVar (NCBI E-utilities), and the challenge's own "
        "clinical phenotype. **No proprietary databases** (e.g., commercial "
        "drug databases) were used.",
        "",
        "## Q9. Mechanism characterization",
        "",
        "- The variant-driven mechanism is characterized **loss-of-function / "
        "pathway-disruption** in the spindle assembly checkpoint (SAC) and "
        "centrosome biology (MVA is autosomal-recessive compound-het in "
        "BUB1B/CEP57/TRIP13/MAD1L1/MAD2L1BP/CEP192-class genes), leading to "
        "chromosome missegregation, aneuploidy, growth delay, and cancer "
        "predisposition (e.g., rhabdomyosarcoma). Downstream consequences "
        "considered: aneuploid-cell proteotoxic stress, mitotic-checkpoint "
        "dependency, and centrosome instability.",
        "",
        "## Q10. Time / effort estimate",
        "",
        "- One agent run: ~5-15 min at ~10-15 iterations of tool + reasoning "
        "steps (~$0.10-0.15 in API cost). The full analysis (mechanism "
        "definition, agent runs, report generation, methods writeup) is "
        "~1-2 person-days.",
        "",
        "## Q11. Method abstract (≤500 words)",
        "",
        "I propose existing, approved medications for further investigation "
        "in a single MVA proband using a tool-using ReActV2 agent over a "
        "clinical LLM. Given the variant-driven SAC/centrosome mechanism and "
        "the child's phenotype (growth delay, failure to thrive, "
        "rhabdomyosarcoma), the agent searches PubMed, FDA drug labels, and "
        "ClinVar to identify and evidence candidate repurposing hypotheses. "
        "The search is deliberately steered toward aneuploid-cell-selective "
        "vulnerabilities (proteotoxic-stress response, HSP90 inhibition, "
        "autophagy modulation) and away from broadly cytotoxic mitotic "
        "poisons, reflecting the pediatric, systemically-aneuploid context.",
        "",
        "Strengths: (1) evidence-backed — every candidate carries citations "
        "from public sources rather than unsupported drug suggestions; "
        "(2) mechanism-anchored — candidates must link to the disrupted "
        "pathway, not just the disease name; (3) safety-aware — the "
        "therapeutic-window guidance prevents clinically implausible "
        "suggestions; (4) reproducible and auditable — structured "
        "`DrugCandidate` output and a written report artifact; (5) the "
        "evidence-gathering agent is disease-agnostic — swapping the "
        "mechanism description and gene context adapts it to another "
        "variant-driven disorder without code changes.",
        "",
        "Limitations: (1) depends on public-API availability and the primary "
        "LLM's retrieval synthesis; (2) no experimental validation — "
        "candidates are hypotheses for follow-up; (3) OpenFDA covers approved "
        "labels, so promising off-label or investigational candidates outside "
        "that surface are not surfaced; (4) tool call history is bounded by "
        "the agent's iteration cap.",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out

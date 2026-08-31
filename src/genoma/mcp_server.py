"""MCP server for the genoma pipeline.

Exposes all pipeline commands as typed MCP tools so an agentic system can
drive the pipeline through structured tool calls instead of parsing CLI output.

Installation:
    uv sync --group mcp-server
    # or: uv pip install -e ".[mcp-server]"

Running:
    python -m genoma.mcp_server          # stdio mode (default)
    genoma-mcp                           # via entry-point

This module is optional: it only loads when the MCP server is started.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from genoma import config as cfg

# Lazy import gate — keeps the rest of the pipeline fast.
_mcp: Any = None


def _get_mcp():
    global _mcp
    if _mcp is None:
        from mcp.server.mcpserver import MCPServer

        _mcp = MCPServer(name="genoma")
    return _mcp


def _run(cmd_fn, **kwargs) -> str:
    """Run a cmd_* function, capture its printed output, return as tool result."""
    ns = SimpleNamespace(**kwargs)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_fn(ns)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Thin wrappers for commands that don't have a cmd_* in cli.py
# ---------------------------------------------------------------------------


def _cmd_cost(args) -> None:
    """Dry-run cost estimator: per-command budget breakdown."""
    from genoma import cost as cost_mod

    max_c = getattr(args, "max_candidates", 500)
    print("=== Track 1: score ===")
    for line in cost_mod.plan_score(max_c):
        print(cost_mod._fmt_line(line))
    print("\n=== Track 1: panel (3-model consensus) ===")
    for line in cost_mod.plan_panel(max_c):
        print(cost_mod._fmt_line(line))
    print("\n=== Track 1: optimize (GEPA) ===")
    for line in cost_mod.plan_optimize():
        print(cost_mod._fmt_line(line))
    print("\n=== Track 2: ReActV2 ===")
    for line in cost_mod.plan_track2():
        print(cost_mod._fmt_line(line))


def _cmd_baml(args) -> None:
    """Run BAMLAdapter variant scoring (single-model, no panel)."""
    args.rlm = False
    from genoma.cli import cmd_score

    cmd_score(args)


def _cmd_rlm(args) -> None:
    """Run RLM variant scoring with sandbox domain tools."""
    args.rlm = True
    from genoma.cli import cmd_score

    cmd_score(args)


def _cmd_phenopacket(args) -> None:
    """Generate an Exomiser phenopacket YAML + genes.tsv + variants.tsv."""
    from genoma.cli import cmd_exomiser

    cmd_exomiser(args)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def _register_tools():
    from genoma.cli import (
        cmd_compliance,
        cmd_demo,
        cmd_evaluate,
        cmd_exomiser,
        cmd_hpo,
        cmd_ingest,
        cmd_optimize,
        cmd_panel,
        cmd_phase,
        cmd_phen2gene,
        cmd_planner,
        cmd_purge_cache,
        cmd_report,
        cmd_score,
        cmd_track2,
        cmd_validate,
    )

    mcp = _get_mcp()

    # ── Free / planning tools (no API cost) ──────────────────────────────

    @mcp.tool()
    def demo(max_candidates: int = 50) -> str:
        """Run demo mode offline (no API keys needed). Validates the full
        pipeline end-to-end using DummyLM. Returns scored candidates, panel,
        calibrated submission, and compliance check. Safe to call freely."""
        return _run(cmd_demo, max_candidates=max_candidates)

    @mcp.tool()
    def cost(max_candidates: int = cfg.CANDIDATE_MAX_DEFAULT) -> str:
        """Dry-run cost estimator. Returns per-command cost breakdown WITHOUT
        making any API calls. Use this to preview budget before running score."""
        return _run(_cmd_cost, max_candidates=max_candidates)

    @mcp.tool()
    def compliance() -> str:
        """Print the Sage Bionetworks deletion checklist and compliance
        reminders. No API cost. Run before every submission."""
        return _run(cmd_compliance)

    @mcp.tool()
    def validate(csv_path: str = "") -> str:
        """Score an existing submission CSV against the synthetic answer key.
        Returns rank-points and F-max checks. No API cost."""
        return _run(cmd_validate, csv=csv_path)

    @mcp.tool()
    def hpo(phenotype_doc: str = "") -> str:
        """Extract HPO terms from a phenotype docx file. Returns the list
        of HPO IDs and labels used for scoring. Falls back to demo HPO
        if no docx is available."""
        return _run(cmd_hpo, phenotype_doc=phenotype_doc)

    @mcp.tool()
    def planner(
        record: str = "",
        matched: str = "",
        note: str = "",
        state: str = "",
    ) -> str:
        """Load and print the SubmissionPlanner config: pair row budget,
        singleton rows, tier cutoffs, panel size. Optionally record a
        submission's feedback (CSV + matched coords). No API cost."""
        return _run(
            cmd_planner,
            record=record,
            matched=matched,
            note=note,
            state=state or str(Path("submission/planner_state.json")),
        )

    @mcp.tool()
    def purge_cache() -> str:
        """Purge the DSPy disk cache (~/.dspy_cache) which stores LM
        responses verbatim. Run after changing models or to free disk."""
        return _run(cmd_purge_cache)

    @mcp.tool()
    def evaluate(program_path: str = "", n: int = 10) -> str:
        """Evaluate a saved/compiled program on the synthetic devset with the
        judge. Returns accuracy metrics. No API cost unless the judge calls LM."""
        return _run(cmd_evaluate, program=program_path, n=n)

    # ── Data preparation tools (no LM API cost) ──────────────────────────

    @mcp.tool()
    def ingest(force: bool = False) -> str:
        """Ingest the gated dataset: download VCF from HuggingFace,
        extract phenotype docx. Requires HF_TOKEN in .env and accepted
        terms on the dataset page. No LM API cost, but downloads ~85GB."""
        return _run(cmd_ingest, force=force)

    @mcp.tool()
    def phenopacket(patient_id: str = "PA001") -> str:
        """Generate an Exomiser phenopacket YAML + genes.tsv + variants.tsv
        from the ingested data. No LM API cost."""
        return _run(_cmd_phenopacket, patient_id=patient_id)

    @mcp.tool()
    def annotate(max_candidates: int = cfg.CANDIDATE_MAX_DEFAULT, top: int = 10) -> str:
        """Enrich candidate variants via Ensembl VEP REST (REVEL /
        AlphaMissense / SpliceAI / CADD). Key-free; fails fast on network
        error (transient errors retry with backoff). No LM API cost.
        Requires ingested data."""
        from genoma.cli import cmd_annotate

        return _run(cmd_annotate, max_candidates=max_candidates, top=top)

    @mcp.tool()
    def candidates(
        gene: str = "",
        max_candidates: int = cfg.CANDIDATE_MAX_DEFAULT,
        top: int = 20,
        min_af: float = 0.0,
        max_af: float = 1.0,
    ) -> str:
        """List/filter the deterministic candidate pool (pre-scoring). Filters
        by gene (substring), allele-frequency range, and prints the top rows
        with gene / consequence / AF / CADD / REVEL / AlphaMissense. No LM API
        cost. Requires ingested data — use `ingest` first."""
        from genoma.data.annotate import build_candidate_table

        cands = build_candidate_table(max_candidates=max_candidates)
        if gene:
            cands = [c for c in cands if gene.lower() in c.gene.lower()]
        cands = [c for c in cands if (c.af is None or min_af <= c.af <= max_af)]
        lines = [
            f"{len(cands)} candidates"
            + (f" for gene '{gene}'" if gene else "")
            + (" (AF-filtered)" if min_af > 0 or max_af < 1 else "")
        ]
        for c in cands[:top]:
            af = f"{c.af:.3g}" if c.af is not None else "-"
            cadd = f"{c.cadd:.1f}" if c.cadd is not None else "-"
            revel = f"{c.revel:.2f}" if c.revel is not None else "-"
            am = f"{c.alphamissense:.2f}" if c.alphamissense is not None else "-"
            lines.append(
                f"{c.coord}  gene={c.gene or '-'}  cons={c.consequence}  "
                f"AF={af}  CADD={cadd}  REVEL={revel}  AM={am}"
            )
        if len(cands) > top:
            lines.append(f"... ({len(cands) - top} more)")
        return "\n".join(lines)

    @mcp.tool()
    def mechanism(gene: str) -> str:
        """Query the MVA disease-mechanism knowledge base for a gene. Returns
        the gene's known MVA mechanism + clinical clue (empty if not one of
        the six established MVA genes), whether it is a known MVA gene, and
        its EPCR prior boost fraction. Pure lookup, no LM API cost, offline."""
        from genoma.data.annotate import KNOWN_MVA_GENES, MVA_GENE_MECHANISMS

        known = gene.upper() in {g.upper() for g in KNOWN_MVA_GENES}
        mechanism_text = MVA_GENE_MECHANISMS.get(gene, "")
        boost = cfg.GENE_PRIOR_BOOST if known else 0.0
        lines = [
            f"gene={gene}",
            f"known_mva_gene={known}",
            f"mechanism={mechanism_text or 'not in MVA mechanism KB (could still be a novel SAC/mitotic gene)'}",
            f"prior_boost={boost:.2f}",
        ]
        return "\n".join(lines)

    @mcp.tool()
    def phen2gene(patient_id: str = "PA001", top_n: int = 20) -> str:
        """Rank candidate genes by phenotype similarity using Phen2Gene
        REST API. No LM API cost (external REST call only)."""
        return _run(cmd_phen2gene, patient_id=patient_id, top_n=top_n)

    @mcp.tool()
    def exomiser(patient_id: str = "PA001") -> str:
        """Run Exomiser comp-het mode (requires Java 21+ and EXOMISER_JAR).
        No LM API cost. Returns Exomiser-ranked candidates."""
        return _run(cmd_exomiser, patient_id=patient_id)

    @mcp.tool()
    def phase(patient_id: str = "PA001", emit_pipeline: bool = False) -> str:
        """Phase-aware analysis: trans/cis from VCF tags (PS/HP/GT), or emit
        the WhatsHap pipeline script. No LM API cost."""
        return _run(cmd_phase, patient_id=patient_id, pipeline=emit_pipeline)

    @mcp.tool()
    def report(
        csv_path: str = "",
        output_path: str = "",
    ) -> str:
        """Generate the Track 1 methods report (Markdown) for a submission.
        No LM API cost."""
        return _run(cmd_report, csv=csv_path, out=output_path)

    # ── LM-powered tools (REAL API COST per run) ─────────────────────────
    @mcp.tool()
    def score(
        model: str = "",
        max_candidates: int = cfg.CANDIDATE_MAX_DEFAULT,
        top: int = 10,
        max_pairs: int = 10,
        annotate: bool = True,
        strategy: str = "mixed",
        self_consistency: int = 0,
        calibrate: bool = True,
        resilient: bool = False,
        checkpoint: str = "",
    ) -> str:
        """Run BAMLAdapter variant scoring with consensus panel.
        ⚠️ COSTS REAL API BUDGET: ~$2-3 (500 candidates) depending on model.
        Requires ANTHROPIC_API_KEY (or OPENROUTER_API_KEY).
        annotate (default on) enriches via VEP REST before scoring.
        resilient=True (production mode): per-candidate failures degrade to a
        loud warning + safe default instead of aborting; checkpoint=<path>
        enables JSONL resume across interrupted runs."""
        return _run(
            cmd_score,
            model=model,
            max_candidates=max_candidates,
            top=top,
            max_pairs=max_pairs,
            annotate=annotate,
            strategy=strategy,
            self_consistency=self_consistency,
            calibrate=calibrate,
            checkpoint=checkpoint,
            resilient=resilient,
        )

    @mcp.tool()
    def panel(
        model: str = "",
        max_candidates: int = cfg.CANDIDATE_MAX_DEFAULT,
        n_calls: int = 20,
        resilient: bool = False,
        checkpoint: str = "",
    ) -> str:
        """Run the panel consensus scorer (4-model config, weighted median
        aggregation, spread-triggered debate + abstention).
        ⚠️ COSTS REAL API BUDGET: ~$2-3 (single model) or ~$6-9 (full panel).
        Requires ANTHROPIC_API_KEY (or OPENROUTER_API_KEY).
        resilient=True + checkpoint=<path>: production-mode degradation +
        JSONL resume (same semantics as the score tool)."""
        return _run(
            cmd_panel,
            model=model,
            max_candidates=max_candidates,
            n_calls=n_calls,
            resilient=resilient,
            checkpoint=checkpoint,
        )

    @mcp.tool()
    def rlm(
        max_candidates: int = cfg.CANDIDATE_MAX_DEFAULT,
        top: int = 10,
        annotate: bool = True,
        calibrate: bool = True,
        self_consistency: int = 0,
        resilient: bool = False,
        checkpoint: str = "",
    ) -> str:
        """Run RLM variant scoring with sandbox domain tools (mechanism_lookup,
        known_mva_gene, gene_prior_fraction). ⚠️ COSTS REAL API BUDGET: ~$2-3
        per run. Requires ANTHROPIC_API_KEY (or OPENROUTER_API_KEY).
        self_consistency defaults to 0 (off) — each sample costs an LM call;
        resilient/checkpoint same semantics as score."""
        return _run(
            _cmd_rlm,
            max_candidates=max_candidates,
            top=top,
            annotate=annotate,
            self_consistency=self_consistency,
            calibrate=calibrate,
            resilient=resilient,
            checkpoint=checkpoint,
            model="",
            max_pairs=10,
            strategy="mixed",
        )

    @mcp.tool()
    def optimize(
        n: int = 12,
        max_metric_calls: int = 30,
        show_src: bool = False,
    ) -> str:
        """Run GEPA optimization (reflection + gradient-free search).
        ⚠️ COSTS REAL API BUDGET: ~$0.85-1.30 (GEPA light) depending on metric_calls.
        Requires ANTHROPIC_API_KEY (or OPENROUTER_API_KEY)."""
        return _run(cmd_optimize, n=n, max_metric_calls=max_metric_calls, show_src=show_src)

    @mcp.tool()
    def track2(
        mechanism: str = "spindle assembly checkpoint failure / chromosome missegregation",
        gene: str = "BUB1B",
        hpo: str = "HP:0001510 Growth delay,HP:0000252 Microcephaly",
    ) -> str:
        """Run Track 2 ReActV2 drug repurposing agent.
        ⚠️ COSTS REAL API BUDGET: ~$0.10-0.15 (10 iterations).
        Requires ANTHROPIC_API_KEY (or OPENROUTER_API_KEY)."""
        return _run(cmd_track2, mechanism=mechanism, gene=gene, hpo=hpo)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main():
    """Run the MCP server in stdio mode."""
    _register_tools()
    mcp = _get_mcp()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

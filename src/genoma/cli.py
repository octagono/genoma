"""CLI for the DSPy pipeline.

Commands:
  ingest       Download the gated dataset (needs HF_TOKEN)
  hpo          Extract HPO terms from the phenotype docx
  demo         Run the full Track 1 pipeline on synthetic data offline
  score        Score candidates (Predict + RLM + compound-het pairs)
  panel        Run the heterogeneous consensus panel
  optimize     Optimize the scorer with GEPA (synthetic trainset)
  track2       Run the ReActV2 drug-repurposing agent
  compliance   Print the data-deletion checklist
  purge-cache  Purge DSPy memory/disk caches
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# NixOS quirk: tokenizers/transformers (via litellm) need libstdc++.so.6 on the
# loader path. Pick the newest gcc lib from the nix store if present.
if "LD_LIBRARY_PATH" not in os.environ or "libstdc++" not in os.environ.get("LD_LIBRARY_PATH", ""):
    import glob

    _libs = sorted(
        glob.glob("/nix/store/*gcc-*-lib/lib/libstdc++.so.6"),
        reverse=True,
    )
    if _libs:
        _libdir = os.path.dirname(_libs[0])
        os.environ["LD_LIBRARY_PATH"] = _libdir + (
            os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
        )

import dspy

from genoma import config as cfg
from genoma.lm import configure_defaults
from genoma.track1.scoring import (
    CompoundHetPairScorer,
    PredictVariantScorer,
    RLMVariantScorer,
)


def _require_real_data():
    from genoma.data.ingest import vcf_path

    if not vcf_path().exists():
        raise SystemExit(
            "Real data not present. Run `genoma ingest` first (needs HF_TOKEN), "
            "or use `genoma demo` for offline development."
        )


def cmd_ingest(args) -> None:
    from genoma.data.ingest import download_dataset

    dest = download_dataset(files=args.files, force=args.force)
    print(f"Dataset files in {dest}")


def cmd_hpo(args) -> None:
    from genoma.data.ingest import extract_hpo_terms

    terms = extract_hpo_terms()
    print("\n".join(terms) if terms else "(no HPO terms found)")


def cmd_demo(args) -> None:
    """Offline end-to-end Track 1 smoke run on synthetic data."""
    from genoma.data.demo import DEMO_HPO, demo_candidates
    from genoma.data.submission import write_submission
    from genoma.lm import default_adapter
    from genoma.track1.consensus import PanelConsensus

    adapter = default_adapter()
    answer = {"reasoning": "demo reasoning", "epcr": 0.5, "rationale": "demo rationale"}
    dspy.configure(
        lm=dspy.utils.DummyLM([answer] * 200, adapter=adapter),
        adapter=adapter,
    )
    print(f"Demo mode: DummyLM configured ({type(adapter).__name__}, no API calls).")

    candidates = demo_candidates()
    print(f"{len(candidates)} synthetic candidates loaded")
    print("HPO:", DEMO_HPO)

    # Per-variant Predict scoring.
    scorer = PredictVariantScorer()
    scores = []
    for c in candidates:
        out = scorer(c, DEMO_HPO)
        scores.append((c, out))
    print("\nPredictVariantScorer (DummyLM — shape only):")
    for c, out in scores[:3]:
        print(f"  {c.coord:24s} -> epcr={getattr(out, 'epcr', 'n/a')}")

    # RLM needs a real LM to emit sandboxed code — DummyLM can't. Gate on an
    # explicit check (no try/except): if a real LM is configured, run it.
    if not isinstance(dspy.settings.lm, dspy.utils.DummyLM):
        rlm = RLMVariantScorer()
        res = rlm(candidates, DEMO_HPO)
        print(f"\nRLMVariantScorer ranked {len(getattr(res, 'ranked', []))} variants")
    else:
        print("\nRLMVariantScorer skipped in demo (DummyLM can't emit sandboxed code)")

    # Compound-het pairs (works with DummyLM — fails fast if broken).
    pairs = CompoundHetPairScorer(max_pairs=5)
    pr = pairs(candidates, DEMO_HPO)
    print(f"CompoundHetPairScorer: {len(getattr(pr, 'pairs', []))} pairs scored")

    # Consensus panel (DummyLM members in demo — no API keys needed).
    def _dummy_factory(slug: str) -> dspy.LM:
        return dspy.utils.DummyLM(
            [{"reasoning": "r", "epcr": 0.5, "rationale": "demo"}] * 200,
            adapter=adapter,
        )

    panel = PanelConsensus(
        panel={cfg.PRIMARY_LM: 1.0},
        lm_factory=_dummy_factory,
    )
    ranked = panel.score_candidates(candidates[:3], DEMO_HPO)
    print(f"\nPanel consensus ranked {len(ranked)} (median EPCR demo)")

    # Submission CSV: the compound-het pair in ONE row is the full-match shape.
    from genoma.data.demo import DEMO_CAUSAL_PAIR
    from genoma.data.submission import pair_row, single_variant_row
    from genoma.track1.metrics import AnswerKey, score_submission

    rows = [
        pair_row(
            "chr2",
            47_908_000,
            "G",
            "A",
            "chr2",
            47_912_000,
            "C",
            "T",
            0.95,
            note="compound-het pair (full-match shape)",
        ),
        single_variant_row("chr2", 47_908_000, "G", "A", 0.90),
        single_variant_row("chr2", 47_912_000, "C", "T", 0.85),
    ]
    out = write_submission(rows, cfg.SUBMISSION_DIR / "demo_submission.csv")
    print(f"\nDemo submission written: {out}")

    answer = AnswerKey(
        causal_variants={f"{r.chrom_1}:{r.pos_1}:{r.ref_1}>{r.alt_1}" for r in DEMO_CAUSAL_PAIR},
        compound_het=True,
        note="demo synthetic answer key",
    )
    print("\nLocal metric check (demo submission vs synthetic answer):")
    print(" ", score_submission(rows, answer))


def _score_candidates_parallel(
    scorer, candidates, hpo, *, num_threads: int = 8, checkpoint_path="", resilient=False
):
    """Score candidates through a PredictVariantScorer in parallel.

    Fail-fast by default (a broken prediction aborts loudly — the old path
    silently dropped failed candidates, which could hide the true causal
    pair). With ``resilient=True`` (production mode), per-candidate failures
    degrade to a loud warning + conservative default and the batch continues;
    ``checkpoint_path`` enables JSONL resume across interrupted runs.
    """
    from genoma.checkpoint import score_chunked

    def _score_one(candidate, hpo_terms):
        out = scorer(candidate, hpo_terms)
        return {"epcr": float(out.epcr), "rationale": str(out.rationale)}

    def _default_record(candidate, err: str) -> dict:
        return {"epcr": 0.0, "rationale": f"SAFE-DEFAULT: {err}"}

    records = score_chunked(
        candidates,
        hpo,
        _score_one,
        _default_record,
        checkpoint_path=checkpoint_path or None,
        num_threads=num_threads,
        resilient=resilient,
    )
    return [
        (c, dspy.Prediction(epcr=rec["epcr"], rationale=rec["rationale"])) for c, rec in records
    ]


def _apply_priors_and_calibrate(
    ranked, candidates_by_coord, *, self_consistency: int = 0, scorer=None, hpo=None
):
    """Gene prior + optional self-consistency mean EPCR for the top candidates."""
    from genoma.data.annotate import apply_gene_prior
    from genoma.track1.calibrate import self_consistency_epcr

    adjusted = []
    # Self-consistency is for calibration of the rows that will actually be
    # submitted — applying it to all candidates is 3x the scoring cost for no
    # ranking benefit. Cap the re-sampling to the submission-row budget.
    sc_scope = min(len(ranked), cfg.MAX_SUBMISSION_ROWS * 3) if self_consistency > 0 else 0
    for i, (c, out) in enumerate(ranked):
        epcr = float(out.epcr)
        if i < sc_scope and scorer is not None and hpo is not None:
            mean, _ = self_consistency_epcr(scorer, c, hpo, n=self_consistency)
            epcr = mean
        epcr = apply_gene_prior(epcr, c.gene)
        out.epcr = epcr
        adjusted.append((c, out))
    adjusted.sort(key=lambda x: x[1].epcr, reverse=True)
    return adjusted


def cmd_score(args) -> None:
    _require_real_data()
    configure_defaults()
    from genoma.data.annotate import build_candidate_table, enrich_via_vep_rest
    from genoma.data.ingest import extract_hpo_terms
    from genoma.data.submission import build_submission_rows, write_submission
    from genoma.track1.scoring import CompoundHetPairScorer

    hpo = extract_hpo_terms()
    candidates = build_candidate_table(max_candidates=args.max_candidates)
    if args.annotate:
        candidates = enrich_via_vep_rest(candidates)
    print(
        f"{len(candidates)} candidates; {len(hpo)} HPO terms"
        + (" (VEP-annotated)" if args.annotate else "")
    )

    scorer = PredictVariantScorer()
    ranked = _score_candidates_parallel(
        scorer,
        candidates,
        hpo,
        checkpoint_path=args.checkpoint,
        resilient=args.resilient,
    )
    ranked = _apply_priors_and_calibrate(
        ranked,
        {c.coord: c for c in candidates},
        self_consistency=args.self_consistency,
        scorer=scorer,
        hpo=hpo,
    )
    print("\nTop candidates:")
    for c, out in ranked[: args.top]:
        print(f"  {c.coord:24s} gene={c.gene:8s} epcr={out.epcr:.3f}")

    # Compound-het pair scoring (the causal mechanism is a pair). Fails fast.
    # Feed the per-variant EPCRs into the pair LM so a strong single-variant
    # signal (e.g. a known-gene stop_gained scoring 0.85) is not discarded
    # when the pair is re-judged from scratch.
    pair_scorer = CompoundHetPairScorer(max_pairs=args.max_pairs)
    single_epcrs = {c.coord: float(out.epcr) for c, out in ranked}
    pairs_res = pair_scorer(candidates, hpo, variant_epcrs=single_epcrs)
    pairs = pairs_res.pairs
    if pairs:
        print("\nTop compound-het pairs:")
        for p in pairs[: min(args.top, len(pairs))]:
            print(
                f"  {p.variant_1.chrom}:{p.variant_1.pos} + "
                f"{p.variant_2.chrom}:{p.variant_2.pos} "
                f"epcr={p.epcr:.3f}"
            )

    if args.rlm:
        rlm = RLMVariantScorer()
        res = rlm(candidates, hpo)
        print(f"\nRLM ranked {len(res.ranked)}")

    # Build submission (pairs + singles per strategy) with calibrated EPCRs.
    from genoma.track1.calibrate import calibrate_submission_epcrs

    singles = [ScoredVariantFromPredict(c, out) for c, out in ranked]
    pairs, singles = calibrate_submission_epcrs(pairs, singles)
    rows = build_submission_rows(singles, pairs, strategy=args.strategy)
    out_path = write_submission(rows, args.out)
    print(f"\nSubmission written: {out_path} ({len(rows)} rows, strategy={args.strategy})")

    if args.validate:
        _validate_rows(rows)


def ScoredVariantFromPredict(c, out):
    """Shim: adapt (candidate, PredictVariantScorer.prediction) to ScoredVariant-like."""
    from genoma.track1.scoring import ScoredVariant

    return ScoredVariant(
        chrom=c.chrom,
        pos=c.pos,
        ref=c.ref,
        alt=c.alt,
        epcr=float(out.epcr),
        rationale=str(out.rationale),
    )


def _validate_rows(rows, answer=None) -> dict:
    """Score submission rows against a (synthetic) answer key; print + return."""
    from genoma.data.demo import DEMO_CAUSAL_PAIR
    from genoma.track1.metrics import AnswerKey, score_submission

    if answer is None:
        answer = AnswerKey(
            causal_variants={
                f"{r.chrom_1}:{r.pos_1}:{r.ref_1}>{r.alt_1}" for r in DEMO_CAUSAL_PAIR
            },
            compound_het=True,
            note="demo synthetic answer key",
        )
    result = score_submission(rows, answer)
    print(
        f"  LOCAL SCORE -> rank_points={result['rank_points']:.1f} "
        f"f_max={result['f_max']:.3f} combined={result['combined']:.3f} "
        f"({result['rows']} rows, compound_het={result['compound_het']})"
    )
    return result


def cmd_panel(args) -> None:
    _require_real_data()
    configure_defaults()
    from genoma.data.annotate import build_candidate_table
    from genoma.data.ingest import extract_hpo_terms
    from genoma.data.submission import write_submission
    from genoma.track1.consensus import PanelConsensus

    hpo = extract_hpo_terms()
    candidates = build_candidate_table(max_candidates=args.max_candidates)
    panel = PanelConsensus()
    ranked = panel.score_candidates(
        candidates, hpo, checkpoint_path=args.checkpoint, resilient=args.resilient
    )
    print(f"Panel members: {list(panel.members)}")
    for v in ranked[: args.top]:
        flag = "DEBATED" if v.debated else ""
        print(f"  {v.chrom}:{v.pos} {v.ref}>{v.alt} epcr={v.epcr:.3f} {flag}")

    # Build submission (pairs + singles per strategy) and persist.
    from genoma.data.submission import build_submission_rows
    from genoma.track1.scoring import CompoundHetPairScorer

    pair_scorer = CompoundHetPairScorer(max_pairs=args.max_pairs)
    # score_candidates returns ScoredVariant objects (epcr attached), not
    # (candidate, output) tuples — key the map by the same coord format the
    # pair scorer uses (chrom:pos:ref>alt).
    single_epcrs = {f"{v.chrom}:{v.pos}:{v.ref}>{v.alt}": v.epcr for v in ranked}
    pairs_res = pair_scorer(candidates, hpo, variant_epcrs=single_epcrs)
    pairs = pairs_res.pairs
    from genoma.track1.calibrate import calibrate_submission_epcrs

    pairs, ranked = calibrate_submission_epcrs(pairs, ranked)
    rows = build_submission_rows(ranked, pairs, strategy=args.strategy)
    out_path = write_submission(rows, args.out)
    print(f"\nSubmission written: {out_path} ({len(rows)} rows, strategy={args.strategy})")

    if args.validate:
        _validate_rows(rows)


def cmd_candidates(args) -> None:
    """List/filter the deterministic candidate pool (pre-scoring, read-only)."""
    from genoma.data.annotate import build_candidate_table

    cands = build_candidate_table(max_candidates=args.max_candidates)
    if args.gene:
        cands = [c for c in cands if args.gene.lower() in (c.gene or "").lower()]
    af_filtered = args.min_af > 0 or args.max_af < 1.0
    if af_filtered:
        cands = [c for c in cands if c.af is None or args.min_af <= c.af <= args.max_af]
    print(
        f"{len(cands)} candidates"
        + (f" for gene '{args.gene}'" if args.gene else "")
        + (" (AF-filtered)" if af_filtered else "")
    )
    for c in cands[: args.top]:
        af = f"{c.af:.3g}" if c.af is not None else "-"
        cadd = f"{c.cadd:.1f}" if c.cadd is not None else "-"
        revel = f"{c.revel:.2f}" if c.revel is not None else "-"
        am = f"{c.alphamissense:.2f}" if c.alphamissense is not None else "-"
        print(
            f"{c.coord}  gene={c.gene or '-'}  cons={c.consequence}  "
            f"AF={af}  CADD={cadd}  REVEL={revel}  AM={am}"
        )
    if len(cands) > args.top:
        print(f"... ({len(cands) - args.top} more)")


def cmd_annotate(args) -> None:
    """Annotate candidate variants via Ensembl VEP REST (key-free)."""
    from genoma.data.annotate import build_candidate_table, enrich_via_vep_rest
    from genoma.data.ingest import extract_hpo_terms

    extract_hpo_terms()  # ensure phenotype present (error path surfaces data need)
    candidates = build_candidate_table(max_candidates=args.max_candidates)
    print(f"{len(candidates)} candidates before VEP")
    enriched = enrich_via_vep_rest(candidates)
    n_gene = sum(1 for c in enriched if c.gene)
    n_af = sum(1 for c in enriched if c.af is not None)
    n_cadd = sum(1 for c in enriched if c.cadd is not None)
    print(
        f"After VEP: {len(enriched)} candidates | gene={n_gene} af={n_af} "
        f"cadd={n_cadd} (fields populated)"
    )
    for c in enriched[: args.top]:
        print(
            f"  {c.coord:24s} gene={c.gene or 'NA':8s} AF={c.af if c.af is not None else 'NA'} "
            f"CADD={c.cadd if c.cadd is not None else 'NA'}"
        )

    if args.out:
        import json

        out = [{**c.model_dump(), "coord": c.coord} for c in enriched]
        args.out.parent.mkdir(parents=True, exist_ok=True) if args.out else None
        args.out.write_text(json.dumps(out, indent=2))
        print(f"Annotated candidates written: {args.out}")


def cmd_validate(args) -> None:
    """Score an existing submission CSV against a synthetic answer key."""
    from genoma.data.submission import load_submission

    rows = load_submission(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")
    _validate_rows(rows)


def cmd_planner(args) -> None:
    """Submission strategy planner: record feedback, re-weight, suggest pairs."""
    from genoma.data.demo import demo_candidates
    from genoma.data.submission import load_submission
    from genoma.track1.strategy import SubmissionPlanner

    planner = (
        SubmissionPlanner.load(args.state) if Path(args.state).exists() else SubmissionPlanner()
    )

    if args.record and args.matched:
        rows = load_submission(args.record)
        matched = args.matched.split(",")
        planner.record(
            rows,
            [m.strip() for m in matched if m.strip()],
            note=args.note or "live feedback",
        )
        planner.save(args.state)
        print(
            f"Recorded submission ({len(rows)} rows, {len(matched)} matched). State saved to {args.state}."
        )

    # Seed candidates from demo (real data would come from build_candidate_table).
    candidates = demo_candidates()
    planner.set_candidates(candidates)
    pairs = planner.suggest_pairs(candidates, n=args.top)
    print(f"\nSuggested compound-het pairs for next submission (top {len(pairs)}):")
    for a, b in pairs:
        print(
            f"  {a.coord} + {b.coord} "
            f"(combined epcr={planner.epcr.get(a.coord, 0):.2f}+{planner.epcr.get(b.coord, 0):.2f})"
        )
    print(f"\nSubmission history: {len(planner.history)} record(s)")
    if planner.history:
        last = planner.history[-1]
        print(f"  last: {len(last.rows)} rows, matched={last.matched}")


def cmd_optimize(args) -> None:
    configure_defaults()
    from genoma.optimize.gepa_optimize import (
        build_synthetic_trainset,
        optimize_with_gepa,
    )
    from genoma.track1.scoring import PredictVariantScorer

    trainset = build_synthetic_trainset(n=args.n)
    scorer = PredictVariantScorer()
    optimized = optimize_with_gepa(
        scorer,
        trainset,
        auto=cfg.GEPA_AUTO,
        out_path=args.out,
        judge_classification_weight=args.judge_cls_weight,
    )
    print(f"Optimized program saved to {args.out}")
    if args.show_src and hasattr(optimized, "module_src"):
        print(optimized.module_src)


def cmd_track2(args) -> None:
    configure_defaults()
    from genoma.track2.agent import Track2Agent
    from genoma.track2.report import generate_track2_methods_report, generate_track2_report

    agent = Track2Agent()
    res = agent(
        mechanism=args.mechanism,
        gene=args.gene,
        hpo_terms=args.hpo.split(","),
    )
    for cand in res.drug_candidates:
        print(f"- {cand.name}: {cand.mechanism} ({cand.evidence})")
    print(f"termination: {res.termination_reason}")
    out = generate_track2_report(
        res.drug_candidates,
        mechanism=args.mechanism,
        gene=args.gene,
        hpo_terms=args.hpo.split(","),
        termination_reason=res.termination_reason,
        stall_diagnosis=getattr(res, "stall_diagnosis", ""),
    )
    print(f"Report written: {out}")
    methods_out = generate_track2_methods_report(
        drug_candidates=res.drug_candidates,
        mechanism=args.mechanism,
        gene=args.gene,
        hpo_terms=args.hpo.split(","),
        team_name=getattr(args, "team", ""),
    )
    print(f"Methods report written: {methods_out}")


def cmd_evaluate(args) -> None:
    """Score a saved/compiled program on the synthetic devset with the judge."""
    configure_defaults()
    from genoma.optimize.gepa_optimize import (
        build_synthetic_trainset,
        evaluate_with_judge,
    )

    devset = build_synthetic_trainset(n=args.n)
    program = dspy.load(args.program, allow_pickle=True) if args.program else None
    if program is None:
        from genoma.track1.scoring import PredictVariantScorer

        program = PredictVariantScorer()
    score = evaluate_with_judge(
        program,
        devset,
        num_threads=args.threads,
        judge_classification_weight=args.judge_cls_weight,
    )
    print(f"Judge score on {len(devset)} devset examples: {score:.3f}")


def cmd_phen2gene(args) -> None:
    """Rank MVA genes by phenotype fit via the Phen2Gene REST API (fail-fast)."""
    from genoma.data.ingest import extract_hpo_terms
    from genoma.data.phen2gene import rank_mva_genes

    hpo = extract_hpo_terms()
    ranked = rank_mva_genes(hpo)
    print(f"Phenotype ranking of the 6 MVA genes (HPO terms: {len(hpo)}):")
    for gene, score in sorted(ranked.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(score * 40)
        print(f"  {gene:10s} {score:.3f} {bar}")


def cmd_exomiser(args) -> None:
    """Run Exomiser comp-het analysis on the VCF (external Java tool)."""
    from genoma.data.exomiser import (
        merge_exomiser_scores,
        run_exomiser,
        write_analysis_yaml,
    )
    from genoma.data.ingest import extract_hpo_terms, vcf_path

    if not vcf_path().exists():
        raise SystemExit(f"VCF not found at {vcf_path()} — run `genoma ingest` first.")
    hpo = extract_hpo_terms()
    if args.yaml_only:
        yaml_path = write_analysis_yaml(vcf_path(), hpo, cfg.ROOT / "exomiser")
        print(f"Analysis YAML written: {yaml_path} (run Exomiser manually)")
        return
    result = run_exomiser(vcf_path(), hpo, cfg.ROOT / "exomiser")
    genes = result["genes"]
    print(f"Exomiser scored {len(genes)} genes. Comp-het genes:")
    for gene, entry in sorted(genes.items(), key=lambda kv: -kv[1]["exomiser_score"])[: args.top]:
        flag = "COMPHET" if entry["comp_het"] else ""
        print(
            f"  {gene:10s} score={entry['exomiser_score']:.3f} "
            f"pheno={entry['phenotype_score']:.3f} var={entry['variant_score']:.3f} {flag}"
        )
    # Couple the gene-level scores onto the candidate table so downstream
    # RLM/panel scoring can reason over Exomiser's ranking as a feature.
    if genes:
        from genoma.data.annotate import build_candidate_table

        candidates = build_candidate_table(max_candidates=args.max_candidates)
        merged = merge_exomiser_scores(candidates, result)
        n_annot = sum(1 for c in merged if c.exomiser_score is not None)
        print(f"Coupled Exomiser scores onto {n_annot}/{len(merged)} candidates.")


def cmd_phase(args) -> None:
    """Phase-aware analysis: trans/cis from VCF tags, or emit the WhatsHap pipeline."""
    from genoma.data.annotate import build_candidate_table
    from genoma.data.ingest import load_vcf_records, vcf_path
    from genoma.data.phasing import phase_pairs, trio_phase_pairs, whatshap_pipeline

    if args.pipeline:
        fastqs = sorted((cfg.DATA_DIR).glob("*.fastq.gz"))
        if not fastqs:
            raise SystemExit(f"No FASTQs in {cfg.DATA_DIR} — run `genoma ingest` with files.")
        script = whatshap_pipeline(fastqs, vcf_path(), cfg.ROOT / "phasing", region=args.region)
        print(script)
        return

    # VCF-tag-based phasing (PS/HP fields, if the provider phased).
    records = load_vcf_records()
    candidates = build_candidate_table(max_candidates=args.max_candidates)
    statuses = phase_pairs(candidates, records)

    # Trio Mendelian phasing: deterministic from parental genotypes when the
    # VCF carries family samples (trans = alts from different parents). No
    # phase tags needed — overrides tag-based when informative.
    trio = trio_phase_pairs(candidates, records)
    statuses = {**statuses, **trio}

    if not statuses:
        print(
            "No phase tags (PS/HP) and no family samples in the VCF. Use "
            "`genoma phase --pipeline` to emit the WhatsHap read-backed "
            "phasing pipeline (needs minimap2/samtools/whatshap)."
        )
        return
    for key, ph in sorted(statuses.items()):
        print(f"  {key}  ->  {ph.status.upper():8s} ({ph.evidence or 'no evidence'})")


def cmd_report(args) -> None:
    """Generate the Track 1 methods report (Markdown) for a submission."""
    from genoma.data.ingest import extract_hpo_terms
    from genoma.track1.report import generate_track1_report

    hpo = extract_hpo_terms()
    out = generate_track1_report(
        submission_csv=args.csv,
        hpo_terms=hpo,
        panel_members=list(args.panel) or None,
        out_path=args.out or (cfg.SUBMISSION_DIR / "track1_report.md"),
        team_name=args.team,
    )
    print(
        f"Report written ({len(out)} chars): {args.out or cfg.SUBMISSION_DIR / 'track1_report.md'}"
    )


def cmd_compliance(args) -> None:
    from genoma.compliance import deletion_checklist

    print(deletion_checklist())


def cmd_purge_cache(args) -> None:
    from genoma.compliance import purge_caches

    result = purge_caches(
        confirm=not args.yes,
        delete_disk=not args.keep_disk,
        delete_data=not args.keep_data,
    )
    print(f"Purged: {result}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MVA Hackathon DSPy pipeline")
    p.add_argument("--cost", action="store_true", help="report token usage after the command")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest", help="download gated dataset")
    sp.add_argument(
        "--files",
        nargs="*",
        default=[cfg.VCF_FILENAME, cfg.VCF_TBI_FILENAME, cfg.PHENOTYPE_FILENAME],
    )
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_ingest)

    sub.add_parser("hpo", help="extract HPO terms").set_defaults(func=cmd_hpo)
    sub.add_parser("demo", help="offline Track 1 demo on synthetic data").set_defaults(
        func=cmd_demo
    )

    sp = sub.add_parser("score", help="score candidates (singles + compound-het pairs)")
    sp.add_argument("--max-candidates", type=int, default=cfg.CANDIDATE_MAX_DEFAULT)
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument(
        "--submit-top", type=int, default=10, help="rows to write to the submission CSV"
    )
    sp.add_argument(
        "--max-pairs", type=int, default=cfg.COMPHET_MAX_PAIRS, help="compound-het pairs to score"
    )
    sp.add_argument("--rlm", action="store_true", help="also run RLM scorer")
    sp.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enrich candidates via Ensembl VEP REST (default: on; --no-annotate to skip)",
    )
    sp.add_argument(
        "--strategy",
        default=cfg.SUBMISSION_STRATEGY,
        choices=("mixed", "pairs", "singles"),
        help="row strategy: pairs first (compound-het expected), then singles",
    )
    sp.add_argument(
        "--self-consistency",
        type=int,
        default=cfg.SELF_CONSISTENCY_N,
        help="N temperature=1.0 samples for the TOP candidates' EPCR calibration "
        "(applies to the submission-row budget only, not all 500 candidates — "
        "full-pool self-consistency is 3x the scoring cost and is not needed "
        "for ranking the top rows)",
    )
    sp.add_argument(
        "--validate",
        action="store_true",
        help="score the CSV against the synthetic answer key",
    )
    sp.add_argument("--out", default=str(cfg.SUBMISSION_DIR / "score_submission.csv"))
    sp.add_argument(
        "--checkpoint",
        default="",
        help="JSONL checkpoint path (resume from interrupted run). Default: "
        "submission/score_checkpoint.jsonl when --resilient is set.",
    )
    sp.add_argument(
        "--resilient",
        action="store_true",
        help="PRODUCTION MODE: per-candidate failures degrade to a loud warning "
        "+ conservative safe-default instead of aborting the run; enables "
        "checkpointing. Fail-fast remains the default.",
    )
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("panel", help="heterogeneous consensus panel")
    sp.add_argument("--max-candidates", type=int, default=cfg.CANDIDATE_MAX_DEFAULT)
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument(
        "--submit-top", type=int, default=10, help="rows to write to the submission CSV"
    )
    sp.add_argument("--max-pairs", type=int, default=cfg.COMPHET_MAX_PAIRS)
    sp.add_argument(
        "--strategy",
        default=cfg.SUBMISSION_STRATEGY,
        choices=("mixed", "pairs", "singles"),
    )
    sp.add_argument(
        "--validate",
        action="store_true",
        help="score the CSV against the synthetic answer key",
    )
    sp.add_argument("--out", default=str(cfg.SUBMISSION_DIR / "panel_submission.csv"))
    sp.add_argument(
        "--checkpoint",
        default="",
        help="JSONL checkpoint path (resume from interrupted run). Default: "
        "submission/panel_checkpoint.jsonl when --resilient is set.",
    )
    sp.add_argument(
        "--resilient",
        action="store_true",
        help="PRODUCTION MODE: per-candidate failures degrade to a loud warning "
        "+ conservative safe-default (abstained) instead of aborting; enables "
        "checkpointing. Fail-fast remains the default.",
    )
    sp.set_defaults(func=cmd_panel)

    sp = sub.add_parser(
        "candidates", help="list/filter the deterministic candidate pool (read-only)"
    )
    sp.add_argument("--max-candidates", type=int, default=cfg.CANDIDATE_MAX_DEFAULT)
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--gene", default="", help="filter by gene (substring)")
    sp.add_argument("--min-af", type=float, default=0.0)
    sp.add_argument("--max-af", type=float, default=1.0)
    sp.set_defaults(func=cmd_candidates)

    sp = sub.add_parser("annotate", help="enrich candidate variants via Ensembl VEP REST")
    sp.add_argument("--max-candidates", type=int, default=cfg.CANDIDATE_MAX_DEFAULT)
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--out", default="", help="write annotated candidates JSON to this path")
    sp.set_defaults(func=cmd_annotate)

    sp = sub.add_parser("validate", help="score a submission CSV against the synthetic answer key")
    sp.add_argument("csv", help="path to a submission CSV")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("planner", help="submission strategy planner (active learning)")
    sp.add_argument("--record", default="", help="submission CSV to record with live feedback")
    sp.add_argument(
        "--matched",
        default="",
        help="comma-separated variant coords the leaderboard confirmed",
    )
    sp.add_argument("--note", default="", help="note for this submission record")
    sp.add_argument("--state", default=str(cfg.SUBMISSION_DIR / "planner_state.json"))
    sp.add_argument("--top", type=int, default=5, help="how many suggested pairs to print")
    sp.set_defaults(func=cmd_planner)

    sp = sub.add_parser("optimize", help="GEPA optimize the scorer")
    sp.add_argument("--n", type=int, default=12, help="synthetic trainset size")
    sp.add_argument("--out", default=str(cfg.ROOT / "optimized_program"))
    sp.add_argument("--show-src", action="store_true")
    sp.add_argument(
        "--judge-cls-weight",
        type=float,
        default=0.7,
        help="judge metric weight on the binary causal call (calibration = 1-w); "
        "raise toward robust reasoning if noisy-data optimization degrades EPCR fit",
    )
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("track2", help="ReActV2 drug repurposing agent")
    sp.add_argument(
        "--mechanism",
        default="spindle assembly checkpoint failure / chromosome missegregation",
    )
    sp.add_argument("--gene", default="BUB1B")
    sp.add_argument("--team", default="", help="team/username for the methods report")
    from genoma.data.demo import DEMO_HPO as _demo_hpo

    sp.add_argument("--hpo", default=",".join(_demo_hpo))
    sp.set_defaults(func=cmd_track2)

    sp = sub.add_parser("evaluate", help="score a program on the devset with the judge")
    sp.add_argument(
        "--program",
        default="",
        help="path to a saved program (blank = fresh PredictVariantScorer)",
    )
    sp.add_argument("--n", type=int, default=8, help="devset size")
    sp.add_argument("--threads", type=int, default=4)
    sp.add_argument(
        "--judge-cls-weight",
        type=float,
        default=0.7,
        help="judge metric weight on the binary causal call (calibration = 1-w); "
        "must match the weight used at optimize time for comparable scores",
    )
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("phen2gene", help="rank MVA genes by phenotype fit (HPO -> gene)")
    sp.set_defaults(func=cmd_phen2gene)

    sp = sub.add_parser("exomiser", help="run Exomiser comp-het analysis (external Java tool)")
    sp.add_argument("--yaml-only", action="store_true", help="just write the analysis YAML")
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument(
        "--max-candidates",
        type=int,
        default=500,
        help="candidate-table size for coupling Exomiser scores onto variants",
    )
    sp.set_defaults(func=cmd_exomiser)

    sp = sub.add_parser("phase", help="trans/cis phasing for compound-het pairs")
    sp.add_argument(
        "--pipeline",
        action="store_true",
        help="emit the WhatsHap read-backed pipeline script",
    )
    sp.add_argument("--region", default="", help="region for WhatsHap (e.g. chr2:47.4Mb-48.4Mb)")
    sp.add_argument("--max-candidates", type=int, default=cfg.CANDIDATE_MAX_DEFAULT)
    sp.set_defaults(func=cmd_phase)

    sp = sub.add_parser("report", help="generate the Track 1 methods report (Markdown)")
    sp.add_argument("csv", help="path to the submission CSV")
    sp.add_argument("--out", default="", help="output path (default submissions/track1_report.md)")
    sp.add_argument("--team", default="", help="team/username for the report header")
    sp.add_argument(
        "--panel",
        nargs="*",
        default=[],
        help="consensus panel member slugs (report reflects the panel run; "
        "omit for a single-model run)",
    )
    sp.set_defaults(func=cmd_report)

    sub.add_parser("compliance", help="print data-deletion checklist").set_defaults(
        func=cmd_compliance
    )

    sp = sub.add_parser("purge-cache", help="purge DSPy caches + gated dataset + HF hub cache")
    sp.add_argument("--yes", action="store_true", help="skip confirmation")
    sp.add_argument("--keep-disk", action="store_true", help="keep DSPy disk cache")
    sp.add_argument("--keep-data", action="store_true", help="keep gated dataset + HF hub cache")
    sp.set_defaults(func=cmd_purge_cache)

    from genoma.cost import add_cost_subcommand

    add_cost_subcommand(sub, p)

    args = p.parse_args(argv)

    if args.cost:
        with dspy.track_usage() as tracker:
            args.func(args)
        totals = tracker.get_total_tokens()
        for lm, counts in totals.items():
            print(f"  tokens {lm}: {counts}")
    else:
        args.func(args)


if __name__ == "__main__":
    main()

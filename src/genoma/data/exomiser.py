"""Exomiser: phenotype-driven variant/comp-het scoring (external tool wrapper).

Exomiser 15.1.1 (data release 2602) jointly scores biallelic pairs under a
gene via AUTOSOMAL_RECESSIVE_COMP_HET. This wrapper:

- writes the Phenopacket-schema analysis YAML (current format; the legacy
  ``sampleId`` schema is gone),
- runs the Exomiser CLI if configured (EXOMISER_JAR env or exomiser on PATH),
- parses ``*.genes.tsv`` / ``*.variants.tsv`` into gene + comp-het scores,
- degrades gracefully (clear error) when the tool is absent.

Install: Java 21+, exomiser-cli zip + 2602_phenotype.zip (13.5 GB) +
2602_hg38.zip (23.6 GB) from data.monarchinitiative.org/exomiser.
Docker: exomiser/exomiser-cli (data must be mounted separately).
Run: java -Xmx12G -XX:+UseG1GC -jar exomiser-cli-*.jar analyse --analysis job.yml
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from genoma import config as cfg

# Phenopacket-schema analysis YAML (Exomiser 15.1.x current format).
ANALYSIS_TEMPLATE = """\
phenopacket:
  id: {sample_id}
  subject:
    id: {sample_id}
  phenotypicFeatures:
{hpo_features}
  htsFiles:
    - uri: file://{vcf}
      htsFormat: VCF
      genomeAssembly: {assembly}
analysis:
  analysisMode: PASS_ONLY
  inheritanceModes:
    AUTOSOMAL_RECESSIVE_COMP_HET: 2.0
    AUTOSOMAL_RECESSIVE_HOM_ALT: 0.1
  frequencySources: [THOUSAND_GENOMES, TOPMED, UK10K,
    GNOMAD_E_AFR, GNOMAD_E_AMR, GNOMAD_E_EAS, GNOMAD_E_NFE, GNOMAD_E_SAS,
    GNOMAD_G_AFR, GNOMAD_G_AMR, GNOMAD_G_EAS, GNOMAD_G_NFE, GNOMAD_G_SAS]
  pathogenicitySources: [REVEL, MVP, ALPHA_MISSENSE]
  steps: [
    failedVariantFilter: {{}},
    variantEffectFilter: {{remove: [FIVE_PRIME_UTR_EXON_VARIANT, THREE_PRIME_UTR_EXON_VARIANT,
      NON_CODING_TRANSCRIPT_EXON_VARIANT, CODING_TRANSCRIPT_INTRON_VARIANT,
      UPSTREAM_GENE_VARIANT, DOWNSTREAM_GENE_VARIANT, INTERGENIC_VARIANT, REGULATORY_REGION_VARIANT]}},
    frequencyFilter: {{maxFrequency: {max_freq_percent}}},
    pathogenicityFilter: {{keepNonPathogenic: true}},
    inheritanceFilter: {{}},
    omimPrioritiser: {{}},
    hiPhivePrioritiser: {{}}
  ]
outputOptions:
  outputContributingVariantsOnly: false
  numGenes: 50
  outputDirectory: {out_dir}
  outputFileName: {out_name}
  outputFormats: [TSV_GENE, TSV_VARIANT]
"""


def exomiser_jar() -> Path | None:
    """Locate the Exomiser CLI jar (EXOMISER_JAR env)."""
    env = os.environ.get("EXOMISER_JAR", "")
    if env and Path(env).exists():
        return Path(env)
    return None


def exomiser_available() -> bool:
    return exomiser_jar() is not None or shutil.which("exomiser") is not None


def write_analysis_yaml(
    vcf: Path | str,
    hpo_terms: list[str],
    out_dir: Path | str,
    *,
    sample_id: str = "PROBAND01",
    max_af: float = 0.01,
    candidate_genes: list[str] | None = None,
) -> Path:
    """Write the Phenopacket-schema analysis YAML for comp-het prioritization.

    Note: Exomiser frequency values are MAF PERCENT (2.0 = 2%); we convert
    from the pipeline's fraction (0.01 = 1%).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = []
    for t in hpo_terms:
        parts = t.split(None, 1)
        hid = parts[0].strip()
        if not hid.upper().startswith("HP:"):
            continue
        label = parts[1].strip().rstrip(",") if len(parts) > 1 else ""
        label_field = f", label: {label}" if label else ""
        features.append(f"    - type: {{id: '{hid}'{label_field}}}")
    if not features:
        features = ["    - type: {id: 'HP:0000252', label: Microcephaly}"]

    yaml_path = out_dir / "analysis.yaml"
    yaml_path.write_text(
        ANALYSIS_TEMPLATE.format(
            sample_id=sample_id,
            hpo_features="\n".join(features),
            vcf=str(Path(vcf).resolve()),
            assembly=cfg.EXOMISER_ASSEMBLY,
            max_freq_percent=max_af * 100.0,
            out_dir=str(out_dir.resolve()),
            out_name="exomiser_result",
        )
    )
    return yaml_path


def run_exomiser(
    vcf: Path | str,
    hpo_terms: list[str],
    out_dir: Path | str,
    *,
    max_af: float = 0.01,
    timeout: int = 3600,
) -> dict:
    """Run Exomiser comp-het analysis. Returns parsed gene/pair scores."""
    if not exomiser_available():
        raise RuntimeError(
            "Exomiser not configured. Install (https://exomiser.readthedocs.io): "
            "Java 21+, exomiser-cli zip + 2602_phenotype.zip (13.5GB) + 2602_hg38.zip "
            "(23.6GB) from data.monarchinitiative.org/exomiser/latest, then set "
            "EXOMISER_JAR=/path/to/exomiser-cli-*.jar. Docker: exomiser/exomiser-cli "
            "(mount the data dir)."
        )

    yaml_path = write_analysis_yaml(vcf, hpo_terms, out_dir, max_af=max_af)
    out_dir = Path(out_dir)
    jar = exomiser_jar()
    cmd = (
        [
            "java",
            "-Xmx12G",
            "-XX:+UseG1GC",
            "-jar",
            str(jar),
            "analyse",
            "--analysis",
            str(yaml_path),
        ]
        if jar
        else ["exomiser", "analyse", "--analysis", str(yaml_path)]
    )
    subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
    return parse_exomiser_output(out_dir)


def parse_exomiser_output(out_dir: Path | str) -> dict:
    """Parse Exomiser genes.tsv + variants.tsv into structured scores.

    Comp-het extraction: variants.tsv rows with MOI == 'AR' and
    CONTRIBUTING_VARIANT == 1, grouped by gene, are the biallelic pairs.
    """
    out_dir = Path(out_dir)
    genes_tsv = _find(out_dir, ".genes.tsv")
    variants_tsv = _find(out_dir, ".variants.tsv")

    genes: dict[str, dict] = {}
    if genes_tsv:
        for row in _read_tsv(genes_tsv):
            gene = row.get("GENE_SYMBOL", "").strip()
            if not gene:
                continue
            genes.setdefault(
                gene,
                {
                    "gene": gene,
                    "phenotype_score": _f(row.get("EXOMISER_GENE_PHENO_SCORE")),
                    "variant_score": _f(row.get("EXOMISER_VARIANT_SCORE")),
                    "exomiser_score": _f(row.get("EXOMISER_GENE_COMBINED_SCORE")),
                    "comp_het": "AR" in row.get("MOI", ""),
                    "variants": [],
                },
            )
    if variants_tsv:
        for row in _read_tsv(variants_tsv):
            gene = row.get("GENE_SYMBOL", "").strip()
            if not gene or gene not in genes:
                continue
            if row.get("MOI") == "AR" and row.get("CONTRIBUTING_VARIANT") == "1":
                genes[gene]["comp_het"] = True
                genes[gene]["variants"].append(row)
    return {"genes": genes}


def _find(out_dir: Path, suffix: str) -> Path | None:
    matches = sorted(out_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _read_tsv(path: Path):
    with open(path) as f:
        header = [h.lstrip("#").strip() for h in f.readline().split("\t")]
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            yield dict(zip(header, parts, strict=True))


def _f(v: str | None) -> float:
    if v in (None, "") or not str(v).replace(".", "", 1).isdigit():
        return 0.0
    return float(v)


def merge_exomiser_scores(candidates: list, parsed: dict) -> list:
    """Couple Exomiser's gene-level scores onto candidate variants.

    ``parsed`` is the dict from ``parse_exomiser_output`` (genes.tsv +
    variants.tsv). Matches by gene symbol; the gene-level combined/phenotype
    scores and the comp-het flag are copied onto every candidate in that
    gene so the RLM/panel can reason over Exomiser's specialized ranking
    alongside CADD/REVEL/AlphaMissense. Unmatched genes keep their fields.
    """
    genes = parsed.get("genes", {})
    out = []
    for c in candidates:
        g = genes.get(c.gene or "")
        if not g:
            out.append(c)
            continue
        out.append(
            c.model_copy(
                update={
                    "exomiser_score": g.get("exomiser_score"),
                    "exomiser_pheno_score": g.get("phenotype_score"),
                    "exomiser_comp_het": bool(g.get("comp_het")),
                }
            )
        )
    return out

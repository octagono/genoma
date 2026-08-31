"""Phasing: trans/cis determination for compound-het candidate pairs.

A true compound-het requires the two variants to be in TRANS (on opposite
homologous chromosomes). Cis pairs (same haplotype) are NOT compound-hets.

Determining phase with a single proband:
1. **VCF phase tags (if present)** — WhatsHap-style output phases GT with a
   pipe (``0|1`` / ``1|0``) plus a ``PS`` (phase set) FORMAT field; GATK
   style keeps ``0/1`` with ``HP`` haplotype tags. Both are parsed here:
   - same PS, phased GT with opposite haplotype-1 alleles -> TRANS
   - same PS, same haplotype -> CIS
   - otherwise -> unknown
2. **Read-backed phasing** — WhatsHap 2.8 on an alignment of the FASTQs.
   Alignment is required first (no BAM is provided); ``whatshap_pipeline()``
   emits the exact shell pipeline.
   NOTE: WhatsHap has NO ``--region`` option — the fast path is pre-subsetting
   the VCF (bcftools view -r) and BAM (samtools view) to the target genes.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PairPhase:
    """Phase status of two heterozygous variants in the same gene."""

    status: str  # "trans" | "cis" | "unknown"
    evidence: str = ""  # e.g. "PS:12345 GT:0|1/1|0" or "PS:12345 HP:1/2"


def _sample_fields(record: dict) -> dict:
    """Extract sample/FORMAT-level fields (GT, PS, HP) from a VCF record dict."""
    fmt = record.get("format") or record.get("sample") or {}
    if isinstance(fmt, dict):
        return {k.upper(): v for k, v in fmt.items()}
    return {}


def _gt_hap1(gt: str | None) -> int | None:
    """Haplotype-1 allele from a GT string ('0|1' -> 0, '1|0' -> 1, '0/1' -> None)."""
    if not gt or "|" not in str(gt):
        return None
    hap = str(gt).split("|")[0]
    if hap.isdigit():
        return int(hap)
    return None


def phase_status(rec_a: dict, rec_b: dict) -> PairPhase:
    """Trans/cis call from PS/HP/GT phase tags (WhatsHap or GATK style).

    Rules (WhatsHap default ``--tag=PS`` output):
    - same PS, phased GT (0|1 / 1|0): opposite haplotype-1 alleles -> TRANS
    - same PS, phased GT with the same haplotype-1 allele -> CIS
    GATK-style (``--tag=HP``): same PS, different HP -> trans; same HP -> cis.
    - No tags / unphased -> unknown.
    """
    fa, fb = _sample_fields(rec_a), _sample_fields(rec_b)
    ps_a, ps_b = fa.get("PS"), fb.get("PS")
    hp_a, hp_b = fa.get("HP"), fb.get("HP")
    gt_a, gt_b = fa.get("GT"), fb.get("GT")

    same_ps = ps_a is not None and ps_b is not None and str(ps_a) == str(ps_b)

    # WhatsHap style: phased GT.
    h1a, h1b = _gt_hap1(gt_a), _gt_hap1(gt_b)
    if same_ps and h1a is not None and h1b is not None:
        if h1a != h1b:
            return PairPhase("trans", f"PS:{ps_a} GT:{gt_a}/{gt_b}")
        return PairPhase("cis", f"PS:{ps_a} GT:{gt_a}/{gt_b}")

    # GATK style: HP tags.
    if same_ps and hp_a is not None and hp_b is not None:
        if str(hp_a) != str(hp_b):
            return PairPhase("trans", f"PS:{ps_a} HP:{hp_a}/{hp_b}")
        return PairPhase("cis", f"PS:{ps_a} HP:{hp_a}")

    return PairPhase("unknown")


def parse_phase_sets(
    records: list[dict],
) -> dict[tuple[tuple[str, int], tuple[str, int]], PairPhase]:
    """Group records by phase set; return pairwise status for co-phased hets."""
    by_ps: dict[str, list[dict]] = {}
    for r in records:
        f = _sample_fields(r)
        ps = f.get("PS")
        if ps is not None:
            by_ps.setdefault(str(ps), []).append(r)

    out: dict[tuple[tuple[str, int], tuple[str, int]], PairPhase] = {}
    for recs in by_ps.values():
        if len(recs) < 2:
            continue
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                ka = (a["chrom"], int(a["pos"]))
                kb = (b["chrom"], int(b["pos"]))
                key = (ka, kb) if ka <= kb else (kb, ka)
                out[key] = phase_status(a, b)
    return out


# ── Trio-based Mendelian phasing (family VCFs only) ─────────────────────────
# When the VCF carries parental genotypes (a family trio), physical phase
# tags are unnecessary: Mendelian inheritance phases the proband's variants
# deterministically.


def _alleles(gt: str | None) -> tuple[int, int] | None:
    """Normalize a GT to (a, b) allele indices; None if unphased/missing."""
    if not gt:
        return None
    parts = str(gt).replace("|", "/").split("/")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def _is_het(gt: str | None) -> bool:
    ab = _alleles(gt)
    return ab is not None and ab[0] != ab[1]


def _is_hom_ref(gt: str | None) -> bool:
    ab = _alleles(gt)
    return ab is not None and ab[0] == ab[1] == 0


def _carrier_parent(proband: str, mother: str, father: str) -> str | None:
    """Which parent transmitted the proband's rare allele (0=ref, 1=alt).

    Assumes a biallelic rare variant: the proband is het (0/1). If exactly one
    parent is het and the other is homozygous ref, the het parent transmitted
    the alt allele. Returns "mother" / "father" / None (uninformative).
    """
    if not _is_het(proband):
        return None
    if _is_het(mother) and _is_hom_ref(father):
        return "mother"
    if _is_het(father) and _is_hom_ref(mother):
        return "father"
    return None  # both het / both hom-ref / missing: uninformative


def mendelian_phase(
    proband_a: str,
    mother_a: str,
    father_a: str,
    proband_b: str,
    mother_b: str,
    father_b: str,
) -> PairPhase:
    """Deterministic trans/cis call from trio genotypes (no phase tags needed).

    Rule: a compound-het pair is TRANS when the two alt alleles were inherited
    from DIFFERENT parents (maternal + paternal). Both from the same parent is
    uninformative for phase (could be same or opposite homolog) -> unknown.

    Returns:
        PairPhase("trans", evidence)  when alt_a came from one parent and
                                      alt_b from the other
        PairPhase("unknown")          when inheritance is uninformative
    """
    parent_a = _carrier_parent(proband_a, mother_a, father_a)
    parent_b = _carrier_parent(proband_b, mother_b, father_b)
    if parent_a and parent_b and parent_a != parent_b:
        return PairPhase(
            "trans",
            f"trio: {parent_a}-inherited + {parent_b}-inherited (Mendelian)",
        )
    return PairPhase("unknown")


def trio_phase_pairs(
    candidates: list,
    records: list[dict] | None = None,
    *,
    proband: str = "PROBAND",
    mother: str = "MOTHER",
    father: str = "FATHER",
) -> dict:
    """Trio phase for same-gene candidate pairs, keyed by "coordA|coordB".

    Reads each VCF record's sample/FORMAT dict for the proband/mother/father
    genotypes. Returns {} when the VCF has no family samples (caller falls
    back to ``phase_pairs`` -> unknown). Pairs the phase_status call with the
    existing tag-based call: trio evidence wins when it is informative.
    """
    if not records:
        return {}

    def _gt(rec: dict, sample: str) -> str:
        fmt = rec.get("format") or rec.get("sample") or {}
        if not isinstance(fmt, dict):
            return ""
        val = fmt.get(sample.upper()) or fmt.get(sample)
        return str(val) if val is not None else ""

    rec_by_pos = {(r["chrom"], int(r["pos"])): r for r in records}
    by_gene: dict[str, list] = {}
    for c in candidates:
        by_gene.setdefault(c.gene or c.chrom, []).append(c)

    result: dict[str, PairPhase] = {}
    for cs in by_gene.values():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                ra = rec_by_pos.get((a.chrom, a.pos))
                rb = rec_by_pos.get((b.chrom, b.pos))
                if ra is None or rb is None:
                    continue
                key = f"{a.coord}|{b.coord}"
                # Physical tags first (phase_status), then trio Mendelian.
                tag = phase_status(ra, rb)
                trio = mendelian_phase(
                    _gt(ra, proband),
                    _gt(ra, mother),
                    _gt(ra, father),
                    _gt(rb, proband),
                    _gt(rb, mother),
                    _gt(rb, father),
                )
                if trio.status == "trans":
                    result[key] = trio  # definitive; overrides unknown/cis
                elif tag.status != "unknown":
                    result[key] = tag
                # both unknown: leave absent (caller treats as unknown)
    return result


def phase_pairs(candidates: list, records: list[dict] | None = None) -> dict:
    """Phase status for same-gene candidate pairs, keyed by "coordA|coordB"."""
    if not records:
        return {}
    rec_by_pos = {(r["chrom"], int(r["pos"])): r for r in records}
    by_gene: dict[str, list] = {}
    for c in candidates:
        by_gene.setdefault(c.gene or c.chrom, []).append(c)

    result: dict[str, PairPhase] = {}
    for cs in by_gene.values():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                ra = rec_by_pos.get((a.chrom, a.pos))
                rb = rec_by_pos.get((b.chrom, b.pos))
                if ra is None or rb is None:
                    continue
                result[f"{a.coord}|{b.coord}"] = phase_status(ra, rb)
    return result


# ── WhatsHap external pipeline (documented; requires external tools) ───────


def _tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def whatshap_pipeline(
    fastqs: list[Path | str] | list[Path],
    vcf: Path | str,
    out_dir: Path | str,
    *,
    region: str = "",
) -> str:
    """Emit the exact shell pipeline for read-backed phasing (does NOT run it).

    Pipeline:
      1. Align FASTQs to GRCh38 (minimap2 -ax sr; lighter RAM than bwa-mem2).
         Multiple per-lane BAMs are accepted by whatshap directly — no merge
         needed. Use --ignore-read-groups for single-sample sloppy headers.
      2. WhatsHap phase the VCF against the BAM(s).

    NOTE: WhatsHap has NO --region option. For a fast region-limited run
    (gene ± 1Mb), pass ``region="chrN:start-end"`` — the script pre-subsets
    the VCF (bcftools view -r) and each BAM (samtools view) first, which
    reduces phasing from hours to minutes.

    Returns the shell script text. Raises RuntimeError listing missing tools.
    """
    tools = ("minimap2", "samtools", "whatshap")
    extra = ("bcftools",) if region else ()
    missing = [t for t in tools + extra if not _tool_available(t)]
    if missing:
        raise RuntimeError(
            f"Read-backed phasing needs external tools not on PATH: {missing}. "
            "Install them (conda install -c bioconda minimap2 samtools whatshap"
            + (" bcftools" if region else "")
            + "), then call whatshap_pipeline() again."
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bams: list[str] = []
    align_lines: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# 1. Align WGS read pairs to GRCh38 (minimap2 short-read preset, per lane).",
    ]
    for i, fq in enumerate(fastqs):
        bam = str(out / f"proband.lane{i}.bam")
        bams.append(bam)
        align_lines.append(
            f"minimap2 -ax sr -t 16 --MD -R '@RG\\tID:lane{i}\\tSM:PROBAND' "
            f"GRCh38.fa {fq} | samtools sort -@ 8 -o {bam} -"
        )
    align_lines.append("samtools index " + " ".join(bams))

    phase_vcf = str(vcf)
    phase_bams = list(bams)
    if region:
        phase_vcf = str(out / "region.vcf.gz")
        region_bams = []
        for i in range(len(bams)):
            rb = str(out / f"region.lane{i}.bam")
            region_bams.append(rb)
        align_lines.append(
            "# 2. Region-limited fast path: subset VCF + BAMs (WhatsHap has no --region)."
        )
        align_lines.append(
            f"bcftools view -r {region} -Oz -o {phase_vcf} {vcf} && tabix -p vcf {phase_vcf}"
        )
        for b, rb in zip(bams, region_bams, strict=True):
            align_lines.append(f"samtools view -b -o {rb} {b} {region} && samtools index {rb}")
        phase_bams = region_bams

    phased = out / "proband.phased.vcf.gz"
    align_lines.append("# 3. WhatsHap read-backed phasing (per-lane BAMs accepted directly).")
    align_lines.append(
        f"whatshap phase -o {phased} --reference GRCh38.fa --ignore-read-groups "
        f"{phase_vcf} {' '.join(phase_bams)}"
    )
    align_lines.append(f"tabix -p vcf {phased}")
    align_lines.append("")
    align_lines.append(f"# Phased output: {phased}")
    align_lines.append(
        "# TRANS (compound-het): same PS with GT 0|1 and 1|0 (ALTs on opposite haplotypes)."
    )
    align_lines.append("# CIS (same haplotype): same PS with GT 0|1 and 0|1 — NOT a compound-het.")

    script = "\n".join(align_lines)
    (out / "phase_pipeline.sh").write_text(script)
    return script

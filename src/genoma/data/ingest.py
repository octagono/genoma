"""Dataset ingestion: gated HF download + VCF + phenotype docx loading.

The dataset (SageBio/mva-hackathon-2026-data) is gated. Downloading requires:
1. An HF account that has accepted the terms on the dataset page.
2. An HF token with read access (HF_TOKEN in .env / environment).

Until that is done, the pipeline runs in *demo mode* on synthetic data
(see `demo` module) so the DSPy layer can be developed offline.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

import cyvcf2
import docx  # python-docx

from genoma import config as cfg

# ── Gated download ─────────────────────────────────────────────────────────


def ensure_hf_auth() -> str:
    """Return an HF token or raise with clear remediation steps."""
    token = cfg.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if token:
        return token
    raise RuntimeError(
        "HF_TOKEN not found. To download the gated dataset:\n"
        "  1. Log in to https://huggingface.co and accept the terms on\n"
        "     https://huggingface.co/datasets/SageBio/mva-hackathon-2026-data\n"
        "  2. Create a read token at https://huggingface.co/settings/tokens\n"
        "  3. Add HF_TOKEN=... to .env\n"
        "The pipeline runs in demo mode (synthetic variants) until then."
    )


def required_download_bytes(files: Iterable[str], token: str) -> int:
    """Total byte size of the requested dataset files (HF metadata lookup).

    Fails fast: the metadata query hits the same endpoint the download
    needs, so a network/auth error here would fail the download anyway.
    Unknown file sizes count as 0 (worst case: preflight underestimates and
    the download itself surfaces ENOSPC).
    """
    from huggingface_hub import HfApi

    info = HfApi(token=token).dataset_info(cfg.HF_DATASET, files_metadata=True)
    sizes = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
    return sum(sizes.get(f, 0) for f in files)


def assert_free_space(dest: Path, needed_bytes: int, *, headroom: float = 1.2) -> None:
    """Fail fast if ``dest`` lacks space for ``needed_bytes`` (+headroom).

    The headroom covers the tbi index, VEP cache growth, and intermediate
    files next to the dataset.
    """
    import shutil

    free = shutil.disk_usage(str(dest)).free
    if free < needed_bytes * headroom:
        raise RuntimeError(
            f"Insufficient disk space at {dest}: need ~{needed_bytes * headroom / 1e9:.1f} GB "
            f"(incl. {headroom:.0%} headroom), have {free / 1e9:.1f} GB free. "
            "Free space or relocate DATA_DIR before downloading the gated dataset."
        )


def download_dataset(
    files: Iterable[str] = (
        cfg.VCF_FILENAME,
        cfg.VCF_TBI_FILENAME,
        cfg.PHENOTYPE_FILENAME,
    ),
    dest_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download gated dataset files into DATA_DIR via hf_hub_download.

    Only downloads what's needed for the pipeline (VCF + index + phenotype).
    The 8 raw FASTQs (~85 GB) are deliberately NOT pulled unless explicitly
    requested (pass files=...). Storage preflight: verifies free disk space
    against the actual HF-reported file sizes (+20% headroom) before
    starting the multi-GB download. The .tbi index ships with the download —
    cyvcf2 needs it for region queries (full-file iteration scans
    sequentially and works either way).
    """
    from huggingface_hub import hf_hub_download

    token = ensure_hf_auth()
    dest_dir = dest_dir or cfg.DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = list(files)
    missing = [f for f in files if force or not (dest_dir / f).exists()]
    if missing:
        needed = required_download_bytes(missing, token)
        if needed:
            print(f"Preflight: {needed / 1e9:.1f} GB to download for {len(missing)} file(s)")
        assert_free_space(dest_dir, needed)
    for fname in files:
        target = dest_dir / fname
        if target.exists() and not force:
            continue
        hf_hub_download(
            repo_id=cfg.HF_DATASET,
            filename=fname,
            repo_type="dataset",
            local_dir=str(dest_dir),
            token=token,
        )
    return dest_dir


def vcf_path() -> Path:
    """The VCF the pipeline consumes.

    Prefers the VEP-annotated VCF (data/annotated.vcf.gz) when it exists —
    gene/consequence come from there. Falls back to the raw downloaded VCF
    so ingest + demo still work before annotation. Single source of truth
    for the pipeline input.
    """
    annotated = cfg.DATA_DIR / cfg.ANNOTATED_VCF_FILENAME
    if annotated.exists():
        return annotated
    return cfg.DATA_DIR / cfg.VCF_FILENAME


def phenotype_path() -> Path:
    return cfg.DATA_DIR / cfg.PHENOTYPE_FILENAME


# ── VCF parsing (cyvcf2) ──────────────────────────────────────────────────


def _maybe_capture_vep_csq_header(reader) -> None:
    """Parse VEP's ``##INFO=<ID=CSQ,Description="Format: ...">`` header, if
    present, and publish the field order for annotate.py's named extractors.

    VEP CSQ is pipe-delimited with release-dependent field order; the header
    declares it explicitly. cyvcf2 exposes raw header lines via
    ``reader.header`` (a list of strings).
    """
    try:
        header_lines = reader.header
    except AttributeError:
        return
    for line in header_lines:
        if "ID=CSQ" not in line or "Format:" not in line:
            continue
        # "##INFO=<ID=CSQ,Description="Format: Allele|Consequence|...">"
        fmt = line.split("Format:")[-1].split('"')[0]
        fields = [f.strip() for f in fmt.split("|") if f.strip()]
        if fields:
            from genoma.data.annotate import _set_vep_csq_fields

            _set_vep_csq_fields(fields)
        return


def _normalize_chrom(chrom: str) -> str:
    """Normalize a VCF contig name to GRCh38 ``chrN`` form.

    VCFs may use bare contigs (``1``..``22``, ``X``, ``Y``, ``M``) while
    chromosome strings are compared verbatim in ``chrN`` form (``chr2``,
    ``chr15``). Normalize here at the VCF boundary so every downstream
    consumer (candidate table, phasing keys, annot-cache keys, submission
    rows) agrees on ``chrN``. Idempotent: already-prefixed contigs pass
    through untouched.
    """
    c = str(chrom)
    if c.startswith("chr"):
        return c
    if c == "M" or c == "MT":
        return "chrM"
    return f"chr{c}"


def load_vcf_records(
    path: Path | None = None,
    *,
    chroms: set[str] | None = None,
    max_records: int | None = None,
    pass_only: bool = True,
) -> list[dict]:
    """Parse a VCF into lightweight record dicts.

    Uses cyvcf2 (declared dependency); fails fast if the VCF cannot be
    opened. Only PASS (or '.') records are kept when ``pass_only``.

    Contigs are normalized to GRCh38 ``chrN`` form (``1`` -> ``chr1``,
    ``M`` -> ``chrM``) so downstream consumers (candidate table, phasing
    keys, annot-cache keys, submission rows) all agree on ``chrN``.
    """
    path = path or vcf_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `genoma ingest` first (requires HF_TOKEN), or use demo mode."
        )

    # VCF parsing uses cyvcf2 (declared dependency). Fails fast on absence.
    reader = cyvcf2.VCF(str(path))
    # If this is a VEP-annotated VCF, capture the CSQ field order from the
    # header so annotate.py can extract gene/consequence by NAME (VEP's CSQ
    # field positions vary by release).
    _maybe_capture_vep_csq_header(reader)
    records: list[dict] = []
    for v in reader:
        chrom = _normalize_chrom(v.CHROM)
        if chroms and chrom not in chroms:
            continue
        if pass_only and v.FILTER is not None and str(v.FILTER) not in ("PASS", "."):
            continue
        records.append(
            {
                "chrom": chrom,
                "pos": v.POS,
                "ref": v.REF,
                "alt": ",".join(v.ALT),
                "qual": v.QUAL,
                "filter": str(v.FILTER),
                "info": dict(v.INFO),
            }
        )
        if max_records and len(records) >= max_records:
            break
    return records


# ── Phenotype / HPO extraction ─────────────────────────────────────────────

# Negation markers used by clinical phenotype sheets ("Microcephaly: None",
# "Not observed", "HP:0000252 | absent"). HPO encodes these as NOT-terms;
# a naive parser extracting every HP id would count excluded phenotypes
# as positive findings and poison the Phen2Gene/prioritization input.
_NEGATION = re.compile(
    r"\b(none|not observed|not present|absent|negative|denied|"
    r"ruled out|excluded|no)\b",
    re.IGNORECASE,
)

# Offline clinical-synonym -> canonical HPO map (lightweight semantic
# fallback; no model dependency). Clinical notes phrase phenotypes as
# "small head", "developmental lag", "failure to thrive" instead of HPO ids;
# this maps free-text lines to the HPO terms Phen2Gene can index.
_HPO_SYNONYMS: dict[str, str] = {
    "small head": "HP:0000252 Microcephaly",
    "microcephaly": "HP:0000252 Microcephaly",
    "growth delay": "HP:0001510 Growth delay",
    "growth retardation": "HP:0001510 Growth delay",
    "failure to thrive": "HP:0001508 Failure to thrive",
    "short stature": "HP:0004322 Short stature",
    "small for gestational age": "HP:0001518 Small for gestational age",
    "developmental delay": "HP:0001263 Global developmental delay",
    "developmental lag": "HP:0001263 Global developmental delay",
    "intellectual disability": "HP:0001249 Intellectual disability",
    "premature chromatid separation": "HP:0009820 Premature chromatid separation",
    "rhabdomyosarcoma": "HP:0006738 Rhabdomyosarcoma",
    "cancer": "HP:0002664 Neoplasm of the immune system",
    "tumor": "HP:0002664 Neoplasm of the immune system",
    "seizures": "HP:0001250 Seizure",
    "epilepsy": "HP:0001250 Seizure",
    "muscle weakness": "HP:0001324 Muscle weakness",
    "hypotonia": "HP:0001252 Hypotonia",
}

# Semantic-stem normalization: collapse morphological variants of the
# synonyms so "seizure", "seizures", "seizuring" all match. Offline and
# dependency-free — a deterministic stemmer, no model download.
_STEM_RULES = [
    (re.compile(r"ing$"), ""),  # delaying -> delay
    (re.compile(r"ies$"), "y"),  # seizures is handled below, but e.g. "anomalies" -> anomaly
    (re.compile(r"es$"), ""),  # seizures -> seizure
    (re.compile(r"s$"), ""),  # heads -> head
]


def _stem(word: str) -> str:
    out = word
    for pat, repl in _STEM_RULES:
        out = pat.sub(repl, out)
    return out


def _semantic_hpo_match(text: str) -> list[str]:
    """Match free text to canonical HPO via normalized synonyms.

    Normalizes both the input and the synonym keys (lowercase + stemming),
    so morphological variants ("growth delays", "seizuring") resolve to the
    same HPO term — the offline semantic layer without a model dependency.
    """
    low = text.lower()
    normalized = " ".join(_stem(w) for w in low.split())
    hits: set[str] = set()
    for syn, hp in _HPO_SYNONYMS.items():
        key = " ".join(_stem(w) for w in syn.split())
        if key in normalized:
            hits.add(hp)
    return sorted(hits)


def _map_free_text(text: str) -> list[str]:
    """Map free-text phenotype lines to canonical HPO terms via synonyms."""
    return _semantic_hpo_match(text)


def extract_hpo_terms(path: Path | None = None) -> list[str]:
    """Extract HPO terms from the clinical phenotype docx.

    Parses paragraphs AND tables (the clinical phenotype sheet often uses a
    table with HPO ID + label columns). Returns normalized "HP:xxxxxxx Label"
    strings. Lines/rows carrying a negation marker (none / not observed /
    absent / ...) are skipped — excluded phenotypes are NOT positive findings
    (row-level granularity: one negated marker negates that row's HP ids).
    Falls back to a curated MVA-relevant HPO starter list so the pipeline is
    testable before the gated docx is available.
    """
    path = path or phenotype_path()
    if path.exists() and path.suffix.lower() == ".docx":
        doc = docx.Document(str(path))
        texts: list[str] = []
        # Paragraphs.
        for p in doc.paragraphs:
            if p.text.strip():
                texts.append(p.text)
        # Tables (rows -> cells).
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    texts.append(" | ".join(cells))

        terms: list[str] = []
        seen: set[str] = set()

        def _add(hid: str, label: str) -> None:
            hid = hid.replace(" ", "").upper()
            if hid in seen:
                return
            seen.add(hid)
            label = re.sub(r"^[\s\-\:\)\]]+", "", label).strip()
            terms.append(f"{hid} {label}".strip() if label else hid)

        # Tables: cell-aware so the HPO Term label (often the cell BEFORE the
        # ID, e.g. "HPO Term | HPO ID") is captured, not the trailing Notes.
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if not cells:
                    continue
                joined = " | ".join(cells)
                if _NEGATION.search(joined):
                    continue
                for m in re.finditer(r"(HP:\s*\d{7,})", joined):
                    hid = m.group(1)
                    # Label = the cell immediately before the ID if it looks
                    # like a term, else the text right after the ID.
                    label = ""
                    pos_in_row = joined[: m.start()].count("|")
                    if pos_in_row >= 1:
                        label = cells[pos_in_row - 1]
                    if not label:
                        label = joined[m.end() :].split("|")[0]
                    _add(hid, label)

        # Paragraphs: "HP:xxxxxxx" with an optional nearby label.
        for text in texts:
            if _NEGATION.search(text):
                continue
            for m in re.finditer(r"(HP:\s*\d{7,})", text):
                label = text[m.end() :].split("|")[0].split("\n")[0].strip()
                _add(m.group(1), label)
        if terms:
            return sorted(terms)
        # No ids: map free-text phenotype notes through the offline synonym
        # table (clinical notes say "small head"/"developmental lag", not HPO
        # ids); unmatched lines are kept verbatim as free-text notes.
        mapped: list[str] = []
        for ln in texts:
            mapped.extend(_map_free_text(ln))
        mapped = sorted(set(mapped))
        if mapped:
            return mapped
        return [ln.strip() for ln in texts if len(ln.strip()) > 3][:100]

    # Fallback starter HPO set (curated for MVA) so the pipeline is testable
    # before the gated docx is available. SSOT: the demo module's canonical list.
    from genoma.data.demo import DEMO_HPO

    return list(DEMO_HPO)

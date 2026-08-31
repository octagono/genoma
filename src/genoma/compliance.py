"""Compliance helpers — dataset deletion rules.

Key obligations:
- All data must be deleted within 30 days of close.
- Delete genome-carrying material: VCF/BAM/CRAM + copies/indexes, genotype-scale
  tables, stored prompts/logs with pasted variant blocks, model weights trained
  on raw data. KEEP: ranked candidate list, HPO terms, gene/pathway rankings,
  drug candidates, code, reports.
- DSPy's disk cache (~/.dspy_cache, 30 GB cap) stores LM responses verbatim —
  including whatever variant data was sent. It MUST be purged.
- The gated dataset itself lives in DATA_DIR **and** the Hugging Face hub
  cache (hf_hub_download leaves blobs under data/.cache/huggingface and
  ~/.cache/huggingface/hub/datasets--SageBio--...). Both are genome-carrying
  material and MUST be purged.
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import dspy

from genoma import config as cfg

# Required acknowledgement for any publication, preprint, abstract, or public
# communication arising from participation.
ACKNOWLEDGEMENT = (
    "This work was made possible through the Hackathon, organized by Sage "
    "Bionetworks in partnership with the MVA Society, Hugging Face, and "
    "BEACON (The Benchmarking, Evaluation, and Assessment Consortium for "
    "Science), with prize sponsorship from AWS and Anthropic. We are deeply "
    "grateful to the child and their family who generously contributed their "
    "data and their story to advance research into this rare disease. We "
    "acknowledge their trust in making this Hackathon possible."
)


def dspy_disk_cache_dir() -> Path:
    """Resolve the DSPy disk cache dir (single source of truth: cfg.CACHE_DIR,
    which honors $DSPY_CACHEDIR and defaults to ~/.dspy_cache)."""
    return cfg.CACHE_DIR


def hf_dataset_cache_dirs() -> list[Path]:
    """HF hub-cache locations that may hold gated-dataset blobs.

    Honors $HF_HUB_CACHE / $HF_HOME (huggingface_hub's own resolution order);
    defaults to ~/.cache/huggingface/hub. Only this dataset's entry is
    targeted — other datasets in the cache are untouched.
    """
    repo_dir = f"datasets--{cfg.HF_DATASET.replace('/', '--')}"
    if env_hub := os.environ.get("HF_HUB_CACHE"):
        return [Path(env_hub) / repo_dir]
    base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return [base / "hub" / repo_dir]


def purge_caches(
    *,
    confirm: bool = True,
    delete_disk: bool = True,
    delete_data: bool = True,
) -> dict[str, bool]:
    """Purge genome-carrying caches. Returns what was cleared.

    - memory: DSPy LRU cache (safe, always)
    - disk: DSPy disk cache (LM responses verbatim)
    - data_dir: DATA_DIR — gated VCF/docx + the data/.cache/huggingface
      download blobs hf_hub_download leaves behind
    - hf_cache: this dataset's entry in the HF hub cache
    - submissions: SUBMISSION_DIR — derived CSVs + checkpoint JSONL
      (ranked variant calls are derived genome data per the rules)

    Deletion is destructive — set confirm=False only when you intend it.
    """
    result: dict[str, bool] = {
        "memory": False,
        "disk": False,
        "data_dir": False,
        "hf_cache": False,
        "submissions": False,
    }

    # Memory cache (LRU) — safe, always. Fails fast on error.
    dspy.cache.reset_memory_cache()
    result["memory"] = True

    targets: dict[str, list[Path]] = {}
    if delete_disk:
        targets["disk"] = [dspy_disk_cache_dir()]
    if delete_data:
        targets["data_dir"] = [cfg.DATA_DIR]
        targets["hf_cache"] = hf_dataset_cache_dirs()
        targets["submissions"] = [cfg.SUBMISSION_DIR]

    if targets:
        listing = "\n".join(f"  - {p}" for paths in targets.values() for p in paths)
        if confirm:
            answer = input(f"Delete:\n{listing}\n? [y/N] ")
            if answer.strip().lower() != "y":
                print("  nothing deleted")
                return result
        for key, paths in targets.items():
            ok = True
            for p in paths:
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)
                    ok = ok and not p.exists()
                # Nonexistent target = nothing to delete = satisfied.
            result[key] = ok

    # The annotation cache (VEP/Phen2Gene JSON blobs) lives inside DATA_DIR
    # and is deleted with it, but also purge it explicitly so the WAL sidecar
    # files (-wal/-shm) don't survive a partial rmtree.
    from genoma.data.annot_cache import cache_path as _annot_cache_path

    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{_annot_cache_path()}{suffix}")
        if p.exists():
            p.unlink(missing_ok=True)
    return result


def deletion_checklist() -> str:
    """Print the DELETE/KEEP checklist + submission quotas."""
    now = datetime.now(UTC)
    close = datetime.fromisoformat(cfg.HACKATHON_CLOSE_ISO.replace("Z", "+00:00"))
    days_left = (close - now).days
    deadline = close.timestamp() + cfg.DATA_DELETE_DAYS * 86400
    deadline_dt = datetime.fromtimestamp(deadline, tz=UTC)
    return f"""\
MVA Hackathon — Data Deletion Checklist
=======================================
Submissions close: {close:%Y-%m-%d %H:%M UTC} ({days_left} days left)
All data must be deleted by: {deadline_dt:%Y-%m-%d} (close + {cfg.DATA_DELETE_DAYS} days)
Confirm deletion by emailing: {cfg.DATA_DELETE_EMAIL}

Submission quotas (track before each submit):
  [ ] Track 1: {cfg.MAX_TRACK1_SUBMISSIONS} max submissions (leaderboard keeps best)
  [ ] Track 2: {cfg.TRACK2_SUBMISSIONS} max submissions (expert panel reviews only the LATEST)

DELETE (genome-carrying material):
  [ ] VCF / BAM / CRAM + any copy, subset, slice, index
  [ ] filtered/annotated variant tables at genotype scale + caches/notebook state
  [ ] stored prompts/logs containing pasted blocks of variant data
  [ ] any model weights/embeddings/fine-tunes trained on raw genomic data
  [ ] DSPy disk cache: {dspy_disk_cache_dir()}
  [ ] gated dataset dir: {cfg.DATA_DIR} (incl. data/.cache/huggingface blobs)
  [ ] VEP-annotated VCF (derived): {cfg.DATA_DIR / cfg.ANNOTATED_VCF_FILENAME}
  [ ] HF hub cache entries: {", ".join(str(p) for p in hf_dataset_cache_dirs())}
  [ ] submission outputs (derived): {cfg.SUBMISSION_DIR} (CSVs + checkpoints)
  [ ] planner state + any local copies of the submission CSV
  [ ] `./run.sh purge-cache` clears all of the above automatically
  [ ] email {cfg.DATA_DELETE_EMAIL} to confirm deletion within 30 days

ACKNOWLEDGEMENT (required in any publication/preprint/abstract/communication):
{ACKNOWLEDGEMENT}

KEEP (your findings):
  [ ] ranked candidate variant list (the submission)
  [ ] HPO terms
  [ ] gene/pathway rankings, mechanism analyses, drug candidates
  [ ] code, report, pitch, leaderboard entry (no prohibited data inside)
"""

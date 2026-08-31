"""Central configuration for the DSPy pipeline.

All knobs live here so the pipeline is tunable from one place.
Model IDs are the DSPy/LiteLLM form ``"provider/model"``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (API keys, HF token, overrides).
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"  # where dataset files land
SUBMISSION_DIR = ROOT / "submissions"  # generated submission CSVs

# DSPy disk cache location. DSPy honors $DSPY_CACHEDIR, defaulting to
# ~/.dspy_cache (30 GB cap). compliance.purge_caches() resolves the same way;
# set DSPY_CACHEDIR in .env to relocate (e.g. inside the project).
CACHE_DIR = Path(os.environ.get("DSPY_CACHEDIR") or Path.home() / ".dspy_cache")

# ── Hugging Face dataset (gated) ───────────────────────────────────────────
HF_DATASET = "SageBio/mva-hackathon-2026-data"
VCF_FILENAME = "WGS_EX2312012_HGWCNDSX7.vcf.gz"
# VEP-annotated VCF (produced by vep_data/vep_run.sh). The pipeline consumes
# this when present (gene/consequence from CSQ); falls back to VCF_FILENAME.
ANNOTATED_VCF_FILENAME = "annotated.vcf.gz"
VCF_TBI_FILENAME = "WGS_EX2312012_HGWCNDSX7.vcf.gz.tbi"
PHENOTYPE_FILENAME = "Challenge_Clinical_Phenotype_1.docx"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
PROBAND_ID = "PROBAND01"
# Version key for the local annotation cache: bump when the dataset is
# re-downloaded/replaced so stale cached VEP/Phen2Gene entries are never read.
DATA_VERSION = "2026-08-30"

# ── Models ─────────────────────────────────────────────────────────────────
# Anthropic 5.x dateless IDs ARE the pinned snapshot (Anthropic self-pins
# dateless IDs from 4.6 on) — there is no -YYYYMMDD form for 5.x. Sonnet 5
# ships "adaptive thinking" (always-on; effort defaults to high) — reasoning
# tokens count toward max_tokens, so keep max_tokens generous.
#
# Primary reasoning model (Anthropic). Provider-side prompt caching is enabled
# via cache_control_injection_points.
PRIMARY_LM = os.environ.get("PRIMARY_LM", "anthropic/claude-sonnet-5")
# Client-side token-bucket budget (tokens/minute) for LM calls; 0 = unlimited.
# Estimates input tokens per request and delays host-side at ~80% of budget to
# avoid provider 429s / Tier-limit blocks during high-volume parallel runs.
LM_TPM = float(os.environ.get("LM_TPM", "0"))
PRIMARY_LM_KWARGS = {
    "temperature": 0.2,
    "max_tokens": 8000,
    "cache": True,
    "cache_control_injection_points": [
        {"location": "message", "role": "system"},
        {"location": "message", "index": -1},
    ],
}

# Cheap model for RLM sub-queries (llm_query/llm_query_batched); dated pin.
SUB_LM = os.environ.get("SUB_LM", "anthropic/claude-haiku-4-5-20251001")

# Strong model for GEPA reflection (required by GEPA constructor).
REFLECTION_LM = os.environ.get("REFLECTION_LM", "anthropic/claude-sonnet-5")
REFLECTION_LM_KWARGS = {"temperature": 1.0, "max_tokens": 16000}

# Heterogeneous consensus panel: model slug -> weight. OpenRouter via LiteLLM.
# Leave OPENROUTER_API_KEY unset to collapse to a single-model panel.
# Four distinct providers for independent errors.
PANEL = {
    "openrouter/anthropic/claude-sonnet-5": 2.0,
    "openrouter/openai/gpt-5.6-sol": 1.0,
    "openrouter/deepseek/deepseek-v4-pro": 1.0,
    "openrouter/google/gemini-3.7-flash": 1.0,
}
PANEL_TEMPERATURE = 0.7
PANEL_DEBATE_THRESHOLD = 0.35  # spread above this triggers the debate round
# Spread that stays above this even AFTER the debate round means the panel
# could not converge — the call is flagged for review instead of being
# presented as a confident EPCR.
PANEL_ABSTAIN_THRESHOLD = 0.55
# GEPA judge model. Default (None) = settings.lm (the proposer's family).
# Cross-family judging avoids self/family-bias inflating the optimization
# signal — set this to a DIFFERENT family than PRIMARY_LM when a second
# provider key is available (e.g. deepseek-v4-pro via OpenRouter).
GEPA_JUDGE_LM = os.environ.get("GEPA_JUDGE_LM", "") or None

# ── Track 1 ────────────────────────────────────────────────────────────────
MAX_SUBMISSION_ROWS = 10  # up to 10 candidate rows accepted
MAX_TRACK1_SUBMISSIONS = 6  # per-participant submission quota
CANDIDATE_PRE_FILTER = {  # deterministic VCF pre-filter (see data/annotate.py)
    "max_af": 0.01,
    "min_cadd": 10.0,
    "keep_splice": True,
}
# Candidate-pool cap for the deterministic pre-filter. Single source of truth
# (was hardcoded 500 in 7 places). Kept high so a causal pair can never be
# truncated out of the pool.
CANDIDATE_MAX_DEFAULT = 2000
# Compound-het pairs to score per run. Single source of truth (was hardcoded
# 10 in cli + scorer). Novel-gene coverage: the pair scorer is gene-agnostic
# (pairs within ANY gene), so the cap must not truncate a novel causal gene's
# pair out of the pool.
COMPHET_MAX_PAIRS = 50
SUBMISSION_STRATEGY = "mixed"  # pairs-first, then singles (compound-het is the expected mechanism)

# Known-gene prior: how strongly a variant in a known MVA gene is boosted.
GENE_PRIOR_BOOST = 0.15
# Panel spread-as-confidence: how far high disagreement pulls EPCR toward 0.5.
SPREAD_SHRINK = 0.30
# Self-consistency calibration: number of temperature=1.0 samples for top candidates.
SELF_CONSISTENCY_N = 0  # 0 = off (each sample costs an LM call)

# Ensembl VEP REST (key-free) annotation.
GENOME_ASSEMBLY = "GRCh38"  # VEP REST default; matches the submission schema
EXOMISER_ASSEMBLY = "hg38"  # Exomiser's token for the same GRCh38 build (2602_hg38 data)
VEP_REST_URL = "https://rest.ensembl.org/vep/human/region"
VEP_BATCH_SIZE = 25  # variants per POST request (200-variant batches take ~100s
# with REVEL/AlphaMissense/SpliceAI predictors — over the 30s timeout; 25 keeps
# each request well under the timeout while staying polite to the endpoint)
VEP_RATE_LIMIT_S = 0.15  # seconds between requests (be polite)

# ── GEPA optimization ──────────────────────────────────────────────────────
GEPA_AUTO = "light"  # light/medium/heavy or None (then set max_metric_calls)
GEPA_MAX_METRIC_CALLS = 200
GEPA_TRACK_STATS = True

# ── Track 2 ────────────────────────────────────────────────────────────────
TRACK2_MAX_ITERS = 10
TRACK2_SUBMISSIONS = 3  # the expert panel reviews only the LATEST submission

# ── Compliance ─────────────────────────────────────────────────────────────
HACKATHON_CLOSE_ISO = "2026-10-24T23:59:00Z"  # submissions close (frozen leaderboard)
DATA_DELETE_EMAIL = "RarediseaserealkidMVAhackathon2026@synapse.org"
DATA_DELETE_DAYS = 30

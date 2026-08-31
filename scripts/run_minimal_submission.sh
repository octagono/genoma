#!/usr/bin/env bash
# Real-data single-model submission run.
#
# Runs AFTER the VEP cache download finishes. Sequence:
#   1. move the finished cache tarball from the legacy mva-agent/vep_data/ path
#   2. extract + verify it
#   3. VEP-annotate the VCF (podman, offline)
#   4. build the candidate pool, score ~500 candidates single-model (no panel), calibrate
#   5. write the submission CSV + methods report
#
# Usage:  bash scripts/run_minimal_submission.sh
# Set USERNAME to your HF username (filenames must follow
# "<username>_<model>.csv"). Default: octagono.
set -euo pipefail
cd "$(dirname "$0")/.."

USERNAME="${USERNAME:-octagono}"
MODEL_TAG="genoma-dspy-minimal"
OUT_CSV="submissions/${USERNAME}_${MODEL_TAG}.csv"
CKPT="submissions/${USERNAME}_${MODEL_TAG}_ckpt.jsonl"

LEGACY_DIR="mva-agent/vep_data"
DATA_DIR="vep_data"
VEP_VERSION=110
VEP_ASSEMBLY=GRCh38

# ── 1. Locate + move the finished cache ─────────────────────────────────────
if [[ -d "$DATA_DIR/homo_sapiens" ]]; then
	echo "Cache already extracted at $DATA_DIR/homo_sapiens — skipping download/move."
elif [[ -f "$LEGACY_DIR/homo_sapiens_vep_${VEP_VERSION}_${VEP_ASSEMBLY}.tar.gz" ]]; then
	echo "Moving finished cache tarball from legacy path..."
	mkdir -p "$DATA_DIR"
	mv "$LEGACY_DIR/homo_sapiens_vep_${VEP_VERSION}_${VEP_ASSEMBLY}.tar.gz" "$DATA_DIR/"
	echo "Extracting (large, several minutes)..."
	tar -xzf "$DATA_DIR/homo_sapiens_vep_${VEP_VERSION}_${VEP_ASSEMBLY}.tar.gz" -C "$DATA_DIR"
	rm -f "$DATA_DIR/homo_sapiens_vep_${VEP_VERSION}_${VEP_ASSEMBLY}.tar.gz"
	echo "Cache ready."
else
	echo "ERROR: cache tarball not found at $LEGACY_DIR. Run vep_data/vep_cache_download.sh first."
	exit 1
fi

# ── 2. VEP-annotate the real VCF ────────────────────────────────────────────
if [[ -f "data/annotated.vcf.gz" ]]; then
	echo "Annotated VCF already exists — skipping annotation."
else
	echo "Annotating the VCF with VEP ${VEP_VERSION} (offline, ~1-3 h)..."
	bash vep_data/vep_run.sh
fi

# ── 3. Candidate pool + minimal single-model score ──────────────────────────
echo "Building candidate pool + scoring (single-model, ~500 candidates, ~\$3)..."
export LD_LIBRARY_PATH="$(ls -d /nix/store/*gcc-*-lib/lib 2>/dev/null | sort -V | tail -1)"

# Score the top candidates single-model.
./run.sh score \
	--max-candidates 500 \
	--max-pairs 10 \
	--annotate \
	--self-consistency 3 \
	--checkpoint "${CKPT}" \
	--out "${OUT_CSV}"

# ── 4. Validate + report ────────────────────────────────────────────────────
echo "Validating the submission..."
./run.sh validate "${OUT_CSV}" || echo "(validate uses the synthetic key — real leaderboard is the source of truth)"

echo
echo "=== MINIMAL SUBMISSION READY ==="
echo "  CSV:  ${OUT_CSV}"
echo "  Next: inspect rows, then submit on the HF Space, then read the"
echo "        full/partial/no-match verdict and iterate (feedback oracle)."
echo "  Budget so far: ~\$3-5 (single-model, 500 candidates)"
echo "  Filename follows the required convention: <username>_<model>.csv"

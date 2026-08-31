#!/usr/bin/env bash
# Ensembl VEP GRCh38 annotation via Podman (official image) — optimized.
#
# Why podman, not nixpkgs: the nixpkgs `vep` package fails to build (kent
# dependency compile error); SnpEff 4.3t predates current GRCh38 gene models.
# The official ensemblorg/ensembl-vep image bundles a current VEP and is
# Ensembl's recommended deployment.
#
# One-time cache download (~25 GB, free, matches VEP release 110):
#   bash vep_data/vep_cache_download.sh
#
# Optimizations for a whole-genome VCF:
#   --fork 8      parallelize across cores (host has 16; 8 is a safe balance)
#                 SV annotation is OFF by default in VEP 110 (--check_sv is
#                 opt-in; no SV cache is downloaded) — do NOT pass --no_sv,
#                 it is not recognized by VEP 110 and aborts the run.
#   --pick        one representative consequence per variant (pre-filter just
#                 needs gene + consequence; --per_gene would be redundant)
#   --symbol      put SYMBOL into CSQ (our parser extracts gene by name)
#   tabix index   after bgzip so cyvcf2 can region-query the annotated VCF
#
#   bash vep_data/vep_run.sh
#
# Rootless podman quirk: the ensembl-vep image runs as uid 999 (vep) by
# default; rootless podman maps that to an unrelated subuid, so the container
# cannot write the output into the host's data/ dir. --user 0 maps container
# root to the host user in rootless mode (writes land with correct ownership).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="docker.io/ensemblorg/ensembl-vep:release_110.0"
DATA_DIR="$PWD/vep_data"
VCF="data/WGS_EX2312012_HGWCNDSX7.vcf.gz"
OUT="data/annotated.vcf.gz"

if ! podman image exists "$IMAGE" 2>/dev/null; then
	echo "Pulling Ensembl VEP image (first run only)..."
	podman pull "$IMAGE"
fi

mkdir -p "$DATA_DIR"

echo "Annotating $(basename "$VCF") with VEP 110 GRCh38 (offline, fork=8)..."
podman run --rm --user 0 \
	-v "$DATA_DIR":/data \
	-v "$PWD/data":/vep_in \
	"$IMAGE" vep \
	-i /vep_in/"$(basename "$VCF")" \
	-o /vep_in/"$(basename "$OUT")" \
	--cache --offline --dir_cache /data \
	--assembly GRCh38 \
	--vcf --compress_output bgzip \
	--fork 8 \
	--symbol \
	--pick \
	--buffer_size 5000

echo "Indexing the annotated VCF with tabix..."
podman run --rm --user 0 \
	-v "$PWD/data":/vep_in \
	"$IMAGE" tabix -p vcf /vep_in/"$(basename "$OUT")"

echo "Done: ${OUT} (+ .tbi)"
echo "Pipeline next: ./run.sh candidates -> annotate -> score"

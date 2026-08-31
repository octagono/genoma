#!/usr/bin/env bash
# One-time Ensembl VEP release-110 GRCh38 cache download + extract (~25 GB).
# Must match the VEP image in vep_run.sh (release_110.0).
#   bash vep_data/vep_cache_download.sh
set -euo pipefail
cd "$(dirname "$0")"

VERSION=110
ASSEMBLY=GRCh38
URL="https://ftp.ensembl.org/pub/release-${VERSION}/variation/indexed_vep_cache/homo_sapiens_vep_${VERSION}_${ASSEMBLY}.tar.gz"
TARBALL="homo_sapiens_vep_${VERSION}_${ASSEMBLY}.tar.gz"

if [[ -d "homo_sapiens" ]]; then
	echo "Cache already extracted (homo_sapiens/ present). Nothing to do."
	exit 0
fi

# Download (resumable: -C - continues a partial download; no-op if complete).
echo "Downloading VEP ${VERSION} ${ASSEMBLY} cache from Ensembl FTP (free)..."
curl -fL --retry 3 -C - -o "${TARBALL}" "${URL}"

echo "Extracting (large — several minutes)..."
# Parallel decompression when pigz is available, else plain tar.
if command -v pigz >/dev/null 2>&1; then
	pigz -dc "${TARBALL}" | tar -xf -
else
	tar -xzf "${TARBALL}"
fi
rm -f "${TARBALL}"

echo "Cache ready in $(pwd)/homo_sapiens"
echo "Next: bash vep_data/vep_run.sh"

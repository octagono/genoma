#!/usr/bin/env bash
# Wait for the VEP cache download to finish, then run the minimal submission.
# Polls the legacy-path tarball until it stops growing (download complete),
# then hands off to run_minimal_submission.sh.
#
# Usage:  nohup bash scripts/wait_for_cache_and_run.sh > /tmp/genoma_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

TARBALL="mva-agent/vep_data/homo_sapiens_vep_110_GRCh38.tar.gz"
EXPECTED_BYTES=$((20 * 1024 * 1024 * 1024)) # ~20 GB (exact size checked by tar)
STALL_SECONDS=90
last_size=0
stable_for=0

echo "[$(date -u +%H:%M:%S)] Waiting for VEP cache download to finish..."
while :; do
	if [[ ! -f "$TARBALL" ]]; then
		echo "[$(date -u +%H:%M:%S)] tarball not present yet; waiting..."
		sleep 60
		continue
	fi
	size=$(stat -c %s "$TARBALL" 2>/dev/null || echo 0)
	if [[ "$size" -eq "$last_size" ]]; then
		stable_for=$((stable_for + 60))
	else
		stable_for=0
	fi
	last_size=$size
	gb=$(awk "BEGIN{printf \"%.1f\", $size/1e9}")
	echo "[$(date -u +%H:%M:%S)] ${gb} GB (stable ${stable_for}s)"
	if [[ "$stable_for" -ge "$STALL_SECONDS" && "$size" -gt 8000000000 ]]; then
		echo "[$(date -u +%H:%M:%S)] Download appears complete (${gb} GB, stable). Starting run."
		break
	fi
	sleep 60
done

echo "[$(date -u +%H:%M:%S)] Running minimal submission..."
bash scripts/run_minimal_submission.sh
echo "[$(date -u +%H:%M:%S)] DONE. See submissions/minimal_submission.csv"

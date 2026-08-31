#!/usr/bin/env bash
# Pipeline launcher.
#
# On NixOS, the uv venv's Python can't find libstdc++.so.6 (needed by
# tokenizers/transformers via litellm). Pick the newest gcc lib from the
# nix store and prepend it to LD_LIBRARY_PATH before running.
set -euo pipefail
cd "$(dirname "$0")"

if ! [[ "${LD_LIBRARY_PATH:-}" == *libstdc++* ]]; then
	GCCLIB="$(ls -d /nix/store/*gcc-*-lib/lib 2>/dev/null | sort -V | tail -1 || true)"
	if [[ -n "${GCCLIB}" ]]; then
		export LD_LIBRARY_PATH="${GCCLIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
	fi
fi

exec uv run genoma "$@"

#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT="${1:-/data}"
ARCHIVE_NAME="long-llm.tar.gz"
ARCHIVE_PATH="${DATA_ROOT}/${ARCHIVE_NAME}"
DATA_URL="${DATA_URL:-https://huggingface.co/datasets/namespace-Pt/projects/resolve/main/long-llm.tar.gz?download=true}"

mkdir -p "${DATA_ROOT}"

if [[ ! -w "${DATA_ROOT}" ]]; then
  echo "ERROR: '${DATA_ROOT}' is not writable. Try another path or run with proper permissions." >&2
  exit 1
fi

echo "Downloading dataset archive to: ${ARCHIVE_PATH}"
if command -v wget >/dev/null 2>&1; then
  wget --tries=3 --timeout=30 "${DATA_URL}" -O "${ARCHIVE_PATH}"
elif command -v curl >/dev/null 2>&1; then
  curl --fail --location --retry 3 --connect-timeout 30 "${DATA_URL}" -o "${ARCHIVE_PATH}"
else
  echo "ERROR: neither 'wget' nor 'curl' is available." >&2
  exit 1
fi

echo "Extracting archive into: ${DATA_ROOT}"
tar -xzvf "${ARCHIVE_PATH}" -C "${DATA_ROOT}"

echo "Done. Data extracted under: ${DATA_ROOT}"

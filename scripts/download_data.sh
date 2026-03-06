#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Auto-load environment variables from .env if present.
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

DATA_ROOT="${1:-/data}"
ARCHIVE_NAME="long-llm.tar.gz"
ARCHIVE_PATH="${DATA_ROOT}/${ARCHIVE_NAME}"

# Support Hugging Face mirror endpoints via HF_ENDPOINT/HF_HUB_ENDPOINT.
HF_BASE_URL="${HF_ENDPOINT:-${HF_HUB_ENDPOINT:-https://huggingface.co}}"
HF_BASE_URL="${HF_BASE_URL%/}"
DEFAULT_DATA_URL="${HF_BASE_URL}/datasets/namespace-Pt/projects/resolve/main/long-llm.tar.gz?download=true"
DATA_URL="${DATA_URL:-${DEFAULT_DATA_URL}}"

mkdir -p "${DATA_ROOT}"

if [[ ! -w "${DATA_ROOT}" ]]; then
  echo "ERROR: '${DATA_ROOT}' is not writable. Try another path or run with proper permissions." >&2
  exit 1
fi

echo "Downloading dataset archive to: ${ARCHIVE_PATH}"
echo "Using DATA_URL: ${DATA_URL}"
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

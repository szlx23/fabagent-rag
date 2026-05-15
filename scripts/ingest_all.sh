#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RAG_PYTHON:-/home/szlx23/conda/envs/rag/bin/python}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m fabagent_rag.cli ingest-all "$@"

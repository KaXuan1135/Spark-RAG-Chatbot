#!/usr/bin/env bash
set -euo pipefail

QUESTION="${1:-What is this document about?}"

curl -sS http://localhost:${API_PORT:-8000}/query \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"${QUESTION}\"}"

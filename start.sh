#!/usr/bin/env bash
# Start the whole Graphiti Memory System stack and open the status
# dashboard in the browser once it's ready.
#
# Usage: ./start.sh [--no-open]
#   --no-open   start the stack but skip opening a browser tab (e.g. when
#               running headless/over SSH).
#
# What it does:
#   1. `docker compose up -d --build` (--build is cheap when nothing
#      changed — Docker layer cache skips the rebuild)
#   2. Polls the ingest service's /healthz until it responds (or times out)
#   3. Opens http://localhost:8100/dashboard in the default browser
#
# Rationale for polling /healthz specifically rather than just waiting a
# fixed number of seconds: ingest is usually the fastest of the 5 services
# to become ready, but on a cold start (fresh volumes, first-ever
# `docker compose up`) Postgres/Neo4j can take 10-30s to initialize, and
# ingest's own FastAPI process starts near-instantly regardless — so
# /healthz alone doesn't guarantee Postgres/Neo4j/Qdrant are up yet. The
# dashboard page itself handles that gracefully (see dashboard_metrics.py
# — each store's check fails independently and shows as "DOWN" without
# crashing the page), so opening the browser as soon as ingest responds
# is good enough: worst case, the user briefly sees a few services still
# starting up, which resolves within the first couple of /metrics polls.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DASHBOARD_URL="http://localhost:8100/dashboard"
HEALTHZ_URL="http://localhost:8100/healthz"
MAX_WAIT_SECONDS=90
OPEN_BROWSER=1

for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN_BROWSER=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [ ! -f .env ]; then
  echo "error: .env not found (copy .env.example and fill in secrets first)" >&2
  exit 1
fi

echo "==> docker compose up -d --build"
docker compose up -d --build

echo "==> waiting for ingest service to become healthy (timeout: ${MAX_WAIT_SECONDS}s)"
elapsed=0
until curl -sf "$HEALTHZ_URL" > /dev/null 2>&1; do
  if [ "$elapsed" -ge "$MAX_WAIT_SECONDS" ]; then
    echo "error: ingest service did not become healthy within ${MAX_WAIT_SECONDS}s" >&2
    echo "       check logs with: docker compose logs ingest" >&2
    exit 1
  fi
  sleep 2
  elapsed=$((elapsed + 2))
  printf '.'
done
echo " ready (${elapsed}s)"

echo "==> stack is up. Dashboard: $DASHBOARD_URL"

if [ "$OPEN_BROWSER" -eq 1 ]; then
  if command -v open > /dev/null 2>&1; then
    open "$DASHBOARD_URL"          # macOS
  elif command -v xdg-open > /dev/null 2>&1; then
    xdg-open "$DASHBOARD_URL"      # Linux
  else
    echo "note: could not detect a way to open a browser automatically — open $DASHBOARD_URL manually"
  fi
fi

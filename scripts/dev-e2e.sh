#!/usr/bin/env bash
#
# Dev end-to-end smoke: start a local tanseki-daemon on a throwaway vault, capture
# a decision through Kojutsu, read it back (client + MCP surface), then stop.
#
# Usage:  scripts/dev-e2e.sh
# Env:    TANSEKI_DIR   path to the Tanseki repo   (default: ../tanseki)
#         TANSEKI_PORT  HTTP port              (default: 8099)
#         PYTHON     interpreter with the package installed (default: python3)
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TANSEKI_DIR="${TANSEKI_DIR:-$(cd "$REPO_DIR/../tanseki" 2>/dev/null && pwd || true)}"
PORT="${TANSEKI_PORT:-8099}"
PYTHON="${PYTHON:-python3}"
WORK="$(mktemp -d)"

DAEMON_PID=""
cleanup() {
  [[ -n "$DAEMON_PID" ]] && kill "$DAEMON_PID" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

if [[ ! -d "$TANSEKI_DIR" ]]; then
  echo "Tanseki repo not found at '${TANSEKI_DIR:-<none>}' — set TANSEKI_DIR" >&2
  exit 1
fi

echo "Building tanseki-daemon…"
(cd "$TANSEKI_DIR" && ./gradlew :service:installDist -q)
# installDist names the distribution after the application, not the product.
DIST="$TANSEKI_DIR/service/build/install/tanseki-daemon/bin/tanseki-daemon"

mkdir -p "$WORK/vault"
TANSEKI_PATH="$WORK/vault" \
  TANSEKI_INDEX_DIR="$WORK/index" \
  TANSEKI_SOCKET="$WORK/tanseki.sock" \
  TANSEKI_HTTP_PORT="$PORT" \
  "$DIST" >"$WORK/daemon.log" 2>&1 &
DAEMON_PID=$!

code=""
for _ in $(seq 1 30); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/health" || true)"
  [[ "$code" == "200" ]] && break
  sleep 1
done
if [[ "$code" != "200" ]]; then
  echo "tanseki-daemon did not become healthy on :$PORT" >&2
  tail -20 "$WORK/daemon.log" >&2
  exit 1
fi
echo "tanseki-daemon healthy on :$PORT"

TANSEKI_URL="http://127.0.0.1:$PORT" \
  TANSEKI_COLLECTION=kojutsu-pilot \
  GITHUB_WEBHOOK_ALLOWED_REPOSITORIES=demo/repo \
  TANSEKI_OUTBOX_PATH="$WORK/outbox.db" \
  "$PYTHON" "$REPO_DIR/scripts/dev_e2e.py"

echo "OK — capture → Tanseki → read verified."

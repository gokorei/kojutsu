#!/usr/bin/env bash
#
# Start the Kojutsu dev console, optionally with a local Tanseki store.
#
# Usage:  ./start.sh [--with-tanseki] [--host H] [--port P] [console args...]
#
#   (default)     console only; TANSEKI_URL must point at a running Tanseki store
#   --with-tanseki   build (if needed) and start a local tanseki-daemon, point the
#                 console at it, and stop the daemon on exit
#
# Env:  CONSOLE_HOST (127.0.0.1), CONSOLE_PORT (8090), PYTHON (python3)
#       TANSEKI_DIR (../tanseki), TANSEKI_HTTP_PORT (8088), TANSEKI_PATH (~/.kojutsu/tanseki-vault),
#       TANSEKI_INDEX_DIR, TANSEKI_SOCKET, TANSEKI_LOG
#
# See docs/tanseki-quickstart.md.
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST="${CONSOLE_HOST:-127.0.0.1}"
PORT="${CONSOLE_PORT:-8090}"
PYTHON="${PYTHON:-python3}"

WITH_TANSEKI=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --with-tanseki) WITH_TANSEKI=1 ;;
    *) ARGS+=("$arg") ;;
  esac
done

if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
fi

DAEMON_PID=""
CONSOLE_PID=""
cleanup() {
  if [[ -n "$CONSOLE_PID" ]]; then
    kill "$CONSOLE_PID" 2>/dev/null || true
  fi
  if [[ -n "$DAEMON_PID" ]]; then
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if (( WITH_TANSEKI )); then
  TANSEKI_DIR="${TANSEKI_DIR:-$(cd "$REPO_DIR/../tanseki" 2>/dev/null && pwd || true)}"
  if [[ ! -d "$TANSEKI_DIR" ]]; then
    echo "Tanseki repo not found at '${TANSEKI_DIR:-<none>}' — set TANSEKI_DIR" >&2
    exit 1
  fi

  TANSEKI_HTTP_PORT="${TANSEKI_HTTP_PORT:-8088}"
  TANSEKI_PATH="${TANSEKI_PATH:-$HOME/.kojutsu/tanseki-vault}"
  TANSEKI_INDEX_DIR="${TANSEKI_INDEX_DIR:-$HOME/.kojutsu/tanseki-index}"
  TANSEKI_SOCKET="${TANSEKI_SOCKET:-$HOME/.kojutsu/tanseki.sock}"
  TANSEKI_LOG="${TANSEKI_LOG:-$HOME/.kojutsu/tanseki-daemon.log}"
  export TANSEKI_PATH TANSEKI_INDEX_DIR TANSEKI_SOCKET TANSEKI_HTTP_PORT

  # installDist names the distribution after the application, not the product:
  # `tanseki` is the product, the distribution and its binaries are `tanseki-*`.
  BIN="$TANSEKI_DIR/service/build/install/tanseki-daemon/bin/tanseki-daemon"
  if [[ ! -x "$BIN" ]]; then
    echo "Building tanseki-daemon…"
    (cd "$TANSEKI_DIR" && ./gradlew :service:installDist -q)
  fi

  mkdir -p "$TANSEKI_PATH" "$TANSEKI_INDEX_DIR" "$(dirname "$TANSEKI_LOG")"
  echo "tanseki: starting on :$TANSEKI_HTTP_PORT (vault $TANSEKI_PATH)"
  "$BIN" >"$TANSEKI_LOG" 2>&1 &
  DAEMON_PID=$!

  TANSEKI_URL="http://127.0.0.1:$TANSEKI_HTTP_PORT"
  export TANSEKI_URL

  code=""
  for _ in $(seq 1 30); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "$TANSEKI_URL/v1/health" || true)"
    [[ "$code" == "200" ]] && break
    kill -0 "$DAEMON_PID" 2>/dev/null || break
    sleep 1
  done
  if [[ "$code" != "200" ]]; then
    echo "tanseki-daemon did not become healthy on :$TANSEKI_HTTP_PORT — log tail:" >&2
    tail -20 "$TANSEKI_LOG" >&2
    exit 1
  fi
  echo "tanseki: reachable at $TANSEKI_URL"
else
  if [[ -z "${TANSEKI_URL:-}" ]]; then
    echo "TANSEKI_URL is unset — set it, or pass --with-tanseki (see docs/tanseki-quickstart.md)." >&2
    exit 1
  fi
  if curl -sf "${TANSEKI_URL%/}/v1/health" >/dev/null 2>&1; then
    echo "tanseki: reachable at $TANSEKI_URL"
  else
    echo "warning: Tanseki not reachable at $TANSEKI_URL (console will start anyway)" >&2
  fi
fi

echo "console: http://$HOST:$PORT"
if (( ${#ARGS[@]} )); then
  "$PYTHON" -m kojutsu.cli console "${ARGS[@]}" &
else
  "$PYTHON" -m kojutsu.cli console --host "$HOST" --port "$PORT" &
fi
CONSOLE_PID=$!
wait "$CONSOLE_PID" || true

#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT_DIR/.run"

echo "Stopping LLM Fine-tuning Workbench..."

stop_pid_file() {
  local file="$1"
  if [ -f "$file" ]; then
    local pid
    pid="$(cat "$file")"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "Stopping process $pid"
      kill "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$file"
  fi
}

stop_port() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Stopping service on port $port"
    echo "$pids" | xargs kill >/dev/null 2>&1 || true
  fi
}

stop_pid_file "$RUN_DIR/backend.pid"
stop_pid_file "$RUN_DIR/frontend.pid"
stop_port 8000
stop_port 8501

echo "Done."
read "?Press Enter to close this window. "

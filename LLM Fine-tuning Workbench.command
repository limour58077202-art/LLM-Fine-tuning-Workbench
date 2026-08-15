#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
LOG_DIR="$ROOT_DIR/logs"
RUN_DIR="$ROOT_DIR/.run"
BACKEND_URL="http://127.0.0.1:8000/health"
FRONTEND_URL="http://127.0.0.1:8501"

mkdir -p "$LOG_DIR" "$RUN_DIR"
cd "$ROOT_DIR"

echo "Starting LLM Fine-tuning Workbench..."
echo "Project: $ROOT_DIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Virtual environment not found. Creating .venv..."
  python3 -m venv "$VENV_DIR"
fi

if [ ! -x "$VENV_DIR/bin/uvicorn" ] || [ ! -x "$VENV_DIR/bin/streamlit" ]; then
  echo "Installing required Python packages. This may take a while..."
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VENV_DIR/bin/python" -m pip install \
    fastapi uvicorn streamlit pandas requests python-multipart \
    torch transformers datasets peft accelerate huggingface_hub \
    sentencepiece protobuf
fi

port_is_listening() {
  local port="$1"
  lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

wait_for_url() {
  local url="$1"
  local name="$2"
  local tries=40

  for _ in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is ready."
      return 0
    fi
    sleep 1
  done

  echo "$name did not become ready in time. Check logs in: $LOG_DIR"
  return 1
}

if port_is_listening 8000; then
  echo "Backend already appears to be running on port 8000."
else
  echo "Starting backend on http://127.0.0.1:8000 ..."
  nohup "$VENV_DIR/bin/uvicorn" backend.app:app \
    --host 127.0.0.1 \
    --port 8000 \
    --ws none \
    > "$LOG_DIR/backend.log" 2>&1 &
  backend_pid=$!
  echo "$backend_pid" > "$RUN_DIR/backend.pid"
  disown "$backend_pid" >/dev/null 2>&1 || true
fi

if port_is_listening 8501; then
  echo "Frontend already appears to be running on port 8501."
else
  echo "Starting frontend on http://127.0.0.1:8501 ..."
  nohup "$VENV_DIR/bin/streamlit" run frontend/streamlit_app.py \
    --server.headless true \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    > "$LOG_DIR/frontend.log" 2>&1 &
  frontend_pid=$!
  echo "$frontend_pid" > "$RUN_DIR/frontend.pid"
  disown "$frontend_pid" >/dev/null 2>&1 || true
fi

wait_for_url "$BACKEND_URL" "Backend"
wait_for_url "$FRONTEND_URL/_stcore/health" "Frontend"

echo "Opening browser..."
open "$FRONTEND_URL"

echo ""
echo "LLM Fine-tuning Workbench is running."
echo "Frontend: $FRONTEND_URL"
echo "Backend:  http://127.0.0.1:8000"
echo "Logs:     $LOG_DIR"
echo ""
echo "To stop it, double-click: Stop LLM Fine-tuning Workbench.command"
echo ""
read "?Press Enter to close this launcher window. The platform will keep running. "

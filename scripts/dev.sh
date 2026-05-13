#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="/home/szlx23/conda/envs/rag/bin/python"

cd "$ROOT_DIR"

echo "启动 Milvus..."
docker compose up -d

echo "启动后端：http://127.0.0.1:8000"
PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m uvicorn \
  fabagent_rag.api:app \
  --host 127.0.0.1 \
  --port 8000 &
BACKEND_PID=$!

echo "启动前端：http://127.0.0.1:5173"
cd "$ROOT_DIR/frontend"
npm run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

echo
echo "服务已启动："
echo "- 后端：http://127.0.0.1:8000"
echo "- 前端：http://127.0.0.1:5173"
echo
echo "按 Ctrl+C 停止前后端。Milvus 容器不会自动停止。"

trap 'kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true' INT TERM EXIT
wait

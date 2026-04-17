#!/usr/bin/env bash
set -euo pipefail

: "${INDEXTTS_MODEL_DIR:=/checkpoints}"
: "${INDEXTTS_VOICES_DIR:=/voices}"
: "${INDEXTTS_DEVICE:=auto}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"

export INDEXTTS_MODEL_DIR INDEXTTS_VOICES_DIR INDEXTTS_DEVICE HOST PORT LOG_LEVEL

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"

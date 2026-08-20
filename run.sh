#!/usr/bin/env sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNTIME="$PROJECT_ROOT/vendor/index-tts/.venv/bin/python"
if [ ! -x "$RUNTIME" ]; then
    "$PROJECT_ROOT/setup.sh"
fi
if [ -d "$PROJECT_ROOT/vendor/ffmpeg/bin" ]; then
    PATH="$PROJECT_ROOT/vendor/ffmpeg/bin:$PATH"
    export PATH
fi
cd "$PROJECT_ROOT"
exec "$RUNTIME" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

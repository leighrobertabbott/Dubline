#!/usr/bin/env sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_URL="http://127.0.0.1:8000"

open_browser() {
    attempt=0
    while [ "$attempt" -lt 180 ]; do
        if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 2 "$APP_URL" >/dev/null 2>&1; then
            if command -v xdg-open >/dev/null 2>&1; then xdg-open "$APP_URL" >/dev/null 2>&1 || true; fi
            return
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
}

if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 2 "$APP_URL" >/dev/null 2>&1; then
    if command -v xdg-open >/dev/null 2>&1; then xdg-open "$APP_URL" >/dev/null 2>&1 || true; fi
    exit 0
fi

printf '%s\n' "Starting Dubline. Keep this window open while you use the app."
open_browser &
BROWSER_PID=$!
trap 'kill "$BROWSER_PID" 2>/dev/null || true' EXIT HUP INT TERM
"$PROJECT_ROOT/run.sh"

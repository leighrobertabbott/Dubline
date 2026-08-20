#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PLATFORM=$(uname -s)
ARCHITECTURE=$(uname -m)
RUNTIME_REVISION="4f8792ff120cd3ea470dd511e997a17c86cddd10"
UV_VERSION="0.12.5"

if [ "$PLATFORM" = "Darwin" ]; then
    printf '%s\n' "Dubline detected macOS ($ARCHITECTURE)."
    printf '%s\n' "This release requires an NVIDIA CUDA GPU. Modern Macs use Apple Metal, so model installation is not available on macOS yet."
    exit 2
fi
if [ "$PLATFORM" != "Linux" ]; then
    printf '%s\n' "Dubline currently supports Windows and Linux with NVIDIA CUDA. Detected: $PLATFORM $ARCHITECTURE"
    exit 2
fi
case "$ARCHITECTURE" in
    x86_64|amd64) ;;
    *) printf '%s\n' "This release supports 64-bit Intel/AMD Linux. Detected: $ARCHITECTURE"; exit 2 ;;
esac

VENDOR_ROOT="$PROJECT_ROOT/vendor"
TOOLS_ROOT="$VENDOR_ROOT/tools"
UPSTREAM="$VENDOR_ROOT/index-tts"
RUNTIME="$UPSTREAM/.venv/bin/python"
mkdir -p "$TOOLS_ROOT"
chmod +x "$PROJECT_ROOT/setup.sh" "$PROJECT_ROOT/run.sh" "$PROJECT_ROOT/start-dubline.sh" 2>/dev/null || true

if command -v uv >/dev/null 2>&1; then
    UV=$(command -v uv)
else
    UV="$TOOLS_ROOT/uv"
    if [ ! -x "$UV" ]; then
        printf '%s\n' "Preparing Dubline's setup helper..."
        if command -v curl >/dev/null 2>&1; then
            curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_UNMANAGED_INSTALL="$TOOLS_ROOT" sh
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_UNMANAGED_INSTALL="$TOOLS_ROOT" sh
        else
            printf '%s\n' "Install curl or wget, then run setup.sh again."
            exit 1
        fi
    fi
fi

printf '%s\n' "Preparing Python 3.11..."
"$UV" python install 3.11

if [ ! -f "$UPSTREAM/pyproject.toml" ]; then
    printf '%s\n' "Downloading the pinned IndexTTS runtime..."
    if command -v git >/dev/null 2>&1; then
        git clone --filter=blob:none --no-checkout https://github.com/index-tts/index-tts.git "$UPSTREAM"
        git -C "$UPSTREAM" checkout --detach "$RUNTIME_REVISION"
    else
        ARCHIVE="$VENDOR_ROOT/index-tts-$RUNTIME_REVISION.zip"
        TEMPORARY=$(mktemp -d "$VENDOR_ROOT/index-tts-download.XXXXXX")
        trap 'rm -rf -- "$TEMPORARY"' EXIT HUP INT TERM
        if command -v curl >/dev/null 2>&1; then
            curl -L --fail --retry 3 -o "$ARCHIVE" "https://github.com/index-tts/index-tts/archive/$RUNTIME_REVISION.zip"
        else
            wget -O "$ARCHIVE" "https://github.com/index-tts/index-tts/archive/$RUNTIME_REVISION.zip"
        fi
        PYTHON=$("$UV" python find 3.11)
        "$PYTHON" -m zipfile -e "$ARCHIVE" "$TEMPORARY"
        DOWNLOADED_ROOT=$(find "$TEMPORARY" -mindepth 1 -maxdepth 1 -type d | head -n 1)
        [ -n "$DOWNLOADED_ROOT" ] || { printf '%s\n' "The IndexTTS download had an unexpected layout."; exit 1; }
        mv "$DOWNLOADED_ROOT" "$UPSTREAM"
        rm -f -- "$ARCHIVE"
        rmdir "$TEMPORARY"
        trap - EXIT HUP INT TERM
    fi
    printf '%s' "$RUNTIME_REVISION" > "$UPSTREAM/.dubline-runtime-revision"
fi

printf '%s\n' "Installing the tested local CUDA runtime. This first step can take several minutes..."
"$UV" sync --project "$UPSTREAM" --python 3.11 --frozen
"$RUNTIME" "$PROJECT_ROOT/scripts/patch_index_tts.py" "$UPSTREAM"
printf '%s\n' "Installing Dubline's local service..."
"$UV" pip install --python "$RUNTIME" -r "$PROJECT_ROOT/requirements.txt"
"$UV" pip install --python "$RUNTIME" "uv==$UV_VERSION" "hf-xet==1.6.0" "melband-roformer-infer==0.1.5" "numpy==2.2.6" "opencv-python==4.12.0.88"
"$UV" pip install --python "$RUNTIME" "llama-cpp-python==0.3.34" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
"$RUNTIME" -c "import torch, llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), 'CUDA LLM offload is unavailable'; print('CUDA translation offload ready')"
"$RUNTIME" -c "import fastapi, torch, uvicorn; print('Local service ready · PyTorch', torch.__version__)"

printf '%s\n' "Bootstrap complete. Run ./start-dubline.sh to open the setup wizard."

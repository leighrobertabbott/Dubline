from __future__ import annotations

"""Shared local-LLM helpers: tolerant JSON decoding and GPU-sized model loading.

Small quantized models emit *almost* valid JSON.  The common failures are an
unescaped quote inside a value, a truncated tail, and prose or reasoning text
wrapped around the object.  A worker must degrade to the safe faithful text on
those, never abort a two-hour dub with a JSONDecodeError.
"""

import json
import os
import re
from pathlib import Path


def decode_json(text: str, want: type | tuple[type, ...] = (dict, list)):
    """Return the first decodable JSON value of a wanted type, or None.

    Never raises.  Scans every opening bracket so reasoning blocks, markdown
    fences and trailing commentary around the payload are all tolerated.
    """
    if not text:
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, want):
            return value
    return None


def scrape_string_fields(text: str, names) -> dict[str, str]:
    """Recover "key": "value" pairs from JSON a decoder already rejected.

    The value is closed at the quote that is actually followed by the next key
    or the end of the object, so an unescaped quote inside dialogue (the single
    most common local-model defect) is kept as part of the text.
    """
    values: dict[str, str] = {}
    for name in names:
        match = re.search(
            rf'"{re.escape(str(name))}"\s*:\s*"(.*?)"\s*(?=,\s*"[^"]*"\s*:|\s*[}}\]]|\s*$)',
            text, re.DOTALL,
        )
        if not match:
            continue
        value = " ".join(_unescape(match.group(1)).split())
        if value:
            values[str(name)] = value
    return values


def _unescape(value: str) -> str:
    return (value.replace('\\"', '"').replace("\n", " ")
            .replace("\t", " ").replace("\\\\", "\\"))


def response_tokens(source_text: str, *, floor: int, ceiling: int, factor: float = 1.6) -> int:
    """Budget output tokens from input size.

    A fixed cap silently truncates long scenes, and a truncated JSON reply costs
    far more than the tokens it saved: the batch falls back to context-free
    per-line translation, which is exactly where accuracy collapses.
    """
    estimated = int(len(source_text) / 3.2 * factor) + 120
    return max(floor, min(ceiling, estimated))


_GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_block_count(path: str | Path) -> int | None:
    """Read a GGUF model's transformer block count from its header.

    llama.cpp needs to be told how many layers to place on the GPU, and the only
    honest source for that number is the file itself.  Guessing from the file
    size mis-sizes every offload on an 8 GB card, where being one layer wrong is
    the difference between a full offload and a spill to system RAM.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"GGUF":
                return None
            version = int.from_bytes(handle.read(4), "little")
            if version not in (2, 3):
                return None
            handle.read(8)  # tensor count
            pairs = int.from_bytes(handle.read(8), "little")

            def read_string() -> str:
                size = int.from_bytes(handle.read(8), "little")
                return handle.read(size).decode("utf-8", "replace")

            def skip_value(kind: int) -> None:
                if kind == 8:
                    handle.seek(int.from_bytes(handle.read(8), "little"), 1)
                elif kind == 9:
                    element = int.from_bytes(handle.read(4), "little")
                    count = int.from_bytes(handle.read(8), "little")
                    if element == 8:
                        for _ in range(count):
                            handle.seek(int.from_bytes(handle.read(8), "little"), 1)
                    else:
                        handle.seek(_GGUF_SCALAR_SIZES.get(element, 0) * count, 1)
                else:
                    handle.seek(_GGUF_SCALAR_SIZES.get(kind, 0), 1)

            for _ in range(min(pairs, 4096)):
                key = read_string()
                kind = int.from_bytes(handle.read(4), "little")
                if key.endswith(".block_count") and kind in {4, 5, 10, 11}:
                    size = _GGUF_SCALAR_SIZES[kind]
                    return int.from_bytes(handle.read(size), "little")
                skip_value(kind)
    except (OSError, ValueError, UnicodeError):
        return None
    return None


def gpu_layer_count(model_path: str | Path, n_ctx: int = 8192) -> int:
    """Place as many layers on the GPU as measurably fit, and no more.

    An explicit DUB_LLAMA_GPU_LAYERS wins.  Otherwise this sizes the split from
    the real block count and the real free VRAM, so a model larger than the card
    still runs at the best split the hardware allows instead of falling back to
    an arbitrary fixed number.
    """
    setting = str(os.getenv("DUB_LLAMA_GPU_LAYERS", "auto")).strip().lower()
    if setting not in {"", "auto"}:
        try:
            return int(setting)
        except ValueError:
            pass
    try:
        from app.services.gpu_safety import query_nvidia

        free_mb = int(query_nvidia()["free_mb"])
    except Exception:
        return 20
    try:
        weights_mb = Path(model_path).stat().st_size / 1024 / 1024
    except OSError:
        return 20
    blocks = gguf_block_count(model_path)
    # KV cache for a 7B/8B class model is roughly 0.13 MB per token of context;
    # the remainder is compute buffers and llama.cpp's own overhead.
    overhead_mb = n_ctx * 0.13 + 600
    budget_mb = free_mb - overhead_mb - _vram_headroom_mb()
    if blocks is None:
        return -1 if budget_mb >= weights_mb else 20
    if budget_mb >= weights_mb:
        return -1
    # Non-block tensors (embeddings, output head) stay resident too, so charge
    # the split against the per-block share rather than the whole file.
    per_block_mb = weights_mb / max(1, blocks + 2)
    return max(0, min(blocks, int(budget_mb / max(per_block_mb, 1e-6))))


def _vram_headroom_mb() -> int:
    try:
        return max(0, int(os.getenv("DUB_LLAMA_VRAM_HEADROOM_MB", "400")))
    except ValueError:
        return 400


def load_llama(model_path: str, *, n_ctx: int = 8192):
    """Construct a llama.cpp model with Dubline's shared runtime settings."""
    from llama_cpp import Llama

    return Llama(
        model_path=model_path, n_ctx=n_ctx, n_batch=512,
        n_threads=int(os.getenv("DUB_LLAMA_THREADS", "10")),
        n_threads_batch=int(os.getenv("DUB_LLAMA_BATCH_THREADS", "12")),
        n_gpu_layers=gpu_layer_count(model_path, n_ctx), verbose=False,
    )

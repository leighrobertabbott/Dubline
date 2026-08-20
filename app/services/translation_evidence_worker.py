from __future__ import annotations

"""Translate selective second-ASR evidence without adapting it for dub timing."""

import argparse
import json
import os
import sys
from pathlib import Path

from app.services.llm import decode_json, load_llama, response_tokens


def parse_object(text: str) -> dict:
    value = decode_json(text, dict)
    return value if isinstance(value, dict) else {}


def ask(llm, prompt: str, max_tokens: int = 700) -> str:
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], temperature=0.0,
        top_p=.85, max_tokens=max_tokens,
    )
    return str(result["choices"][0]["message"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = list(spec.get("items") or [])
    allowed_ids = {int(item["id"]) for item in items}
    results: list[dict] = []
    if args.output.is_file():
        try:
            prior = json.loads(args.output.read_text(encoding="utf-8"))
            results = [item for item in prior if isinstance(item, dict)
                       and int(item.get("id", -1)) in allowed_ids]
        except (OSError, ValueError, json.JSONDecodeError):
            results = []
    completed = {int(item["id"]) for item in results}
    pending = [item for item in items if int(item["id"]) not in completed]
    if not pending:
        print(json.dumps({"progress": 1.0, "index": max(0, len(results) - 1)}), flush=True)
        return
    import torch  # registers CUDA/cuBLAS DLLs before llama.cpp on Windows  # noqa: F401
    llm = load_llama(spec["model"], n_ctx=4096)
    for batch_start in range(0, len(pending), 8):
        batch = pending[batch_start:batch_start + 8]
        lines = "\n".join(
            f"ID {item['id']} ({item.get('language') or 'auto'}): {item.get('source', '')}"
            for item in batch
        )
        wanted = ", ".join(str(item["id"]) for item in batch)
        prompt = f"""Translate each source-language film line literally but naturally into English.
Preserve polarity, quantities, intent, names and who does what. Do not adapt for timing, add context,
or copy any subtitle. Return only one JSON object mapping these IDs to English strings: {wanted}
{lines}"""
        values = parse_object(ask(llm, prompt,
            response_tokens(lines, floor=600, ceiling=1600)))
        for item in batch:
            cue_id = int(item["id"])
            translated = values.get(str(cue_id))
            if not isinstance(translated, str) or not translated.strip():
                translated = ask(llm, "Translate to English only:\n" + str(item.get("source", "")), 160)
            results.append({
                "id": cue_id, "text": " ".join(str(translated).strip().strip('"').split()),
                "model": "Hy-MT2-7B Q4 independent second-ASR translation",
            })
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": len(results) / max(1, len(items)),
                          "index": len(results) - 1}), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

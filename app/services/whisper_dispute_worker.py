from __future__ import annotations

"""Selective CUDA Whisper transcription for disputed subtitle/ASR lines."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import whisper


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
    model = whisper.load_model(spec.get("model", "turbo"), download_root=spec["cache"])
    use_cuda = bool(torch.cuda.is_available())
    for item in pending:
        response = model.transcribe(
            str(item["path"]), task="transcribe", language=item.get("language") or None,
            fp16=use_cuda, verbose=False, temperature=0, condition_on_previous_text=False,
            word_timestamps=True,
        )
        segments = list(response.get("segments") or [])
        text = " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()
        weights = [max(.05, float(segment.get("end", 0)) - float(segment.get("start", 0)))
                   for segment in segments]
        weight_sum = sum(weights)
        avg_logprob = (sum(float(segment.get("avg_logprob", -5.0)) * weight
                           for segment, weight in zip(segments, weights)) / weight_sum
                       if weight_sum else -5.0)
        no_speech = (sum(float(segment.get("no_speech_prob", 1.0)) * weight
                         for segment, weight in zip(segments, weights)) / weight_sum
                     if weight_sum else 1.0)
        words = [word for segment in segments for word in (segment.get("words") or [])]
        word_confidence = (sum(float(word.get("probability", 0.0)) for word in words) / len(words)
                           if words else 0.0)
        results.append({
            "id": int(item["id"]), "text": text,
            "language": response.get("language") or item.get("language"),
            "confidence": round(math.exp(min(0.0, avg_logprob)), 4),
            "word_confidence": round(word_confidence, 4),
            "no_speech_probability": round(no_speech, 6),
            "model": "Whisper large-v3-turbo selective second opinion",
        })
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": len(results) / max(1, len(items)),
                          "index": len(results) - 1}), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

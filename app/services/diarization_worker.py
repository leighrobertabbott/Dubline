from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Community-1 is used entirely offline; disable anonymous metrics as well.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

import torch
import soundfile as sf


class JsonProgressHook:
    """Translate pyannote's internal batches into stable whole-stage progress."""

    stages = {
        "segmentation": (0.01, 0.56),
        "speaker_counting": (0.57, 0.03),
        "embeddings": (0.60, 0.37),
        "discrete_diarization": (0.97, 0.02),
    }

    def __init__(self) -> None:
        self.last_value = 0.01
        self.last_emit = 0.0

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None) -> None:
        base, span = self.stages.get(str(step_name), (self.last_value, 0.0))
        fraction = 1.0 if completed is None or total in {None, 0} else min(1.0, completed / total)
        value = max(self.last_value, min(.99, base + span * fraction))
        now = time.monotonic()
        if value - self.last_value >= .002 or now - self.last_emit >= 3.0 or fraction >= 1.0:
            print(json.dumps({"progress": round(value, 5), "step": str(step_name),
                              "completed": int(completed) if completed is not None else None,
                              "total": int(total) if total is not None else None}), flush=True)
            self.last_emit = now
        self.last_value = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    from pyannote.audio import Pipeline

    if args.device == "cpu":
        torch.set_num_threads(max(1, min(12, args.cpu_threads)))
        torch.set_num_interop_threads(1)
    pipeline = Pipeline.from_pretrained(str(args.model))
    # Community-1 ships with batch sizes of 32.  This worker only receives a
    # bounded film chunk and is destroyed after it, so model memory and CUDA
    # context cannot accumulate across a feature film.
    pipeline.segmentation_batch_size = max(1, args.batch_size)
    pipeline.embedding_batch_size = max(1, args.batch_size)
    pipeline.to(torch.device(args.device))
    print(json.dumps({"progress": 0.01, "device": args.device,
                      "batch_size": args.batch_size}), flush=True)
    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    result = pipeline({"waveform": torch.from_numpy(audio.T).contiguous(),
                       "sample_rate": sample_rate}, hook=JsonProgressHook())
    if hasattr(result, "serialize"):
        payload = result.serialize()
        embeddings = getattr(result, "speaker_embeddings", None)
        annotation = getattr(result, "speaker_diarization", None)
        if embeddings is not None and annotation is not None:
            labels = list(annotation.labels())
            payload["speaker_embeddings"] = {
                str(label): [round(float(value), 8) for value in embeddings[index].tolist()]
                for index, label in enumerate(labels) if index < len(embeddings)
            }
    else:
        payload = {
            "diarization": [{"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
                            for turn, _, speaker in result.itertracks(yield_label=True)],
            "exclusive_diarization": [],
        }
    payload["version"] = 2
    payload["analysis"] = {"model": "pyannote-community-1", "device": args.device,
                           "batch_size": args.batch_size}
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"progress": 1.0}), flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()

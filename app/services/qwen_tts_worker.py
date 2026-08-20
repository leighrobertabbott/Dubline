from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch

from app.services.tts_worker import fit_audio, line_cooldown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        spec["model"], device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    for position, item in enumerate(spec["items"]):
        raw = Path(item["raw"]); fitted = Path(item["fitted"])
        raw.parent.mkdir(parents=True, exist_ok=True); fitted.parent.mkdir(parents=True, exist_ok=True)
        reference_text = str(item.get("reference_text") or "").strip()
        clone = {"ref_audio": item["reference"]}
        if reference_text:
            clone.update({"ref_text": reference_text, "x_vector_only_mode": False})
        else:
            # This remains an explicit last resort for an uncertain cue whose
            # source transcript could not be recovered.  It is surfaced in QC.
            clone["x_vector_only_mode"] = True
        wavs, sample_rate = model.generate_voice_clone(
            text=item["text"], language=item.get("language", "English"), **clone,
        )
        sf.write(raw, wavs[0], sample_rate)
        metrics = fit_audio(raw, fitted, float(item["target"]),
                            float(item.get("fit_limit_percent", 8.0)))
        print(json.dumps({"progress": (position + 1) / max(1, len(spec["items"])),
                          "cue_index": item["cue_index"], "full_reference_clone": bool(reference_text),
                          **metrics}), flush=True)
        line_cooldown()
    os._exit(0)


if __name__ == "__main__":
    main()

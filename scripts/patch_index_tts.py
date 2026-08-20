from __future__ import annotations

"""Apply Dubline's small, pinned laptop-runtime adjustment idempotently."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    args = parser.parse_args()
    target = args.upstream / "indextts" / "infer_v2_5.py"
    source = target.read_text(encoding="utf-8")
    # Default to the model's own value. Making the step count configurable is
    # the point of the hook; silently shipping a lower one trades away voice
    # quality in the mel stage for speed the pipeline no longer needs.
    replacement = (
        'diffusion_steps = max(8, min(25, int(os.getenv('
        '"INDEXTTS_DIFFUSION_STEPS", "25"))))'
    )
    superseded = replacement.replace('"25"))))', '"16"))))')
    if superseded in source:
        target.write_text(source.replace(superseded, replacement), encoding="utf-8")
        return
    if replacement in source:
        return
    original = "diffusion_steps = 25"
    if source.count(original) != 1:
        raise RuntimeError("The pinned IndexTTS diffusion hook no longer matches its tested revision")
    target.write_text(source.replace(original, replacement), encoding="utf-8")


if __name__ == "__main__":
    main()

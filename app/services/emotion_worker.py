from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from app.services.tts_worker import line_cooldown
from app.services.tts import ContextEmotionAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contexts = manifest["contexts"]
    analyzer = ContextEmotionAnalyzer(manifest["mode"], preview=manifest.get("preview", False))
    results = []
    for index, context in enumerate(contexts):
        results.append(analyzer.analyze(context))
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": (index + 1) / max(1, len(contexts)), "index": index}), flush=True)
        if not manifest.get("preview", False):
            line_cooldown()
    # PyTorch/Transformers finalizers have caused torch_cpu.dll access violations on
    # Windows after moving Qwen between CPU and CUDA. The worker owns the process, so
    # an immediate successful exit is the deterministic and safe teardown boundary.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

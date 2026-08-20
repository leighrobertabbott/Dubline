from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from app.services.subprocess_control import controlled_lines, terminate_process


TRANSLATION_QC_REVISION = "blind-source-translation-and-english-gate-v3"


def validate_translations(cues: list[dict], folder: Path,
                          progress: Callable[[float, int], None],
                          checkpoint: Callable[[], None]) -> list[dict]:
    """Run an independent bilingual judge (Qwen3), never the Hy-MT2 generator."""
    model = Path(os.getenv("TRANSLATION_QC_MODEL", "vendor/qwen3-8b/Qwen3-8B-Q4_K_M.gguf")).resolve()
    if not model.is_file():
        for cue in cues:
            cue["translation_qc"] = {"available": False, "passed": False,
                                     "reason": "independent bilingual QC model is missing"}
        return cues
    manifest = folder / "translation-qc-manifest.json"
    output = folder / "translation-qc.json"
    # Keep the prompt/evidence policy in the cache signature. A judge result from
    # an older policy must never survive a semantic-QC upgrade.
    manifest_value = {"model": str(model), "judge_revision": TRANSLATION_QC_REVISION, "cues": cues}
    manifest_text = json.dumps(manifest_value, ensure_ascii=False, sort_keys=True)
    signature = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    signature_path = folder / "translation-qc-input.sha256"
    prior_signature = (signature_path.read_text(encoding="utf-8").strip()
                       if signature_path.is_file() else "")
    if prior_signature and prior_signature != signature:
        output.unlink(missing_ok=True)
    # A partial result created by the immediately preceding worker predates the
    # signature sidecar; adopt it once. Future cue edits invalidate it exactly.
    signature_path.write_text(signature, encoding="utf-8")
    manifest.write_text(manifest_text, encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.services.translation_qc_worker", "--manifest", str(manifest),
         "--output", str(output)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-20:]
            try:
                event = json.loads(line)
                progress(float(event["progress"]), int(event.get("index", 0)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        code = process.wait()
    except BaseException:
        if process.poll() is None:
            terminate_process(process)
        raise
    if code != 0 or not output.is_file():
        raise RuntimeError("Independent translation QC failed: " + "\n".join(tail[-10:]))
    judged = json.loads(output.read_text(encoding="utf-8"))
    by_id = {int(item["id"]): item for item in judged}
    for cue in cues:
        cue["translation_qc"] = by_id.get(int(cue["id"]), {
            "available": True, "passed": False, "adequacy": 0.0,
            "reason": "independent judge returned no evidence",
        })
    return cues

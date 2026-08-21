from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

from app.services.analysis_cache import restore_json_artifact, store_json_artifact
from app.services.subprocess_control import controlled_lines, terminate_process


FACE_REGISTRY_VERSION = 2
FACE_REGISTRY_CACHE_ARTIFACT = "face-registry-v2.json"


def upgrade_legacy_face_registry(registry: Path) -> bool:
    """Safely migrate a completed pre-cache registry without rescanning video."""
    try:
        import numpy as np
        from app.services.visual_speakers_worker import consolidate_identities, unit
        value = json.loads(registry.read_text(encoding="utf-8"))
        if int(value.get("version", -1)) == FACE_REGISTRY_VERSION:
            return True
        identities = value.get("identities", [])
        prototypes = [unit(np.asarray(item["embedding"], dtype=np.float32)) for item in identities]
        counts = [max(1, int(item.get("observations", 1))) for item in identities]
        prototypes, counts = consolidate_identities(prototypes, counts)
        migrated = {
            "version": FACE_REGISTRY_VERSION,
            "model": "OpenCV SFace 2021dec",
            "similarity_threshold": .37,
            "consolidation_similarity": .42,
            "minimum_observations": 2,
            "identities": [
                {"face_id": index + 1, "observations": counts[index],
                 "embedding": [round(float(component), 7) for component in prototype]}
                for index, prototype in enumerate(prototypes)
            ],
        }
        temporary = registry.with_suffix(registry.suffix + ".tmp")
        temporary.write_text(json.dumps(migrated, ensure_ascii=False), encoding="utf-8")
        temporary.replace(registry)
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _run_worker(command: list[str], output: Path, progress: Callable[[float], None],
                checkpoint: Callable[[], None]) -> tuple[bool, str]:
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-12:]
            try:
                progress(float(json.loads(line)["progress"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        return process.wait() == 0 and output.is_file(), "\n".join(tail)
    except BaseException:
        if process.poll() is None:
            terminate_process(process)
        raise
    finally:
        if process.poll() is None:
            terminate_process(process)
        if process.stdout:
            process.stdout.close()


def build_face_registry(video: Path, folder: Path, duration: float,
                        progress: Callable[[float], None], checkpoint: Callable[[], None],
                        cache_key: str | None = None) -> dict:
    """Cluster recurring faces across the whole original film before cue assignment."""
    model_dir = Path(os.getenv("OPENCV_FACE_MODEL_DIR", "vendor/opencv-face")).resolve()
    detector = model_dir / "face_detection_yunet_2023mar.onnx"
    recognizer = model_dir / "face_recognition_sface_2021dec.onnx"
    registry = folder / "face-registry.json"
    if registry.is_file():
        data = json.loads(registry.read_text(encoding="utf-8"))
        if (int(data.get("version", -1)) == FACE_REGISTRY_VERSION
                or upgrade_legacy_face_registry(registry)):
            data = json.loads(registry.read_text(encoding="utf-8"))
            cached = bool(cache_key and store_json_artifact(
                folder, cache_key, FACE_REGISTRY_CACHE_ARTIFACT, registry,
                expected_version=FACE_REGISTRY_VERSION))
            return {"available": True, "identities": len(data.get("identities", [])),
                    "path": str(registry), "cache_hit": False, "cached": cached}
        registry.unlink(missing_ok=True)
    if cache_key and restore_json_artifact(
            folder, cache_key, FACE_REGISTRY_CACHE_ARTIFACT, registry,
            expected_version=FACE_REGISTRY_VERSION):
        data = json.loads(registry.read_text(encoding="utf-8"))
        return {"available": True, "identities": len(data.get("identities", [])),
                "path": str(registry), "cache_hit": True}
    if not detector.is_file() or not recognizer.is_file():
        return {"available": False, "identities": 0}
    manifest = folder / "face-registry-scan.json"
    manifest.write_text(json.dumps([{"id": 0, "start": 0.0, "end": duration}]), encoding="utf-8")
    diagnostic = folder / "face-registry-scan-result.json"
    command = [
        sys.executable, "-m", "app.services.visual_speakers_worker", "--video", str(video),
        "--cues", str(manifest), "--detector", str(detector), "--recognizer", str(recognizer),
        "--output", str(diagnostic), "--registry-output", str(registry), "--sample-fps", ".75",
    ]
    ok, tail = _run_worker(command, registry, progress, checkpoint)
    if not ok:
        return {"available": False, "identities": 0, "error": tail}
    data = json.loads(registry.read_text(encoding="utf-8"))
    cached = bool(cache_key and store_json_artifact(
        folder, cache_key, FACE_REGISTRY_CACHE_ARTIFACT, registry,
        expected_version=FACE_REGISTRY_VERSION))
    return {"available": True, "identities": len(data.get("identities", [])),
            "path": str(registry), "cache_hit": False, "cached": cached}


def fuse_visual_speakers(cues: list[dict], visual: dict[int, dict]) -> dict:
    """Fuse reliable face activity with audio clusters without requiring a face."""
    votes: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for cue in cues:
        evidence = visual.get(int(cue["id"]), {})
        cue["visual_speaker"] = evidence
        face_id = evidence.get("active_face_id")
        if (face_id and float(evidence.get("active_speaker_confidence", 0)) >= .70
                and float(cue.get("speaker_confidence", 0)) >= .75):
            votes[int(face_id)][int(cue["speaker_id"])] += (
                float(evidence["active_speaker_confidence"]) * float(cue["speaker_confidence"]))
    anchors = {}
    for face_id, weights in votes.items():
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        total = sum(weights.values())
        if ranked and total and ranked[0][1] / total >= .60:
            anchors[face_id] = (ranked[0][0], ranked[0][1] / total)
    reassigned = 0
    for cue in cues:
        evidence = cue.get("visual_speaker", {})
        face_id = evidence.get("active_face_id")
        anchor = anchors.get(int(face_id)) if face_id else None
        if (anchor and float(evidence.get("active_speaker_confidence", 0)) >= .72
                and (float(cue.get("speaker_confidence", 0)) < .62
                     or float(evidence.get("active_speaker_confidence", 0)) > float(cue.get("speaker_confidence", 0)) + 0.15)):
            cue["speaker_id"] = anchor[0]
            cue["speaker"] = f"Voice {anchor[0]}"
            cue["speaker_confidence"] = round(.6 * float(evidence["active_speaker_confidence"])
                                               + .4 * anchor[1], 3)
            cue["speaker_assignment"] = "audiovisual anchor"
            reassigned += 1
        cue["mouth_visible"] = bool(evidence.get("mouth_visible"))

    # A short diagnostic scene can collapse two actors into one audio cluster.
    # Repeated, decisive mouth activity from an otherwise unanchored recurring
    # face is enough to create an anonymous voice identity when visual evidence is strong.
    candidates: dict[int, list[dict]] = defaultdict(list)
    for cue in cues:
        evidence = cue.get("visual_speaker", {})
        face_id = evidence.get("active_face_id")
        if (face_id and int(face_id) not in anchors and cue.get("mouth_visible")
                and float(evidence.get("active_speaker_confidence", 0)) >= .82
                and (float(cue.get("speaker_confidence", 0)) < .62
                     or float(evidence.get("active_speaker_confidence", 0)) > float(cue.get("speaker_confidence", 0)) + 0.18)):
            candidates[int(face_id)].append(cue)
    next_speaker = max((int(cue.get("speaker_id", 0)) for cue in cues), default=0) + 1
    created = 0
    for face_id, lines in candidates.items():
        total_duration = sum(max(0.0, float(cue["end"]) - float(cue["start"])) for cue in lines)
        if len(lines) < 2 or total_duration < .65:
            continue
        confidence = min(.86, .62 + .08 * len(lines))
        for cue in lines:
            cue["speaker_id"] = next_speaker
            cue["speaker"] = f"Voice {next_speaker}"
            cue["speaker_confidence"] = round(confidence, 3)
            cue["speaker_assignment"] = "repeated active face; anonymous visual voice"
        next_speaker += 1; created += 1
    return {"anchors": len(anchors), "reassigned": reassigned,
            "created_visual_voices": created,
            "visible_cues": sum(bool(c.get("visual_speaker", {}).get("active_face_id")) for c in cues)}


def register_visual_speakers(video: Path, cues: list[dict], folder: Path,
                             progress: Callable[[float], None], checkpoint: Callable[[], None]) -> dict:
    model_dir = Path(os.getenv("OPENCV_FACE_MODEL_DIR", "vendor/opencv-face")).resolve()
    detector = model_dir / "face_detection_yunet_2023mar.onnx"
    recognizer = model_dir / "face_recognition_sface_2021dec.onnx"
    if not detector.is_file() or not recognizer.is_file():
        return {"available": False, "anchors": 0, "reassigned": 0}
    manifest = folder / "visual-speaker-cues.json"
    manifest.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
    output = folder / "visual-speaker-analysis.json"
    command = [sys.executable, "-m", "app.services.visual_speakers_worker", "--video", str(video),
         "--cues", str(manifest), "--detector", str(detector), "--recognizer", str(recognizer),
         "--output", str(output)]
    registry = folder / "face-registry.json"
    if registry.is_file():
        command += ["--registry", str(registry)]
    ok, tail = _run_worker(command, output, progress, checkpoint)
    if not ok:
        return {"available": False, "anchors": 0, "reassigned": 0, "error": tail}
    visual = {int(item["id"]): item for item in json.loads(output.read_text(encoding="utf-8"))}
    return {"available": True, **fuse_visual_speakers(cues, visual)}

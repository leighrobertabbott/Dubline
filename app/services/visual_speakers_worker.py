from __future__ import annotations

"""CPU audiovisual speaker anchors using OpenCV's local YuNet/SFace models.

This is deliberately confidence-gated.  It does not assign a speaker merely
because a face is present; it requires repeated face visibility, measurable
mouth motion, and a clear margin over every other visible face.
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


def unit(value: np.ndarray) -> np.ndarray:
    value = value.reshape(-1).astype(np.float32)
    return value / (np.linalg.norm(value) + 1e-9)


def consolidate_identities(prototypes: list[np.ndarray], counts: list[int],
                           *, minimum_observations: int = 2,
                           merge_similarity: float = .42) -> tuple[list[np.ndarray], list[int]]:
    """Merge convincing duplicate face centroids and discard one-frame noise."""
    retained = [(unit(prototype), max(1, int(count)))
                for prototype, count in zip(prototypes, counts)
                if int(count) >= minimum_observations]
    while len(retained) > 1:
        best_pair: tuple[int, int] | None = None
        best_score = merge_similarity
        for left in range(len(retained)):
            for right in range(left + 1, len(retained)):
                score = float(np.dot(retained[left][0], retained[right][0]))
                if score > best_score:
                    best_score = score
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        left_embedding, left_count = retained[left]
        right_embedding, right_count = retained[right]
        combined_count = left_count + right_count
        combined = unit((left_embedding * left_count + right_embedding * right_count) / combined_count)
        retained[left] = (combined, combined_count)
        retained.pop(right)
    retained.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in retained], [item[1] for item in retained]


def mouth_patch(gray: np.ndarray, face: np.ndarray) -> np.ndarray | None:
    x, y, w, h = [float(v) for v in face[:4]]
    if w < 34 or h < 34:
        return None
    # YuNet exposes both mouth corners.  Expand around their centre while
    # remaining inside the detected face so distance shots fail safely.
    mouth = face[10:14].reshape(2, 2)
    cx, cy = mouth.mean(axis=0)
    mw = max(10.0, abs(float(mouth[0, 0] - mouth[1, 0])) * 1.65)
    mh = max(8.0, h * 0.22)
    frame_height, frame_width = gray.shape[:2]
    # YuNet may return a box/landmark a few pixels outside the decoded frame at
    # hard cuts and anamorphic edges.  Clamp both axes to the actual image, not
    # merely to the (possibly out-of-frame) face box.
    x1 = max(0, min(frame_width, int(max(x, cx - mw))))
    x2 = max(0, min(frame_width, int(min(x + w, cx + mw))))
    y1 = max(0, min(frame_height, int(max(y, cy - mh * .55))))
    y2 = max(0, min(frame_height, int(min(y + h, cy + mh * 1.05))))
    if x2 - x1 < 8 or y2 - y1 < 6:
        return None
    return cv2.resize(gray[y1:y2, x1:x2], (40, 20), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cues", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--registry-output", type=Path)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    args = parser.parse_args()

    cues = json.loads(args.cues.read_text(encoding="utf-8"))
    result = {int(cue.get("id", i + 1)): {"samples": 0, "faces": {},
                                                   "max_concurrent_faces": 0, "single_face_samples": 0}
              for i, cue in enumerate(cues)}
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit("Could not open video for audiovisual speaker analysis")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = cv2.FaceDetectorYN.create(str(args.detector), "", (width, height), .72, .3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(args.recognizer), "")
    step = max(1, round(fps / max(.5, args.sample_fps)))
    prototypes: list[np.ndarray] = []
    prototype_counts: list[int] = []
    if args.registry and args.registry.is_file():
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        for identity in registry.get("identities", []):
            prototypes.append(unit(np.asarray(identity["embedding"], dtype=np.float32)))
            prototype_counts.append(max(1, int(identity.get("observations", 1))))
    previous_mouth: dict[int, tuple[np.ndarray, int]] = {}
    frame_index = -1
    cue_index = 0
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    registry_scan = bool(args.registry_output and len(cues) == 1 and int(cues[0].get("id", -1)) == 0)
    next_sample = 0
    partial = args.registry_output.with_suffix(".partial.json") if args.registry_output else None
    if registry_scan and partial and partial.is_file():
        try:
            saved = json.loads(partial.read_text(encoding="utf-8"))
            prototypes = [unit(np.asarray(item["embedding"], dtype=np.float32))
                          for item in saved.get("identities", [])]
            prototype_counts = [max(1, int(item.get("observations", 1)))
                                for item in saved.get("identities", [])]
            next_sample = max(0, min(total_frames, int(saved.get("next_frame", 0))))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            prototypes, prototype_counts, next_sample = [], [], 0

    def save_partial() -> None:
        if not registry_scan or not partial:
            return
        value = {"version": 1, "next_frame": next_sample, "total_frames": total_frames,
                 "identities": [
                     {"observations": int(prototype_counts[index]),
                      "embedding": [round(float(component), 7) for component in prototype]}
                     for index, prototype in enumerate(prototypes)
                 ]}
        temporary = partial.with_suffix(partial.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temporary.replace(partial)

    registry_checkpoint_interval = max(step, round(total_frames / 200))
    progress_interval = max(step, round(total_frames / 100))
    next_registry_checkpoint = next_sample + registry_checkpoint_interval
    next_progress_frame = next_sample

    while True:
        if registry_scan:
            if next_sample >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_sample)
            ok, frame = cap.read()
            frame_index = next_sample
            next_sample += step
        else:
            ok, frame = cap.read()
            frame_index += 1
        if not ok:
            break
        if not registry_scan and frame_index % step:
            continue
        now = frame_index / fps
        while cue_index < len(cues) and float(cues[cue_index]["end"]) < now:
            cue_index += 1
        active = []
        probe = cue_index
        while probe < len(cues) and float(cues[probe]["start"]) <= now:
            if now <= float(cues[probe]["end"]):
                active.append(cues[probe])
            probe += 1
        if not active:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, faces = detector.detect(frame)
        observations: dict[int, tuple[float, float]] = {}
        for face in ([] if faces is None else faces):
            x, y, w, h = face[:4]
            if min(w, h) < 34 or w * h < width * height * .00045:
                continue
            try:
                aligned = recognizer.alignCrop(frame, face)
                embedding = unit(recognizer.feature(aligned))
            except cv2.error:
                continue
            scores = [float(np.dot(embedding, value)) for value in prototypes]
            # SFace cosine boundary: use 0.52 to prevent merging different faces across scenes,
            # and only update the prototype representation on high-confidence matches (>= 0.60).
            identity = int(np.argmax(scores)) if scores and max(scores) >= .52 else len(prototypes)
            if identity == len(prototypes):
                prototypes.append(embedding); prototype_counts.append(1)
            else:
                count = prototype_counts[identity]
                if scores[identity] >= .60:
                    # Bound maximum sample count to prevent prototype immutability while preventing drift
                    effective_count = min(count, 20)
                    prototypes[identity] = unit((prototypes[identity] * effective_count + embedding) / (effective_count + 1))
                prototype_counts[identity] += 1
            patch = mouth_patch(gray, face)
            motion = 0.0
            if (patch is not None and identity in previous_mouth
                    and frame_index - previous_mouth[identity][1] <= step * 2):
                motion = float(np.mean(cv2.absdiff(patch, previous_mouth[identity][0])) / 255.0)
            if patch is not None:
                previous_mouth[identity] = (patch, frame_index)
            identity += 1
            area = float(w * h / (width * height))
            if identity not in observations or area > observations[identity][1]:
                observations[identity] = (motion, area)
        for cue in active:
            target = result[int(cue["id"])]
            target["samples"] += 1
            target["max_concurrent_faces"] = max(target["max_concurrent_faces"], len(observations))
            target["single_face_samples"] += int(len(observations) == 1)
            for identity, (motion, area) in observations.items():
                face_data = target["faces"].setdefault(str(identity), {"visible": 0, "motions": [], "area": 0.0})
                face_data["visible"] += 1
                face_data["motions"].append(motion)
                face_data["area"] = max(face_data["area"], area)
        if registry_scan and frame_index >= next_registry_checkpoint:
            save_partial()
            next_registry_checkpoint = frame_index + registry_checkpoint_interval
        if frame_index >= next_progress_frame:
            print(json.dumps({"progress": min(1.0, frame_index / total_frames)}), flush=True)
            next_progress_frame = frame_index + progress_interval
    cap.release()

    # Online centroids make the whole-film scan cheap and restartable.  This
    # conservative final pass joins fragments caused by profiles/lighting and
    # prevents isolated detector mistakes from becoming "characters".
    if registry_scan:
        prototypes, prototype_counts = consolidate_identities(prototypes, prototype_counts)

    output = []
    for cue in cues:
        data = result[int(cue["id"])]
        samples = max(1, int(data["samples"]))
        ranked = []
        for identity, value in data["faces"].items():
            visible = min(1.0, value["visible"] / samples)
            motions = [float(v) for v in value["motions"] if v > 0]
            motion = float(np.percentile(motions, 70)) if motions else 0.0
            score = motion * min(1.0, visible / .55) * min(1.0, value["area"] / .003)
            ranked.append((score, int(identity), visible, motion, value["area"]))
        ranked.sort(reverse=True)
        best = ranked[0] if ranked else (0.0, 0, 0.0, 0.0, 0.0)
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = max(0.0, best[0] - second)
        confidence = min(1.0, best[0] / .035) * min(1.0, margin / .012) * min(1.0, best[2] / .55)
        # Mouth-pixel motion is a safe anchor only when one face dominates the
        # shot.  In multi-face film coverage it is useful visibility evidence,
        # but not a substitute for a trained active-speaker model.
        single_face_ratio = min(1.0, float(data["single_face_samples"]) / samples)
        confidence *= min(1.0, single_face_ratio / .60)
        output.append({
            "id": cue["id"], "visible_faces": int(data["max_concurrent_faces"]),
            "distinct_face_tracks": len(ranked),
            "active_face_id": best[1] if confidence >= .56 else None,
            "active_speaker_confidence": round(confidence, 3),
            "single_face_ratio": round(single_face_ratio, 3),
            "mouth_visibility": round(best[2], 3), "mouth_motion": round(best[3], 4),
            "mouth_visible": bool(confidence >= .56 and best[2] >= .55 and best[4] >= .0012),
            "face_area_ratio": round(best[4], 5),
        })
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.registry_output:
        registry = {
            "version": 2,
            "model": "OpenCV SFace 2021dec",
            "similarity_threshold": .37,
            "consolidation_similarity": .42,
            "minimum_observations": 2,
            "identities": [
                {"face_id": index + 1, "observations": int(prototype_counts[index]),
                 "embedding": [round(float(value), 7) for value in prototype]}
                for index, prototype in enumerate(prototypes)
            ],
        }
        args.registry_output.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
        if partial:
            partial.unlink(missing_ok=True)
    print(json.dumps({"progress": 1.0, "faces": len(prototypes)}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

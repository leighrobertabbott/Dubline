from __future__ import annotations

import json
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from app.services.gpu_safety import gpu_stage
from app.services.subprocess_control import controlled_lines, terminate_process


def diarization_runtime_settings(environment: dict[str, str] | None = None) -> tuple[str, int, int]:
    environment = environment or os.environ
    device = environment.get("DUB_DIARIZATION_DEVICE", "cuda").strip().lower()
    if device not in {"cpu", "cuda"}:
        device = "cuda"
    try:
        default_batch = "2" if device == "cuda" else "4"
        batch_size = max(1, min(8, int(environment.get("DUB_DIARIZATION_BATCH_SIZE", default_batch))))
    except ValueError:
        batch_size = 4
    try:
        cpu_threads = max(1, min(12, int(environment.get("DUB_DIARIZATION_CPU_THREADS", "8"))))
    except ValueError:
        cpu_threads = 8
    return device, batch_size, cpu_threads


def diarize(audio: Path, folder: Path, progress: Callable[[float], None], checkpoint: Callable[[], None]) -> dict | None:
    model = Path(os.getenv("PYANNOTE_MODEL", "vendor/pyannote-community-1")).resolve()
    if not (model / "config.yaml").is_file():
        return None
    runtime = Path(os.getenv("PYANNOTE_RUNTIME", "vendor/pyannote-env/Scripts/python.exe")).resolve()
    if not runtime.is_file():
        raise RuntimeError("The isolated pyannote runtime is missing")
    output = folder / "speaker-diarization.json"
    if output.is_file():
        try:
            cached = json.loads(output.read_text(encoding="utf-8"))
            if int(cached.get("version", -1)) == 3:
                return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        output.unlink(missing_ok=True)
    device, batch_size, cpu_threads = diarization_runtime_settings()
    if device == "cuda":
        return _diarize_chunked_cuda(audio, model, runtime, output, folder,
                                     batch_size, cpu_threads, progress, checkpoint)
    payload = _run_worker(audio, model, runtime, output, device, batch_size, cpu_threads,
                          progress, checkpoint)
    payload["version"] = 3
    payload.setdefault("analysis", {})["execution"] = "single bounded CPU process"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return payload


def _run_worker(audio: Path, model: Path, runtime: Path, output: Path, device: str,
                batch_size: int, cpu_threads: int, progress: Callable[[float], None],
                checkpoint: Callable[[], None]) -> dict:
    process = subprocess.Popen(
        [str(runtime), "-m", "app.services.diarization_worker", "--audio", str(audio),
         "--model", str(model), "--output", str(output), "--device", device,
         "--batch-size", str(batch_size), "--cpu-threads", str(cpu_threads)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-20:]
            try:
                progress(float(json.loads(line)["progress"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        code = process.wait()
    except BaseException:
        terminate_process(process); raise
    if code != 0 or not output.is_file():
        raise RuntimeError("Speaker diarization worker failed: " + "\n".join(tail[-10:]))
    return json.loads(output.read_text(encoding="utf-8"))


def _duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return max(0.01, float(result.stdout.strip()))


def _diarize_chunked_cuda(audio: Path, model: Path, runtime: Path, output: Path, folder: Path,
                          batch_size: int, cpu_threads: int, progress: Callable[[float], None],
                          checkpoint: Callable[[], None]) -> dict:
    """Diarize bounded CUDA chunks, releasing the driver between every chunk.

    Chunk-local speaker centroids are clustered after all chunks finish, which
    keeps character IDs stable even when the same actor reappears much later.
    Every chunk is a durable restart point.
    """
    duration = _duration(audio)
    chunk_seconds = max(90.0, min(600.0, float(os.getenv("DUB_DIARIZATION_CHUNK_SECONDS", "240"))))
    overlap = max(4.0, min(30.0, float(os.getenv("DUB_DIARIZATION_CHUNK_OVERLAP", "12"))))
    step = max(60.0, chunk_seconds - overlap)
    starts = list(np.arange(0.0, duration, step))
    chunks = folder / "diarization-chunks-v3"
    chunks.mkdir(parents=True, exist_ok=True)
    payloads: list[tuple[float, float, dict]] = []
    for index, start in enumerate(starts):
        checkpoint()
        length = min(chunk_seconds, duration - float(start))
        chunk_audio = chunks / f"chunk-{index:05d}.flac"
        chunk_output = chunks / f"chunk-{index:05d}.json"
        if not chunk_output.is_file():
            if not chunk_audio.is_file():
                subprocess.run([
                    "ffmpeg", "-y", "-v", "error", "-ss", f"{float(start):.3f}",
                    "-t", f"{length:.3f}", "-i", str(audio), "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "flac", str(chunk_audio),
                ], check=True)
            with gpu_stage(folder, f"Speaker registration chunk {index + 1}/{len(starts)}", checkpoint,
                           minimum_free_mb=5200):
                _run_worker(
                    chunk_audio, model, runtime, chunk_output, "cuda", batch_size, cpu_threads,
                    lambda value, i=index: progress((i + value) / max(1, len(starts))), checkpoint,
                )
        payload = json.loads(chunk_output.read_text(encoding="utf-8"))
        payloads.append((float(start), length, payload))
        progress((index + 1) / max(1, len(starts)))
        chunk_audio.unlink(missing_ok=True)

    nodes: list[tuple[int, str, np.ndarray]] = []
    for chunk_index, (_, _, payload) in enumerate(payloads):
        for label, values in (payload.get("speaker_embeddings") or {}).items():
            vector = np.asarray(values, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-6:
                nodes.append((chunk_index, str(label), vector / norm))
    node_clusters: dict[tuple[int, str], int] = {}
    if nodes:
        from sklearn.cluster import AgglomerativeClustering
        matrix = np.stack([node[2] for node in nodes])
        threshold = max(.18, min(.5, float(os.getenv("DUB_DIARIZATION_IDENTITY_DISTANCE", ".32"))))
        if len(nodes) == 1:
            labels = np.zeros(1, dtype=int)
        else:
            labels = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold,
            ).fit_predict(matrix)
        next_cluster = int(max(labels, default=-1)) + 1
        used_in_chunk: dict[tuple[int, int], tuple[int, str]] = {}
        for node, cluster in zip(nodes, map(int, labels)):
            collision = (node[0], cluster)
            if collision in used_in_chunk:
                cluster = next_cluster; next_cluster += 1
            used_in_chunk[(node[0], cluster)] = (node[0], node[1])
            node_clusters[(node[0], node[1])] = cluster

    next_cluster = max(node_clusters.values(), default=-1) + 1
    for chunk_index, (_, _, payload) in enumerate(payloads):
        local_labels = {str(turn["speaker"]) for key in ("diarization", "exclusive_diarization")
                        for turn in payload.get(key, [])}
        for label in sorted(local_labels):
            if (chunk_index, label) not in node_clusters:
                node_clusters[(chunk_index, label)] = next_cluster; next_cluster += 1

    regular: list[dict] = []
    exclusive: list[dict] = []
    for chunk_index, (start, length, payload) in enumerate(payloads):
        core_start = start if chunk_index == 0 else start + overlap / 2
        raw_end = start + length
        core_end = raw_end if chunk_index + 1 == len(payloads) else raw_end - overlap / 2
        for key, target in (("diarization", regular), ("exclusive_diarization", exclusive)):
            for turn in payload.get(key, []):
                absolute_start = start + float(turn["start"])
                absolute_end = start + float(turn["end"])
                clipped_start = max(core_start, absolute_start)
                clipped_end = min(core_end, absolute_end)
                if clipped_end - clipped_start < .012:
                    continue
                cluster = node_clusters[(chunk_index, str(turn["speaker"]))]
                target.append({"start": round(clipped_start, 3), "end": round(clipped_end, 3),
                               "speaker": f"SPEAKER_{cluster:03d}"})
    payload = {
        "version": 3, "diarization": sorted(regular, key=lambda item: (item["start"], item["end"])),
        "exclusive_diarization": sorted(exclusive, key=lambda item: (item["start"], item["end"])),
        "analysis": {"model": "pyannote-community-1", "device": "cuda", "batch_size": batch_size,
                     "execution": "bounded crash-resumable CUDA chunks", "chunk_seconds": chunk_seconds,
                     "overlap_seconds": overlap, "chunks": len(payloads),
                     "identity_distance": float(os.getenv("DUB_DIARIZATION_IDENTITY_DISTANCE", ".32"))},
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    shutil.rmtree(chunks, ignore_errors=True)
    return payload


def assign_diarized_speakers(cues: list[dict], result: dict) -> None:
    exclusive = result.get("exclusive_diarization") or result.get("diarization") or []
    regular = result.get("diarization") or exclusive
    # Keep IDs stable across selected sections by numbering every full-context
    # cluster at its first occurrence, not merely those present in this cue list.
    label_order: dict[str, int] = {}
    for turn in sorted(exclusive, key=lambda item: (float(item["start"]), float(item["end"]))):
        label = str(turn["speaker"])
        label_order.setdefault(label, len(label_order) + 1)
    for cue in cues:
        start, end = float(cue["start"]), float(cue["end"])
        duration = max(0.05, end - start)
        scores: dict[str, float] = {}
        score_turns = regular if cue.get("simultaneous_card") else exclusive
        for turn in score_turns:
            amount = max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"])))
            scores[turn["speaker"]] = scores.get(turn["speaker"], 0.0) + amount
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        overlaps = {turn["speaker"] for turn in regular
                    if max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"]))) > 0.08}
        cue["overlapping_speech"] = len(overlaps) > 1
        if not ranked or ranked[0][1] / duration < 0.3:
            cue["speaker_id"] = 0; cue["speaker"] = "Uncertain voice"; cue["speaker_confidence"] = 0.0
            cue["speaker_assignment"] = "uncertain"
            continue
        requested_rank = int(cue.get("card_speaker_index", 0)) if cue.get("simultaneous_card") else 0
        selected_rank = min(requested_rank, len(ranked) - 1)
        label, amount = ranked[selected_rank]
        cue["speaker_id"] = label_order[label]
        cue["speaker"] = f"Voice {label_order[label]}"
        cue["speaker_confidence"] = round(min(0.98, amount / duration) *
                                           (0.62 if cue["overlapping_speech"] else 1.0), 3)
        cue["diarization_label"] = label
        cue["speaker_assignment"] = "confident" if cue["speaker_confidence"] >= 0.62 else "tentative"

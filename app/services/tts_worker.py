from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.services.gpu_safety import query_nvidia
from app.services.tts import IndexTTSEngine, natural_duration_factor


def synthesis_input_signature(item: dict) -> str:
    """Bind every reusable take to the text and material synthesis controls."""
    relevant = {key: item.get(key) for key in (
        "text", "target", "reference", "reference_text", "emotion_vector",
        "emotion_strength", "emotion_audio", "language", "use_random",
        "duration_factor", "fit_limit_percent",
    )}
    for key in ("reference", "emotion_audio"):
        value = relevant.get(key)
        if value:
            path = Path(str(value))
            try:
                relevant[key + "_file"] = [path.stat().st_size, path.stat().st_mtime_ns]
            except OSError:
                relevant[key + "_file"] = None
    encoded = json.dumps(relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def line_cooldown() -> None:
    """Hold at a line boundary only while the GPU is actually too hot.

    The previous behaviour slept a fixed second before even reading the sensor,
    which on a feature-length film is roughly twenty minutes of unconditional
    waiting whether or not the device needed it.  Health reads are effectively
    free through NVML, so this now measures first and returns immediately on a
    cool card, spending time only where there is real heat to shed.
    """
    settle = max(0.0, float(os.getenv("DUB_GPU_LINE_SETTLE_SECONDS", "0.15")))
    if settle:
        # A brief pause keeps consecutive vocoder/diffusion bursts from stacking
        # into the power spikes seen in testing, and lets the sensor catch up.
        time.sleep(settle)
    try:
        soft_limit = int(os.getenv("DUB_GPU_SOFT_TEMPERATURE_C", "80"))
        target = min(soft_limit, int(os.getenv("DUB_GPU_COOLDOWN_TARGET_C", "77")))
    except ValueError:
        soft_limit, target = 80, 77
    deadline = time.monotonic() + max(0.0, float(os.getenv("DUB_GPU_LINE_COOLDOWN_LIMIT_SECONDS", "45")))
    waited = 0.0
    while True:
        try:
            health = query_nvidia()
        except (OSError, RuntimeError, ValueError):
            return
        temperature, utilization = int(health["temperature_c"]), int(health["utilization"])
        if temperature < soft_limit or (temperature <= target and utilization <= 10):
            if waited:
                print(json.dumps({"cooled": True, "temperature_c": temperature,
                                  "waited_seconds": round(waited, 1)}), flush=True)
            return
        if time.monotonic() >= deadline:
            return
        if waited == 0.0 or waited % 5 < .25:
            print(json.dumps({"cooling": True, "temperature_c": temperature}), flush=True)
        time.sleep(.25)
        waited += .25


def activity(audio: np.ndarray, rate: int) -> tuple[int, int, list[tuple[int, int]]]:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    frame = max(1, round(rate * .01)); count = len(mono) // frame
    if count == 0:
        return 0, len(mono), [(0, len(mono))]
    levels = np.sqrt(np.mean(mono[:count * frame].reshape(count, frame) ** 2, axis=1) + 1e-12)
    threshold = max(10 ** (-48 / 20), float(np.percentile(levels, 90)) * .09)
    active = levels >= threshold
    indices = np.flatnonzero(active)
    if not len(indices):
        return 0, len(mono), [(0, len(mono))]
    start = max(0, int(indices[0] * frame - rate * .02))
    end = min(len(mono), int((indices[-1] + 1) * frame + rate * .035))
    runs: list[tuple[int, int]] = []
    run_start = int(indices[0])
    previous = int(indices[0])
    max_gap = round(.18 / .01)
    for value in map(int, indices[1:]):
        if value - previous > max_gap:
            runs.append((max(start, run_start * frame), min(end, (previous + 1) * frame)))
            run_start = value
        previous = value
    runs.append((max(start, run_start * frame), min(end, (previous + 1) * frame)))
    return start, end, runs


def active_seconds(path: Path) -> float:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    _, _, runs = activity(audio, rate)
    return sum(end - start for start, end in runs) / rate


def speech_measurements(path: Path) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    start, end, runs = activity(audio, rate)
    active = sum(run_end - run_start for run_start, run_end in runs) / rate
    return {"active": active, "span": max(0.0, end - start) / rate,
            "start": start, "end": end, "rate": rate, "runs": runs}


def stretch_phrase(values: np.ndarray, rate: int, tempo: float, folder: Path, index: int) -> np.ndarray:
    if abs(tempo - 1.0) <= .005 or len(values) < round(rate * .08):
        return values.copy()
    source = folder / f"phrase-{index:03d}.wav"; output = folder / f"phrase-{index:03d}-fit.wav"
    sf.write(source, values, rate, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-af",
                    f"rubberband=tempo={tempo:.8f}", "-ar", str(rate), "-ac", "1",
                    "-c:a", "pcm_s16le", str(output)], check=True, capture_output=True)
    result, _ = sf.read(output, dtype="float32", always_2d=True)
    return result.mean(axis=1)


def fit_audio(source: Path, output: Path, target: float, fit_limit_percent: float = 8.0) -> dict:
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1); start, end, runs = activity(audio, rate)
    target_frames = max(1, round(target * rate))
    active_frames = sum(run_end - run_start for run_start, run_end in runs)
    active_duration = active_frames / rate
    trimmed = mono[start:end]
    lead = min(round(rate * .025), max(0, target_frames // 8))
    trail = min(round(rate * .04), max(0, target_frames // 8))
    body_frames = max(1, target_frames - lead - trail)
    requested_tempo = len(trimmed) / body_frames
    limit = max(0.0, min(20.0, float(fit_limit_percent))) / 100.0
    applied_tempo = float(np.clip(requested_tempo, 1.0 - limit, 1.0 + limit))
    with tempfile.TemporaryDirectory(prefix="dubline-fit-") as temp:
        body = stretch_phrase(trimmed, rate, applied_tempo, Path(temp), 0)
    result = np.zeros(target_frames, dtype=np.float32)
    usable = min(len(body), body_frames)
    result[lead:lead + usable] = body[:usable]
    padding_ms = max(0, body_frames - len(body)) / rate * 1000
    truncated_ms = max(0, len(body) - body_frames) / rate * 1000
    tempo_correction = (applied_tempo - 1.0) * 100
    span_duration = len(trimmed) / rate
    duration_error = (span_duration / max(target - (lead + trail) / rate, .05) - 1.0) * 100
    fade = min(round(rate * .015), len(result) // 4)
    if fade:
        result[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        result[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    output_rate = 24_000
    if rate != output_rate:
        import torch
        import torchaudio
        result = torchaudio.functional.resample(torch.from_numpy(result.copy()), rate, output_rate).numpy()
        wanted = max(1, round(target * output_rate))
        result = result[:wanted] if len(result) >= wanted else np.pad(result, (0, wanted - len(result)))
    sf.write(output, np.clip(result * .96, -.97, .97), output_rate, subtype="PCM_16")
    timing_pass = abs(tempo_correction) <= fit_limit_percent + .05 and padding_ms <= 160 and truncated_ms <= 40
    return {"active_duration": round(active_duration, 4), "speech_span_duration": round(span_duration, 4),
            "active_fill_percent": round(active_duration / target * 100, 2),
            "padding_ms": round(padding_ms, 1), "phrase_count": len(runs),
            "truncated_ms": round(truncated_ms, 1), "duration_error_percent": round(duration_error, 2),
            "stretch_percent": round(tempo_correction, 2), "timing_pass": timing_pass}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest["items"]
    engine = IndexTTSEngine(manifest["engine"])
    for index, item in enumerate(items):
        raw = Path(item["raw"])
        fitted = Path(item["fitted"])
        metrics_path = fitted.with_suffix(".json")
        factor = float(item.get("duration_factor", 1.0))
        target = float(item["target"])
        fit_limit = float(item.get("fit_limit_percent", 8.0))
        desired_span = max(.16, target - min(.10, target * .06))
        input_signature = synthesis_input_signature(item)
        cached_metrics = None
        if raw.is_file() and fitted.is_file() and metrics_path.is_file():
            try:
                fitted_info = sf.info(fitted)
                candidate_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                required_metrics = {"speech_span_duration", "padding_ms", "truncated_ms",
                                    "duration_error_percent", "timing_pass"}
                if (fitted_info.samplerate == 24_000
                        and abs(fitted_info.frames / fitted_info.samplerate - target) <= .002
                        and required_metrics.issubset(candidate_metrics)
                        and candidate_metrics.get("input_signature") == input_signature):
                    cached_metrics = candidate_metrics
            except (OSError, ValueError, json.JSONDecodeError):
                cached_metrics = None
        if cached_metrics is None and (raw.exists() or fitted.exists() or metrics_path.exists()):
            # An interrupted/old take without a matching provenance signature is
            # not reusable. This also prevents edited translations from silently
            # reappearing in the final acoustic match and mix.
            raw.unlink(missing_ok=True)
            fitted.unlink(missing_ok=True)
            metrics_path.unlink(missing_ok=True)
            root = raw.parent.parent
            (root / "acoustically-matched" / raw.name).unlink(missing_ok=True)
            (root / "qwen-generated" / raw.name).unlink(missing_ok=True)
            (root / "qwen-fitted" / raw.name).unlink(missing_ok=True)
        if cached_metrics is not None:
            print(json.dumps({"progress": (index + 1) / max(1, len(items)), "index": index,
                              "cue_index": int(item.get("cue_index", index)),
                              "raw_duration": round(duration(raw), 4),
                              "duration_factor": round(factor, 5),
                              **cached_metrics, "attempts": 0, "restored": True}), flush=True)
            continue
        if not raw.exists():
            vector = item.get("emotion_vector")
            emotion_audio = Path(item["emotion_audio"]) if item.get("emotion_audio") else None
            engine.synthesize(
                item["text"], Path(item["reference"]), raw, float(item["target"]),
                vector, float(item["emotion_strength"]), emotion_audio, item.get("language", "EN"),
                bool(item.get("use_random", False)), factor,
            )
            if engine.mode != "preview":
                line_cooldown()
        actual = duration(raw)
        measured = speech_measurements(raw)
        active, span = measured["active"], measured["span"]
        attempts = 1
        best_error = abs(span / desired_span - 1.0)
        # Duration control is calibrated from the preceding real take.  Both
        # undershoots and overshoots are regenerated; text length is never used
        # as a speech clock.
        for retry_number in range(1, 3):
            if best_error <= max(.025, fit_limit / 100 * .55):
                break
            retry = raw.with_name(raw.stem + f".retry-{retry_number}.wav")
            # Searching the full 0.5-2.0 range is what produced half-speed
            # vowels; a line that will not fit at a natural rate is a
            # translation problem, and the timing-QC rewrite owns it.
            factor = natural_duration_factor(factor * desired_span / max(span, .08))
            engine.synthesize(
                item["text"], Path(item["reference"]), retry, target,
                item.get("emotion_vector"), float(item["emotion_strength"]),
                Path(item["emotion_audio"]) if item.get("emotion_audio") else None,
                item.get("language", "EN"), bool(item.get("use_random", False)), factor,
            )
            if engine.mode != "preview":
                line_cooldown()
            attempts += 1
            retry_duration = duration(retry); retry_measurements = speech_measurements(retry)
            retry_active, retry_span = retry_measurements["active"], retry_measurements["span"]
            retry_error = abs(retry_span / desired_span - 1.0)
            if retry_error < best_error:
                os.replace(retry, raw)
                actual, active, span, best_error = retry_duration, retry_active, retry_span, retry_error
            else:
                retry.unlink(missing_ok=True)
        fitted_valid = False
        if fitted.exists():
            info = sf.info(fitted)
            fitted_valid = info.samplerate == 24_000 and abs(info.frames / info.samplerate - target) <= .002
            if not fitted_valid:
                fitted.unlink(missing_ok=True)
        metrics = fit_audio(raw, fitted, target, fit_limit)
        metrics["input_signature"] = input_signature
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": (index + 1) / max(1, len(items)), "index": index,
                          "cue_index": int(item.get("cue_index", index)),
                          "raw_duration": round(actual, 4), "duration_factor": round(factor, 5),
                          **metrics, "attempts": attempts}), flush=True)
        if engine.mode != "preview":
            line_cooldown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return max(0.01, float(result.stdout.strip()))


if __name__ == "__main__":
    main()

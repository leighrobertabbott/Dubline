from __future__ import annotations

import html
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from app.services.subprocess_control import controlled_lines, terminate_process


def _controlled_capture(command: list[str], checkpoint=None) -> str:
    if checkpoint is None:
        result = subprocess.run(command, check=True, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        return result.stdout + result.stderr
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    lines = []
    try:
        for line in controlled_lines(process, checkpoint):
            lines.append(line)
        code = process.wait()
    except BaseException:
        terminate_process(process)
        raise
    output = "".join(lines)
    if code:
        raise subprocess.CalledProcessError(code, command, output=output, stderr=output)
    return output


def backtranscribe_lines(cues: list[dict], fitted_dir: Path, folder: Path,
                        progress, checkpoint) -> None:
    manifest = folder / "qc-asr-manifest.json"
    output = folder / "qc-backtranscription.json"
    items = [{"audio": str(fitted_dir / f"{index:06d}.wav")} for index in range(1, len(cues) + 1)]
    manifest.write_text(json.dumps({"items": items,
                                    "cache": str(Path(os.getenv("WHISPER_CACHE_DIR", "vendor/whisper")).resolve())},
                                   indent=2), encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.services.qc_asr_worker", "--manifest", str(manifest), "--output", str(output)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-15:]
            try:
                event = json.loads(line); progress(float(event["progress"]), int(event["index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        code = process.wait()
    except BaseException:
        terminate_process(process); raise
    if code != 0 or not output.is_file():
        raise RuntimeError("Back-transcription QC worker failed: " + "\n".join(tail[-10:]))
    results = json.loads(output.read_text(encoding="utf-8"))
    for cue, result in zip(cues, results):
        intended = normalize_words(cue.get("spoken_text") or cue.get("english", ""))
        heard = normalize_words(result["text"])
        similarity = __import__('difflib').SequenceMatcher(None, intended, heard).ratio()
        intended_words = intended.split(); heard_words = heard.split()
        wer = edit_distance(intended_words, heard_words) / max(1, len(intended_words))
        intended_chars = list(intended.replace(" ", "")); heard_chars = list(heard.replace(" ", ""))
        cer = edit_distance(intended_chars, heard_chars) / max(1, len(intended_chars))
        cue.setdefault("qc", {}).update({"backtranscription": result["text"],
                                          "word_similarity": round(similarity, 3),
                                          "wer": round(wer, 3), "cer": round(cer, 3),
                                          "asr_confidence": result["confidence"]})


def normalize_words(value: str) -> str:
    value = value.lower()
    numbers = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
               "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    for digit, word in numbers.items():
        value = value.replace(digit, f" {word} ")
    return " ".join(__import__('re').findall(r"[a-z']+", value))


def edit_distance(left: list, right: list) -> int:
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def _performance_metrics(audio: np.ndarray, rate: int) -> dict:
    """Measure coarse performance shape without making QC depend on a large model."""
    if not len(audio):
        return {"energy_contour": [0.0] * 10, "pause_ratio": 1.0, "pitch_hz": 0.0}
    frame = max(256, round(rate * .025))
    hop = max(128, round(rate * .010))
    energies = []
    for start in range(0, max(1, len(audio) - frame + 1), hop):
        values = audio[start:start + frame]
        energies.append(float(np.sqrt(np.mean(values * values) + 1e-12)))
    energy = np.asarray(energies or [0.0], dtype=np.float32)
    voiced_floor = max(10 ** (-48 / 20), float(np.percentile(energy, 70)) * .18)
    pause_ratio = float(np.mean(energy < voiced_floor))
    chunks = np.array_split(energy, 10)
    contour = np.asarray([float(np.mean(chunk)) if len(chunk) else 0.0 for chunk in chunks])
    maximum = float(np.max(contour))
    if maximum > 1e-8:
        contour /= maximum
    pitch = 0.0
    if len(audio) >= rate // 3 and float(np.sqrt(np.mean(audio * audio))) > 2e-4:
        try:
            import librosa
            track = librosa.yin(audio, fmin=65, fmax=500, sr=rate,
                                frame_length=min(2048, max(512, 2 ** int(np.log2(len(audio))))))
            valid = track[np.isfinite(track) & (track >= 65) & (track <= 500)]
            pitch = float(np.median(valid)) if len(valid) else 0.0
        except (ImportError, ValueError, FloatingPointError):
            pass
    return {"energy_contour": [round(float(value), 3) for value in contour],
            "pause_ratio": round(pause_ratio, 3), "pitch_hz": round(pitch, 1)}


# Register is identity, not style: a median pitch a few semitones from the actor
# reads as a different person even when the timbre embedding matches.  These are
# the shared tolerances for the QC gate, the retake trigger and the residual
# pitch correction, kept in one place so they cannot drift apart.
PITCH_TOLERANCE_SEMITONES = 3.5
PITCH_SCORE_RANGE_SEMITONES = 6.0
PERFORMANCE_RETAKE_THRESHOLD = 0.55
# Residual register correction is a last resort applied only after retakes, and
# only inside a band where formant-preserving shifting stays transparent.
PITCH_CORRECTION_FLOOR_SEMITONES = 0.6
PITCH_CORRECTION_CEILING_SEMITONES = 3.0


def register_correction(cue: dict) -> tuple[float, float] | None:
    """Return (semitones, pitch scale) to move a take toward the actor's register.

    None whenever correction is unwarranted or unsafe: no reliable measurement on
    either side, a line too short to have measured a stable median, a difference
    small enough to be inaudible, or one large enough that shifting would cost
    more in artefacts than it recovers in accuracy.
    """
    source = float((cue.get("source_performance") or {}).get("pitch_hz") or 0.0)
    generated = float(((cue.get("qc") or {}).get("generated_performance") or {}).get("pitch_hz") or 0.0)
    if source <= 0 or generated <= 0:
        return None
    if float(cue.get("end", 0)) - float(cue.get("start", 0)) < 1.0:
        return None
    semitones = 12 * math.log2(source / generated)
    if not PITCH_CORRECTION_FLOOR_SEMITONES <= abs(semitones) <= PITCH_CORRECTION_CEILING_SEMITONES:
        return None
    return round(semitones, 2), source / generated


def measure_performance_similarity(cues: list[dict], fitted_dir: Path) -> None:
    """Measure acting-shape evidence early enough to select another take."""
    for index, cue in enumerate(cues, 1):
        path = fitted_dir / f"{index:06d}.wav"
        source = cue.get("source_performance", {})
        if not path.is_file() or not source:
            continue
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        generated = _performance_metrics(values.mean(axis=1), int(rate))
        metrics = cue.setdefault("qc", {})
        metrics["generated_performance"] = generated
        components: list[tuple[float, float]] = []
        source_contour = np.asarray(source.get("energy_contour", []), dtype=np.float32)
        generated_contour = np.asarray(generated["energy_contour"], dtype=np.float32)
        if len(source_contour) == len(generated_contour) and len(source_contour) >= 4:
            if float(np.std(source_contour)) > .03 and float(np.std(generated_contour)) > .03:
                similarity = float(np.corrcoef(source_contour, generated_contour)[0, 1])
            else:
                similarity = 1.0 - float(np.mean(np.abs(source_contour - generated_contour)))
            similarity = float(np.clip(similarity, -1, 1))
            metrics["energy_contour_similarity"] = round(similarity, 3)
            components.append(((similarity + 1) / 2, .40))
        if "pause_ratio" in source:
            delta = abs(float(source["pause_ratio"]) - float(generated["pause_ratio"]))
            metrics["pause_ratio_delta"] = round(delta, 3)
            components.append((max(0.0, 1 - delta / .55), .20))
        source_pitch, generated_pitch = float(source.get("pitch_hz", 0)), float(generated["pitch_hz"])
        if source_pitch and generated_pitch:
            delta = abs(12 * np.log2(generated_pitch / source_pitch))
            metrics["pitch_delta_semitones"] = round(float(delta), 2)
            components.append((max(0.0, 1 - delta / PITCH_SCORE_RANGE_SEMITONES), .40))
        if components:
            total_weight = sum(weight for _, weight in components)
            metrics["performance_similarity"] = round(
                sum(value * weight for value, weight in components) / total_weight, 3)


def inspect_cues(cues: list[dict], fitted_dir: Path) -> dict:
    flagged = 0
    for index, cue in enumerate(cues, 1):
        reasons: list[str] = []
        metrics = cue.setdefault("qc", {})
        required_cue = ("speaker_confidence", "reference_quality", "timing_confidence",
                        "transcription_confidence", "adaptation_confidence", "alignment_confidence")
        required_take = ("stretch_percent", "timing_pass", "padding_ms", "truncated_ms",
                         "word_similarity", "wer", "cer", "backtranscription")
        for name in required_cue:
            if cue.get(name) is None:
                reasons.append(f"QC evidence is missing: {name.replace('_', ' ')}")
        for name in required_take:
            if metrics.get(name) is None:
                reasons.append(f"QC evidence is missing: {name.replace('_', ' ')}")
        translation_qc = cue.get("translation_qc")
        if not isinstance(translation_qc, dict) or not translation_qc.get("available"):
            reasons.append("independent bilingual translation evidence is missing")
        elif not translation_qc.get("passed"):
            detail = str(translation_qc.get("reason") or "meaning may differ from the source")
            reasons.append(f"independent translation QC failed: {detail}")
        if cue.get("overlapping_speech"):
            if cue.get("overlap_source_preserved"):
                reasons.append("simultaneous source speakers were preserved under the dominant English performance")
            else:
                reasons.append("overlapping speakers were not resolved into independent performances")

        stretch_value = metrics.get("stretch_percent")
        stretch = abs(float(stretch_value)) if stretch_value is not None else 999.0
        wer_value, cer_value = metrics.get("wer"), metrics.get("cer")
        wer = float(wer_value) if wer_value is not None else 1.0
        cer = float(cer_value) if cer_value is not None else 1.0
        timing_limit = 5.0 if cue.get("mouth_visible") else 8.0
        if stretch > timing_limit:
            reasons.append(f"timing correction {stretch:.1f}% exceeds {timing_limit:.0f}%")
        if metrics.get("timing_pass") is False:
            reasons.append("measured speech duration did not fit without excessive correction or dead air")
        if float(metrics.get("padding_ms") or 0.0) > 160:
            reasons.append("the take leaves excessive dead air after the spoken performance")
        if float(metrics.get("truncated_ms") or 0.0) > 40:
            reasons.append("the take was truncated to fit its timeline window")
        if (float(metrics.get("word_similarity") or 0.0) < 0.72
                or (wer > 0.4 and cer > 0.35)):
            reasons.append("generated words disagree with the intended dialogue")
        if float(cue.get("speaker_confidence") or 0.0) < 0.5:
            reasons.append("speaker identity is uncertain")
        if float(cue.get("reference_quality", 0.0)) < 0.30:
            reasons.append("source voice reference is weak")
        if float(cue.get("timing_confidence") or 0.0) < 0.5:
            reasons.append("speech timing could not be aligned confidently")
        if float(cue.get("transcription_confidence") or 0.0) < 0.55:
            reasons.append("source transcription confidence is low")
        if float(cue.get("adaptation_confidence") or 0.0) < 0.45:
            reasons.append("adapted translation needs a bilingual check")
        if float(metrics.get("active_duration") or 0.0) >= 0.8 and metrics.get("speaker_similarity") is None:
            reasons.append("QC evidence is missing: generated speaker similarity")
        if (float(metrics.get("speaker_similarity") or 0.0) < 0.55
                and float(metrics.get("active_duration") or 0.0) >= 0.8):
            reasons.append("generated voice identity differs from the source reference")
        if metrics.get("tts_engine", "").startswith("Qwen3") and metrics.get("full_reference_clone") is not True:
            reasons.append("the fallback engine lacked a reference transcript and used reduced voice cloning")
        if float(cue.get("alignment_confidence") or 0.0) < 0.55:
            reasons.append("forced alignment confidence is low")
        leakage = float(cue.get("dialogue_leakage_after_suppression",
                                cue.get("dialogue_leakage", 0.0)))
        metrics["dialogue_leakage"] = round(leakage, 3)
        if leakage > .28:
            reasons.append("original dialogue may remain audible in the background bed")
        path = fitted_dir / f"{index:06d}.wav"
        if not path.is_file():
            reasons.append("generated line is missing")
        else:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
            generated_rms = float(np.sqrt(np.mean(mono * mono) + 1e-12)) if len(mono) else 0.0
            active = float(np.mean(np.abs(mono) > 10 ** (-48 / 20))) if len(mono) else 0.0
            source_rms = float(cue.get("source_performance", {}).get("rms", 0.0))
            level_delta = 20 * np.log10(max(generated_rms, 1e-7) / max(source_rms, 1e-7)) if source_rms else 0.0
            generated_performance = _performance_metrics(mono, int(sample_rate))
            metrics.update({"peak": round(peak, 4), "active_ratio": round(active, 4),
                            "rms": round(generated_rms, 6), "source_level_delta_db": round(float(level_delta), 2),
                            "generated_performance": generated_performance})
            source_performance = cue.get("source_performance", {})
            source_contour = np.asarray(source_performance.get("energy_contour", []), dtype=np.float32)
            generated_contour = np.asarray(generated_performance["energy_contour"], dtype=np.float32)
            if len(source_contour) == len(generated_contour) and len(source_contour) >= 4:
                if float(np.std(source_contour)) > .03 and float(np.std(generated_contour)) > .03:
                    contour_similarity = float(np.corrcoef(source_contour, generated_contour)[0, 1])
                else:
                    contour_similarity = 1.0 - float(np.mean(np.abs(source_contour - generated_contour)))
                metrics["energy_contour_similarity"] = round(float(np.clip(contour_similarity, -1, 1)), 3)
                if float(cue.get("end", 0)) - float(cue.get("start", 0)) >= 1.0 and contour_similarity < .15:
                    reasons.append("generated emphasis differs substantially from the source performance")
            if "pause_ratio" in source_performance:
                pause_delta = abs(float(source_performance["pause_ratio"]) - generated_performance["pause_ratio"])
                metrics["pause_ratio_delta"] = round(pause_delta, 3)
                if pause_delta > .48 and float(cue.get("end", 0)) - float(cue.get("start", 0)) >= 1.2:
                    reasons.append("generated pause pattern differs substantially from the source")
            source_pitch = float(source_performance.get("pitch_hz", 0.0))
            generated_pitch = float(generated_performance["pitch_hz"])
            if source_pitch and generated_pitch:
                pitch_delta = abs(12 * np.log2(generated_pitch / source_pitch))
                metrics["pitch_delta_semitones"] = round(float(pitch_delta), 2)
                if (pitch_delta > PITCH_TOLERANCE_SEMITONES
                        and float(cue.get("end", 0)) - float(cue.get("start", 0)) >= 1.0):
                    reasons.append(f"generated vocal pitch is {pitch_delta:.1f} semitones from the "
                                   f"source performance")
            if peak >= 0.995:
                reasons.append("line is at clipping threshold")
            if active < 0.08:
                reasons.append("line contains unexpected silence")
        cue["review_reasons"] = reasons
        cue["needs_review"] = bool(reasons)
        flagged += int(bool(reasons))
    return {"cue_count": len(cues), "flagged_count": flagged,
            "passed_count": len(cues) - flagged}


def _stream_identity(stream: dict) -> dict:
    tags = stream.get("tags", {})
    disposition = stream.get("disposition", {})
    return {
        "type": stream.get("codec_type"), "codec": stream.get("codec_name"),
        "channels": stream.get("channels"), "layout": stream.get("channel_layout"),
        "width": stream.get("width"), "height": stream.get("height"),
        "language": tags.get("language"), "title": tags.get("title"),
        "disposition": {key: value for key, value in disposition.items() if value},
    }


def media_qc(output: Path, expected_duration: float, source: Path | None = None, checkpoint=None) -> dict:
    probe_output = _controlled_capture(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-show_chapters", "-show_format", "-show_streams",
         "-of", "json", str(output)], checkpoint,
    )
    data = json.loads(probe_output)
    actual = float(data["format"]["duration"])
    streams = data.get("streams", [])
    audio_count = sum(x.get("codec_type") == "audio" for x in streams)
    subtitle_count = sum(x.get("codec_type") == "subtitle" for x in streams)
    loudness_output = _controlled_capture(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(output), "-map", "0:a:0", "-af",
         "loudnorm=I=-24:TP=-2:LRA=7:print_format=json", "-f", "null", os.devnull],
        checkpoint,
    )
    measured = {}
    match = __import__('re').findall(r"\{[^{}]+\}", loudness_output, flags=__import__('re').S)
    if match:
        try:
            values = json.loads(match[-1])
            measured = {"integrated_lufs": float(values.get("input_i", 0)),
                        "true_peak_dbtp": float(values.get("input_tp", 0)),
                        "loudness_range_lu": float(values.get("input_lra", 0))}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    source_streams: list[dict] = []
    source_chapters: list[dict] = []
    source_tags: dict = {}
    if source and source.is_file():
        source_data = json.loads(_controlled_capture(
            ["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-show_streams", "-of", "json", str(source)],
            checkpoint,
        ))
        source_streams = [_stream_identity(item) for item in source_data.get("streams", [])]
        source_chapters = source_data.get("chapters", [])
        source_tags = source_data.get("format", {}).get("tags", {})
    output_identities = [_stream_identity(item) for item in streams]
    unmatched = list(output_identities)
    missing_identities = []
    for identity in source_streams:
        try:
            index = unmatched.index(identity)
        except ValueError:
            missing_identities.append(identity)
        else:
            unmatched.pop(index)
    output_chapters = data.get("chapters", [])
    chapter_signature = lambda chapter: (round(float(chapter.get("start_time", 0)), 3),
                                         round(float(chapter.get("end_time", 0)), 3),
                                         chapter.get("tags", {}))
    metadata_missing = {key: value for key, value in source_tags.items()
                        if key.lower() != "encoder" and data.get("format", {}).get("tags", {}).get(key) != value}
    return {
        "expected_duration": round(expected_duration, 3), "actual_duration": round(actual, 3),
        "duration_error_ms": round(abs(actual - expected_duration) * 1000, 1),
        "video_tracks": sum(x.get("codec_type") == "video" for x in streams),
        "source_has_video": any(item["type"] == "video" for item in source_streams),
        "audio_tracks": audio_count,
        "subtitle_tracks": subtitle_count,
        "original_audio_preserved": not any(item["type"] == "audio" for item in missing_identities),
        "original_subtitles_preserved": not any(item["type"] == "subtitle" for item in missing_identities),
        "streams_preserved_exactly": not missing_identities,
        "missing_stream_identities": missing_identities,
        "chapters_preserved": [chapter_signature(item) for item in source_chapters] ==
                              [chapter_signature(item) for item in output_chapters],
        "metadata_preserved": not metadata_missing, "missing_metadata": metadata_missing,
        "english_lossless_track": any(
            x.get("codec_type") == "audio" and x.get("codec_name") == "flac"
            and x.get("tags", {}).get("language") == "eng" for x in streams
        ), **measured,
    }


def evaluate_media_qc(media: dict, mastering: dict) -> list[str]:
    """Turn delivery measurements into hard failures for the actual output file."""
    failures: list[str] = []
    if media.get("source_has_video") and int(media.get("video_tracks", 0)) < 1:
        failures.append("the delivered file has no video track")
    if float(media.get("duration_error_ms", float("inf"))) > 250:
        failures.append("the delivered duration differs from the selected source by more than 250 ms")
    if not media.get("original_audio_preserved"):
        failures.append("one or more original audio tracks are missing")
    if not media.get("original_subtitles_preserved"):
        failures.append("one or more original subtitle tracks are missing")
    if not media.get("streams_preserved_exactly"):
        failures.append("one or more original streams changed codec, identity, language, title, or disposition")
    if not media.get("chapters_preserved"):
        failures.append("chapter timing or chapter metadata changed")
    if not media.get("metadata_preserved"):
        failures.append("source container metadata was not preserved")
    if not media.get("english_lossless_track"):
        failures.append("the English dub is not present as an English-labelled FLAC delivery track")

    measured_lufs = media.get("integrated_lufs")
    measured_peak = media.get("true_peak_dbtp")
    if measured_lufs is None or measured_peak is None:
        failures.append("the delivered English track loudness could not be measured")
    else:
        try:
            if not math.isfinite(float(measured_lufs)) or not math.isfinite(float(measured_peak)):
                raise ValueError
        except (TypeError, ValueError):
            failures.append("the delivered English track has invalid loudness measurements")
        else:
            peak_target = mastering.get("true_peak_target_dbtp")
            if peak_target is not None and float(measured_peak) > float(peak_target) + .2:
                failures.append(
                    f"the English track true peak is {float(measured_peak):.1f} dBTP, above its delivery limit"
                )
            target_lufs = mastering.get("target_lufs")
            if target_lufs is not None and abs(float(measured_lufs) - float(target_lufs)) > 1.5:
                failures.append(
                    f"the English track loudness is {float(measured_lufs):.1f} LUFS, outside the delivery target"
                )
    if mastering.get("target_dialogue_lufs") is not None:
        after = mastering.get("dialogue_lufs_after_gain")
        if after is None or abs(float(after) - float(mastering["target_dialogue_lufs"])) > .75:
            failures.append("dialogue-gated cinema loudness missed its target")
    return failures


def _value(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.3f}"


def voice_fidelity_summary(cues: list[dict]) -> dict:
    """Aggregate how closely the dub tracks each original voice, across every line.

    The flagged-cue table only shows failures.  Fidelity is a property of the
    whole dub, so the medians are reported over all measured lines.
    """
    def median(values: list[float]) -> float | None:
        return round(float(np.median(values)), 3) if values else None

    measured = [cue.get("qc") or {} for cue in cues]
    identity = [float(item["speaker_similarity"]) for item in measured
                if item.get("speaker_similarity") is not None]
    performance = [float(item["performance_similarity"]) for item in measured
                   if item.get("performance_similarity") is not None]
    pitch = [float(item["pitch_delta_semitones"]) for item in measured
             if item.get("pitch_delta_semitones") is not None]
    verified = [cue for cue in cues if cue.get("source_asr_agreement") is not None]
    return {
        "lines": len(cues),
        "source_lines_independently_verified": len(verified),
        "source_lines_disputed": sum(1 for cue in cues if cue.get("source_asr_disputed")),
        "median_source_asr_agreement": median(
            [float(cue["source_asr_agreement"]) for cue in verified]),
        "source_performance_lines": sum(
            1 for cue in cues if str(cue.get("performance_source")) == "source performance"),
        "median_speaker_similarity": median(identity),
        "median_performance_similarity": median(performance),
        "median_pitch_delta_semitones": median(pitch),
        "lines_beyond_pitch_tolerance": sum(1 for value in pitch if value > PITCH_TOLERANCE_SEMITONES),
        "lines_cloned_from_a_matched_take": sum(
            1 for cue in cues if str(cue.get("reference_basis", "")).startswith("nearest take")),
        "performance_retakes_kept": sum(1 for item in measured if item.get("performance_retake")),
        "register_corrected_lines": sum(
            1 for item in measured if item.get("pitch_correction_semitones") is not None),
        "fallback_engine_lines": sum(1 for item in measured if item.get("tts_engine")),
    }


def write_report(folder: Path, job_id: str, cues: list[dict], cue_summary: dict,
                 media_summary: dict, separation: dict) -> tuple[Path, Path]:
    media_failures = list(media_summary.get("failures", []))
    result = "failed" if media_failures else ("needs_review" if cue_summary["flagged_count"] else "passed")
    fidelity = voice_fidelity_summary(cues)
    report = {
        "version": 3, "job_id": job_id, "result": result,
        "cues": cue_summary, "media": media_summary, "separation": separation,
        "voice_fidelity": fidelity,
        "flagged_cues": [
            {"id": cue.get("id"), "start": cue.get("start"), "end": cue.get("end"),
             "text": cue.get("english"), "reasons": cue.get("review_reasons", []), "metrics": cue.get("qc", {})}
            for cue in cues if cue.get("needs_review")
        ],
    }
    json_path = folder / "qc-report.json"
    html_path = folder / "qc-report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "".join(
        f"<tr><td>{item['id']}</td><td>{item['start']:.3f}–{item['end']:.3f}</td>"
        f"<td>{html.escape(str(item['text']))}</td><td>{html.escape('; '.join(item['reasons']))}</td></tr>"
        for item in report["flagged_cues"]
    ) or '<tr><td colspan="4">All generated lines passed the automatic checks.</td></tr>'
    delivery = ("<ul>" + "".join(f"<li>{html.escape(reason)}</li>" for reason in media_failures) + "</ul>"
                if media_failures else "<p class='ok'>The delivered media passed track, duration, codec, loudness, and peak checks.</p>")
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Dubline QC report</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#202328}"
        "table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #d8dce2;text-align:left}"
        "th{background:#f3f5f7}.ok{color:#19733b}.warn{color:#a34d00}</style>"
        f"<h1>Dubline quality-control report</h1><h2>Delivered media</h2>{delivery}"
        f"<h2>Dialogue lines</h2><p class='{'warn' if cue_summary['flagged_count'] else 'ok'}'>"
        f"{cue_summary['passed_count']} passed · {cue_summary['flagged_count']} need review</p>"
        f"<p>Output duration error: {media_summary['duration_error_ms']} ms · "
        f"audio tracks: {media_summary['audio_tracks']} · subtitles: {media_summary['subtitle_tracks']}</p>"
        "<h2>Source transcript</h2>"
        f"<p>{fidelity['source_lines_independently_verified']} of {fidelity['lines']} line(s) were read "
        "by a second, independent transcriber before anything was translated; "
        f"{fidelity['source_lines_disputed']} disagreed materially and carry both readings forward. "
        f"Median agreement between the two models: {_value(fidelity['median_source_asr_agreement'])}.</p>"
        "<h2>Voice fidelity</h2>"
        f"<p>{fidelity['source_performance_lines']} of {fidelity['lines']} line(s) were voiced from the "
        "original actor's own delivery of that line; the rest fell back to scene-context emotion.</p>"
        "<table><thead><tr><th>Measure</th><th>Median</th></tr></thead><tbody>"
        f"<tr><td>Speaker identity similarity</td><td>{_value(fidelity['median_speaker_similarity'])}</td></tr>"
        f"<tr><td>Performance similarity</td><td>{_value(fidelity['median_performance_similarity'])}</td></tr>"
        f"<tr><td>Pitch difference from the actor</td>"
        f"<td>{_value(fidelity['median_pitch_delta_semitones'])} semitones</td></tr>"
        f"<tr><td>Lines beyond the {PITCH_TOLERANCE_SEMITONES} semitone tolerance</td>"
        f"<td>{fidelity['lines_beyond_pitch_tolerance']}</td></tr>"
        f"<tr><td>Lines cloned from an acoustically matched take</td>"
        f"<td>{fidelity['lines_cloned_from_a_matched_take']}</td></tr>"
        f"<tr><td>Performance retakes kept</td><td>{fidelity['performance_retakes_kept']}</td></tr>"
        f"<tr><td>Lines register-corrected</td><td>{fidelity['register_corrected_lines']}</td></tr>"
        f"<tr><td>Lines voiced by the fallback engine</td><td>{fidelity['fallback_engine_lines']}</td></tr>"
        "</tbody></table>"
        "<table><thead><tr><th>Line</th><th>Time</th><th>Dialogue</th><th>Reason</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>", encoding="utf-8",
    )
    return json_path, html_path

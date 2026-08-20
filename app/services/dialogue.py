from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf


def build_adaptive_dialogue(primary: Path, recovery: Path, cues: list[dict], output: Path,
                            padding: float = 0.18, roformer: Path | None = None) -> dict:
    """Use clean CASS dialogue unless it measurably dropped a timed voice line."""
    main, rate = sf.read(primary, dtype="float32", always_2d=True)
    backup, backup_rate = sf.read(recovery, dtype="float32", always_2d=True)
    if backup_rate != rate:
        raise RuntimeError("Dialogue candidates must have the same sample rate")
    modern = None
    if roformer and roformer.is_file():
        modern_values, modern_rate = sf.read(roformer, dtype="float32", always_2d=True)
        if modern_rate != rate:
            raise RuntimeError("Dialogue candidates must have the same sample rate")
        modern = modern_values.mean(axis=1)
    frames = min(len(main), len(backup), len(modern) if modern is not None else len(main))
    main = main[:frames].mean(axis=1)
    backup = backup[:frames].mean(axis=1)
    if modern is not None:
        modern = modern[:frames]
    adaptive = main.copy()
    recovered = 0
    roformer_recovered = 0
    demucs_recovered = 0

    for cue in cues:
        cue_start = max(0, round(float(cue["start"]) * rate))
        cue_end = min(frames, round(float(cue["end"]) * rate))
        if cue_end <= cue_start:
            continue
        primary_rms = rms(main[cue_start:cue_end])
        recovery_rms = rms(backup[cue_start:cue_end])
        roformer_rms = rms(modern[cue_start:cue_end]) if modern is not None else 0.0
        ratio = primary_rms / max(recovery_rms, 1e-7)
        # A film-specific separator is preferred whenever it retained a credible line.
        # The recovery model is selected only for a near-null primary estimate.
        strongest = max(recovery_rms, roformer_rms)
        use_recovery = strongest >= 2e-4 and primary_rms < max(2e-4, strongest * 0.08)
        selected = backup
        selected_name = "HTDemucs recovery"
        if use_recovery and modern is not None and roformer_rms >= 2e-4:
            modern_score = speech_likeness(modern[cue_start:cue_end], rate)
            legacy_score = speech_likeness(backup[cue_start:cue_end], rate)
            cue["roformer_speech_score"] = round(modern_score, 4)
            cue["demucs_speech_score"] = round(legacy_score, 4)
            if modern_score >= legacy_score * 0.88:
                selected, selected_name = modern, "MelBand-RoFormer recovery"
        cue["dialogue_source"] = selected_name if use_recovery else "Bandit cinematic"
        cue["dialogue_confidence"] = round(float(min(1.0, ratio / 0.08)), 3)
        cue["primary_dialogue_rms"] = round(primary_rms, 7)
        cue["recovery_dialogue_rms"] = round(recovery_rms, 7)
        cue["roformer_dialogue_rms"] = round(roformer_rms, 7)
        if not use_recovery:
            continue
        recovered += 1
        if selected_name.startswith("MelBand"):
            roformer_recovered += 1
        else:
            demucs_recovered += 1
        start = max(0, cue_start - round(padding * rate))
        end = min(frames, cue_end + round(padding * rate))
        fade = min(round(0.04 * rate), (end - start) // 2)
        if fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            adaptive[start:start + fade] = main[start:start + fade] * (1 - ramp) + selected[start:start + fade] * ramp
            adaptive[end - fade:end] = selected[end - fade:end] * (1 - ramp) + main[end - fade:end] * ramp
            adaptive[start + fade:end - fade] = selected[start + fade:end - fade]
        else:
            adaptive[start:end] = selected[start:end]

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, np.clip(adaptive, -1, 1), rate, subtype="PCM_16")
    return {"total_cues": len(cues), "recovered_cues": recovered,
            "roformer_cues": roformer_recovered, "demucs_cues": demucs_recovered,
            "primary_cues": len(cues) - recovered}


def speech_likeness(samples: np.ndarray, rate: int) -> float:
    """Cheap contamination proxy: voiced-band energy with bass/ultrasonic penalties."""
    if len(samples) < 32:
        return 0.0
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(len(samples), 1 / rate)
    total = float(spectrum[(frequencies >= 40) & (frequencies <= 11_000)].sum()) + 1e-12
    speech = float(spectrum[(frequencies >= 90) & (frequencies <= 4_800)].sum()) / total
    low = float(spectrum[(frequencies >= 40) & (frequencies < 90)].sum()) / total
    return rms(samples) * max(0.05, speech - low * 0.7)


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)) + 1e-12))


def analyze_performance(audio: Path, cues: list[dict], spatial_audio: Path | None = None) -> None:
    """Measure source energy, pauses, pitch, and stereo position for performance/mix matching."""
    import librosa

    spatial_path = spatial_audio if spatial_audio and spatial_audio.is_file() else audio
    with sf.SoundFile(audio) as reader, sf.SoundFile(spatial_path) as spatial_reader:
        rate = reader.samplerate
        for cue in cues:
            start = max(0, round(float(cue["start"]) * rate))
            end = min(len(reader), round(float(cue["end"]) * rate))
            if end <= start:
                continue
            reader.seek(start)
            values = reader.read(end - start, dtype="float32", always_2d=True)
            mono = values.mean(axis=1)
            source_rms = rms(mono)
            frame = max(128, round(rate * 0.02))
            usable = len(mono) // frame * frame
            frame_rms = np.sqrt(np.mean(mono[:usable].reshape(-1, frame) ** 2, axis=1) + 1e-12) if usable else np.array([])
            threshold = max(2e-4, source_rms * 0.16)
            pause_ratio = float(np.mean(frame_rms < threshold)) if len(frame_rms) else 1.0
            if len(frame_rms):
                edges = np.linspace(0, len(frame_rms), 11, dtype=int)
                contour = [float(np.mean(frame_rms[edges[i]:max(edges[i] + 1, edges[i + 1])]))
                           for i in range(10)]
                scale = max(contour) or 1.0
                contour = [round(value / scale, 3) for value in contour]
            else:
                contour = []
            spatial_start = max(0, round(float(cue["start"]) * spatial_reader.samplerate))
            spatial_end = min(len(spatial_reader), round(float(cue["end"]) * spatial_reader.samplerate))
            spatial_reader.seek(spatial_start)
            spatial = spatial_reader.read(max(0, spatial_end - spatial_start), dtype="float32", always_2d=True)
            if spatial.shape[1] >= 2 and len(spatial):
                left, right = rms(spatial[:, 0]), rms(spatial[:, 1])
                pan = (right - left) / max(left + right, 1e-7)
            else:
                pan = 0.0
            pitch = 0.0
            if len(mono) >= rate // 3 and source_rms > 2e-4:
                try:
                    track = librosa.yin(mono, fmin=65, fmax=500, sr=rate,
                                        frame_length=min(2048, max(512, 2 ** int(np.log2(len(mono))))))
                    valid = track[np.isfinite(track) & (track >= 65) & (track <= 500)]
                    pitch = float(np.median(valid)) if len(valid) else 0.0
                except (ValueError, FloatingPointError):
                    pass
            word_count = len(str(cue.get("source", "")).split())
            spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) if len(mono) else np.array([])
            frequencies = np.fft.rfftfreq(len(mono), 1 / rate) if len(mono) else np.array([])
            spectral_centroid = (float(np.sum(frequencies * spectrum) / max(np.sum(spectrum), 1e-9))
                                 if len(spectrum) else 0.0)
            speech_band = ((frequencies >= 120) & (frequencies <= 4800)) if len(frequencies) else np.array([], dtype=bool)
            clarity = float(np.sum(spectrum[speech_band]) / max(np.sum(spectrum), 1e-9)) if len(spectrum) else 0.0
            tail_start = spatial_end
            tail_end = min(len(spatial_reader), tail_start + round(spatial_reader.samplerate * .25))
            spatial_reader.seek(tail_start)
            tail = spatial_reader.read(max(0, tail_end - tail_start), dtype="float32", always_2d=True)
            tail_ratio = rms(tail.mean(axis=1)) / max(source_rms, 1e-6) if len(tail) else 0.0
            distance = ("reverberant" if tail_ratio > .55 else
                        "close" if source_rms > .018 and spectral_centroid > 1500 and clarity > .72 else "medium")
            cue["source_performance"] = {
                "rms": round(source_rms, 7), "peak": round(float(np.max(np.abs(mono))), 5),
                "pause_ratio": round(pause_ratio, 3), "pitch_hz": round(pitch, 1),
                "pan": round(float(np.clip(pan, -0.7, 0.7)), 3),
                "words_per_second": round(word_count / max(0.2, float(cue["end"]) - float(cue["start"])), 2),
                "spectral_centroid_hz": round(spectral_centroid, 1),
                "speech_band_ratio": round(clarity, 3), "tail_ratio": round(tail_ratio, 3),
                "distance": distance, "energy_contour": contour,
            }


def measure_dialogue_leakage(cues: list[dict], dialogue_audio: Path,
                             background_audio: Path | None) -> None:
    """Estimate correlated source-dialogue residue left in the M&E bed."""
    if not background_audio or not background_audio.is_file():
        return
    with sf.SoundFile(dialogue_audio) as speech, sf.SoundFile(background_audio) as bed:
        for cue in cues:
            start = max(0, round(float(cue["start"]) * speech.samplerate))
            end = min(len(speech), round(float(cue["end"]) * speech.samplerate))
            bed_start = max(0, round(float(cue["start"]) * bed.samplerate))
            bed_end = min(len(bed), round(float(cue["end"]) * bed.samplerate))
            speech.seek(start); bed.seek(bed_start)
            source = speech.read(max(0, end - start), dtype="float32", always_2d=True).mean(axis=1)
            residue = bed.read(max(0, bed_end - bed_start), dtype="float32", always_2d=True).mean(axis=1)
            if bed.samplerate != speech.samplerate and len(residue):
                positions = np.linspace(0, len(residue) - 1, len(source))
                residue = np.interp(positions, np.arange(len(residue)), residue)
            size = min(len(source), len(residue))
            if size < 256 or rms(source[:size]) < 2e-4:
                cue["dialogue_leakage"] = 0.0; continue
            left = source[:size] - np.mean(source[:size]); right = residue[:size] - np.mean(residue[:size])
            correlation = abs(float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9)))
            level = min(1.0, rms(right) / max(rms(left), 1e-6))
            cue["dialogue_leakage"] = round(correlation * level, 3)


def suppress_dialogue_leakage(cues: list[dict], dialogue_audio: Path,
                              background_audio: Path | None, output: Path) -> Path | None:
    """Remove source-correlated speech residue from the inferred M&E bed.

    This is deliberately streaming so a feature film never enters RAM.  Linear
    source projection removes correlated separator bleed; high-leakage spans
    that are not linearly recoverable receive a conservative local attenuation.
    Music and effects outside aligned speech spans are bit-for-bit untouched at
    the sample-processing level.
    """
    if not background_audio or not background_audio.is_file():
        return background_audio
    output.parent.mkdir(parents=True, exist_ok=True)
    version = output.with_suffix(output.suffix + ".version")
    if output.is_file() and version.is_file() and version.read_text(encoding="utf-8").strip() == "2":
        return output
    with sf.SoundFile(background_audio) as bed:
        rate, channels = bed.samplerate, bed.channels
    aligned_dialogue = output.with_suffix(".dialogue-aligned.wav")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(dialogue_audio), "-ar", str(rate),
        "-ac", "1", "-c:a", "pcm_f32le", str(aligned_dialogue),
    ], check=True)
    spans: list[tuple[int, int, int]] = []
    for cue_index, cue in enumerate(cues):
        words = [(float(word["start"]), float(word["end"])) for word in cue.get("words", [])
                 if word.get("start") is not None and word.get("end") is not None]
        if not words:
            words = [(float(cue["start"]), float(cue["end"]))]
        for start, end in words:
            left = max(0, round((start - .04) * rate))
            right = max(left, round((end + .04) * rate))
            if right > left:
                spans.append((left, right, cue_index))
    spans.sort()
    chunk_frames = rate * 20
    cursor = 0
    cue_reduction: dict[int, list[float]] = {}
    temporary = output.with_suffix(output.suffix + ".tmp.flac")
    try:
        with sf.SoundFile(background_audio) as bed, sf.SoundFile(aligned_dialogue) as dialogue, sf.SoundFile(
            temporary, "w", samplerate=rate, channels=channels, format="FLAC", subtype="PCM_24"
        ) as writer:
            while True:
                values = bed.read(chunk_frames, dtype="float32", always_2d=True)
                if not len(values):
                    break
                speech = dialogue.read(len(values), dtype="float32", always_2d=True).mean(axis=1)
                if len(speech) < len(values):
                    speech = np.pad(speech, (0, len(values) - len(speech)))
                chunk_end = cursor + len(values)
                for left, right, cue_index in spans:
                    if right <= cursor:
                        continue
                    if left >= chunk_end:
                        break
                    local_left = max(0, left - cursor); local_right = min(len(values), right - cursor)
                    if local_right - local_left < 64:
                        continue
                    source = speech[local_left:local_right]
                    source = source - float(np.mean(source))
                    source_energy = float(np.dot(source, source)) + 1e-10
                    fade = min(round(.025 * rate), (local_right - local_left) // 3)
                    envelope = np.ones(local_right - local_left, dtype=np.float32)
                    if fade:
                        envelope[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
                        envelope[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
                    reductions = []
                    for channel in range(channels):
                        bed_segment = values[local_left:local_right, channel]
                        centered = bed_segment - float(np.mean(bed_segment))
                        beta = float(np.dot(centered, source) / source_energy)
                        correlation = abs(float(np.dot(centered, source) /
                            (np.linalg.norm(centered) * np.linalg.norm(source) + 1e-10)))
                        cleaned = bed_segment - np.clip(beta, -1.5, 1.5) * source
                        leakage = float(cues[cue_index].get("dialogue_leakage", 0.0))
                        if correlation < .06 and leakage > .16:
                            cleaned *= .35
                            reductions.append(9.1)
                        else:
                            before = float(np.sqrt(np.mean(bed_segment * bed_segment) + 1e-12))
                            after = float(np.sqrt(np.mean(cleaned * cleaned) + 1e-12))
                            reductions.append(max(0.0, 20 * np.log10(before / max(after, 1e-8))))
                        values[local_left:local_right, channel] = (
                            bed_segment * (1 - envelope) + cleaned * envelope
                        )
                    cue_reduction.setdefault(cue_index, []).append(max(reductions, default=0.0))
                writer.write(np.clip(values, -1, 1))
                cursor = chunk_end
        temporary.replace(output)
        version.write_text("2", encoding="utf-8")
    finally:
        aligned_dialogue.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    for index, cue in enumerate(cues):
        reduction = max(cue_reduction.get(index, [0.0]))
        cue["dialogue_leakage_suppression_db"] = round(reduction, 2)
        cue["dialogue_leakage_after_suppression"] = round(
            float(cue.get("dialogue_leakage", 0.0)) * 10 ** (-reduction / 20), 3)
    return output

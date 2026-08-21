from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def analyze_speakers(dialogue_audio: Path, cues: list[dict], output_dir: Path) -> dict[int, Path]:
    """Cluster cue-level CAMPPlus voiceprints and build a clean reference bank."""
    if not cues:
        return {}
    repo = Path(os.getenv("INDEXTTS_REPO", "vendor/index-tts")).resolve()
    model_dir = Path(os.getenv("INDEXTTS_MODEL_DIR", repo / "checkpoints")).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import torch
    import torchaudio
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(model_dir / "hf_cache" / "campplus_cn_common.bin", map_location="cpu"))
    model.eval().to(device)
    valid_indices: list[int] = []
    embeddings: list[np.ndarray] = []
    qualities: dict[int, dict] = {}

    with sf.SoundFile(dialogue_audio) as reader, torch.inference_mode():
        source_rate = reader.samplerate
        for index, cue in enumerate(cues):
            start = max(0, round((float(cue["start"]) - 0.12) * source_rate))
            end = min(len(reader), round((float(cue["end"]) + 0.12) * source_rate))
            if end <= start:
                continue
            reader.seek(start)
            samples = reader.read(end - start, dtype="float32", always_2d=True).mean(axis=1)
            qualities[index] = reference_metrics(samples, source_rate, cue)
            if len(samples) < source_rate * 0.65 or qualities[index]["rms"] < 2e-4:
                continue
            waveform = torch.from_numpy(samples)[None]
            if source_rate != 16_000:
                waveform = torchaudio.functional.resample(waveform, source_rate, 16_000)
            features = torchaudio.compliance.kaldi.fbank(
                waveform, num_mel_bins=80, dither=0, sample_frequency=16_000,
            )
            if features.shape[0] < 10:
                continue
            features = (features - features.mean(dim=0, keepdim=True)).to(device)
            embedding = model(features.unsqueeze(0)).squeeze(0).cpu().numpy()
            embedding /= np.linalg.norm(embedding) + 1e-9
            valid_indices.append(index)
            embeddings.append(embedding)

    preserve_diarization = all("speaker_id" in cue for cue in cues)
    labels = cluster_embeddings(np.stack(embeddings)) if embeddings and not preserve_diarization else np.zeros(0, dtype=int)
    if preserve_diarization:
        for index, cue in enumerate(cues):
            metrics = qualities.get(index, {})
            cue["reference_quality"] = float(metrics.get("score", 0.0))
            cue["reference_metrics"] = metrics
        output_dir.mkdir(parents=True, exist_ok=True)
        references = build_reference_bank(dialogue_audio, cues, output_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return references
    assignments: dict[int, int] = {index: int(label) for index, label in zip(valid_indices, labels)}
    valid_assignment_indices = set(assignments)
    for index in range(len(cues)):
        assignments.setdefault(index, -1)

    # Stable, human-readable numbering follows first appearance rather than cluster internals.
    ordered: dict[int, int] = {}
    for index in range(len(cues)):
        cluster = assignments[index]
        if cluster == -1:
            cues[index]["speaker_id"] = 0
            cues[index]["speaker"] = "Uncertain voice"
            cues[index]["speaker_confidence"] = 0.0
            metrics = qualities.get(index, {})
            cues[index]["reference_quality"] = float(metrics.get("score", 0.0))
            cues[index]["reference_metrics"] = metrics
            continue
        if cluster not in ordered:
            ordered[cluster] = len(ordered) + 1
        speaker_id = ordered[cluster]
        assignments[index] = speaker_id
        cues[index]["speaker_id"] = speaker_id
        cues[index]["speaker"] = f"Voice {speaker_id}"
        cues[index]["speaker_confidence"] = 0.9 if index in valid_assignment_indices else 0.0
        metrics = qualities.get(index, {})
        cues[index]["reference_quality"] = float(metrics.get("score", 0.0))
        cues[index]["reference_metrics"] = metrics

    for index, cue in enumerate(cues):
        overlaps = any(
            other != index and min(float(cue["end"]), float(value["end"]))
            - max(float(cue["start"]), float(value["start"])) > 0.08
            for other, value in enumerate(cues)
        )
        cue["overlapping_speech"] = overlaps
        if overlaps:
            cue["speaker_confidence"] = min(float(cue.get("speaker_confidence", 0.0)), 0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    references = build_reference_bank(dialogue_audio, cues, output_dir)
    del model
    return references


def cluster_embeddings(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    # CAMPPlus cosine distance: conservative threshold avoids merging distinct actors.
    clustering = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.24,
    )
    labels = clustering.fit_predict(values)
    # Avoid pathological over-segmentation caused by isolated, very short film cues.
    if len(set(labels)) > max(2, len(values) // 3):
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.34,
        )
        labels = clustering.fit_predict(values)
    return labels


def reference_metrics(samples: np.ndarray, rate: int, cue: dict | None = None) -> dict:
    """Score clone references for speech cleanliness, not mere loudness."""
    cue = cue or {}
    mono = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12)) if len(mono) else 0.0
    frame = max(1, round(rate * .02))
    usable = len(mono) // frame * frame
    levels = (np.sqrt(np.mean(mono[:usable].reshape(-1, frame) ** 2, axis=1) + 1e-12)
              if usable else np.zeros(0, dtype=np.float32))
    noise = float(np.percentile(levels, 20)) if len(levels) else 1e-6
    speech = float(np.percentile(levels, 90)) if len(levels) else 0.0
    snr_db = float(np.clip(20 * np.log10(max(speech, 1e-7) / max(noise, 1e-7)), 0, 60))
    threshold = max(10 ** (-48 / 20), noise * 2.5, speech * .10)
    active_ratio = float(np.mean(levels >= threshold)) if len(levels) else 0.0
    clipping = float(np.mean(np.abs(mono) >= .985)) if len(mono) else 1.0
    speech_band_ratio = 0.0
    if len(mono) >= 64:
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
        frequencies = np.fft.rfftfreq(len(mono), 1 / rate)
        total = float(spectrum[(frequencies >= 40) & (frequencies <= min(rate / 2, 11_000))].sum()) + 1e-12
        speech_band_ratio = float(spectrum[(frequencies >= 90) & (frequencies <= 4_800)].sum()) / total
    snr_score = float(np.clip((snr_db - 7) / 23, 0, 1))
    activity_score = float(np.clip(active_ratio / .48, 0, 1) * np.clip((1.02 - active_ratio) / .18, 0, 1))
    band_score = float(np.clip((speech_band_ratio - .48) / .42, 0, 1))
    confidence = float(cue.get("speaker_confidence", 0.0))
    tail = float(cue.get("source_performance", {}).get("tail_ratio", 0.0))
    score = (.32 * snr_score + .24 * activity_score + .22 * band_score + .22 * confidence)
    score *= max(.25, 1 - min(.65, tail * .35))
    score *= max(0.0, 1 - min(1.0, clipping * 35))
    if cue.get("overlapping_speech"):
        score *= .2
    return {"score": round(float(np.clip(score, 0, 1)), 4), "rms": round(rms, 7),
            "snr_db": round(snr_db, 2), "active_ratio": round(active_ratio, 3),
            "speech_band_ratio": round(speech_band_ratio, 3), "clipping_ratio": round(clipping, 6)}


def _vad_trim(samples: np.ndarray, rate: int) -> np.ndarray:
    frame = max(1, round(rate * .02)); usable = len(samples) // frame * frame
    if not usable:
        return samples
    levels = np.sqrt(np.mean(samples[:usable].reshape(-1, frame) ** 2, axis=1) + 1e-12)
    threshold = max(10 ** (-48 / 20), float(np.percentile(levels, 20)) * 2.5,
                    float(np.percentile(levels, 90)) * .10)
    active = np.flatnonzero(levels >= threshold)
    if not len(active):
        return samples
    start = max(0, int(active[0] * frame - rate * .06))
    end = min(len(samples), int((active[-1] + 1) * frame + rate * .08))
    return samples[start:end]


def build_reference_bank(dialogue_audio: Path, cues: list[dict], output_dir: Path) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[tuple[float, int, dict]]] = defaultdict(list)
    for index, cue in enumerate(cues):
        duration = float(cue["end"]) - float(cue["start"])
        quality = float(cue.get("reference_quality", 0.0))
        # A long but wrongly assigned line is far more damaging than a short
        # reference bank.  Only clean, confident full-film assignments are
        # allowed to teach a recurring character voice.  Tentative lines still
        # retain their candidate cluster for later reconciliation, but TTS uses
        # the line's own source voice instead of contaminating another actor.
        if (int(cue.get("speaker_id", 0)) <= 0
                or float(cue.get("speaker_confidence", 0.0)) < 0.62
                or bool(cue.get("overlapping_speech"))
                or duration < 0.65 or quality < .30):
            continue
        duration_score = min(duration, 6.0) / 6.0
        score = quality * .82 + duration_score * .18
        groups[int(cue["speaker_id"])].append((score, index, cue))

    references: dict[int, Path] = {}
    variants: dict[int, list[dict]] = {}
    with sf.SoundFile(dialogue_audio) as reader:
        rate = reader.samplerate
        for speaker_id, candidates in groups.items():
            parts = []; used_cues = []
            seconds = 0.0
            ordered = sorted(candidates, reverse=True)
            # A single pristine utterance is more coherent than a montage.  A
            # VAD-cleaned collection is only used when no 3–8 second take exists.
            pristine = [item for item in ordered
                        if 3.0 <= float(item[2]["end"]) - float(item[2]["start"]) <= 8.0
                        and float(item[2].get("reference_quality", 0.0)) >= .48]
            selected = pristine[:1] if pristine else ordered
            variants[speaker_id] = _acoustic_variants(reader, rate, pristine, selected,
                                                      output_dir, speaker_id, cues)
            for _, cue_index, cue in selected:
                start = max(0, round((float(cue["start"]) - 0.1) * rate))
                end = min(len(reader), round((float(cue["end"]) + 0.1) * rate))
                if end <= start:
                    continue
                reader.seek(start)
                samples = reader.read(end - start, dtype="float32", always_2d=True).mean(axis=1)
                samples = _vad_trim(samples, rate)
                if len(samples) < rate * 0.4:
                    continue
                parts.append(samples)
                used_cues.append(cue_index)
                seconds += len(samples) / rate
                if pristine or seconds >= 8.0:
                    break
                parts.append(np.zeros(round(rate * 0.08), dtype=np.float32))
            if not parts:
                continue
            montage = np.concatenate(parts)
            montage = montage[: round(rate * 12.0)]
            rms = float(np.sqrt(np.mean(montage * montage) + 1e-12))
            gain = min(12.0, 0.075 / max(rms, 1e-5))
            montage = np.clip(montage * gain, -0.95, 0.95)
            path = output_dir / f"voice-{speaker_id:03d}.wav"
            sf.write(path, montage, rate, subtype="PCM_16")
            transcript = ". ".join(str(cues[index].get("source", "")).strip(" .")
                                     for index in used_cues if str(cues[index].get("source", "")).strip())
            metadata = {"version": 2, "speaker_id": speaker_id, "reference_text": transcript,
                        "cue_indices": used_cues, "seconds": round(len(montage) / rate, 3),
                        "quality": round(max(float(cues[index].get("reference_quality", 0.0))
                                             for index in used_cues), 4)}
            path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
            references[speaker_id] = path
            # The curated clip competes as a candidate in its own right, so a
            # variant is measured against it rather than silently replacing it.
            variants.setdefault(speaker_id, []).insert(0, {
                "path": str(path), "canonical": True,
                "profile": _clip_profile(montage, rate),
                "reference_text": transcript,
                "quality": float(metadata["quality"]),
            })
    _write_variant_index(output_dir, variants)
    return references


# Below this, a speaker-embedding comparison carries more duration noise than
# speaker information, so it is reported but never used as a gate.
RELIABLE_EMBEDDING_SECONDS = 2.0


def _measured_seconds(path: Path) -> float:
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:
        return 0.0


VARIANT_QUALITY_FLOOR = .48
VARIANT_SWITCH_MARGIN = 1.5
VARIANT_ABSOLUTE_LIMIT = 4.0


def _clip_profile(samples: np.ndarray, rate: int) -> dict:
    """Describe a reference clip the same way cues describe source performance."""
    mono = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12)) if len(mono) else 0.0
    centroid = 0.0
    if len(mono) >= 64:
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        frequencies = np.fft.rfftfreq(len(mono), 1 / rate)
        centroid = float(np.sum(frequencies * spectrum) / max(float(np.sum(spectrum)), 1e-9))
    pitch = 0.0
    try:
        import librosa

        track = librosa.yin(mono, fmin=65, fmax=500, sr=rate,
                            frame_length=min(2048, max(512, 2 ** int(np.log2(max(len(mono), 512))))))
        valid = track[np.isfinite(track) & (track >= 65) & (track <= 500)]
        pitch = float(np.median(valid)) if len(valid) else 0.0
    except (ImportError, ValueError, FloatingPointError):
        pass
    return {"pitch_hz": round(pitch, 1), "rms": round(rms, 6),
            "spectral_centroid_hz": round(centroid, 1)}


def _acoustic_variants(reader, rate: int, pristine: list, selected: list, output_dir: Path,
                       speaker_id: int, cues: list[dict]) -> list[dict]:
    """Keep a few acoustically distinct takes per character, not just the best one.

    An actor does not sound like one five-second clip for a whole feature: they
    shout, whisper, and play scenes at different distances from the mic.  Cloning
    every line from a single take caps timbre fidelity no matter how good that
    take is, so the bank keeps spread and each line picks its nearest match.
    """
    keep = max(1, min(6, int(os.getenv("DUB_REFERENCE_VARIANTS", "4"))))
    # Only clips good enough to have been the bank clip are eligible. A variant
    # that matches a line's pitch but is short or noisy clones a worse voice
    # than the curated take it would displace.
    pool = [item for item in (pristine or selected)[:24]
            if float(item[2].get("reference_quality", 0.0)) >= VARIANT_QUALITY_FLOOR
            and float(item[2]["end"]) - float(item[2]["start"]) >= 2.0]
    chosen: list[dict] = []
    for _, cue_index, cue in pool:
        if len(chosen) >= keep:
            break
        start = max(0, round((float(cue["start"]) - 0.1) * rate))
        end = min(len(reader), round((float(cue["end"]) + 0.1) * rate))
        if end <= start:
            continue
        reader.seek(start)
        samples = _vad_trim(reader.read(end - start, dtype="float32", always_2d=True).mean(axis=1), rate)
        if len(samples) < rate * 2.0:
            continue
        profile = _clip_profile(samples, rate)
        if not profile["pitch_hz"]:
            continue
        # Spread, not repetition: a near-duplicate of a kept take teaches nothing
        # new about how this character sounds.
        if any(abs(profile["pitch_hz"] - item["profile"]["pitch_hz"]) < 6
               and abs(profile["spectral_centroid_hz"] - item["profile"]["spectral_centroid_hz"]) < 220
               for item in chosen):
            continue
        rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
        normalized = np.clip(samples * min(12.0, 0.075 / max(rms, 1e-5)), -0.95, 0.95)
        path = output_dir / f"take-{speaker_id:03d}-{len(chosen):02d}.wav"
        sf.write(path, normalized, rate, subtype="PCM_16")
        transcript = str(cues[cue_index].get("source", "")).strip()
        path.with_suffix(".json").write_text(json.dumps(
            {"speaker_id": speaker_id, "reference_text": transcript, "cue_index": cue_index,
             "profile": profile}, ensure_ascii=False, indent=2), encoding="utf-8")
        chosen.append({"path": str(path), "cue_index": cue_index, "profile": profile,
                       "reference_text": transcript,
                       "quality": round(float(cue.get("reference_quality", 0.0)), 4)})
    return chosen


def _write_variant_index(output_dir: Path, variants: dict[int, list[dict]]) -> None:
    payload = {str(speaker_id): items for speaker_id, items in variants.items() if items}
    (output_dir / "reference-variants.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_reference_variants(output_dir: Path) -> dict[int, list[dict]]:
    path = output_dir / "reference-variants.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {int(key): value for key, value in payload.items() if isinstance(value, list)}


def select_reference(cue: dict, variants: list[dict], default: Path) -> tuple[Path, str]:
    """Pick the take of this character that sounds most like this line.

    The bank clip is the curated one: chosen for cleanliness and length, and it
    is what identity scoring measures against.  A variant therefore has to earn
    its place -- it must be clearly closer to how this line was actually
    delivered, not merely closer.  Matching the pitch of a line with a worse
    recording clones a worse voice, which is a poor trade for a closer number.
    """
    performance = cue.get("source_performance") or {}
    pitch = float(performance.get("pitch_hz") or 0.0)
    if not variants or pitch <= 0:
        return default, "character bank"
    centroid = float(performance.get("spectral_centroid_hz") or 0.0)
    level = float(performance.get("rms") or 0.0)

    def distance(profile: dict) -> float | None:
        candidate_pitch = float(profile.get("pitch_hz") or 0.0)
        if candidate_pitch <= 0:
            return None
        value = abs(12 * math.log2(pitch / candidate_pitch))
        candidate_centroid = float(profile.get("spectral_centroid_hz") or 0.0)
        if centroid > 0 and candidate_centroid > 0:
            value += abs(math.log2(centroid / candidate_centroid)) * 2.0
        candidate_level = float(profile.get("rms") or 0.0)
        if level > 0 and candidate_level > 0:
            value += abs(20 * math.log10(level / candidate_level)) * .12
        return value

    baseline = None
    for item in variants:
        if item.get("canonical"):
            baseline = distance(item.get("profile") or {})
            break
    best, best_distance = None, float("inf")
    for item in variants:
        if item.get("canonical"):
            continue
        value = distance(item.get("profile") or {})
        if value is not None and value < best_distance:
            best, best_distance = item, value
    if best is None:
        return default, "character bank"
    # Never swap for a marginal gain, and never for a poorer recording.
    if baseline is not None and best_distance > baseline - VARIANT_SWITCH_MARGIN:
        return default, "character bank"
    if baseline is None and best_distance > VARIANT_ABSOLUTE_LIMIT:
        return default, "character bank"
    path = Path(str(best["path"]))
    if not path.is_file():
        return default, "character bank"
    return path, f"nearest take (±{best_distance:.1f})"


def score_speaker_similarity(cues: list[dict], generated_dir: Path,
                             references: dict[int, Path]) -> None:
    """Add CAMPPlus cosine similarity between each take and its character bank."""
    if not cues or not references:
        return
    repo = Path(os.getenv("INDEXTTS_REPO", "vendor/index-tts")).resolve()
    model_dir = Path(os.getenv("INDEXTTS_MODEL_DIR", repo / "checkpoints")).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import torch
    import torchaudio
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(model_dir / "hf_cache" / "campplus_cn_common.bin", map_location="cpu"))
    model.eval().to(device)

    def embed(path: Path) -> np.ndarray | None:
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(values.mean(axis=1))[None]
        if waveform.shape[-1] < rate * .35:
            return None
        if rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, rate, 16_000)
        features = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, dither=0,
                                                     sample_frequency=16_000)
        features = (features - features.mean(dim=0, keepdim=True)).to(device)
        with torch.inference_mode():
            value = model(features.unsqueeze(0)).squeeze(0).cpu().numpy()
        return value / (np.linalg.norm(value) + 1e-9)

    banks = {speaker_id: embed(path) for speaker_id, path in references.items() if path.is_file()}
    for index, cue in enumerate(cues, 1):
        bank = banks.get(int(cue.get("speaker_id", 0))); line = generated_dir / f"{index:06d}.wav"
        value = embed(line) if bank is not None and line.is_file() else None
        if bank is not None and value is not None:
            metrics = cue.setdefault("qc", {})
            metrics["speaker_similarity"] = round(float(np.dot(bank, value)), 3)
            # Speaker-embedding scores are only meaningful above a couple of
            # seconds: published verification results degrade ~46% relatively
            # between 3.6s and 2.0s utterances, and cropping one of our own
            # takes to different lengths moves this score by up to 0.15 with no
            # change of voice at all. Record how far the measurement can be
            # trusted so nothing downstream treats short-take noise as evidence.
            seconds = _measured_seconds(line)
            metrics["speaker_similarity_seconds"] = round(seconds, 2)
            metrics["speaker_similarity_reliable"] = bool(seconds >= RELIABLE_EMBEDDING_SECONDS)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

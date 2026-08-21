from __future__ import annotations

import json
import os
import queue
import copy
import csv
import zipfile
import shutil
import subprocess
import sys
import threading
import time
import statistics
import hashlib
from pathlib import Path
from typing import Callable, Sequence

from app.services.cinematic import recover_roformer, recover_vocals, separate_cinematic_audio
from app.services.analysis_cache import media_fingerprint, restore_json_artifact, store_json_artifact
from app.services.gpu_safety import (GPUStageUnsafe, gpu_checkpoint, gpu_safety_summary,
                                     gpu_stage, query_nvidia)
from app.services.job_lock import acquire_job_lock, job_is_running, release_job_lock
from app.services.asr import transcribe_aligned
from app.services.adapter import adapt_dialogue
from app.services.dialogue import (analyze_performance, build_adaptive_dialogue,
                                   measure_dialogue_leakage, suppress_dialogue_leakage)
from app.services.diarization import assign_diarized_speakers, diarize
from app.services.qc import (PERFORMANCE_RETAKE_THRESHOLD, PITCH_TOLERANCE_SEMITONES,
                             backtranscribe_lines, evaluate_media_qc, inspect_cues,
                             measure_performance_similarity, media_qc, register_correction,
                             write_report)
from app.services.speakers import (analyze_speakers, load_reference_variants,
                                   score_speaker_similarity, select_reference)
from app.services.visual_speakers import build_face_registry, register_visual_speakers
from app.services.lipsync import apply_selective_lipsync
from app.services.subtitles import (
    audio_streams, choose_audio, choose_embedded, extract_embedded, looks_english, media_duration, normalize_sidecar,
    parse_srt, probe_media, subtitle_streams,
    write_srt,
)
from app.services.tts import analyze_context_emotions, synthesize_qwen_fallback, synthesize_voice_lines
from app.services.translation_qc import TRANSLATION_QC_REVISION, validate_translations
from app.services.subprocess_control import controlled_lines, terminate_process
from app.store import JobStore


SAMPLE_RATE = 24_000
# Conditioning strength when the emotion prompt is the actor's own delivery of
# this exact line, and the value a retake escalates to.
#
# Raising this above 0.6 was an untested judgement call and coincided with a
# measured fall in speaker similarity (0.755 -> 0.700 on the same clip), so it
# is back at the value that produced the better identity. Stronger emotion
# conditioning appears to pull timbre away from the speaker; re-raise it only
# with a same-clip measurement showing it does not.
SOURCE_EMOTION_STRENGTH = 0.6
RETAKE_EMOTION_STRENGTH = 0.6
_whisper_models: dict[str, object] = {}
_whisper_lock = threading.Lock()
_run_context = threading.local()


class PipelinePause(Exception):
    pass


class PipelineCancel(Exception):
    pass


class PipelineApproval(Exception):
    pass


def run(*args: str) -> subprocess.CompletedProcess:
    checkpoint = getattr(_run_context, "checkpoint", None)
    if checkpoint is None:
        return subprocess.run(args, check=True, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1,
    )
    output: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            output.append(line)
        code = process.wait()
    except BaseException:
        terminate_process(process)
        # A cancelled FFmpeg command must never leave a partial file that a
        # resume path mistakes for a committed stem or mux.
        if args and Path(args[0]).name.lower().startswith("ffmpeg") and "-y" in args:
            candidate = Path(args[-1])
            if candidate.name.upper() != "NUL" and candidate.is_file():
                candidate.unlink(missing_ok=True)
        raise
    stdout = "".join(output)
    if code:
        raise subprocess.CalledProcessError(code, args, output=stdout, stderr=stdout)
    return subprocess.CompletedProcess(args, code, stdout=stdout, stderr="")


def inspect_system() -> dict:
    status = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "cuda": False,
        "gpu": "No CUDA GPU detected",
        "gpu_free_mb": 0, "disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / 1024 ** 3, 1),
        "engine": os.getenv("DUB_ENGINE", "indextts"),
        "model_ready": False, "emotion_ready": False, "aux_models_ready": False,
        "whisper_ready": False, "separator_ready": False, "recovery_ready": False,
        "roformer_ready": False, "asr_ready": False, "aligner_ready": False,
        "adapter_ready": False, "diarization_ready": False, "visual_speaker_ready": False,
        "translation_qc_ready": False,
        "asr_escalation_ready": False, "tts_fallback_ready": False, "musetalk_ready": False,
    }
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if process.returncode == 0 and process.stdout.strip():
            fields = [value.strip() for value in process.stdout.strip().splitlines()[0].split(",")]
            status["gpu"] = f"{fields[0]}, {fields[1]} MiB"
            if len(fields) > 2:
                status["gpu_free_mb"] = int(fields[2])
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        import torch
        status["cuda"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        pass
    repo = Path(os.getenv("INDEXTTS_REPO", "vendor/index-tts")).resolve()
    model_dir = Path(os.getenv("INDEXTTS_MODEL_DIR", repo / "checkpoints")).resolve()
    status["emotion_ready"] = (model_dir / "qwen0.6bemo4-merge" / "model.safetensors").is_file()
    status["aux_models_ready"] = all(path.exists() for path in (
        model_dir / "hf_cache" / "w2v-bert-2.0",
        model_dir / "hf_cache" / "semantic_codec_model.safetensors",
        model_dir / "hf_cache" / "campplus_cn_common.bin",
        model_dir / "hf_cache" / "bigvgan" / "bigvgan_generator.pt",
    ))
    whisper_dir = Path(os.getenv("WHISPER_CACHE_DIR", "vendor/whisper")).resolve()
    status["whisper_ready"] = (whisper_dir / "large-v3-turbo.pt").is_file()
    bandit_repo = Path(os.getenv("BANDIT_REPO", "vendor/bandit-v2")).resolve()
    status["separator_ready"] = (
        (bandit_repo / "src" / "models" / "bandit" / "bandit.py").is_file()
        and (Path(os.getenv("BANDIT_CHECKPOINT", bandit_repo / "checkpoints" / "checkpoint-multi.ckpt"))).is_file()
    )
    try:
        import importlib.util
        demucs_weights = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "955717e8-8726e21a.th"
        status["recovery_ready"] = importlib.util.find_spec("demucs") is not None and demucs_weights.is_file()
    except (ImportError, ValueError):
        pass
    roformer_dir = Path(os.getenv(
        "ROFORMER_MODEL_DIR", "vendor/melband-roformer/melband-roformer-kim-vocals"
    )).resolve()
    status["roformer_ready"] = (roformer_dir / "MelBandRoformer.ckpt").is_file() and (
        roformer_dir / "config_vocals_mel_band_roformer.yaml").is_file()
    status["asr_ready"] = (Path(os.getenv(
        "QWEN_ASR_MODEL", "vendor/qwen3-asr-0.6b-qwen"
    )).resolve() / "model.safetensors").is_file()
    escalation_dir = Path(os.getenv(
        "QWEN_ASR_ESCALATION_MODEL", "vendor/qwen3-asr-1.7b-qwen"
    )).resolve()
    status["asr_escalation_ready"] = ((escalation_dir / "model.safetensors").is_file()
                                       or (escalation_dir / "model.safetensors.index.json").is_file())
    status["aligner_ready"] = (Path(os.getenv(
        "QWEN_ALIGNER_MODEL", "vendor/qwen3-forced-aligner-0.6b-qwen"
    )).resolve() / "model.safetensors").is_file()
    status["adapter_ready"] = Path(os.getenv(
        "TRANSLATION_MODEL", "vendor/hy-mt2-7b/Hy-MT2-7B-Q4_K_M.gguf"
    )).resolve().is_file()
    status["translation_qc_ready"] = Path(os.getenv(
        "TRANSLATION_QC_MODEL", "vendor/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"
    )).resolve().is_file()
    status["diarization_ready"] = (Path(os.getenv(
        "PYANNOTE_MODEL", "vendor/pyannote-community-1"
    )).resolve() / "config.yaml").is_file() and Path(os.getenv(
        "PYANNOTE_RUNTIME", "vendor/pyannote-env/Scripts/python.exe"
    )).resolve().is_file()
    face_dir = Path(os.getenv("OPENCV_FACE_MODEL_DIR", "vendor/opencv-face")).resolve()
    status["visual_speaker_ready"] = all((face_dir / name).is_file() for name in (
        "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"))
    status["tts_fallback_ready"] = (Path(os.getenv(
        "QWEN_TTS_MODEL", "vendor/qwen3-tts-1.7b-base"
    )).resolve() / "model.safetensors").is_file()
    status["musetalk_ready"] = (Path(os.getenv(
        "MUSETALK_MODEL_DIR", "vendor/MuseTalk/models/musetalkV15"
    )).resolve() / "unet.pth").is_file()
    primary_ready = all((model_dir / name).is_file() for name in ("config.yaml", "gpt.pth", "s2mel.pth", "codec.pth"))
    status["model_ready"] = primary_ready and status["emotion_ready"] and status["aux_models_ready"]
    status["gpu_safety"] = gpu_safety_summary()
    return status


class PipelineWorker:
    def __init__(self, store: JobStore, data_root: Path):
        self.store = store
        self.data_root = data_root
        self.jobs: queue.Queue[str | None] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stopping.clear()
        self.thread = threading.Thread(target=self._work, name="dubline-gpu-worker", daemon=True)
        self.thread.start()
        for job_id in self.store.recover_interrupted(
            lambda job: job_is_running(Path(job["folder"])) if job.get("folder") else False
        ):
            self.submit(job_id)
        for job in self.store.list_summaries():
            if job.get("status") == "paused" and job.get("auto_resume_pending"):
                self.submit(str(job["id"]))

    def stop(self) -> None:
        self.stopping.set()
        self.jobs.put(None)
        if self.thread:
            self.thread.join(timeout=3)

    def submit(self, job_id: str) -> None:
        # Duplicate queue notices are harmless because run_pipeline accepts only
        # an actually queued job.  Avoiding an "already enqueued" flag closes a
        # pause/resume race where resume could arrive just before the old queue
        # notice was retired and then be lost permanently.
        self.jobs.put(job_id)

    def _work(self) -> None:
        while not self.stopping.is_set():
            job_id = self.jobs.get()
            if job_id is None:
                break
            try:
                current = self.store.get_summary(job_id) or {}
                if current.get("status") == "paused" and current.get("auto_resume_pending"):
                    self._resume_after_cooldown(job_id)
                else:
                    run_pipeline(job_id, self.store)
                    current = self.store.get_summary(job_id) or {}
                    if current.get("status") == "paused" and current.get("auto_resume_pending"):
                        self._resume_after_cooldown(job_id)
            finally:
                self.jobs.task_done()

    def _resume_after_cooldown(self, job_id: str) -> None:
        """Resume only watchdog-created pauses after stable thermal recovery."""
        stable_samples = 0
        deadline = time.monotonic() + float(os.getenv("DUB_GPU_AUTO_RESUME_TIMEOUT_SECONDS", "600"))
        last_label = ""
        while not self.stopping.is_set() and time.monotonic() < deadline:
            current = self.store.get_summary(job_id) or {}
            if current.get("status") != "paused" or not current.get("auto_resume_pending"):
                return
            try:
                health = query_nvidia()
                safe = (health["temperature_c"] <= 70 and health["utilization"] <= 10
                        and health["free_mb"] >= health["total_mb"] - 900)
                stable_samples = stable_samples + 1 if safe else 0
                label = (f"Cooling GPU safely · {health['temperature_c']}°C · "
                         f"auto-resume after {max(0, 3 - stable_samples)} stable check(s)")
            except Exception as exc:
                stable_samples = 0
                label = f"Waiting for NVIDIA health monitoring · {exc}"
            if label != last_label:
                self.store.update(job_id, stage=label)
                last_label = label
            if stable_samples >= 3:
                with self.store.lock:
                    current = self.store.get_summary(job_id) or {}
                    if current.get("status") != "paused" or not current.get("auto_resume_pending"):
                        return
                    self.store.update(job_id, status="queued", stage="GPU cooled · resuming automatically",
                                      error=None, control=None, auto_resume_pending=False)
                self.submit(job_id)
                return
            self.stopping.wait(2.0)
        current = self.store.get_summary(job_id) or {}
        if current.get("status") == "paused" and current.get("auto_resume_pending"):
            self.store.update(job_id, stage="GPU safety pause · automatic cooldown timed out",
                              auto_resume_pending=False)


def run_pipeline(job_id: str, store: JobStore) -> None:
    run_started_at = time.time()
    initial = store.get(job_id)
    if not initial or initial.get("status") != "queued":
        return
    execution_lock = acquire_job_lock(Path(initial["folder"]))
    if execution_lock is None:
        return
    # Claim and transition under the same store lock used by pause/cancel.  A
    # control request can therefore happen entirely before or after this claim,
    # never in a gap where the worker overwrites it.
    with store.lock:
        job = store.get(job_id)
        if not job or job.get("status") != "queued":
            release_job_lock(execution_lock)
            return
        job = store.update(
            job_id, status="processing", control=None, error=None,
            processing_started_at=job.get("processing_started_at") or run_started_at,
            active_run_started_at=run_started_at,
        )
    folder = Path(job["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    original_source = Path(job["input_path"])
    source = original_source
    options = job.get("options", {})
    accumulated_active = float(job.get("active_processing_seconds") or 0.0)

    def active_seconds() -> float:
        return accumulated_active + max(0.0, time.time() - run_started_at)

    def update(progress: float, stage: str, **extra) -> dict:
        return store.update(job_id, status="processing", progress=round(progress, 1),
                            stage=stage, error=None, **extra)

    def log(message: str) -> None:
        store.append_log(job_id, message)

    def checkpoint() -> None:
        gpu_checkpoint()
        current = store.get(job_id) or {}
        action = current.get("control")
        if action == "pause":
            raise PipelinePause()
        if action == "cancel":
            raise PipelineCancel()

    _run_context.checkpoint = checkpoint
    try:
        checkpoint()
        preflight = inspect_system()
        if not preflight["cuda"]:
            raise RuntimeError("CUDA is unavailable; this local film pipeline requires the NVIDIA GPU")
        if 0 < preflight.get("gpu_free_mb", 0) < 6500:
            log(f"Only {preflight['gpu_free_mb']} MB GPU memory is free; close GPU-heavy applications if a model fails")
        update(2, "Inspecting media")
        original_probe = probe_media(original_source)
        source_duration = media_duration(original_probe)
        if source_duration <= 0:
            raise RuntimeError("The source has no readable duration")
        range_start = float(options.get("range_start") or 0.0)
        range_end = float(options.get("range_end") or source_duration)
        if range_start >= source_duration:
            raise RuntimeError("The selected section starts after the film ends")
        range_end = min(range_end, source_duration)
        if range_end <= range_start:
            raise RuntimeError("The selected section end must be after its start")
        selected_range = range_start > .001 or range_end < source_duration - .001
        if selected_range:
            source = folder / "selected-source.mkv"
            if not source.is_file():
                update(2.5, f"Extracting selected section · {format_duration(range_start)} to {format_duration(range_end)}")
                run("ffmpeg", "-y", "-v", "error", "-i", str(original_source),
                    "-ss", f"{range_start:.3f}", "-t", f"{range_end - range_start:.3f}",
                    "-map", "0", "-c", "copy", "-avoid_negative_ts", "make_zero", str(source))
            log(f"Selected {format_duration(range_start)} to {format_duration(range_end)}; "
                "the delivered MKV will contain this section")
        probe = probe_media(source)
        duration = media_duration(probe)
        if duration <= 0:
            raise RuntimeError("The selected source section has no readable duration")
        video_stream = next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"), None)
        free_bytes = shutil.disk_usage(folder).free
        estimated_bytes = source.stat().st_size + int(duration / 60 * 55 * 1024 ** 2) + 2 * 1024 ** 3
        store.update(job_id, storage={"free_gb": round(free_bytes / 1024 ** 3, 1),
                                      "estimated_gb": round(estimated_bytes / 1024 ** 3, 1)})
        if free_bytes < estimated_bytes:
            raise RuntimeError(
                f"This film needs about {estimated_bytes / 1024 ** 3:.1f} GB of working space, "
                f"but only {free_bytes / 1024 ** 3:.1f} GB is free. Move DUB_WORKDIR to a larger SSD."
            )
        subs = subtitle_streams(probe)
        available_audio = audio_streams(probe)
        working_audio, audio_warnings = choose_audio(available_audio, options.get("audio_stream_index"))
        media = {
            "duration": duration, "source_duration": source_duration,
            "range_start": range_start, "range_end": range_end,
            "selection": "section" if selected_range else "full film",
            "width": video_stream.get("width") if video_stream else None,
            "height": video_stream.get("height") if video_stream else None,
            "video_codec": video_stream.get("codec_name") if video_stream else None,
            "media_kind": "video" if video_stream else "audio", "subtitle_streams": subs,
            "audio_streams": available_audio, "working_audio_stream": working_audio,
            "audio_selection_warnings": audio_warnings,
        }
        store.update(job_id, media=media)
        historical_factors = []
        for previous in store.list_summaries(50):
            previous_media = previous.get("media") or {}
            if (previous.get("status") not in {"complete", "needs_review"}
                    or previous_media.get("media_kind") != media["media_kind"]):
                continue
            factor = (previous.get("throughput") or {}).get("realtime_factor")
            if factor is None and previous.get("processing_started_at") and previous.get("updated_at"):
                prior_duration = float(previous_media.get("duration") or 0)
                if prior_duration > 0:
                    factor = (float(previous["updated_at"]) - float(previous["processing_started_at"])) / prior_duration
            if factor is not None and .2 <= float(factor) <= 2000:
                historical_factors.append(float(factor))
        if historical_factors:
            factor = statistics.median(historical_factors)
            store.update(job_id, eta={"predicted_seconds": round(duration * factor),
                                      "historical_realtime_factor": round(factor, 2),
                                      "sample_jobs": len(historical_factors)})
        else:
            store.update(job_id, eta={"predicted_seconds": None, "sample_jobs": 0})
        log(f"Media inspected: {format_duration(source_duration)} source, {format_duration(duration)} selected, "
            f"{len(subs)} embedded subtitle track(s)")
        log(f"Working soundtrack: stream {working_audio['index']} · {working_audio['title']} · "
            f"{working_audio['language']} · {working_audio['channels']} channel(s)")
        for warning in audio_warnings:
            log(f"Audio-track warning: {warning}")

        fingerprint = media_fingerprint(original_source, folder)
        store.update(job_id, media_fingerprint=fingerprint)
        update(3, "Registering recurring characters across the whole film")
        face_registry = build_face_registry(
            original_source, folder, source_duration,
            lambda value: update(3 + value * 1.5,
                f"Scanning the whole film for recurring characters · {value * 100:.0f}%"), checkpoint,
            cache_key=fingerprint,
        ) if video_stream else {"available": False, "identities": 0, "reason": "audio-only source"}
        store.update(job_id, face_registry=face_registry)
        if face_registry.get("available"):
            if face_registry.get("cache_hit"):
                log(f"Reused the prior whole-film character analysis for this exact media file "
                    f"({face_registry['identities']} recurring visual identity candidates)")
            else:
                log(f"Whole-film face registry clustered {face_registry['identities']} recurring visual identity candidate(s); "
                    "speaker names remain confidence-gated until mouth and voice evidence agree")
        else:
            log("Whole-film face registry was unavailable; audio-only speaker mapping remains active")

        audio_mode = options.get("audio_mode", "separate")
        background_stem: Path | None = None
        working_soundtrack = folder / "working-soundtrack-48k.flac"
        if not working_soundtrack.is_file():
            run("ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", f"0:{working_audio['index']}",
                "-vn", "-ar", "48000", "-c:a", "flac", str(working_soundtrack))
        dialogue_source: Path = working_soundtrack
        if audio_mode == "separate":
            source_audio_streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"]
            me_stream = next((stream for stream in source_audio_streams if any(
                marker in (str(stream.get("tags", {}).get("title", "")) + " " +
                           str(stream.get("tags", {}).get("handler_name", ""))).lower()
                for marker in ("music & effects", "music and effects", "m&e", "dme", "clean effects")
            )), None)
            film_mix = working_soundtrack
            dialogue_stem = folder / "cinema-dialogue.flac"
            background_stem = folder / "cinema-background.flac"
            separated_background = folder / "cinema-background-separated.flac" if me_stream else background_stem
            recovery_stem = folder / "recovery-vocals.flac"
            roformer_stem = folder / "roformer-vocals.flac"
            if not dialogue_stem.exists() or not separated_background.exists():
                update(7, "Separating dialogue, music, and effects")
                with gpu_stage(folder, "Bandit cinematic separation", checkpoint):
                    separate_cinematic_audio(
                        film_mix, dialogue_stem, separated_background,
                        lambda value: update(7 + value * 11, f"Separating dialogue from the film mix · {value * 100:.0f}%"),
                        checkpoint,
                    )
                log("Bandit v2 multilingual separation produced clean dialogue and music/effects stems")
            if me_stream and not background_stem.exists():
                update(18, "Using the supplied music-and-effects track")
                run("ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", f"0:{me_stream['index']}",
                    "-vn", "-ar", "48000", "-c:a", "flac", str(background_stem))
                log(f"Preferred supplied M&E/DME track: {me_stream.get('tags', {}).get('title', 'untitled')}")
            if not recovery_stem.exists():
                update(18, "Checking for dialogue hidden by music and effects")
                with gpu_stage(folder, "HTDemucs dialogue recovery", checkpoint):
                    recover_vocals(
                        film_mix, recovery_stem,
                        lambda value: update(18 + value * 6, f"Recovering difficult voice lines · {value * 100:.0f}%"),
                        checkpoint,
                    )
                log("HTDemucs produced an independent vocal recovery stem for confidence fallback")
            if not roformer_stem.exists():
                update(22, "Recovering masked dialogue with MelBand-RoFormer")
                with gpu_stage(folder, "MelBand-RoFormer dialogue recovery", checkpoint):
                    recover_roformer(
                        film_mix, roformer_stem,
                        lambda value: update(22 + value * 2,
                            f"Running modern vocal recovery · {value * 100:.0f}%"), checkpoint,
                    )
                log("MelBand-RoFormer produced a modern vocal recovery candidate")
            dialogue_source = dialogue_stem
            store.update(job_id, separation={
                "model": "Bandit v2 multilingual + MelBand-RoFormer + HTDemucs", "sample_rate": 48000,
                "dialogue": True, "background": True, "recovery": True, "roformer": True,
                "bed_source": "supplied M&E/DME" if me_stream else "Bandit music + effects",
            })

        primary_audio = folder / "dialogue-primary-24k.flac"
        if not primary_audio.exists():
            update(24, "Preparing clean dialogue for understanding")
            run("ffmpeg", "-y", "-v", "error", "-i", str(dialogue_source), "-vn", "-ac", "1",
                "-ar", str(SAMPLE_RATE), "-c:a", "flac", str(primary_audio))
        recovery_audio = folder / "dialogue-recovery-24k.flac"
        recovery_stem = folder / "recovery-vocals.flac"
        if audio_mode == "separate" and recovery_stem.exists() and not recovery_audio.exists():
            run("ffmpeg", "-y", "-v", "error", "-i", str(recovery_stem), "-vn", "-ac", "1",
                "-ar", str(SAMPLE_RATE), "-c:a", "flac", str(recovery_audio))
        roformer_audio = folder / "dialogue-roformer-24k.flac"
        roformer_stem = folder / "roformer-vocals.flac"
        if audio_mode == "separate" and roformer_stem.exists() and not roformer_audio.exists():
            run("ffmpeg", "-y", "-v", "error", "-i", str(roformer_stem), "-vn", "-ac", "1",
                "-ar", str(SAMPLE_RATE), "-c:a", "flac", str(roformer_audio))
        understanding_audio = recovery_audio if recovery_audio.exists() else primary_audio
        checkpoint()

        cues_path = folder / "cues.json"
        if cues_path.exists():
            cues = json.loads(cues_path.read_text(encoding="utf-8"))
            cue_source = (store.get(job_id) or {}).get("cue_source", "Recovered cue sheet")
            log(f"Recovered {len(cues)} timed lines")
        else:
            update(26, "Finding the best dialogue source")
            with gpu_stage(folder, "Qwen source-language ASR", checkpoint):
                cues, cue_source = build_cues(source, understanding_audio, folder, job, probe, options,
                                              checkpoint, update, log)
            if not cues:
                raise RuntimeError("No spoken dialogue or readable subtitles were found")
            for index, cue in enumerate(cues, 1):
                cue.update({"id": index, "speaker": "Identifying…", "status": "waiting"})
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
            shutil.rmtree(folder / "speech-chunks", ignore_errors=True)
        alignment_restored = hydrate_alignment_confidence(cues)
        if alignment_restored:
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"Restored measured forced-alignment evidence for {alignment_restored} cached line(s)")
        if prepare_translation_revision(folder, cues):
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
            log("Upgraded cached dialogue to source-faithful translation with independent subtitle evidence")
        speaker_placeholders_changed = False
        for cue in cues:
            if cue.get("speaker_id") is None and cue.get("speaker") != "Identifying…":
                cue["speaker"] = "Identifying…"
                speaker_placeholders_changed = True
        if speaker_placeholders_changed:
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
        update(34, f"Prepared {len(cues)} English voice lines", cues=cues, cue_source=cue_source)
        log(f"Dialogue source: {cue_source}")
        checkpoint()

        reference_audio = primary_audio
        if recovery_audio.exists():
            reference_audio = folder / "dialogue-adaptive-24k.flac"
            if not reference_audio.exists() or any("dialogue_source" not in cue for cue in cues):
                update(34.5, "Verifying every source voice line")
                selection = build_adaptive_dialogue(primary_audio, recovery_audio, cues, reference_audio,
                                                    roformer=roformer_audio if roformer_audio.exists() else None)
                cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
                store.update(job_id, separation={**((store.get(job_id) or {}).get("separation") or {}),
                                                  **selection})
                log(f"Dialogue confidence gate recovered {selection['recovered_cues']} of "
                    f"{selection['total_cues']} lines missed by the cinematic stem "
                    f"({selection.get('roformer_cues', 0)} RoFormer, {selection.get('demucs_cues', 0)} HTDemucs)")
        if any("source_performance" not in cue for cue in cues):
            update(34.8, "Measuring source rhythm, pitch, pauses, and position")
            # Measure the same per-cue source selected by the confidence gate.  The
            # pristine cinematic stem can be silent for distant/PA dialogue that was
            # deliberately recovered from the vocal model.
            analyze_performance(reference_audio, cues,
                                film_mix if audio_mode == "separate" else source)
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
        if any("dialogue_leakage" not in cue for cue in cues):
            measure_dialogue_leakage(cues, reference_audio, background_stem)
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")

        # Confirm the source transcript before anything is built on top of it.
        # Everything downstream -- translation, its independent QC, the dub -- is
        # only as true as this text, and until now nothing checked it against a
        # second model except reactively, after a translation had already failed.
        if any("source_asr_agreement" not in cue for cue in cues):
            unverified = source_asr_verification_candidates(cues)
            if unverified:
                update(34.9, f"Confirming {len(unverified)} source line(s) with a second transcriber")
                verification_folder = folder / "source-verification"
                verification_folder.mkdir(exist_ok=True)
                with gpu_stage(folder, "Whisper source transcript verification", checkpoint,
                               minimum_free_mb=5000):
                    opinions = verify_translation_disputes(
                        reference_audio, unverified, verification_folder,
                        lambda value, index: update(34.9 + value * .05,
                            f"Second transcription · line {index + 1} of {len(unverified)}"),
                        checkpoint,
                    )
                disputed = apply_source_asr_agreement(cues, opinions)
                for cue in cues:
                    cue.setdefault("source_asr_agreement", None)
                log(f"Independent transcription disputed {disputed} of {len(unverified)} "
                    f"source line(s); both readings now travel forward as evidence")
                cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")

        # Translation, emotion analysis, and TTS are deliberately sequential. This keeps
        # the full pipeline within the RTX 4060 Laptop GPU's 8 GB VRAM budget.
        release_whisper_models()
        speaker_reference_dir = folder / "speaker-references"
        reference_marker = speaker_reference_dir / "reference-version.txt"
        reference_version = "acoustic-variant-reference-v3"
        prior_reference_version = (reference_marker.read_text(encoding="utf-8").strip()
                                   if reference_marker.is_file() else "")
        if prior_reference_version != reference_version:
            shutil.rmtree(speaker_reference_dir, ignore_errors=True)
            for cue in cues:
                cue.pop("reference_quality", None)
                cue.pop("reference_metrics", None)
        if any("speaker_id" not in cue for cue in cues) or not list(speaker_reference_dir.glob("voice-*.wav")):
            update(35, "Matching lines to consistent voices")
            diarization_audio = reference_audio
            diarization_folder = folder
            diarization_cues = cues
            if selected_range:
                diarization_folder = folder / "full-film-speaker-registry"
                diarization_folder.mkdir(exist_ok=True)
                diarization_audio = diarization_folder / "full-film-context-16k.flac"
                diarization_cues = copy.deepcopy(cues)
                for cue in diarization_cues:
                    cue["start"] = float(cue["start"]) + range_start
                    cue["end"] = float(cue["end"]) + range_start
                update(35, "Registering voices from the whole film")
                diarization_cache_artifact = (
                    f"speaker-diarization-community1-v3-stream-{working_audio['index']}.json"
                )
                restored_diarization = restore_json_artifact(
                    folder, fingerprint, diarization_cache_artifact,
                    diarization_folder / "speaker-diarization.json", expected_version=3,
                )
                if not restored_diarization:
                    legacy_artifact = f"speaker-diarization-community1-v2-stream-{working_audio['index']}.json"
                    restored_legacy = restore_json_artifact(
                        folder, fingerprint, legacy_artifact,
                        diarization_folder / "speaker-diarization.json", expected_version=2,
                    )
                    if restored_legacy:
                        migrated = json.loads((diarization_folder / "speaker-diarization.json").read_text(
                            encoding="utf-8"))
                        migrated["version"] = 3
                        migrated.setdefault("analysis", {})["execution"] = (
                            "reused trusted whole-film result created before chunked CUDA"
                        )
                        temporary = diarization_folder / "speaker-diarization.json.tmp"
                        temporary.write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")
                        temporary.replace(diarization_folder / "speaker-diarization.json")
                        store_json_artifact(
                            folder, fingerprint, diarization_cache_artifact,
                            diarization_folder / "speaker-diarization.json", expected_version=3,
                        )
                        restored_diarization = True
                        log("Upgraded and reused the prior completed whole-film speaker analysis")
                if restored_diarization:
                    log("Reused the prior whole-film voice-turn analysis for this exact soundtrack")
                elif not diarization_audio.is_file():
                    update(34.9, "Preparing whole-film voice context")
                    run("ffmpeg", "-y", "-v", "error", "-i", str(original_source),
                        "-map", f"0:{working_audio['index']}",
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(diarization_audio))
            diarization = diarize(
                diarization_audio, diarization_folder,
                lambda value: update(35 + value * 1.2, f"Mapping speaker turns · {value * 100:.0f}%"), checkpoint,
            )
            if diarization and selected_range and not restored_diarization:
                store_json_artifact(
                    folder, fingerprint, diarization_cache_artifact,
                    diarization_folder / "speaker-diarization.json", expected_version=3,
                )
            if diarization:
                assign_diarized_speakers(diarization_cues, diarization)
                if diarization_cues is not cues:
                    for cue, registered in zip(cues, diarization_cues):
                        for key in ("speaker_id", "speaker", "speaker_confidence", "diarization_label",
                                    "speaker_assignment", "overlapping_speech"):
                            if key in registered:
                                cue[key] = registered[key]
                labels = sorted({str(turn["speaker"]) for turn in
                                 (diarization.get("exclusive_diarization") or diarization.get("diarization") or [])})
                store.update(job_id, speaker_registry={
                    "scope": "whole film" if selected_range else "processed film",
                    "voice_clusters": len(labels), "labels": labels,
                })
                confident_voices = len({int(cue["speaker_id"]) for cue in cues
                                        if float(cue.get("speaker_confidence", 0.0)) >= 0.62})
                log(f"pyannote Community-1 used {(source_duration if selected_range else duration) / 60:.1f} minutes of film context "
                    f"and found {confident_voices} confident voices in the dialogue map")
            else:
                log("pyannote Community-1 is unavailable; using CAMPPlus confidence-gated speaker clustering")
            update(36.3, "Registering visible speakers and mouth motion")
            visual = register_visual_speakers(
                source, cues, folder,
                lambda value: update(36.3 + value * .5,
                    f"Checking visible speakers · {value * 100:.0f}%"), checkpoint,
            )
            if visual.get("available"):
                log(f"Audiovisual registration found {visual['anchors']} trusted face/voice anchor(s); "
                    f"reconciled {visual['reassigned']} tentative line(s) and created "
                    f"{visual.get('created_visual_voices', 0)} anonymous visual voice(s)")
            else:
                log("No reliable visual speaker anchors; retained full-film audio assignments")
            with gpu_stage(folder, "CAMPPlus speaker reference scoring", checkpoint, minimum_free_mb=3000):
                analyze_speakers(reference_audio, cues, speaker_reference_dir)
            reference_marker.write_text(reference_version, encoding="utf-8")
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
            update(37, f"Matched {len(set(cue['speaker_id'] for cue in cues))} source voices", cues=cues)
        reference_variants = load_reference_variants(speaker_reference_dir)
        speaker_references = {
            int(path.stem.split("-")[-1]): path for path in speaker_reference_dir.glob("voice-*.wav")
        }
        allocate_dub_windows(cues, duration)
        cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
        if any("adaptation_confidence" not in cue for cue in cues):
            update(37.2, "Adapting dialogue for meaning, timing, and natural speech")
            with gpu_stage(folder, "Hy-MT2 dialogue adaptation", checkpoint, minimum_free_mb=6000):
                cues = adapt_dialogue(
                    cues, folder,
                    lambda value, index: update(37.2 + value * 0.7,
                        f"Translating and adapting dialogue · line {index + 1} of {len(cues)}"), checkpoint,
                )
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
        if any((cue.get("translation_qc") or {}).get("revision") != TRANSLATION_QC_REVISION
               for cue in cues):
            update(37.8, "Independently checking translation accuracy")
            with gpu_stage(folder, "Qwen bilingual translation QC", checkpoint, minimum_free_mb=6000):
                cues = validate_translations(
                    cues, folder,
                    lambda value, index: update(37.8 + value * .2,
                        f"Independent bilingual QC · line {index + 1} of {len(cues)}"), checkpoint,
                )
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
        disputes = translation_dispute_candidates(cues)
        if disputes:
            update(37.81, f"Resolving {len(disputes)} uncertain subtitle/ASR disagreement(s)")
            with gpu_stage(folder, "Whisper selective ASR dispute check", checkpoint,
                           minimum_free_mb=5000):
                second_opinions = verify_translation_disputes(
                    reference_audio, disputes, folder,
                    lambda value, index: update(37.81 + value * .01,
                        f"Second transcription · line {index + 1} of {len(disputes)}"), checkpoint,
                )
            disputed_ids = set(second_opinions)
            for cue in cues:
                cue_id = int(cue["id"])
                if cue_id in disputed_ids:
                    cue["asr_second_opinion"] = second_opinions[cue_id]
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
        evidence_cues = second_asr_evidence_candidates(cues)
        if evidence_cues:
            with gpu_stage(folder, "Hy-MT2 second-ASR evidence translation", checkpoint,
                           minimum_free_mb=6000):
                evidence = translate_second_asr_evidence(
                    evidence_cues, folder,
                    lambda value, index: update(37.82 + value * .01,
                        f"Translating second transcription · {index + 1} of {len(evidence_cues)}"),
                    checkpoint,
                )
            evidence_ids = set(evidence)
            for cue in cues:
                cue_id = int(cue["id"])
                if cue_id in evidence_ids:
                    cue["asr_second_opinion"]["english_translation"] = evidence[cue_id]["text"]
                    cue["asr_second_opinion"]["translation_model"] = evidence[cue_id]["model"]
                    cue.pop("translation_qc", None)
            evidence_subset = [cue for cue in cues if int(cue["id"]) in evidence_ids]
            with gpu_stage(folder, "Qwen second-opinion translation QC", checkpoint,
                           minimum_free_mb=6000):
                validate_translations(
                    evidence_subset, folder,
                    lambda value, index: update(37.83 + value * .01,
                        f"Rechecking disputed translation · {index + 1} of {len(evidence_subset)}"),
                    checkpoint,
                )
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
        for correction_round in range(3):
            failed_ids = {int(cue["id"]) for cue in cues
                          if (not bool((cue.get("translation_qc") or {}).get("passed"))
                              and int(cue.get("translation_correction_attempts") or 0) < 3)}
            if not failed_ids:
                break
            log(f"Independent bilingual QC rejected {len(failed_ids)} line(s); "
                f"running automatic correction pass {correction_round + 1} of 3")
            for cue in cues:
                if int(cue["id"]) not in failed_ids:
                    cue["_skip_adaptation"] = True
                    continue
                cue.pop("_skip_adaptation", None)
                cue["translation_qc_feedback"] = str(
                    (cue.get("translation_qc") or {}).get("reason") or "meaning was not independently verified"
                )
                cue["force_translation_correction"] = True
                cue["force_adaptation"] = True
                cue["translation_correction_attempts"] = int(
                    cue.get("translation_correction_attempts") or 0) + 1
                cue.pop("translation_qc", None)
            with gpu_stage(folder, f"Hy-MT2 translation correction {correction_round + 1}/3",
                           checkpoint, minimum_free_mb=6000):
                cues = adapt_dialogue(
                    cues, folder,
                    lambda value, index: update(37.82 + correction_round * .05 + value * .02,
                        f"Correcting rejected translation · line {index + 1} of {len(cues)}"), checkpoint,
                )
            for cue in cues:
                cue.pop("_skip_adaptation", None)
            failed_subset = [cue for cue in cues if int(cue["id"]) in failed_ids]
            with gpu_stage(folder, f"Qwen translation correction QC {correction_round + 1}/3",
                           checkpoint, minimum_free_mb=6000):
                judged_subset = validate_translations(
                    failed_subset, folder,
                    lambda value, index: update(37.86 + correction_round * .05 + value * .02,
                        f"Rechecking corrected translation · {index + 1} of {len(failed_subset)}"), checkpoint,
                )
            judged_by_id = {int(cue["id"]): cue.get("translation_qc") for cue in judged_subset}
            for cue in cues:
                if int(cue["id"]) in judged_by_id:
                    cue["translation_qc"] = judged_by_id[int(cue["id"])]
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
        allocate_dub_windows(cues, duration)
        cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
        emotion_mode = options.get("emotion_mode", "auto")
        if any("emotion_vector" not in cue for cue in cues):
            update(38, "Reading emotion from each scene")
            with gpu_stage(folder, "IndexTTS context emotion analysis", checkpoint):
                results = analyze_context_emotions(
                    [emotion_context(cues, index) for index in range(len(cues))], emotion_mode,
                    options.get("engine") == "preview", folder,
                    lambda value, index: update(38 + value * 3,
                        f"Understanding scene emotion {index + 1} of {len(cues)}"),
                    checkpoint,
                )
            for cue, detected in zip(cues, results):
                cue["emotion"] = detected["label"]
                cue["emotion_vector"] = detected["vector"]
            cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")

        if options.get("workflow_mode") == "approval" and not job.get("analysis_approved"):
            store.update(job_id, status="paused", progress=41, stage="Translation ready for approval",
                         cues=cues, control=None)
            log("Approval workflow paused before the lengthy synthesis stage")
            raise PipelineApproval()

        update(42, "Loading IndexTTS in 8 GB VRAM mode")
        fallback_refs = folder / "references"
        emotion_refs = folder / "emotion-references"
        generated = folder / "generated"
        fitted = folder / "fitted"
        fallback_refs.mkdir(exist_ok=True)
        emotion_refs.mkdir(exist_ok=True)
        generated.mkdir(exist_ok=True)
        fitted.mkdir(exist_ok=True)
        timing_version = "measured-duration-v3"
        timing_marker = folder / "tts-timing-version.txt"
        prior_timing_version = (timing_marker.read_text(encoding="utf-8").strip()
                                if timing_marker.is_file() else "")
        if prior_timing_version != timing_version:
            for stale in (generated, fitted, folder / "qwen-generated", folder / "qwen-fitted",
                          folder / "acoustically-matched"):
                shutil.rmtree(stale, ignore_errors=True)
                stale.mkdir(exist_ok=True)
        timing_marker.write_text(timing_version, encoding="utf-8")
        total = len(cues)
        glossary = options.get("glossary") or {}
        def apply_glossary(text: str) -> str:
            spoken = str(text)
            for term, pronunciation in sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True):
                spoken = __import__('re').sub(
                    rf"\b{__import__('re').escape(term)}\b", pronunciation,
                    spoken, flags=__import__('re').I,
                )
            return spoken

        def persist_cues(current_index: int | None = None, force: bool = False) -> None:
            """Checkpoint cue state in bounded batches instead of rewriting it for every line."""
            should_flush = force or current_index is None or current_index + 1 >= total
            if current_index is not None and (current_index + 1) % 25 == 0:
                should_flush = True
            if should_flush:
                cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
                store.update(job_id, cues=cues)

        tts_items = []
        for index, cue in enumerate(cues):
            checkpoint()
            line_number = index + 1
            # Low-confidence diarization must never clone a neighbouring actor.
            # Preserve the tentative cluster for reporting and later full-film
            # reconciliation, but synthesize from this cue's own source voice.
            reference = (speaker_references.get(int(cue.get("speaker_id", 0)))
                         if float(cue.get("speaker_confidence", 0.0)) >= 0.62 else None)
            if reference is not None:
                # Clone this line from the take of this character that actually
                # sounds like this line, rather than from one clip for the film.
                reference, basis = select_reference(
                    cue, reference_variants.get(int(cue.get("speaker_id", 0)), []), reference)
                cue["reference_basis"] = basis
            else:
                reference = fallback_refs / f"{line_number:06d}.wav"
                cue["reference_basis"] = "this line's own source voice"
            raw_line = generated / f"{line_number:06d}.wav"
            fitted_line = fitted / f"{line_number:06d}.wav"
            emotion_reference = emotion_refs / f"{line_number:06d}.wav"
            target = max(0.24, float(cue.get("dub_end", cue["end"]))
                         - float(cue.get("dub_start", cue["start"])))
            if not reference.exists():
                make_reference(reference_audio, cue, reference, duration)
            reference_text = str(cue.get("source") or "").strip()
            reference_metadata = reference.with_suffix(".json")
            if reference_metadata.is_file():
                try:
                    reference_text = str(json.loads(reference_metadata.read_text(
                        encoding="utf-8")).get("reference_text") or reference_text).strip()
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            source_performance_ok = (
                emotion_mode in {"auto", "source"}
                and float(cue.get("reference_quality", 0.0)) >= 0.30
                and target >= 0.5
            )
            if source_performance_ok and not emotion_reference.exists():
                make_reference(reference_audio, cue, emotion_reference, duration, minimum=0.5, padding=0.15)
            cue["performance_source"] = "source performance" if source_performance_ok else "scene context"
            # Conditioning strength must follow the evidence, not the mode name.
            # The actor's own delivery of this line is the ground truth and is
            # trusted hard; a vector inferred from scene text is a guess and is
            # deliberately not over-committed to.
            emotion_strength = (SOURCE_EMOTION_STRENGTH if source_performance_ok
                                else 0.6 if emotion_mode == "auto" else 0.82)
            cue["emotion_strength"] = emotion_strength
            spoken_text = apply_glossary(cue["english"])
            cue["spoken_text"] = spoken_text
            tts_items.append({
                "text": spoken_text, "reference": str(reference), "raw": str(raw_line),
                "fitted": str(fitted_line), "target": target,
                "emotion_vector": None if source_performance_ok else cue.get("emotion_vector"),
                "emotion_strength": emotion_strength,
                "emotion_audio": str(emotion_reference) if source_performance_ok else None,
                "language": target_language_code(options.get("target_language", "English")),
                "cue_index": index, "reference_text": reference_text,
                "fit_limit_percent": 5.0 if cue.get("mouth_visible") else 8.0,
            })

        def voice_progress(event: dict) -> None:
            value = float(event["progress"])
            completed_index = int(event.get("cue_index", event["index"]))
            for cue_index in range(completed_index + 1):
                cues[cue_index]["status"] = "voiced"
                cues[cue_index]["audio"] = f"{cue_index + 1:06d}.wav"
            cues[completed_index]["qc"] = {
                "tts_attempts": int(event.get("attempts", 1)),
                "raw_duration": event.get("raw_duration"),
                "stretch_percent": event.get("stretch_percent"),
                "active_duration": event.get("active_duration"),
                "speech_span_duration": event.get("speech_span_duration"),
                "active_fill_percent": event.get("active_fill_percent"),
                "padding_ms": event.get("padding_ms"),
                "truncated_ms": event.get("truncated_ms"),
                "duration_error_percent": event.get("duration_error_percent"),
                "duration_factor": event.get("duration_factor"),
                "timing_pass": event.get("timing_pass"),
                "phrase_count": event.get("phrase_count"),
            }
            persist_cues(completed_index)
            update(44 + value * 36, f"Voicing line {completed_index + 1} of {total}",
                   current_cue=completed_index + 1)

        with gpu_stage(folder, "IndexTTS primary synthesis", checkpoint):
            synthesize_voice_lines({"engine": options.get("engine", "indextts"), "items": tts_items},
                                   folder, voice_progress, checkpoint)
        persist_cues(force=True)

        timing_failures = [
            index for index, cue in enumerate(cues)
            if (cue.get("qc", {}).get("timing_pass") is not True
                or float(cue.get("qc", {}).get("padding_ms") or 0.0) > 160
                or float(cue.get("qc", {}).get("truncated_ms") or 0.0) > 40)
        ]
        if timing_failures:
            update(80.2, f"Rewriting {len(timing_failures)} line(s) that failed timing QC")
            failure_set = set(timing_failures)
            for index, cue in enumerate(cues):
                cue["force_adaptation"] = index in failure_set
                cue["_skip_adaptation"] = index not in failure_set
            cues = adapt_dialogue(
                cues, folder,
                lambda value, index: update(80.2 + value * 0.3,
                    f"Rewriting timing outlier {index + 1}"), checkpoint,
            )
            for cue in cues:
                cue.pop("_skip_adaptation", None)
            retry_items = []
            rewritten: list[int] = []
            for index in timing_failures:
                raw = generated / f"{index + 1:06d}.wav"
                fitted_line = fitted / f"{index + 1:06d}.wav"
                raw.unlink(missing_ok=True)
                fitted_line.unlink(missing_ok=True)
                previous_text = cues[index].get("spoken_text")
                cues[index]["spoken_text"] = apply_glossary(cues[index]["english"])
                target = max(0.24, float(cues[index].get("dub_end", cues[index]["end"]))
                             - float(cues[index].get("dub_start", cues[index]["start"])))
                tts_items[index]["text"] = cues[index]["spoken_text"]
                tts_items[index]["target"] = target
                item = dict(tts_items[index])
                if cues[index]["spoken_text"] != previous_text:
                    # The wording changed under timing pressure, which is exactly
                    # when meaning gets dropped. The verdict this line earned
                    # belongs to the old wording and must not survive the rewrite.
                    cues[index].pop("translation_qc", None)
                    if cues[index].get("performance_source") != "source performance":
                        cues[index].pop("emotion_vector", None)
                        tts_items[index]["emotion_vector"] = None
                        item["emotion_vector"] = None
                    rewritten.append(index)
                retry_items.append(item)

            def retry_progress(event: dict) -> None:
                cue_index = int(event.get("cue_index", event["index"]))
                cues[cue_index]["qc"] = {
                    "tts_attempts": int(event.get("attempts", 1)) + 2,
                    "raw_duration": event.get("raw_duration"),
                    "stretch_percent": event.get("stretch_percent"),
                    "active_duration": event.get("active_duration"),
                    "speech_span_duration": event.get("speech_span_duration"),
                    "active_fill_percent": event.get("active_fill_percent"),
                    "padding_ms": event.get("padding_ms"),
                    "truncated_ms": event.get("truncated_ms"),
                    "duration_error_percent": event.get("duration_error_percent"),
                    "duration_factor": event.get("duration_factor"),
                    "timing_pass": event.get("timing_pass"),
                    "phrase_count": event.get("phrase_count"),
                    "adapted_retry": True,
                }
                update(80.5 + float(event["progress"]) * 0.5,
                       f"Resynthesizing corrected line {cue_index + 1} of {total}",
                       current_cue=cue_index + 1)

            with gpu_stage(folder, "IndexTTS timing retries", checkpoint):
                synthesize_voice_lines({"engine": options.get("engine", "indextts"), "items": retry_items},
                                       folder, retry_progress, checkpoint)
            persist_cues(force=True)
            if rewritten:
                update(80.85, f"Rechecking {len(rewritten)} line(s) rewritten for timing")
                rewritten_subset = [cues[index] for index in rewritten]
                with gpu_stage(folder, "Qwen QC for timing rewrites", checkpoint,
                               minimum_free_mb=6000):
                    validate_translations(
                        rewritten_subset, folder,
                        lambda value, index: update(80.85 + value * .05,
                            f"Rechecking rewritten meaning · {index + 1} of {len(rewritten_subset)}"),
                        checkpoint,
                    )
                persist_cues(force=True)

        update(80.9, "Checking generated words against the intended dialogue")
        with gpu_stage(folder, "Whisper spoken-word QC", checkpoint):
            backtranscribe_lines(
                cues, fitted, folder,
                lambda value, index: update(80.9 + value * 0.1,
                    f"Verifying spoken words · line {index + 1} of {total}"), checkpoint,
            )
        word_failures = [
            index for index, cue in enumerate(cues)
            if float(cue.get("qc", {}).get("word_similarity") or 0.0) < 0.58
        ]
        if word_failures:
            update(81, f"Regenerating {len(word_failures)} line(s) that failed word QC")
            word_retry_items = []
            for index in word_failures:
                (generated / f"{index + 1:06d}.wav").unlink(missing_ok=True)
                (fitted / f"{index + 1:06d}.wav").unlink(missing_ok=True)
                item = dict(tts_items[index])
                item["text"] = cues[index].get("spoken_text") or apply_glossary(cues[index]["english"])
                item["use_random"] = False
                word_retry_items.append(item)

            def word_retry_progress(event: dict) -> None:
                cue_index = int(event.get("cue_index", event["index"]))
                cues[cue_index].setdefault("qc", {}).update({
                    "tts_attempts": int(cues[cue_index].get("qc", {}).get("tts_attempts", 1))
                                    + int(event.get("attempts", 1)),
                    "raw_duration": event.get("raw_duration"),
                    "stretch_percent": event.get("stretch_percent"),
                    "active_duration": event.get("active_duration"),
                    "speech_span_duration": event.get("speech_span_duration"),
                    "active_fill_percent": event.get("active_fill_percent"),
                    "padding_ms": event.get("padding_ms"),
                    "truncated_ms": event.get("truncated_ms"),
                    "duration_error_percent": event.get("duration_error_percent"),
                    "duration_factor": event.get("duration_factor"),
                    "timing_pass": event.get("timing_pass"),
                    "phrase_count": event.get("phrase_count"),
                    "word_retry": True,
                })
                update(81 + float(event["progress"]) * 0.3,
                       f"Regenerating word mismatch · line {cue_index + 1}",
                       current_cue=cue_index + 1)

            with gpu_stage(folder, "IndexTTS intelligibility retries", checkpoint):
                synthesize_voice_lines({"engine": options.get("engine", "indextts"), "items": word_retry_items},
                                       folder, word_retry_progress, checkpoint)
            persist_cues(force=True)
            with gpu_stage(folder, "Whisper retry QC", checkpoint):
                backtranscribe_lines(
                    cues, fitted, folder,
                    lambda value, index: update(81.3 + value * 0.2,
                        f"Rechecking spoken words · line {index + 1} of {total}"), checkpoint,
                )
        # Measure identity before deciding whether the secondary engine is useful;
        # fallback is not limited to pronunciation failures.
        score_speaker_similarity(cues, fitted, speaker_references)
        measure_performance_similarity(cues, fitted)
        # Acting and register failures are answered by asking the *cloning*
        # engine again with the actor's own delivery weighted harder, before any
        # second engine is considered.  A different engine carries no source
        # performance prompt at all, so reaching for it first would trade the
        # thing being repaired for the thing that was already right.
        performance_failures = [
            index for index, cue in enumerate(cues)
            if float(cue["end"]) - float(cue["start"]) >= 1.0
            and str(cue.get("performance_source")) == "source performance"
            and (float(cue.get("qc", {}).get("performance_similarity", 1.0)) < PERFORMANCE_RETAKE_THRESHOLD
                 or float(cue.get("qc", {}).get("pitch_delta_semitones") or 0.0) > PITCH_TOLERANCE_SEMITONES)
        ]
        if performance_failures:
            update(81.4, f"Re-performing {len(performance_failures)} line(s) to match the original delivery")
            retake_dir = folder / "performance-retakes"; retake_raw = folder / "performance-retake-raw"
            retake_dir.mkdir(exist_ok=True); retake_raw.mkdir(exist_ok=True)
            retake_items = []
            for index in performance_failures:
                item = dict(tts_items[index])
                item.update({
                    "raw": str(retake_raw / f"{index + 1:06d}.wav"),
                    "fitted": str(retake_dir / f"{index + 1:06d}.wav"),
                    "emotion_strength": RETAKE_EMOTION_STRENGTH,
                    # Sampling variance fights identity; a retake for accuracy is
                    # never a random reroll.
                    "use_random": False, "cue_index": index,
                })
                retake_items.append(item)
            retake_metrics: dict[int, dict] = {}

            def retake_progress(event: dict) -> None:
                cue_index = int(event.get("cue_index", event["index"]))
                retake_metrics[cue_index] = event
                update(81.4 + float(event["progress"]) * .1,
                       f"Re-performing line {cue_index + 1}", current_cue=cue_index + 1)

            with gpu_stage(folder, "IndexTTS performance retakes", checkpoint):
                synthesize_voice_lines(
                    {"engine": options.get("engine", "indextts"), "items": retake_items},
                    folder, retake_progress, checkpoint,
                )
            retake_cues = copy.deepcopy(cues)
            for index, metrics in retake_metrics.items():
                retake_cues[index].setdefault("qc", {}).update(metrics)
            mirror_current_takes(fitted, retake_dir, total)
            retake_qc = folder / "performance-retake-qc"; retake_qc.mkdir(exist_ok=True)
            with gpu_stage(folder, "Whisper performance-retake QC", checkpoint):
                backtranscribe_lines(retake_cues, retake_dir, retake_qc,
                    lambda value, index: update(81.5 + value * .05,
                        f"Checking re-performed line {index + 1}"), checkpoint)
            score_speaker_similarity(retake_cues, retake_dir, speaker_references)
            measure_performance_similarity(retake_cues, retake_dir)
            kept = 0
            for index in performance_failures:
                current = cues[index].get("qc", {}); retake = retake_cues[index].get("qc", {})
                # Expression is never bought with intelligibility or identity: a retake must
                # not drop spoken accuracy or discard the cloned actor timbre.
                current_word = float(current.get("word_similarity") or 0.0)
                retake_word = float(retake.get("word_similarity") or 0.0)
                intelligible = retake_word >= max(0.72, current_word - 0.04)
                retains_identity = (float(retake.get("speaker_similarity", .5))
                                    >= float(current.get("speaker_similarity", .5)) - 0.03)
                current_score = (.35 * float(current.get("performance_similarity", .5))
                                 + .35 * float(current.get("speaker_similarity", .5))
                                 + .30 * current_word)
                retake_score = (.35 * float(retake.get("performance_similarity", .5))
                                + .35 * float(retake.get("speaker_similarity", .5))
                                + .30 * retake_word)
                if intelligible and retains_identity and retake_score > current_score + .03:
                    shutil.copy2(retake_dir / f"{index + 1:06d}.wav", fitted / f"{index + 1:06d}.wav")
                    cues[index]["qc"] = retake
                    cues[index]["qc"]["performance_retake"] = True
                    cues[index]["qc"]["performance_retake_margin"] = round(retake_score - current_score, 3)
                    kept += 1
            log(f"Performance retakes improved {kept} of {len(performance_failures)} line(s)")
            persist_cues(force=True)
        fallback_failures = [
            index for index, cue in enumerate(cues)
            if ((float(cue.get("qc", {}).get("word_similarity") or 0.0) < .8
                 and int(cue.get("qc", {}).get("tts_attempts", 1)) >= 2)
                or (float(cue.get("qc", {}).get("speaker_similarity") or 0.0) < .55
                    and float(cue.get("qc", {}).get("active_duration", 0.0)) >= .8)
                or (float(cue.get("qc", {}).get("performance_similarity", 1.0)) < .45
                    and float(cue["end"]) - float(cue["start"]) >= 1.0))
        ]
        if fallback_failures:
            update(81.45, f"Comparing a second speech engine for {len(fallback_failures)} difficult line(s)")
            qwen_dir = folder / "qwen-fitted"; qwen_raw = folder / "qwen-generated"
            qwen_dir.mkdir(exist_ok=True); qwen_raw.mkdir(exist_ok=True)
            mirror_current_takes(fitted, qwen_dir, total)
            qwen_items = []
            qwen_metrics: dict[int, dict] = {}
            for index in fallback_failures:
                item = dict(tts_items[index])
                item["text"] = cues[index].get("spoken_text") or apply_glossary(cues[index]["english"])
                item.update({"raw": str(qwen_raw / f"{index + 1:06d}.wav"),
                             "fitted": str(qwen_dir / f"{index + 1:06d}.wav"),
                             "language": "English", "cue_index": index})
                qwen_items.append(item)

            def qwen_progress(event: dict) -> None:
                cue_index = int(event["cue_index"]); qwen_metrics[cue_index] = event
                update(81.45 + float(event["progress"]) * .15,
                       f"Generating alternative engine take · line {cue_index + 1}")

            try:
                with gpu_stage(folder, "Qwen3-TTS fallback synthesis", checkpoint):
                    qwen_available = synthesize_qwen_fallback(qwen_items, folder, qwen_progress, checkpoint)
            except Exception as exc:
                qwen_available = False
                log(f"Alternative Qwen3-TTS take generation was skipped after a local worker error: {exc}")
            if qwen_available:
                qwen_cues = copy.deepcopy(cues)
                for index, metrics in qwen_metrics.items():
                    qwen_cues[index].setdefault("qc", {}).update(metrics)
                qwen_qc = folder / "qwen-qc"; qwen_qc.mkdir(exist_ok=True)
                with gpu_stage(folder, "Whisper alternate-take QC", checkpoint):
                    backtranscribe_lines(qwen_cues, qwen_dir, qwen_qc,
                        lambda value, index: update(81.6 + value * .08,
                            f"Checking alternative take · line {index + 1} of {total}"), checkpoint)
                score_speaker_similarity(cues, fitted, speaker_references)
                score_speaker_similarity(qwen_cues, qwen_dir, speaker_references)
                measure_performance_similarity(cues, fitted)
                measure_performance_similarity(qwen_cues, qwen_dir)
                for index in fallback_failures:
                    current = cues[index].get("qc", {}); alternate = qwen_cues[index].get("qc", {})
                    current_score = (.4 * float(current.get("word_similarity", 0))
                                     + .25 * float(current.get("speaker_similarity", .5))
                                     + .25 * float(current.get("performance_similarity", .5))
                                     + .1 * float(current.get("timing_pass", False)))
                    alternate_score = (.4 * float(alternate.get("word_similarity", 0))
                                       + .25 * float(alternate.get("speaker_similarity", .5))
                                       + .25 * float(alternate.get("performance_similarity", .5))
                                       + .1 * float(alternate.get("timing_pass", False)))
                    # The fallback engine clones timbre but receives no source
                    # performance prompt, so it must not win a line by fluency
                    # while quietly discarding the actor's delivery or register.
                    keeps_performance = (
                        float(alternate.get("performance_similarity", .5))
                        >= float(current.get("performance_similarity", .5)) - .02)
                    keeps_register = (
                        float(alternate.get("pitch_delta_semitones") or 0.0)
                        <= max(PITCH_TOLERANCE_SEMITONES,
                               float(current.get("pitch_delta_semitones") or 0.0)))
                    if (alternate_score > current_score + .025
                            and keeps_performance and keeps_register):
                        shutil.copy2(qwen_dir / f"{index + 1:06d}.wav", fitted / f"{index + 1:06d}.wav")
                        cues[index]["qc"] = alternate
                        cues[index]["qc"]["tts_engine"] = "Qwen3-TTS 1.7B Base fallback"
                        cues[index]["qc"]["engine_selection_margin"] = round(alternate_score - current_score, 3)
        update(81.5, "Checking voice identity against each character")
        score_speaker_similarity(cues, fitted, speaker_references)
        checkpoint()
        matched = folder / "acoustically-matched"
        update(82, "Matching dialogue tone, distance, and room sound")
        match_acoustics(cues, fitted, matched, reference_audio)
        # Re-score speaker similarity on the delivered matched audio so QC reflects real output
        score_speaker_similarity(cues, matched, speaker_references)
        measure_performance_similarity(cues, matched)
        clean_background = background_stem
        if audio_mode == "separate" and background_stem.is_file():
            update(82.4, "Removing residual source dialogue from the music-and-effects bed")
            clean_background = suppress_dialogue_leakage(
                cues, reference_audio, background_stem, folder / "background-dialogue-suppressed.flac"
            ) or background_stem
        for cue in cues:
            cue["overlap_source_preserved"] = False
        update(82.6, "Running automatic line quality checks")
        cue_qc = inspect_cues(cues, matched)
        cues_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
        voice = folder / "english-dialogue.flac"
        update(83, "Placing every line on the original timeline")
        render_timeline(cues, matched, voice, duration,
                        reference_audio if audio_mode == "separate" else None)
        checkpoint()

        mixed = folder / "english-mix.flac"
        update(89, "Balancing dialogue with the source soundtrack")
        mastering = mix_audio(working_soundtrack, voice, mixed, duration, audio_mode, clean_background,
                              options.get("mastering_preset", "cinema"), cues)

        output = folder / "dubbed-english.mkv"
        try:
            if not video_stream:
                video_override, lipsync_summary = None, {"enabled": False, "selected": 0, "completed": 0,
                                                          "reason": "audio-only source"}
            else:
                with gpu_stage(folder, "MuseTalk selective lip finishing", checkpoint):
                    video_override, lipsync_summary = apply_selective_lipsync(
                        source, cues, matched, folder,
                        lambda value, index: update(89 + value * 5,
                            f"Finishing visible lip sync · line {index + 1} of {total}"), checkpoint,
                    )
        except Exception as exc:
            video_override = None
            lipsync_summary = {"enabled": True, "selected": 0, "completed": 0,
                               "error": str(exc)}
            log(f"Optional visual finishing was skipped after a local worker error: {exc}")
        if lipsync_summary.get("enabled"):
            log(f"MuseTalk finishing selected {lipsync_summary['selected']} clean visible shot(s); "
                f"completed {lipsync_summary['completed']}")
        update(95, "Rejoining the full-length video")
        # The dialogue soundtrack becomes track 2, ahead of any commentary or
        # description track that happened to sit earlier in the container.
        original_order = [working_audio["index"]] + [
            stream["index"] for stream in available_audio
            if stream["index"] != working_audio["index"]
        ] if working_audio else []
        remux(source, mixed, output, video_override, original_order)
        update(98, "Verifying the finished film and writing its QC report")
        final_qc = media_qc(
            output, duration, source, checkpoint,
        )
        final_qc["mastering"] = mastering
        final_qc["visual_lipsync"] = lipsync_summary
        final_qc["failures"] = evaluate_media_qc(final_qc, mastering)
        final_qc["passed"] = not final_qc["failures"]
        update(98.5, "Writing captions, cue data, edit list, and speaker stems")
        exports = write_exports(folder, cues, matched, duration)
        report_json, report_html = write_report(
            folder, job_id, cues, cue_qc, final_qc, (store.get(job_id) or {}).get("separation", {}),
        )
        for cue in cues:
            cue["status"] = "needs_review" if cue.get("needs_review") else "complete"
        if final_qc["failures"]:
            stage = f"English dub produced · delivery QC failed ({len(final_qc['failures'])} issue(s))"
        elif cue_qc["flagged_count"]:
            stage = f"English dub ready · {cue_qc['flagged_count']} line(s) need review"
        else:
            stage = "English dub ready · QC passed"
        final_status = "needs_review" if final_qc["failures"] or cue_qc["flagged_count"] else "complete"
        elapsed = active_seconds()
        throughput = {"wall_seconds": round(elapsed, 1), "media_seconds": round(duration, 3),
                      "realtime_factor": round(elapsed / max(duration, .001), 2)}
        store.update(job_id, status=final_status, progress=100, stage=stage, cues=cues,
                     output_path=str(output), output_size=output.stat().st_size, control=None,
                     qc={**cue_qc, **final_qc}, qc_report_json=str(report_json), qc_report_html=str(report_html),
                     throughput=throughput, active_processing_seconds=round(elapsed, 1),
                     active_run_started_at=None)
        store.update(job_id, exports=exports)
        log(f"Finished the dubbed Matroska video; QC flagged {cue_qc['flagged_count']} line(s) "
            f"and {len(final_qc['failures'])} delivery issue(s)")
    except PipelineApproval:
        store.update(job_id, active_processing_seconds=round(active_seconds(), 1), active_run_started_at=None)
    except PipelinePause:
        store.update(job_id, status="paused", stage="Paused safely", control=None,
                     active_processing_seconds=round(active_seconds(), 1), active_run_started_at=None)
        log("Paused at a safe checkpoint; completed lines were kept")
    except PipelineCancel:
        store.update(job_id, status="cancelled", stage="Cancelled", control=None,
                     active_processing_seconds=round(active_seconds(), 1), active_run_started_at=None)
        log("Job cancelled; intermediate files were kept until removal")
    except GPUStageUnsafe as exc:
        attempts = int((store.get(job_id) or {}).get("thermal_resume_count") or 0) + 1
        try:
            limit = max(0, int(os.getenv("DUB_GPU_AUTO_RESUME_LIMIT", "12")))
        except ValueError:
            limit = 12
        automatic = attempts <= limit
        stage = ("Cooling GPU safely · will resume automatically" if automatic else
                 "Paused to protect the GPU · automatic retry limit reached")
        store.update(job_id, status="paused", stage=stage, error=str(exc), control=None,
                     auto_resume_pending=automatic, thermal_resume_count=attempts,
                     active_processing_seconds=round(active_seconds(), 1), active_run_started_at=None)
        log(f"CUDA safety pause: {exc}. Completed analysis and lines were kept; "
            + ("the job will resume after a stable cooldown." if automatic else
               "manual review is required after repeated thermal stops."))
    except Exception as exc:
        store.update(job_id, status="error", stage="Needs attention", error=f"{type(exc).__name__}: {exc}", control=None,
                     active_processing_seconds=round(active_seconds(), 1), active_run_started_at=None)
        log(f"Pipeline stopped: {type(exc).__name__}: {exc}")
    finally:
        _run_context.checkpoint = None
        release_job_lock(execution_lock)


def build_cues(source: Path, audio: Path, folder: Path, job: dict, probe: dict, options: dict,
               checkpoint: Callable, update: Callable, log: Callable) -> tuple[list[dict], str]:
    mode = options.get("subtitle_mode", "auto")
    sidecars = [Path(path) for path in job.get("sidecars", [])]
    sidecars += [Path(item["path"]) for item in job.get("uploads", []) if item.get("kind") == "subtitle"]
    sidecars = [path for path in sidecars if path.is_file() and path.suffix.lower() != ".idx"]
    embedded = choose_embedded(subtitle_streams(probe), options.get("subtitle_stream_index"))
    selected: Path | None = None
    language = None
    source_label = ""
    if mode in {"auto", "sidecar"} and sidecars:
        def sidecar_rank(path: Path) -> tuple:
            label = path.stem.lower()
            unwanted = any(marker in label for marker in
                           ("forced", "signs", "songs", "commentary", "director", "trivia"))
            full = any(marker in label for marker in ("full", "complete", "dialogue"))
            english = any(__import__('re').search(pattern, label) for pattern in
                          (r"(^|[._ -])en(g)?($|[._ -])", r"english"))
            text_priority = {".srt": 0, ".vtt": 1, ".ass": 2, ".ssa": 3, ".sub": 4}.get(path.suffix.lower(), 8)
            return (unwanted, not full, not english, text_priority, path.name.lower())
        for candidate in sorted(sidecars, key=sidecar_rank):
            try:
                normalized = normalize_sidecar(candidate, folder / "selected-subtitles.srt", video_fps(probe))
                if parse_srt(normalized):
                    selected = normalized
                    source_label = f"Sidecar subtitles · {candidate.name}"
                    break
                log(f"{candidate.name} contained no readable dialogue cards; trying the next subtitle source")
            except subprocess.CalledProcessError:
                log(f"Could not read {candidate.name}; trying the next subtitle source")
    if selected is None and mode in {"auto", "embedded"} and embedded:
        try:
            selected = extract_embedded(source, embedded, folder / "selected-subtitles.srt")
            language = embedded.get("language")
            source_label = f"Embedded subtitles · {embedded.get('title')} ({language})"
        except subprocess.CalledProcessError:
            log("The embedded subtitle track is image-based or unreadable; using speech recognition")
    parsed = parse_srt(selected) if selected else []
    if parsed and source_label.startswith("Sidecar"):
        range_start = float(options.get("range_start") or 0.0)
        range_end = options.get("range_end")
        if range_start > 0 or range_end is not None:
            finish = float(range_end) if range_end is not None else float("inf")
            shifted = []
            for cue in parsed:
                start = max(float(cue["start"]), range_start)
                end = min(float(cue["end"]), finish)
                if end > start:
                    shifted.append({**cue, "start": round(start - range_start, 3),
                                    "end": round(end - range_start, 3)})
            parsed = shifted
            source_label += " · trimmed to selected section"
    (folder / "source-language.txt").write_text(str(options.get("source_language", "auto")), encoding="utf-8")
    checkpoint()
    update(28, "Building a source-language dialogue map with word timing")
    cues = transcribe_long_audio(audio, folder, options.get("whisper_model", "turbo"), checkpoint, update,
                                 subtitle_hints=parsed)
    if parsed and looks_english(parsed, language):
        reconciled = reconcile_subtitles_with_asr(parsed, cues)
        return reconciled, source_label + " · word-aligned to source speech"
    if parsed:
        for cue in cues:
            matching = [card for card in parsed if
                        min(float(cue["end"]), float(card["end"])) -
                        max(float(cue["start"]), float(card["start"])) > .05]
            if matching:
                evidence = " ".join(str(card["text"]) for card in matching)
                cue["subtitle_source_evidence"] = evidence
                cue["source"] = evidence
        log("The non-English subtitle wording is attached as translation evidence and ASR supplies word timing")
    engine = cues[0].get("asr_engine", "Whisper Turbo fallback") if cues else "local ASR"
    return cues, f"{engine} source transcript" + (f" · guided by {source_label}" if source_label else "")


def prepare_translation_revision(folder: Path, cues: list[dict]) -> bool:
    """Invalidate only translation-dependent work when semantics are upgraded."""
    revision = "subtitle-provenance-source-faithful-v4"
    marker = folder / "translation-logic-version.txt"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == revision:
        return False
    cards = parse_srt(folder / "selected-subtitles.srt") if (folder / "selected-subtitles.srt").is_file() else []
    for cue in cues:
        cue_start = float(cue.get("subtitle_start", cue.get("start", 0.0)))
        cue_end = float(cue.get("subtitle_end", cue.get("end", cue_start)))
        candidates = []
        if cue.get("subtitle_start") is not None:
            for card in cards:
                overlap = min(cue_end, float(card["end"])) - max(cue_start, float(card["start"]))
                if overlap > 0:
                    candidates.append((overlap, str(card.get("text", ""))))
        supplied = max(candidates)[1] if candidates else (
            cue.get("supplied_translation") if cue.get("subtitle_start") is not None else None
        )
        if (supplied and len(cue.get("words", [])) <= 2
                and __import__('re').search(
                    r"(?:\d|[$£€¥]|\byen\b|\btickets?\b|\badmission\b|\s[-–—]\s)",
                    str(supplied), flags=__import__('re').I)):
            cue["display_caption_ignored_for_dubbing"] = supplied
            supplied = None
        if supplied:
            cue["supplied_translation"] = supplied
            cue["translation_was_supplied"] = True
            cue["literal_translation"] = supplied
            cue["english"] = supplied
            cue["adapted_dialogue"] = supplied
        else:
            cue.pop("supplied_translation", None)
            cue["translation_was_supplied"] = False
            cue["translation_is_target"] = False
            cue["literal_translation"] = str(cue.get("source") or "")
            cue["english"] = str(cue.get("source") or "")
            cue["adapted_dialogue"] = str(cue.get("source") or "")
        for key in ("faithful_translation", "translation_model", "translation_candidates",
                    "candidate_semantic_scores", "adaptation_confidence", "adaptation_attempts",
                    "adaptation_model", "adaptation_scoring", "translation_qc",
                    "translation_qc_feedback", "force_translation_correction",
                    "asr_second_opinion"):
            cue.pop(key, None)
        cue.pop("translation_correction_attempts", None)
        cue["force_adaptation"] = False
    for directory in ("generated", "fitted", "qwen-generated", "qwen-fitted", "acoustically-matched"):
        shutil.rmtree(folder / directory, ignore_errors=True)
    for name in ("english-dialogue.flac", "clean-background.flac", "dubbed-mix.flac",
                 "dubbed-english.mkv", "qc-report.json", "qc-report.html"):
        (folder / name).unlink(missing_ok=True)
    marker.write_text(revision, encoding="utf-8")
    return True


def reconcile_subtitles_with_asr(subtitles: list[dict], asr: list[dict]) -> list[dict]:
    """Treat subtitle cards as translation evidence and ASR/VAD as speech timing evidence."""
    words_by_card: dict[int, list[dict]] = {index: [] for index in range(len(subtitles))}
    for asr_cue in asr:
        for word in asr_cue.get("words", []):
            midpoint = (float(word["start"]) + float(word["end"])) / 2
            distances = []
            for index, subtitle in enumerate(subtitles):
                start, end = float(subtitle["start"]), float(subtitle["end"])
                distance = 0.0 if start <= midpoint <= end else min(abs(midpoint - start), abs(midpoint - end))
                distances.append((distance, index))
            distance, index = min(distances) if distances else (999, -1)
            if distance <= 0.55:
                words_by_card[index].append(word)
    result = []
    spoken_subtitle_indices: set[int] = set()
    for subtitle_index, subtitle in enumerate(subtitles):
        start, end = float(subtitle["start"]), float(subtitle["end"])
        matching = [cue for cue in asr if min(end, float(cue["end"])) - max(start, float(cue["start"])) > 0.04]
        text = subtitle["text"]
        if matching:
            confidence = sum(float(cue.get("transcription_confidence", 0.0)) for cue in matching) / len(matching)
            alignment_confidence = sum(float(cue.get("alignment_confidence", 0.0))
                                       for cue in matching) / len(matching)
            words = sorted(words_by_card[subtitle_index], key=lambda word: float(word["start"]))
            if words:
                speech_start = max(start - 0.5, min(float(word["start"]) for word in words))
                speech_end = min(end + 0.5, max(float(word["end"]) for word in words))
                source_text = "".join(str(word.get("word", "")) for word in words).strip()
            else:
                speech_start, speech_end = start, end
                source_text = " ".join(str(cue.get("source", "")) for cue in matching).strip()
            timing_confidence = min(1.0, 0.55 + confidence * 0.45)
        else:
            speech_start, speech_end, source_text, words = start, end, "", []
            timing_confidence, alignment_confidence = 0.25, 0.0
        # Full subtitle tracks often contain title cards and character-name
        # captions.  When local ASR finds no speech or aligned words, concise
        # display-style text must remain a caption rather than becoming a TTS
        # line (e.g. "WATERBOYS" or "Suzuki").
        plain_words = __import__('re').findall(r"[A-Za-zÀ-ÿ0-9']+", text)
        sign_like = bool(__import__('re').search(
            r"(?:\d|[$£€¥]|\byen\b|\btickets?\b|\badmission\b|\s[-–—]\s)", text,
            flags=__import__('re').I,
        ))
        display_caption = (
            not matching and not words and 1 <= len(plain_words) <= 6
            and not __import__('re').search(r"[.!?。！？]", text)
            and (text.upper() == text or len(plain_words) == 1)
        ) or (sign_like and len(words) <= 2 and len(plain_words) <= 8
              and not __import__('re').search(r"[.!?。！？]", text))
        if display_caption:
            continue
        spoken_subtitle_indices.add(subtitle_index)
        result.append({
            "start": round(speech_start, 3), "end": round(speech_end, 3),
            "subtitle_start": round(start, 3), "subtitle_end": round(end, 3),
            "source": source_text or text, "literal_translation": text,
            # The distributor's own translation, recorded where nothing
            # downstream can overwrite it. Every later stage treats this as
            # independent human evidence, so it must never be back-filled from
            # a machine translation of the same line.
            "supplied_translation": text,
            "english": text, "adapted_dialogue": text, "words": words,
            "translation_is_target": True,
            "source_language": matching[0].get("source_language") if matching else None,
            "timing_confidence": round(timing_confidence, 3),
            "alignment_confidence": round(alignment_confidence, 3),
            "confidence_source": "forced-alignment validity reconciled with subtitle timing",
            "transcription_confidence": round(confidence if matching else 0.0, 3),
            "subtitle_speaker_hint": subtitle.get("subtitle_speaker_hint"),
            "card_speaker_index": subtitle.get("card_speaker_index", 0),
            "simultaneous_card": bool(subtitle.get("simultaneous_card")),
            "nonverbal_annotations": subtitle.get("nonverbal_annotations", []),
        })
    # Forced/signs-only English tracks must not suppress ordinary dialogue.
    for cue in asr:
        overlap = max((min(float(cue["end"]), float(card["end"])) -
                       max(float(cue["start"]), float(card["start"]))
                       for index, card in enumerate(subtitles)
                       if index in spoken_subtitle_indices), default=0.0)
        if overlap <= .05:
            extra = dict(cue)
            extra.setdefault("english", extra.get("source", ""))
            extra.setdefault("adapted_dialogue", extra.get("english", ""))
            extra["subtitle_gap_recovered_by_asr"] = True
            result.append(extra)
    result.sort(key=lambda item: (float(item["start"]), int(item.get("card_speaker_index", 0))))
    return result


def hydrate_alignment_confidence(cues: list[dict]) -> int:
    """Migrate cached reconciled cues using their retained aligner evidence.

    Qwen's aligned word objects already contain the validity-derived probability;
    older reconciliation dropped only the cue-level aggregate. Reconstructing the
    mean is measured evidence, not a favourable default.
    """
    restored = 0
    for cue in cues:
        if cue.get("alignment_confidence") is not None:
            continue
        probabilities = []
        for word in cue.get("words", []):
            try:
                value = float(word["probability"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0.0 <= value <= 1.0:
                probabilities.append(value)
        if probabilities:
            cue["alignment_confidence"] = round(sum(probabilities) / len(probabilities), 3)
            cue["alignment_evidence"] = "mean retained Qwen forced-aligner word validity"
            restored += 1
    return restored


def allocate_dub_windows(cues: list[dict], media_duration: float) -> None:
    """Use measured speech plus subtitle/scene gaps to give English a safe window."""
    ordered = sorted(cues, key=lambda item: (float(item["start"]), int(item.get("card_speaker_index", 0))))
    for index, cue in enumerate(ordered):
        source_start, source_end = float(cue["start"]), float(cue["end"])
        next_start = min((float(other["start"]) for other in ordered[index + 1:]
                          if float(other["start"]) > source_start + .02), default=media_duration)
        subtitle_end = float(cue.get("subtitle_end") or source_end)
        mouth_visible = bool(cue.get("mouth_visible"))
        extension_cap = .25 if mouth_visible else 1.2
        candidate_end = max(source_end, min(subtitle_end, source_end + extension_cap))
        # Short acknowledgements need a producible window. They may overlap a
        # different active speaker, but never steal time from the next turn.
        minimum_end = source_start + (.45 if mouth_visible else .62)
        if source_end - source_start < .55:
            candidate_end = max(candidate_end, minimum_end)
        if not cue.get("overlapping_speech"):
            candidate_end = min(candidate_end, max(source_end, next_start - .03))
        cue["dub_start"] = round(source_start, 3)
        cue["dub_end"] = round(min(media_duration, max(source_end, candidate_end)), 3)
        cue["dub_timing_basis"] = (
            "forced-aligned source + subtitle/scene gap" if cue["dub_end"] > source_end + .01
            else "forced-aligned source"
        )


def video_fps(probe: dict) -> float:
    stream = next((item for item in probe.get("streams", []) if item.get("codec_type") == "video"), {})
    raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "24000/1001"
    try:
        top, bottom = raw.split("/")
        return float(top) / float(bottom)
    except (ValueError, ZeroDivisionError):
        return 23.976


def whisper_model(name: str):
    with _whisper_lock:
        if name not in _whisper_models:
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError("OpenAI Whisper is not installed") from exc
            download_root = Path(os.getenv("WHISPER_CACHE_DIR", "vendor/whisper")).resolve()
            download_root.mkdir(parents=True, exist_ok=True)
            _whisper_models[name] = whisper.load_model(name, download_root=str(download_root))
        return _whisper_models[name]


def release_whisper_models() -> None:
    with _whisper_lock:
        for model in _whisper_models.values():
            try:
                model.cpu()
            except Exception:
                pass
        _whisper_models.clear()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def translation_dispute_candidates(cues: list[dict]) -> list[dict]:
    """Select only failed, low-confidence bilingual conflicts for expensive second ASR."""
    selected = []
    for cue in cues:
        language = str(cue.get("source_language") or "").lower()
        second = cue.get("asr_second_opinion") or {}
        try:
            confidence = float(cue.get("transcription_confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if (not bool((cue.get("translation_qc") or {}).get("passed"))
                and bool(cue.get("supplied_translation"))
                and confidence < .93
                and language not in {"en", "eng", "english"}
                and not str(second.get("text") or "").strip()):
            selected.append(cue)
    return selected


def comparable_transcript(value: str) -> str:
    """Reduce a transcript to comparable characters, in any script.

    Punctuation, spacing and case differ freely between two ASR models reading
    the same audio; what matters is whether they heard the same words.
    """
    import unicodedata

    return "".join(character for character in unicodedata.normalize("NFKC", str(value)).casefold()
                   if character.isalnum())


def source_asr_verification_candidates(cues: list[dict]) -> list[dict]:
    """Lines whose source transcript deserves an independent second reading.

    The primary transcript comes from a 0.6B ASR, and its escalation to the
    larger model triggers on alignment confidence: how cleanly the aligner fitted
    the words it was handed, not whether those words were right.  A confidently
    aligned mistranscription therefore never reaches the larger model, and every
    stage downstream inherits it as fact.  An independent reading is the only
    thing that catches it.
    """
    try:
        limit = max(0, int(os.getenv("DUB_SOURCE_VERIFY_MAX_LINES", "1500")))
    except ValueError:
        limit = 1500
    scored: list[tuple[float, int, dict]] = []
    for index, cue in enumerate(cues):
        if str(cue.get("source_language") or "").lower() in {"", "en", "eng", "english"}:
            continue
        if not str(cue.get("source") or "").strip():
            continue
        if float(cue.get("end", 0)) - float(cue.get("start", 0)) < .35:
            continue
        if str((cue.get("asr_second_opinion") or {}).get("text") or "").strip():
            continue
        # Rank by how little the primary can be trusted, so a capped run spends
        # its budget where a mistranscription is most likely to be hiding.
        risk = 1.0 - float(cue.get("transcription_confidence") or 0.0)
        if not cue.get("asr_escalation_consulted"):
            risk += .35
        risk += max(0.0, .85 - float(cue.get("alignment_confidence") or 0.0))
        scored.append((risk, index, cue))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [cue for _, _, cue in sorted(scored[:limit], key=lambda item: item[1])]


def apply_source_asr_agreement(cues: list[dict], opinions: dict[int, dict]) -> int:
    """Record the second reading and de-rate transcripts the two models dispute."""
    import difflib

    disputed = 0
    for cue in cues:
        opinion = opinions.get(int(cue["id"]))
        if not opinion:
            continue
        cue["asr_second_opinion"] = opinion
        primary = comparable_transcript(cue.get("source", ""))
        second = comparable_transcript(opinion.get("text", ""))
        agreement = difflib.SequenceMatcher(None, primary, second).ratio() if primary and second else 0.0
        cue["source_asr_agreement"] = round(agreement, 3)
        usable = (_measure(opinion.get("confidence"), 0.0) >= .5
                  and _measure(opinion.get("word_confidence"), 0.0) >= .7
                  # A perfect no-speech score is 0.0, which is falsy: defaulting
                  # it away would reject exactly the most confident readings.
                  and _measure(opinion.get("no_speech_probability"), 1.0) < .35)
        if not usable or agreement >= .995:
            continue
        # Two independent models did not produce the same words.  Whether the
        # difference changes the meaning is not something string distance can
        # tell us -- in a logographic script one character separates "study"
        # from "compensate", while a politeness particle changes nothing.  So
        # this only records that the reading is contested and hands both to the
        # bilingual judge downstream, which compares meanings rather than
        # characters.  Only a wholesale disagreement de-rates far enough to ask
        # for human eyes directly.
        disputed += 1
        cue["source_asr_disputed"] = True
        ceiling = .5 if agreement < .5 else .7
        cue["transcription_confidence"] = round(
            min(_measure(cue.get("transcription_confidence"), 1.0), ceiling), 3)
        cue["confidence_source"] = "contested by an independent Whisper reading"
    return disputed


def _measure(value, default: float) -> float:
    """Read a numeric measurement, keeping a legitimate zero."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mirror_current_takes(fitted: Path, comparison: Path, total: int) -> int:
    """Fill a comparison directory so it holds one take per line.

    The QC and similarity passes read a whole directory positionally, one file
    per cue.  A directory holding only the lines that were regenerated makes
    them fail on the first line that was not, so every other line is carried
    across as its current take and only the regenerated files differ.
    """
    comparison.mkdir(parents=True, exist_ok=True)
    mirrored = 0
    for index in range(total):
        current = fitted / f"{index + 1:06d}.wav"
        target = comparison / current.name
        if current.is_file() and not target.is_file():
            shutil.copy2(current, target)
            mirrored += 1
    return mirrored


def second_asr_evidence_candidates(cues: list[dict]) -> list[dict]:
    """Find second transcriptions that still need an explicit English meaning."""
    return [cue for cue in cues
            if str((cue.get("asr_second_opinion") or {}).get("text") or "").strip()
            and not str((cue.get("asr_second_opinion") or {}).get("english_translation") or "").strip()]


def translate_second_asr_evidence(cues: list[dict], folder: Path,
                                  progress: Callable[[float, int], None],
                                  checkpoint: Callable[[], None]) -> dict[int, dict]:
    """Translate selective ASR evidence separately so the judge compares English meanings."""
    model = Path(os.getenv(
        "TRANSLATION_MODEL", "vendor/hy-mt2-7b/Hy-MT2-7B-Q4_K_M.gguf"
    )).resolve()
    if not model.is_file():
        raise RuntimeError("The local Hy-MT2 evidence-translation model is missing")
    items = [{
        "id": int(cue["id"]),
        "source": str(cue["asr_second_opinion"]["text"]),
        "language": cue.get("source_language") or cue["asr_second_opinion"].get("language"),
    } for cue in cues]
    manifest_value = {"model": str(model), "items": items, "revision": "literal-evidence-v1"}
    manifest_text = json.dumps(manifest_value, ensure_ascii=False, sort_keys=True)
    signature = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    signature_path = folder / "translation-evidence-input.sha256"
    output = folder / "translation-evidence.json"
    if signature_path.is_file() and signature_path.read_text(encoding="utf-8").strip() != signature:
        output.unlink(missing_ok=True)
    signature_path.write_text(signature, encoding="utf-8")
    manifest = folder / "translation-evidence-manifest.json"
    manifest.write_text(manifest_text, encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.services.translation_evidence_worker", "--manifest", str(manifest),
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
        raise RuntimeError("Second-ASR evidence translation failed: " + "\n".join(tail[-10:]))
    values = json.loads(output.read_text(encoding="utf-8"))
    return {int(item["id"]): item for item in values
            if isinstance(item, dict) and item.get("id") is not None}


def verify_translation_disputes(audio: Path, cues: list[dict], folder: Path,
                                progress: Callable[[float, int], None],
                                checkpoint: Callable[[], None]) -> dict[int, dict]:
    """Re-transcribe only disputed lines with local CUDA Whisper Large-v3-Turbo."""
    clips = folder / "translation-dispute-clips"
    clips.mkdir(exist_ok=True)
    language_codes = {
        "japanese": "ja", "ja": "ja", "chinese": "zh", "zh": "zh",
        "spanish": "es", "es": "es", "arabic": "ar", "ar": "ar",
        "french": "fr", "fr": "fr", "german": "de", "de": "de",
        "korean": "ko", "ko": "ko", "italian": "it", "it": "it",
        "portuguese": "pt", "pt": "pt", "russian": "ru", "ru": "ru",
    }
    items = []
    for cue in cues:
        checkpoint()
        cue_id = int(cue["id"])
        start = max(0.0, float(cue.get("start", 0.0)) - .45)
        end = max(start + .35, float(cue.get("end", start)) + .75)
        target = clips / f"{cue_id:06d}.flac"
        if not target.is_file() or target.stat().st_size < 256:
            run("ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}",
                "-t", f"{end - start:.3f}", "-i", str(audio), "-ac", "1", "-ar", "16000",
                "-c:a", "flac", str(target))
        language = str(cue.get("source_language") or "").lower()
        items.append({"id": cue_id, "path": str(target), "language": language_codes.get(language)})
    manifest_value = {
        "model": "turbo",
        "cache": str(Path(os.getenv("WHISPER_CACHE_DIR", "vendor/whisper")).resolve()),
        "items": items,
    }
    manifest_text = json.dumps(manifest_value, ensure_ascii=False, sort_keys=True)
    signature = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    signature_path = folder / "translation-dispute-input.sha256"
    output = folder / "translation-dispute-asr.json"
    if signature_path.is_file() and signature_path.read_text(encoding="utf-8").strip() != signature:
        output.unlink(missing_ok=True)
    signature_path.write_text(signature, encoding="utf-8")
    manifest = folder / "translation-dispute-manifest.json"
    manifest.write_text(manifest_text, encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.services.whisper_dispute_worker", "--manifest", str(manifest),
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
        raise RuntimeError("Selective Whisper dispute check failed: " + "\n".join(tail[-10:]))
    values = json.loads(output.read_text(encoding="utf-8"))
    return {int(item["id"]): item for item in values
            if isinstance(item, dict) and item.get("id") is not None}


def emotion_context(cues: list[dict], index: int) -> str:
    previous = cues[index - 1]["english"] if index else "(scene begins)"
    current = cues[index]["english"]
    following = cues[index + 1]["english"] if index + 1 < len(cues) else "(scene ends)"
    return (
        "Classify the emotion of the CURRENT spoken line using its scene context. "
        f"Previous line: {previous}\nCURRENT line: {current}\nFollowing line: {following}"
    )


def transcribe_long_audio(audio: Path, folder: Path, model_name: str, checkpoint: Callable,
                          update: Callable, subtitle_hints: list[dict] | None = None) -> list[dict]:
    source_value = "auto"
    language_file = folder / "source-language.txt"
    if language_file.exists():
        source_value = language_file.read_text(encoding="utf-8").strip()
    language_name = None if source_value == "auto" else source_value
    committed_map = folder / "asr-dialogue-map.json"
    if committed_map.is_file():
        try:
            cached = json.loads(committed_map.read_text(encoding="utf-8"))
            if isinstance(cached, list) and cached:
                return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    windows = prepare_asr_windows(audio, folder, subtitle_hints or [])
    try:
        aligned = transcribe_aligned(
            windows, folder, language_name,
            lambda value, index: update(28 + value * 5,
                f"Transcribing and force-aligning scene {index + 1} of {len(windows)}"), checkpoint,
        )
    except RuntimeError as exc:
        # Auto-detected languages outside the forced aligner's set must cross a
        # clean model boundary rather than aborting the film.
        (folder / "qwen-asr-fallback.txt").write_text(str(exc)[-2000:], encoding="utf-8")
        aligned = None
    if aligned:
        return aligned

    # Whisper Turbo remains a robust fallback for unsupported aligner languages and
    # a boundary/QC model.  It is no longer the primary transcript/aligner.
    chunks = folder / "speech-chunks"
    chunks.mkdir(exist_ok=True)
    if not list(chunks.glob("chunk-*.flac")):
        run("ffmpeg", "-y", "-v", "error", "-i", str(audio), "-f", "segment", "-segment_time", "1500",
            "-reset_timestamps", "1", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(chunks / "chunk-%04d.flac"))
    files = sorted(chunks.glob("chunk-*.flac"))
    manifest = folder / "whisper-manifest.json"
    output = folder / "asr-dialogue-map.json"
    language_code = {"Japanese": "ja", "Chinese": "zh", "Spanish": "es", "Arabic": "ar", "English": "en"}.get(source_value)
    manifest.write_text(json.dumps({"files": [str(path) for path in files], "model": model_name,
                                    "cache": str(Path(os.getenv("WHISPER_CACHE_DIR", "vendor/whisper")).resolve()),
                                    "source_language": language_code, "chunk_seconds": 1500}, indent=2), encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.services.whisper_worker", "--manifest", str(manifest), "--output", str(output)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-25:]
            try:
                event = json.loads(line)
                update(28 + float(event["progress"]) * 5,
                       f"Building word-level dialogue map · chunk {int(event['chunk']) + 1} of {len(files)}")
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        code = process.wait()
    except BaseException:
        terminate_process(process); raise
    if code != 0 or not output.is_file():
        raise RuntimeError("Whisper dialogue-map worker failed: " + "\n".join(tail[-12:]))
    return json.loads(output.read_text(encoding="utf-8"))


def prepare_asr_windows(audio: Path, folder: Path, subtitle_hints: list[dict]) -> list[dict]:
    """Create bounded speech/scene windows so ASR never spans long non-speech passages."""
    target = folder / "asr-windows"
    index_path = target / "windows.json"
    if index_path.is_file():
        cached = json.loads(index_path.read_text(encoding="utf-8"))
        if cached and all(Path(item["path"]).is_file() for item in cached):
            return cached
    target.mkdir(exist_ok=True)
    intervals: list[tuple[float, float]] = []
    if subtitle_hints:
        for hint in subtitle_hints:
            start = max(0.0, float(hint["start"]) - 0.65)
            end = float(hint["end"]) + 0.65
            if intervals and start - intervals[-1][1] < 3.0 and end - intervals[-1][0] <= 180.0:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
            else:
                intervals.append((start, end))
    else:
        detected = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio), "-af",
             "silencedetect=noise=-42dB:d=0.35", "-f", "null", os.devnull],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stderr
        silence_starts = [float(x) for x in __import__('re').findall(r"silence_start: ([0-9.]+)", detected)]
        silence_ends = [float(x) for x in __import__('re').findall(r"silence_end: ([0-9.]+)", detected)]
        total = audio_duration(audio)
        position = 0.0
        for silence_start, silence_end in zip(silence_starts, silence_ends):
            if silence_start - position >= 0.24:
                intervals.append((max(0, position - .25), min(total, silence_start + .25)))
            position = silence_end
        if total - position >= 0.24:
            intervals.append((max(0, position - .25), total))
        if not intervals:
            intervals = [(start, min(total, start + 180.0)) for start in __import__('numpy').arange(0, total, 180.0)]
    # Split any continuous scene to stay below the forced aligner's five-minute limit.
    bounded: list[tuple[float, float]] = []
    for start, end in intervals:
        while end - start > 180.0:
            bounded.append((start, start + 180.0)); start += 180.0
        if end - start >= 0.2:
            bounded.append((start, end))
    windows = []
    for index, (start, end) in enumerate(bounded):
        path = target / f"window-{index:05d}.flac"
        if not path.is_file():
            run("ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(audio),
                "-t", f"{end - start:.3f}", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(path))
        windows.append({"path": str(path.resolve()), "offset": round(start, 3),
                        "duration": round(end - start, 3)})
    index_path.write_text(json.dumps(windows, indent=2), encoding="utf-8")
    return windows


def make_reference(audio: Path, cue: dict, output: Path, media_length: float,
                   minimum: float = 0.5, padding: float = 0.12) -> None:
    """Create a cue-contained reference without destructive silence padding."""
    cue_start, cue_end = float(cue["start"]), float(cue["end"])
    start = max(0.0, cue_start - min(.08, padding))
    end = min(media_length, cue_end + min(.08, padding))
    captured = max(.12, end - start)
    run("ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(audio), "-t", f"{captured:.3f}",
        "-af", "loudnorm=I=-20:TP=-2:LRA=11",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(output))
    cue["reference_scope"] = "cue contained; clean dialogue"


def audio_duration(path: Path) -> float:
    result = run("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path))
    return float(result.stdout.strip())


def fit_audio(source: Path, output: Path, target: float) -> None:
    actual = max(0.01, audio_duration(source))
    tempo = actual / target
    run("ffmpeg", "-y", "-v", "error", "-i", str(source), "-af",
        f"rubberband=tempo={tempo:.8f},apad,atrim=duration={target:.6f},afade=t=in:st=0:d=0.015,afade=t=out:st={max(0, target - .03):.6f}:d=0.03",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(output))


def render_timeline(cues: list[dict], fitted_dir: Path, output: Path, duration: float,
                    nonverbal_source: Path | None = None) -> None:
    import numpy as np
    import soundfile as sf
    sample_count = int(round(duration * SAMPLE_RATE))
    scratch = output.with_suffix(".mix.f32")
    stereo = output.suffix.lower() == ".flac"
    shape = (sample_count, 2) if stereo else (sample_count,)
    with scratch.open("wb") as raw:
        raw.truncate(sample_count * 4 * (2 if stereo else 1))
    mix = np.memmap(scratch, dtype=np.float32, mode="r+", shape=shape)
    mix[:] = 0
    if nonverbal_source and nonverbal_source.is_file():
        with sf.SoundFile(nonverbal_source) as reader:
            cursor = 0
            while cursor < sample_count:
                values = reader.read(min(SAMPLE_RATE * 60, sample_count - cursor),
                                     dtype="float32", always_2d=True)
                if not len(values):
                    break
                mono = values.mean(axis=1)
                if stereo:
                    mix[cursor:cursor + len(mono), 0] = mono
                    mix[cursor:cursor + len(mono), 1] = mono
                else:
                    mix[cursor:cursor + len(mono)] = mono
                cursor += len(mono)
        fade = max(1, round(.025 * SAMPLE_RATE))
        for cue in cues:
            cue["overlap_source_preserved"] = False
            cue_start = min(float(cue.get("dub_start", cue["start"])), float(cue["start"]))
            cue_end = max(float(cue.get("dub_end", cue["end"])), float(cue["end"]))
            left = max(0, round((cue_start - .05) * SAMPLE_RATE))
            right = min(sample_count, round((cue_end + .05) * SAMPLE_RATE))
            if right <= left:
                continue
            lead = min(fade, right - left); trail = min(fade, right - left)
            if stereo:
                mix[left:left + lead] *= np.linspace(1, 0, lead)[:, None]
                mix[left + lead:right - trail] = 0
                mix[right - trail:right] *= np.linspace(0, 1, trail)[:, None]
            else:
                mix[left:left + lead] *= np.linspace(1, 0, lead)
                mix[left + lead:right - trail] = 0
                mix[right - trail:right] *= np.linspace(0, 1, trail)
    for index, cue in enumerate(cues, 1):
        line = fitted_dir / f"{int(cue.get('_audio_line', index)):06d}.wav"
        frames, rate = sf.read(line, dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            raise RuntimeError(f"Unexpected generated line sample rate: {rate}")
        frames = frames.mean(axis=1)
        start = max(0, int(round(float(cue.get("dub_start", cue["start"])) * SAMPLE_RATE)))
        end = min(sample_count, start + len(frames))
        if end > start:
            segment = frames[:end - start]
            if stereo:
                generated_rms = float(np.sqrt(np.mean(segment * segment) + 1e-12))
                source = cue.get("source_performance", {})
                gain = np.clip(float(source.get("rms", generated_rms)) / max(generated_rms, 1e-6), 0.5, 2.0)
                pan = float(source.get("pan", 0.0))
                angle = (pan + 1.0) * np.pi / 4
                placed = np.stack((segment * np.cos(angle), segment * np.sin(angle)), axis=1) * gain
                mix[start:end] += placed
            else:
                mix[start:end] += segment
    peak = max(1e-6, max(float(np.max(np.abs(mix[start:start + SAMPLE_RATE * 60])))
                         for start in range(0, sample_count, SAMPLE_RATE * 60)))
    gain = min(1.0, 0.98 / peak)
    subtype = "PCM_24" if output.suffix.lower() == ".flac" else "PCM_16"
    with sf.SoundFile(output, "w", samplerate=SAMPLE_RATE, channels=2 if stereo else 1, subtype=subtype) as writer:
        for start in range(0, sample_count, SAMPLE_RATE * 60):
            writer.write(np.clip(mix[start:start + SAMPLE_RATE * 60] * gain, -0.98, 0.98))
    del mix
    scratch.unlink(missing_ok=True)


def match_acoustics(cues: list[dict], fitted_dir: Path, output_dir: Path,
                    source_audio: Path | None = None) -> None:
    """Apply source-derived match EQ and conservative room/level integration."""
    import soundfile as sf
    import numpy as np

    def band_levels(values, rate, centers):
        if not len(values):
            return [1e-12] * len(centers)
        spectrum = np.abs(np.fft.rfft(values * np.hanning(len(values)))) ** 2
        frequencies = np.fft.rfftfreq(len(values), 1 / rate)
        return [float(np.mean(spectrum[(frequencies >= center / 1.42) &
                                       (frequencies <= center * 1.42)])) + 1e-12
                for center in centers]

    output_dir.mkdir(exist_ok=True)
    for index, cue in enumerate(cues, 1):
        source = fitted_dir / f"{index:06d}.wav"; output = output_dir / source.name
        if output.is_file():
            continue
        performance = cue.get("source_performance", {})
        distance = performance.get("distance", "medium")
        lowpass = {"close": 12_000, "medium": 9_500, "reverberant": 7_200}.get(distance, 9_500)
        generated, generated_rate = sf.read(source, dtype="float32", always_2d=True)
        generated = generated.mean(axis=1)
        centers = [180, 350, 700, 1400, 2800, 5600]
        match_gains = [0.0] * len(centers)
        if source_audio and source_audio.is_file():
            with sf.SoundFile(source_audio) as reader:
                start = max(0, round(float(cue["start"]) * reader.samplerate))
                end = min(len(reader), round(float(cue["end"]) * reader.samplerate))
                reader.seek(start)
                reference = reader.read(max(0, end - start), dtype="float32", always_2d=True).mean(axis=1)
                ref_levels = band_levels(reference, reader.samplerate, centers)
                gen_levels = band_levels(generated, generated_rate, centers)
                raw_gains = [10 * np.log10(ref / gen) for ref, gen in zip(ref_levels, gen_levels)]
                offset = float(np.median(raw_gains))
                match_gains = [float(np.clip(value - offset, -1.5, 1.5)) for value in raw_gains]
        filters = ["highpass=f=70", f"lowpass=f={lowpass}"]
        # Residual register match, after the retakes have had their chance.
        # Formant-preserving shifting moves the fundamental toward the actor
        # without resizing the perceived vocal tract, so the cloned timbre is
        # left intact.  Bounded on both sides: below the floor the difference is
        # inaudible, above the ceiling the artefacts cost more than the match.
        correction = register_correction(cue)
        if correction is not None:
            semitones, scale = correction
            filters.append(f"rubberband=pitch={scale:.6f}:formant=preserved:pitchq=quality")
            cue.setdefault("qc", {})["pitch_correction_semitones"] = semitones
        filters += [f"equalizer=f={center}:t=q:w=0.9:g={gain:.2f}"
                    for center, gain in zip(centers, match_gains) if abs(gain) >= .2]
        filters += ["deesser=i=0.18:m=0.28:f=0.5",
                    "acompressor=threshold=0.14:ratio=2:attack=8:release=110:makeup=1.05"]
        filters += ["alimiter=limit=0.94", "volume=0.98"]
        run("ffmpeg", "-y", "-v", "error", "-i", str(source), "-af", ",".join(filters),
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(output))
        values, rate = sf.read(output, dtype="float32", always_2d=True)
        current_rms = float(np.sqrt(np.mean(values * values) + 1e-12))
        source_rms = float(performance.get("rms", current_rms))
        gain = float(np.clip(source_rms / max(current_rms, 1e-6), .5, 4.0))
        sf.write(output, np.clip(values * gain, -.94, .94), rate, subtype="PCM_16")
        cue["acoustic_match"] = {
            "distance": distance, "lowpass_hz": lowpass,
            "match_eq_db": {str(center): round(value, 2)
                            for center, value in zip(centers, match_gains)},
            "level_match_db": round(20 * np.log10(gain), 2),
        }


def mix_audio(source: Path, voice: Path, output: Path, duration: float, mode: str,
              background: Path | None = None, mastering_preset: str = "cinema",
              cues: list[dict] | None = None) -> dict:
    premaster = output.with_name(output.stem + "-premaster.flac")
    if mode == "replace":
        run("ffmpeg", "-y", "-v", "error", "-i", str(voice), "-t", f"{duration:.3f}",
            "-ar", "48000", "-c:a", "flac", "-sample_fmt", "s32", str(premaster))
        return master_audio(premaster, output, voice, mastering_preset, cues)
    if mode == "separate" and background and background.is_file():
        bed_source = background
        bed_probe = probe_media(background)
        bed_stream = next((item for item in bed_probe.get("streams", []) if item.get("codec_type") == "audio"), {})
        channels = int(bed_stream.get("channels", 2) or 2)
        layout = str(bed_stream.get("channel_layout") or "")
        if channels in {6, 8}:
            if channels == 8:
                surround = layout if layout and layout != "unknown" else "7.1"
                voice_pan = (f"pan={surround}|FL=0.30*FL|FR=0.30*FR|FC=0.5*FL+0.5*FR|LFE=0*FL|"
                             "BL=0*FL|BR=0*FR|SL=0*FL|SR=0*FR")
            else:
                surround = "5.1(side)" if "side" in layout else "5.1"
                voice_pan = (f"pan={surround}|FL=0.35*FL|FR=0.35*FR|FC=0.5*FL+0.5*FR|LFE=0*FL|" +
                             ("SL=0*FL|SR=0*FR" if "side" in surround else "BL=0*FL|BR=0*FR"))
            graph = (
                f"[0:a]aresample=48000[bed];[1:a]aresample=48000,{voice_pan},"
                "volume=1.05[voice];[bed][voice]amix=inputs=2:duration=longest:dropout_transition=0,"
                "alimiter=limit=0.95[mix]"
            )
        else:
            graph = (
                "[0:a]aresample=48000[bed];"
                "[1:a]aresample=48000,volume=1.05,asplit=2[voice][key];"
                "[bed][key]sidechaincompress=threshold=0.02:ratio=2:attack=12:release=220[ducked];"
                "[ducked][voice]amix=inputs=2:duration=longest:dropout_transition=0,alimiter=limit=0.95[mix]"
            )
    else:
        bed_source = source
        graph = (
            "[0:a]aresample=48000,volume=0.48[bed];"
            "[1:a]aresample=48000,volume=1.15,asplit=2[voice][key];"
            "[bed][key]sidechaincompress=threshold=0.012:ratio=10:attack=12:release=320[ducked];"
            "[ducked][voice]amix=inputs=2:duration=longest:dropout_transition=0,alimiter=limit=0.95[mix]"
        )
    run("ffmpeg", "-y", "-v", "error", "-i", str(bed_source), "-i", str(voice), "-filter_complex", graph,
        "-map", "[mix]", "-t", f"{duration:.3f}", "-ar", "48000", "-c:a", "flac", "-sample_fmt", "s32", str(premaster))
    return master_audio(premaster, output, voice, mastering_preset, cues)


def measured_lufs(path: Path) -> float | None:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
         "loudnorm=I=-24:TP=-2:LRA=18:print_format=json", "-f", "null", os.devnull],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    matches = __import__('re').findall(r"\{[^{}]+\}", result.stderr, flags=__import__('re').S)
    if not matches:
        return None
    try:
        return float(json.loads(matches[-1]).get("input_i"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def dialogue_gated_lufs(premaster: Path, cues: list[dict]) -> float | None:
    """Measure the finished mix over the spans where dialogue actually plays.

    Program gain must be derived from the dialogue as it sits *in the mix*, not
    from the isolated stem: the mix graph applies its own gain structure (amix
    scales every input by 1/n), so a stem measurement is several dB adrift from
    the level it is being used to set.  Gating on the real spans removes that
    coupling entirely, and is what dialogue-gated levelling means in practice.
    """
    spans = sorted(((float(cue["end"]) - float(cue["start"]), float(cue["start"]), float(cue["end"]))
                    for cue in cues
                    if float(cue.get("end", 0)) - float(cue.get("start", 0)) >= .6),
                   reverse=True)
    if not spans:
        return None
    selected, total = [], 0.0
    for length, start, end in spans:
        if len(selected) >= 80 or total >= 300:
            break
        selected.append((start, end)); total += length
    if not selected:
        return None
    selected.sort()
    expression = "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in selected)
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(premaster), "-af",
         f"aselect='{expression}',asetpts=N/SR/TB,"
         "loudnorm=I=-24:TP=-2:LRA=18:print_format=json", "-f", "null", os.devnull],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    matches = __import__('re').findall(r"\{[^{}]+\}", result.stderr, flags=__import__('re').S)
    if not matches:
        return None
    try:
        return float(json.loads(matches[-1]).get("input_i"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def master_audio(premaster: Path, output: Path, dialogue: Path, preset: str,
                 cues: list[dict] | None = None) -> dict:
    """Apply an explicit delivery preset rather than merely reporting loudness."""
    if preset == "preserve":
        shutil.copyfile(premaster, output)
        premaster.unlink(missing_ok=True)
        return {"preset": "preserve", "target": None, "dialogue_lufs": measured_lufs(dialogue)}
    if preset == "broadcast":
        audio_filter = "loudnorm=I=-23:TP=-1:LRA=18,alimiter=limit=0.891"
        result = {"preset": "EBU broadcast", "target_lufs": -23.0, "true_peak_target_dbtp": -1.0}
    elif preset == "web":
        audio_filter = "loudnorm=I=-16:TP=-1:LRA=11,alimiter=limit=0.891"
        result = {"preset": "web", "target_lufs": -16.0, "true_peak_target_dbtp": -1.0}
    else:
        # Cinema/localization preset: set program gain from dialogue-gated content,
        # then protect the full program at -2 dBTP.
        gated = dialogue_gated_lufs(premaster, cues) if cues else None
        stem_lufs = measured_lufs(dialogue)
        # The gated measurement is the correct basis; the stem is only a fallback
        # for an audio-only or cue-less job, and is recorded for comparison.
        dialogue_lufs = gated if gated is not None else stem_lufs
        gain = float(__import__('numpy').clip(-27.0 - (dialogue_lufs if dialogue_lufs is not None else -27.0), -12, 12))
        audio_filter = f"volume={gain:.3f}dB,alimiter=limit=0.794"
        result = {"preset": "cinema dialogue-gated", "target_dialogue_lufs": -27.0,
                  "measured_dialogue_lufs": dialogue_lufs,
                  "measurement_basis": "dialogue-gated mix" if gated is not None else "dialogue stem",
                  "dialogue_stem_lufs": stem_lufs,
                  "dialogue_lufs_after_gain": round((dialogue_lufs or -27.0) + gain, 2),
                  "program_gain_db": round(gain, 3), "true_peak_target_dbtp": -2.0}
    run("ffmpeg", "-y", "-v", "error", "-i", str(premaster), "-af", audio_filter,
        "-ar", "48000", "-c:a", "flac", "-sample_fmt", "s32", str(output))
    premaster.unlink(missing_ok=True)
    return result


def remux(source: Path, audio: Path, output: Path, video_override: Path | None = None,
          original_order: Sequence[int] = ()) -> None:
    """Mux the delivery master with the English dub as the first audio track.

    Track 1 is the dub and track 2 is the original dialogue soundtrack the
    pipeline actually worked from, which is not always the source's first audio
    stream: plenty of releases put a commentary track ahead of the programme.
    Remaining source audio follows in its original order.
    """
    inputs = ["-i", str(source), "-i", str(audio)]
    video_map = "0:v?"
    if video_override and video_override.is_file():
        inputs += ["-i", str(video_override)]
        video_map = "2:v:0"
    # An explicit order is used when the caller knows which stream carried the
    # dialogue; otherwise fall back to whatever order the container lists.
    original_maps: list[str] = []
    for index in original_order:
        original_maps += ["-map", f"0:{index}"]
    if not original_maps:
        original_maps = ["-map", "0:a?"]
    run("ffmpeg", "-y", "-v", "error", *inputs,
        "-map", video_map, "-map", "1:a:0", *original_maps,
        "-map", "0:s?", "-map", "0:d?", "-map", "0:t?",
        "-map_metadata", "0", "-map_chapters", "0", "-c", "copy",
        "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=English AI Dub · FLAC delivery master",
        # Clear only the default flag across the audio tracks, then raise it on
        # the dub, so a player opens speaking English. Wiping dispositions
        # wholesale would also strip commentary and hearing-impaired flags.
        "-disposition:a", "-default", "-disposition:a:0", "+default", str(output))


def format_duration(seconds: float) -> str:
    hours, rest = divmod(round(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def target_language_code(language: str) -> str:
    return {"English": "EN", "Chinese": "ZH", "Japanese": "JA", "Spanish": "ES", "Arabic": "AR"}.get(language, "EN")


def write_exports(folder: Path, cues: list[dict], dialogue_dir: Path, duration: float) -> dict:
    export_dir = folder / "exports"; export_dir.mkdir(exist_ok=True)
    srt = export_dir / "english-dub.srt"
    write_srt([{"start": cue["start"], "end": cue["end"], "text": cue.get("english", "")}
               for cue in cues], srt)
    csv_path = export_dir / "cues.csv"
    fields = ("id", "start", "end", "speaker", "source", "faithful_translation", "english",
              "emotion", "needs_review", "review_reasons")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for cue in cues:
            writer.writerow({key: "; ".join(cue.get(key, [])) if isinstance(cue.get(key), list)
                             else cue.get(key, "") for key in fields})
    edl = export_dir / "dub-cues.edl"
    def tc(seconds: float) -> str:
        frames = max(0, round(seconds * 24)); hours, frames = divmod(frames, 24 * 3600)
        minutes, frames = divmod(frames, 24 * 60); secs, frames = divmod(frames, 24)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
    lines = ["TITLE: DUBLINE ENGLISH DUB", "FCM: NON-DROP FRAME", ""]
    for index, cue in enumerate(cues, 1):
        lines += [f"{index:03d}  DUBLINE  A     C        {tc(0)} {tc(float(cue['end'])-float(cue['start']))} {tc(float(cue['start']))} {tc(float(cue['end']))}",
                  f"* FROM CLIP NAME: {index:06d}.wav", f"* COMMENT: {cue.get('speaker','')} | {cue.get('english','')}", ""]
    edl.write_text("\n".join(lines), encoding="utf-8")
    speaker_dir = export_dir / "speaker-stems"; speaker_dir.mkdir(exist_ok=True)
    speaker_files = {}
    for speaker_id in sorted({int(cue.get("speaker_id", 0)) for cue in cues}):
        selected = [{**cue, "_audio_line": index} for index, cue in enumerate(cues, 1)
                    if int(cue.get("speaker_id", 0)) == speaker_id]
        if not selected:
            continue
        path = speaker_dir / f"voice-{speaker_id:03d}.flac"
        render_timeline(selected, dialogue_dir, path, duration)
        speaker_files[str(speaker_id)] = str(path)
    clips = export_dir / "dub-clips.zip"
    with zipfile.ZipFile(clips, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(srt, srt.name); archive.write(csv_path, csv_path.name); archive.write(edl, edl.name)
        for index in range(1, len(cues) + 1):
            path = dialogue_dir / f"{index:06d}.wav"
            if path.is_file(): archive.write(path, f"clips/{path.name}")
        for path in speaker_files.values():
            stem = Path(path)
            if stem.is_file(): archive.write(stem, f"speaker-stems/{stem.name}")
    return {"srt": str(srt), "csv": str(csv_path), "edl": str(edl), "clips": str(clips),
            "speaker_stems": speaker_files,
            "mix": str(folder / "english-mix.flac"), "dialogue": str(folder / "english-dialogue.flac")}

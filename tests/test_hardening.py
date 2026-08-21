from pathlib import Path
import sys
import time

import pytest
from fastapi import HTTPException

from app.main import invalidate_cue_artifacts, normalized_options, persist_job_cues
from app.services.analysis_cache import media_fingerprint, restore_json_artifact, store_json_artifact
from app.services.cinematic import _worker
from app.services.qc import evaluate_media_qc, inspect_cues, media_qc
from app.services.diarization import diarization_runtime_settings
from app.services.pipeline import PipelineWorker, SAMPLE_RATE, remux, render_timeline
from app.services.qwen_asr_worker import grouped_cues
from app.services import gpu_safety
from app.services import pipeline as pipeline_service
from app.services.job_lock import acquire_job_lock, release_job_lock
from app.services.subtitles import choose_audio, choose_embedded, parse_srt
from app.services.subprocess_control import controlled_lines
from app.services.visual_speakers_worker import consolidate_identities, mouth_patch
from app.services.visual_speakers import upgrade_legacy_face_registry
from app.store import JobStore


def test_regeneration_invalidates_acoustic_and_alternative_takes(tmp_path: Path):
    targets = [
        tmp_path / "generated" / "000003.wav",
        tmp_path / "fitted" / "000003.wav",
        tmp_path / "acoustically-matched" / "000003.wav",
        tmp_path / "qwen-generated" / "000003.wav",
        tmp_path / "qwen-fitted" / "000003.wav",
        tmp_path / "english-dialogue.flac",
        tmp_path / "dubbed-english.mkv",
        tmp_path / "qc-report.json",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stale")
    invalidate_cue_artifacts(tmp_path, 3)
    assert not any(target.exists() for target in targets)


def test_job_polling_summary_excludes_large_documents(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create({"id": "film", "filename": "film.mkv", "status": "queued",
                  "stage": "Waiting", "progress": 0, "cues": [{"id": 1}], "logs": ["created"]})
    store.update("film", cues=[{"id": 1}, {"id": 2}])
    summary = store.list_summaries()[0]
    assert summary["cue_count"] == 2 and summary["cue_revision"] == 1
    assert "cues" not in summary and "logs" not in summary


def test_range_options_validate_reverse_range_before_queuing():
    selected = normalized_options({"range_start": 1220, "range_end": 1320})
    assert selected["range_start"] == 1220 and selected["range_end"] == 1320
    with pytest.raises(HTTPException):
        normalized_options({"range_start": 1320, "range_end": 1220})


def test_delivery_failures_are_hard_qc_failures():
    media = {
        "video_tracks": 1, "duration_error_ms": 700, "original_audio_preserved": True,
        "original_subtitles_preserved": True, "english_lossless_track": True,
        "integrated_lufs": -19.0, "true_peak_dbtp": -0.2,
    }
    failures = evaluate_media_qc(media, {"target_lufs": -23.0, "true_peak_target_dbtp": -1.0})
    assert any("duration" in reason for reason in failures)
    assert any("loudness" in reason for reason in failures)
    assert any("true peak" in reason for reason in failures)


def test_recovery_honours_pending_pause_and_cancel(tmp_path: Path):
    store = JobStore(tmp_path / "recovery.sqlite3")
    store.create({"id": "pause", "filename": "a.mkv", "status": "queued", "control": "pause",
                  "stage": "Waiting", "progress": 0, "cues": []})
    store.create({"id": "cancel", "filename": "b.mkv", "status": "processing", "control": "cancel",
                  "stage": "Running", "progress": 1, "cues": []})
    store.create({"id": "resume", "filename": "c.mkv", "status": "processing",
                  "stage": "Running", "progress": 1, "cues": []})
    assert store.recover_interrupted() == ["resume"]
    assert store.get("pause")["status"] == "paused"
    assert store.get("cancel")["status"] == "cancelled"


def test_resume_notice_cannot_be_lost_behind_old_queue_notice(tmp_path: Path):
    store = JobStore(tmp_path / "queue.sqlite3")
    worker = PipelineWorker(store, tmp_path)
    worker.submit("film")
    worker.submit("film")
    assert worker.jobs.qsize() == 2


def test_quiet_model_worker_is_still_cancellable():
    started = time.monotonic()

    def checkpoint():
        if time.monotonic() - started > .35:
            raise RuntimeError("cancel requested")

    with pytest.raises(RuntimeError, match="cancel requested"):
        _worker([sys.executable, "-c", "import time; time.sleep(20)"], [], "quiet worker",
                lambda _: None, checkpoint)
    assert time.monotonic() - started < 4


def test_every_quiet_model_stream_checks_control_without_output():
    import subprocess
    started = time.monotonic()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def checkpoint():
        if time.monotonic() - started > .35:
            raise RuntimeError("pause requested")
    with pytest.raises(RuntimeError, match="pause requested"):
        list(controlled_lines(process, checkpoint))
    assert process.poll() is not None
    assert time.monotonic() - started < 4


def test_qc_fails_closed_when_evidence_is_missing(tmp_path: Path):
    fitted = tmp_path / "fitted"; fitted.mkdir()
    import numpy as np
    import soundfile as sf
    sf.write(fitted / "000001.wav", np.ones(SAMPLE_RATE, dtype=np.float32) * .01, SAMPLE_RATE)
    cues = [{"start": 0, "end": 1, "qc": {}}]
    assert inspect_cues(cues, fitted)["flagged_count"] == 1
    assert any("evidence is missing" in reason for reason in cues[0]["review_reasons"])
    assert any("independent bilingual" in reason for reason in cues[0]["review_reasons"])


def test_sdh_and_two_speaker_cards_are_semantically_split(tmp_path: Path):
    subtitle = tmp_path / "scene.srt"
    subtitle.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n[DOOR SLAMS]\n- ALICE: Wait!\n- BOB: I'm coming.\n",
        encoding="utf-8",
    )
    cues = parse_srt(subtitle)
    assert [cue["text"] for cue in cues] == ["Wait!", "I'm coming."]
    assert [cue["subtitle_speaker_hint"] for cue in cues] == ["Alice", "Bob"]
    assert all(cue["simultaneous_card"] for cue in cues)
    assert cues[0]["nonverbal_annotations"] == ["DOOR SLAMS"]


def test_track_selection_avoids_commentary_and_forced_subtitles():
    audio = [
        {"index": 1, "ordinal": 0, "title": "Director commentary", "language": "eng",
         "channels": 2, "default": True},
        {"index": 2, "ordinal": 1, "title": "Japanese programme 5.1", "language": "jpn",
         "channels": 6, "default": False},
    ]
    selected, _ = choose_audio(audio)
    assert selected["index"] == 2
    subtitles = [
        {"index": 3, "text": True, "title": "English forced signs", "language": "eng",
         "forced": True, "hearing_impaired": False, "default": True},
        {"index": 4, "text": True, "title": "English full", "language": "eng",
         "forced": False, "hearing_impaired": False, "default": False},
    ]
    assert choose_embedded(subtitles)["index"] == 4


def test_nonverbal_performance_is_kept_outside_aligned_words(tmp_path: Path):
    import numpy as np
    import soundfile as sf
    source = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    source[:SAMPLE_RATE // 2] = .03  # a breath/laugh before the aligned spoken word
    source[SAMPLE_RATE // 2:SAMPLE_RATE] = .02
    source_path = tmp_path / "source.flac"; sf.write(source_path, source, SAMPLE_RATE)
    fitted = tmp_path / "fitted"; fitted.mkdir()
    sf.write(fitted / "000001.wav", np.ones(SAMPLE_RATE, dtype=np.float32) * .08, SAMPLE_RATE)
    output = tmp_path / "timeline.wav"
    render_timeline([{"start": .5, "end": 1.5, "words": [{"start": .5, "end": 1.0}]}],
                    fitted, output, 2.0, source_path)
    rendered, _ = sf.read(output)
    assert abs(float(rendered[round(.2 * SAMPLE_RATE)])) > .02
    assert abs(float(rendered[round(.75 * SAMPLE_RATE)])) > .06


def test_structural_edits_update_the_resumable_cue_document(tmp_path: Path):
    job = {"folder": str(tmp_path)}
    persist_job_cues(job, [{"id": 1, "english": "Corrected"}])
    import json
    assert json.loads((tmp_path / "cues.json").read_text(encoding="utf-8"))[0]["english"] == "Corrected"


def test_face_landmarks_outside_frame_do_not_crash_registry_scan():
    import numpy as np
    gray = np.zeros((100, 160), dtype=np.uint8)
    # Face and mouth corners extend beyond the right image edge, as YuNet can
    # report on a hard cut/anamorphic border.
    face = np.array([150, 20, 35, 45, 155, 30, 170, 30, 160, 40, 170, 55, 178, 55, 0], dtype=np.float32)
    patch = mouth_patch(gray, face)
    assert patch is None or patch.shape == (20, 40)


def test_face_registry_consolidates_duplicates_and_discards_one_frame_noise():
    import numpy as np
    first = np.zeros(128, dtype=np.float32); first[0] = 1
    duplicate = np.zeros(128, dtype=np.float32); duplicate[0] = .95; duplicate[1] = .1
    different = np.zeros(128, dtype=np.float32); different[2] = 1
    noise = np.zeros(128, dtype=np.float32); noise[3] = 1

    identities, counts = consolidate_identities(
        [first, duplicate, different, noise], [12, 4, 9, 1]
    )

    assert counts == [16, 9]
    assert len(identities) == 2


def test_qwen_silent_window_is_not_a_global_asr_failure():
    assert grouped_cues(None, "Japanese", 12.0, 3, "Qwen test") == []


def test_media_analysis_cache_follows_content_not_filename(tmp_path: Path):
    first = tmp_path / "first.mkv"
    renamed = tmp_path / "renamed-copy.mkv"
    payload = (b"start" * 1000) + (b"middle" * 1000) + (b"end" * 1000)
    first.write_bytes(payload); renamed.write_bytes(payload)
    assert media_fingerprint(first) == media_fingerprint(renamed)
    changed = bytearray(payload); changed[len(changed) // 2] ^= 1
    renamed.write_bytes(changed)
    assert media_fingerprint(first) != media_fingerprint(renamed)


def test_versioned_json_analysis_artifact_round_trip(tmp_path: Path):
    job = tmp_path / "jobs" / "job-1"; job.mkdir(parents=True)
    source = job / "face-registry.json"
    source.write_text('{"version": 2, "identities": []}', encoding="utf-8")
    assert store_json_artifact(job, "abc", "face-v2.json", source, expected_version=2)
    source.unlink()
    assert restore_json_artifact(job, "abc", "face-v2.json", source, expected_version=2)
    assert source.is_file()


def test_completed_legacy_face_scan_migrates_without_rescanning(tmp_path: Path):
    import json
    import numpy as np
    first = np.zeros(128, dtype=np.float32); first[0] = 1
    duplicate = np.zeros(128, dtype=np.float32); duplicate[0] = .98; duplicate[1] = .1
    registry = tmp_path / "face-registry.json"
    registry.write_text(json.dumps({"version": 1, "identities": [
        {"observations": 8, "embedding": first.tolist()},
        {"observations": 3, "embedding": duplicate.tolist()},
    ]}), encoding="utf-8")
    assert upgrade_legacy_face_registry(registry)
    migrated = json.loads(registry.read_text(encoding="utf-8"))
    assert migrated["version"] == 2
    assert len(migrated["identities"]) == 1
    assert migrated["identities"][0]["observations"] == 11


def test_full_film_diarization_defaults_to_bounded_cuda_execution():
    assert diarization_runtime_settings({}) == ("cuda", 2, 8)
    assert diarization_runtime_settings({"DUB_DIARIZATION_DEVICE": "cuda",
                                         "DUB_DIARIZATION_BATCH_SIZE": "99",
                                         "DUB_DIARIZATION_CPU_THREADS": "99"}) == ("cuda", 8, 12)
    assert diarization_runtime_settings({"DUB_DIARIZATION_DEVICE": "invalid",
                                         "DUB_DIARIZATION_BATCH_SIZE": "bad"}) == ("cuda", 4, 8)


def test_gpu_health_csv_is_fail_closed():
    assert gpu_safety.parse_nvidia_status("7000, 8188, 66, 4, 22.5")["free_mb"] == 7000
    with pytest.raises(RuntimeError):
        gpu_safety.parse_nvidia_status("GPU is lost")


def test_job_execution_lock_rejects_a_duplicate_worker(tmp_path: Path):
    first = acquire_job_lock(tmp_path)
    assert first is not None
    try:
        assert acquire_job_lock(tmp_path) is None
    finally:
        release_job_lock(first)
    second = acquire_job_lock(tmp_path)
    assert second is not None
    release_job_lock(second)


def test_gpu_stage_leaves_crash_evident_state_and_commits_safe_handoff(tmp_path: Path, monkeypatch):
    import json
    state_path = tmp_path / "gpu-safety.json"
    health = {"free_mb": 8000, "total_mb": 8188, "temperature_c": 62,
              "utilization": 0, "power_w": 12.0}
    monkeypatch.setattr(gpu_safety, "_state_path", lambda _: state_path)
    monkeypatch.setattr(gpu_safety, "query_nvidia", lambda: health)
    monkeypatch.setattr(gpu_safety, "_cooldown", lambda checkpoint, seconds=8.0: health)
    monkeypatch.setattr(gpu_safety, "_canary", lambda path, checkpoint: gpu_safety._write_state(
        path, {"status": "idle", "last_canary_at": 1, "boot_token": gpu_safety._boot_token()}))

    with gpu_safety.gpu_stage(tmp_path, "test stage", lambda: None):
        assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "active"
    completed = json.loads(state_path.read_text(encoding="utf-8"))
    assert completed["status"] == "idle" and completed["last_stage"] == "test stage"

    with pytest.raises(RuntimeError):
        with gpu_safety.gpu_stage(tmp_path, "failed stage", lambda: None):
            raise RuntimeError("worker failed")
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "interrupted"

    canaries = []
    monkeypatch.setattr(gpu_safety, "_canary", lambda path, checkpoint: (
        canaries.append(True), gpu_safety._write_state(
            path, {"status": "idle", "last_canary_at": 2, "boot_token": gpu_safety._boot_token()})
    )[-1])
    with gpu_safety.gpu_stage(tmp_path, "recovered stage", lambda: None):
        pass
    assert canaries, "an interrupted CUDA stage must force an isolated canary before reuse"


def test_active_gpu_watchdog_stops_a_hot_worker_at_checkpoint(tmp_path: Path, monkeypatch):
    import json
    import time
    state_path = tmp_path / "gpu-safety.json"
    cool = {"free_mb": 8000, "total_mb": 8188, "temperature_c": 60,
            "utilization": 0, "power_w": 12.0}
    hot = {**cool, "temperature_c": 91, "utilization": 99, "power_w": 95.0}
    monkeypatch.setattr(gpu_safety, "_state_path", lambda _: state_path)
    monkeypatch.setattr(gpu_safety, "_watchdog_interval", lambda: .01)
    monkeypatch.setattr(gpu_safety, "query_nvidia", lambda: (
        hot if __import__("threading").current_thread().name == "cuda-thermal-watchdog" else cool
    ))
    monkeypatch.setattr(gpu_safety, "_cooldown", lambda checkpoint, seconds=8.0: cool)
    monkeypatch.setattr(gpu_safety, "_canary", lambda path, checkpoint: gpu_safety._write_state(
        path, {"status": "idle", "last_canary_at": 1, "boot_token": gpu_safety._boot_token()}))

    with pytest.raises(gpu_safety.GPUStageUnsafe, match="91°C"):
        with gpu_safety.gpu_stage(tmp_path, "hot test stage", lambda: None):
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                gpu_safety.gpu_checkpoint()
                time.sleep(.01)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert "91°C" in state["error"]


def test_thermal_pause_auto_resumes_only_after_stable_cooldown(tmp_path: Path, monkeypatch):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create({"id": "thermal", "filename": "film.mkv", "status": "paused",
                  "stage": "Cooling", "progress": 42, "auto_resume_pending": True,
                  "cues": []})
    worker = PipelineWorker(store, tmp_path)
    cool = {"free_mb": 8000, "total_mb": 8188, "temperature_c": 68,
            "utilization": 0, "power_w": 12.0}
    monkeypatch.setattr(pipeline_service, "query_nvidia", lambda: cool)
    monkeypatch.setattr(worker.stopping, "wait", lambda _: False)

    worker._resume_after_cooldown("thermal")

    resumed = store.get_summary("thermal")
    assert resumed["status"] == "queued"
    assert resumed["auto_resume_pending"] is False
    assert worker.jobs.get_nowait() == "thermal"


def test_remux_preserves_source_stream_identity_and_disposition(tmp_path: Path):
    source = tmp_path / "source.mkv"; mix = tmp_path / "mix.flac"; output = tmp_path / "output.mkv"
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=size=64x64:rate=24:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-map", "0:v", "-map", "1:a",
        "-c:v", "ffv1", "-c:a", "flac", "-metadata", "title=Preservation test",
        "-metadata:s:a:0", "language=jpn", "-metadata:s:a:0", "title=Original programme",
        "-disposition:a:0", "default", str(source),
    ], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=220:duration=1", "-c:a", "flac", str(mix)], check=True)
    remux(source, mix, output)
    evidence = media_qc(output, 1.0, source)
    assert evidence["streams_preserved_exactly"]
    assert evidence["metadata_preserved"]
    assert evidence["english_lossless_track"]
    # Track 1 is the dub and it is the track a player opens on; the original
    # keeps every other flag it arrived with.
    assert evidence["english_dub_is_default"]
    assert not evaluate_media_qc(evidence, {"program_gain_db": 0.0})

def test_no_service_module_references_an_unimported_name():
    """Guard against a name that only fails once a worker is deep into a job.

    Every heavy stage runs in its own subprocess, so a missing import is not a
    syntax error and not an import error -- it is a NameError raised minutes or
    hours in, after the GPU work that preceded it has already been spent.
    """
    import ast
    import builtins

    services = sorted(Path("app/services").glob("*.py"))
    assert services, "no service modules were found to check"
    problems = {}
    for module in services:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        bound = set(dir(builtins))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                bound.update(node.names)
        used = {node.id for node in ast.walk(tree)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        if used - bound:
            problems[module.name] = sorted(used - bound)
    assert not problems, f"unimported names: {problems}"

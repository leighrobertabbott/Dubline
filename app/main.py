from __future__ import annotations

import os
import asyncio
import json
import shutil
import uuid
import copy
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import (load_local_environment, media_lookup_summary, remove_hf_token,
                        remove_setting)

load_local_environment()

from app.services.pipeline import PipelineWorker, inspect_system
from app.services.setup_manager import SetupManager
from app.services.media_identity import MediaLibrary
from app.services.subtitles import audio_streams, media_duration, probe_media, subtitle_streams
from app.store import JobStore


BASE = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("DUB_WORKDIR", BASE / "data")).resolve()
DATA.mkdir(parents=True, exist_ok=True)
WEB = BASE / "web"
store = JobStore(DATA / "dubstudio.sqlite3")
worker = PipelineWorker(store, DATA)
setup = SetupManager(BASE, inspect_system)
library = MediaLibrary(store, DATA)


class FileSpec(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    size: int = Field(ge=0)
    kind: Literal["video", "audio", "subtitle", "subtitle_index"]


class JobCreate(BaseModel):
    files: list[FileSpec]
    options: dict = Field(default_factory=dict)


class LocalJobCreate(BaseModel):
    path: str
    options: dict = Field(default_factory=dict)


class MediaProbeRequest(BaseModel):
    path: str


class MediaTrackSelection(BaseModel):
    audio_index: int = Field(ge=0)
    subtitle_index: int | None = Field(default=None, ge=0)


class SetupInstall(BaseModel):
    enhanced_speakers: bool = True
    selective_lip_sync: bool = False
    media_lookup: bool = True


class MediaIdentityChoice(BaseModel):
    provider: Literal["tmdb", "tvmaze"]
    media_type: Literal["movie", "tv"]
    external_id: int = Field(gt=0)


class CueUpdate(BaseModel):
    source: str | None = Field(default=None, min_length=1, max_length=1200)
    english: str | None = Field(default=None, min_length=1, max_length=1200)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, gt=0)
    speaker_id: int | None = Field(default=None, ge=0)
    speaker_name: str | None = Field(default=None, min_length=1, max_length=80)


class CueSplit(BaseModel):
    at: float = Field(gt=0)
    first_text: str = Field(min_length=1, max_length=1200)
    second_text: str = Field(min_length=1, max_length=1200)


def safe_name(name: str) -> str:
    name = Path(name).name.strip().replace("\x00", "")
    return name[:255] or "source.bin"


def public_project(project: dict | None) -> dict | None:
    if not project:
        return None
    result = {key: project.get(key) for key in (
        "id", "title", "year", "media_type", "provider", "external_id", "overview",
        "site_url", "poster_ready", "created_at", "updated_at",
    )}
    result["poster_url"] = f"/api/projects/{project['id']}/poster" if project.get("poster_ready") else None
    return result


def public_job(job: dict) -> dict:
    result = copy.deepcopy(job)
    result.pop("input_path", None)
    result.pop("folder", None)
    for upload in result.get("uploads", []):
        upload.pop("path", None)
    if result.get("project_id"):
        result["project"] = public_project(store.get_project(result["project_id"]))
    return result


def require_local_studio() -> None:
    if not setup.snapshot()["ready"]:
        raise HTTPException(409, "Finish local studio setup before starting a film")


def invalidate_cue_artifacts(folder: Path, line: int) -> None:
    """Remove every per-line take and every aggregate derived from that take."""
    per_line = (
        folder / "generated" / f"{line:06d}.wav", folder / "fitted" / f"{line:06d}.wav",
        folder / "acoustically-matched" / f"{line:06d}.wav",
        folder / "qwen-generated" / f"{line:06d}.wav",
        folder / "qwen-fitted" / f"{line:06d}.wav",
    )
    existing = [path for path in per_line if path.is_file()]
    if existing:
        take = folder / "take-history" / f"{line:06d}" / str(time.time_ns())
        take.mkdir(parents=True, exist_ok=True)
        for path in existing:
            shutil.move(str(path), str(take / f"{path.parent.name}.wav"))
    for path in (
        folder / "emotion-references" / f"{line:06d}.wav", folder / "english-dialogue.flac",
        folder / "english-mix.flac", folder / "dubbed-english.mkv", folder / "qc-report.json",
        folder / "qc-report.html", folder / "qc-backtranscription.json",
    ):
        path.unlink(missing_ok=True)


def persist_job_cues(job: dict, cues: list[dict]) -> None:
    """Keep the resumable cue document in lockstep with the database copy."""
    folder = Path(job["folder"])
    (folder / "cues.json").write_text(
        __import__("json").dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    library.start()
    worker.start()
    try:
        yield
    finally:
        setup.shutdown()
        library.stop()
        worker.stop()


app = FastAPI(title="Dubline", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def local_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
    )
    return response


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


@app.get("/faq")
async def faq():
    return FileResponse(WEB / "faq.html")


@app.get("/api/system")
async def system_status():
    return inspect_system()


@app.get("/api/setup")
async def setup_status():
    return setup.snapshot()


@app.post("/api/setup/token")
async def configure_huggingface_token(request: Request):
    raw = await request.body()
    if len(raw) > 2048:
        raise HTTPException(413, "The token entry was unexpectedly large")
    try:
        token = json.loads(raw).get("token")
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, "Enter a valid Hugging Face access token") from exc
    if not isinstance(token, str):
        raise HTTPException(400, "Enter a valid Hugging Face access token")
    try:
        return await asyncio.to_thread(setup.save_token, token)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.delete("/api/setup/token")
async def forget_huggingface_token():
    remove_hf_token()
    return {"configured": False, "display": None}


@app.post("/api/setup/install")
async def install_setup(spec: SetupInstall):
    try:
        library.set_enabled(spec.media_lookup)
        return setup.start(include_diarization=spec.enhanced_speakers,
                           include_lip_sync=spec.selective_lip_sync)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/setup/cancel")
async def cancel_setup():
    return setup.cancel()


@app.get("/api/library/settings")
async def library_settings():
    return media_lookup_summary()


@app.post("/api/library/token")
async def configure_tmdb_token(request: Request):
    raw = await request.body()
    if len(raw) > 4096:
        raise HTTPException(413, "The token entry was unexpectedly large")
    try:
        token = json.loads(raw).get("token")
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, "Enter a valid TMDB API Read Access Token") from exc
    if not isinstance(token, str):
        raise HTTPException(400, "Enter a valid TMDB API Read Access Token")
    try:
        return await asyncio.to_thread(library.save_tmdb_token, token)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.delete("/api/library/token")
async def forget_tmdb_token():
    remove_setting("TMDB_TOKEN")
    return media_lookup_summary()


@app.get("/api/library/search")
async def search_media(q: str, media_type: Literal["movie", "tv"] | None = None,
                       year: int | None = None):
    try:
        return await asyncio.to_thread(library.search, q, media_type, year)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/jobs/{job_id}/identify")
async def identify_job(job_id: str, choice: MediaIdentityChoice):
    store.get_or_404(job_id)
    try:
        await asyncio.to_thread(library.choose, job_id, choice.provider, choice.media_type, choice.external_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return public_job(store.get_or_404(job_id))


@app.get("/api/projects/{project_id}/poster")
async def project_poster(project_id: str):
    if not store.get_project(project_id):
        raise HTTPException(404, "Project not found")
    path = library.poster_path(project_id)
    if not path:
        raise HTTPException(404, "This project does not have cached artwork")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=86400",
                                       "X-Content-Type-Options": "nosniff"})


@app.get("/api/jobs")
async def list_jobs():
    return [public_job(item) for item in store.list_summaries(limit=50)]


@app.post("/api/media/probe")
async def probe_local_media(spec: MediaProbeRequest):
    source = Path(spec.path).expanduser().resolve()
    if not source.is_file():
        raise HTTPException(404, "That local video file was not found")
    info = probe_media(source)
    duration = media_duration(info)
    video = next((stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"), None)
    if duration <= 0 or not audio_streams(info):
        raise HTTPException(400, "That file does not contain readable audio or video")
    return {"duration": duration, "width": video.get("width") if video else None,
            "height": video.get("height") if video else None,
            "audio_streams": audio_streams(info), "subtitle_streams": subtitle_streams(info)}


@app.post("/api/jobs")
async def create_job(spec: JobCreate):
    require_local_studio()
    sources = [item for item in spec.files if item.kind in {"video", "audio"}]
    if len(sources) != 1:
        raise HTTPException(400, "Select exactly one source video or audio programme")
    incoming = sum(item.size for item in spec.files)
    free = shutil.disk_usage(DATA).free
    required = int(incoming * 2.2) + 2 * 1024 ** 3
    if free < required:
        raise HTTPException(507, f"Upload needs approximately {required / 1024 ** 3:.1f} GB of free working space, "
                                 f"but only {free / 1024 ** 3:.1f} GB is available")
    options = normalized_options(spec.options)
    if not options["voice_rights_confirmed"]:
        raise HTTPException(400, "Confirm that you have permission to dub the media and reproduce its voices")
    job_id = uuid.uuid4().hex[:12]
    folder = DATA / "jobs" / job_id
    upload_dir = folder / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=False)
    uploads = []
    for index, item in enumerate(spec.files):
        name = safe_name(item.name)
        file_id = f"f{index + 1}"
        path = upload_dir / f"{file_id}-{name}"
        path.touch()
        uploads.append({"id": file_id, "name": name, "kind": item.kind, "size": item.size,
                        "received": 0, "path": str(path)})
    job = store.create({
        "id": job_id,
        "filename": sources[0].name,
        "folder": str(folder),
        "status": "uploading",
        "stage": "Uploading source files",
        "progress": 0,
        "uploads": uploads,
        "options": options,
        "cues": [],
        "logs": ["Job created"],
    })
    library.attach(job_id, sources[0].name)
    return public_job(store.get_or_404(job_id))


@app.put("/api/jobs/{job_id}/files/{file_id}")
async def upload_chunk(job_id: str, file_id: str, request: Request):
    job = store.get(job_id)
    if not job or job.get("status") != "uploading":
        raise HTTPException(409, "This upload is not active")
    upload = next((item for item in job["uploads"] if item["id"] == file_id), None)
    if not upload:
        raise HTTPException(404, "Upload file not found")
    expected_offset = int(upload["received"])
    try:
        supplied_offset = int(request.headers.get("Upload-Offset", expected_offset))
    except ValueError as exc:
        raise HTTPException(400, "Invalid upload offset") from exc
    if supplied_offset != expected_offset:
        return JSONResponse({"offset": expected_offset}, status_code=409)
    path = Path(upload["path"])
    with path.open("ab") as output:
        async for chunk in request.stream():
            if chunk:
                output.write(chunk)
    upload["received"] = path.stat().st_size
    if upload["received"] > upload["size"]:
        raise HTTPException(413, "Received more data than expected")
    total = sum(item["size"] for item in job["uploads"]) or 1
    received = sum(item["received"] for item in job["uploads"])
    job = store.update(job_id, uploads=job["uploads"], progress=round(received / total * 100, 1))
    return {"offset": upload["received"], "job": public_job(job)}


@app.post("/api/jobs/{job_id}/finalize")
async def finalize_upload(job_id: str):
    job = store.get_or_404(job_id)
    incomplete = [item["name"] for item in job.get("uploads", []) if item["received"] != item["size"]]
    if incomplete:
        raise HTTPException(409, f"Upload incomplete: {', '.join(incomplete)}")
    # Reconstitute uploaded VobSub .idx/.sub pairs under one shared basename so
    # FFmpeg can open the bitmap subtitle pair correctly.
    by_stem: dict[str, dict[str, dict]] = {}
    for item in job.get("uploads", []):
        suffix = Path(item["name"]).suffix.lower()
        if suffix in {".sub", ".idx"}:
            by_stem.setdefault(Path(item["name"]).stem.lower(), {})[suffix] = item
    pair_dir = Path(job["folder"]) / "sidecars"; pair_dir.mkdir(exist_ok=True)
    for stem, pair in by_stem.items():
        if not {".sub", ".idx"}.issubset(pair):
            continue
        for suffix, item in pair.items():
            target = pair_dir / f"{safe_name(stem)}{suffix}"
            shutil.copy2(item["path"], target)
            item["path"] = str(target)
    if by_stem:
        store.update(job_id, uploads=job["uploads"])
    source = next(item for item in job["uploads"] if item["kind"] in {"video", "audio"})
    library.attach(job_id, job.get("filename") or source["name"], Path(source["path"]))
    uploaded_probe = probe_media(Path(source["path"]))
    tracks = audio_streams(uploaded_probe)
    text_subtitles = [item for item in subtitle_streams(uploaded_probe) if item.get("text")]
    needs_audio = len(tracks) > 1 and job.get("options", {}).get("audio_stream_index") is None
    needs_subtitle = len(text_subtitles) > 1 and job.get("options", {}).get("subtitle_stream_index") is None
    if needs_audio or needs_subtitle:
        job = store.update(job_id, input_path=source["path"], status="awaiting_selection",
                           stage="Choose the programme tracks", progress=100,
                           media_selection={"audio_streams": tracks,
                                            "subtitle_streams": text_subtitles})
        return public_job(job)
    job = store.update(job_id, input_path=source["path"], status="queued", stage="Waiting for GPU worker", progress=0)
    worker.submit(job_id)
    return public_job(job)


@app.post("/api/jobs/{job_id}/media-tracks")
async def select_job_media_tracks(job_id: str, selection: MediaTrackSelection):
    job = store.get_or_404(job_id)
    if job.get("status") != "awaiting_selection":
        raise HTTPException(409, "This job is not waiting for a soundtrack choice")
    tracks = job.get("media_selection", {}).get("audio_streams", [])
    if not any(int(item["index"]) == selection.audio_index for item in tracks):
        raise HTTPException(400, "That audio stream does not exist")
    subtitles = job.get("media_selection", {}).get("subtitle_streams", [])
    if selection.subtitle_index is not None and not any(
            int(item["index"]) == selection.subtitle_index for item in subtitles):
        raise HTTPException(400, "That subtitle stream does not exist or is not text-readable")
    options = {**job.get("options", {}), "audio_stream_index": selection.audio_index,
               "subtitle_stream_index": selection.subtitle_index}
    job = store.update(job_id, options=options, status="queued", stage="Waiting for GPU worker", progress=0)
    worker.submit(job_id)
    return public_job(job)


@app.post("/api/jobs/local")
async def create_local_job(spec: LocalJobCreate):
    require_local_studio()
    source = Path(spec.path).expanduser().resolve()
    if not source.is_file():
        raise HTTPException(404, "That local video file was not found")
    options = normalized_options(spec.options)
    if not options["voice_rights_confirmed"]:
        raise HTTPException(400, "Confirm that you have permission to dub the media and reproduce its voices")
    job_id = uuid.uuid4().hex[:12]
    folder = DATA / "jobs" / job_id
    folder.mkdir(parents=True)
    sidecars = []
    for suffix in (".srt", ".ass", ".ssa", ".vtt", ".sub", ".idx"):
        candidate = source.with_suffix(suffix)
        if candidate.exists():
            sidecars.append(str(candidate))
    job = store.create({
        "id": job_id, "filename": source.name, "folder": str(folder), "input_path": str(source),
        "status": "queued", "stage": "Waiting for GPU worker", "progress": 0,
        "uploads": [], "sidecars": sidecars, "options": options,
        "cues": [], "logs": ["Local source added without copying"],
    })
    library.attach(job_id, source.name, source)
    worker.submit(job_id)
    return public_job(store.get_or_404(job_id))


@app.post("/api/jobs/{job_id}/control/{action}")
async def control_job(job_id: str, action: Literal["pause", "resume", "cancel"]):
    should_submit = False
    with store.lock:
        job = store.get_or_404(job_id)
        if action == "resume":
            if job["status"] not in {"paused", "error"}:
                raise HTTPException(409, "Only paused or failed jobs can be resumed")
            job = store.update(job_id, status="queued", stage="Waiting to resume", error=None,
                               auto_resume_pending=False,
                               analysis_approved=True if job.get("stage") == "Translation ready for approval" else job.get("analysis_approved"))
            should_submit = True
        elif action == "pause":
            if job["status"] not in {"queued", "processing"}:
                raise HTTPException(409, "This job is not running")
            if job["status"] == "queued":
                job = store.update(job_id, status="paused", stage="Paused before processing", control=None)
            else:
                job = store.update(job_id, control="pause", stage="Pausing safely")
        else:
            if job["status"] in {"complete", "needs_review", "cancelled"}:
                raise HTTPException(409, "This job is no longer running")
            if job["status"] == "processing":
                job = store.update(job_id, control="cancel", stage="Cancelling")
            else:
                job = store.update(job_id, status="cancelled", stage="Cancelled", control=None,
                                   auto_resume_pending=False)
    if should_submit:
        worker.submit(job_id)
    return public_job(store.get_or_404(job_id))


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    job = store.get_or_404(job_id)
    if job["status"] == "processing":
        raise HTTPException(409, "Cancel the running job before removing it")
    folder = Path(job.get("folder", ""))
    store.delete(job_id)
    if folder.is_dir() and DATA in folder.parents:
        shutil.rmtree(folder, ignore_errors=True)
    return {"deleted": True}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    return public_job(store.get_or_404(job_id))


@app.patch("/api/jobs/{job_id}/cues/{cue_id}")
async def edit_cue(job_id: str, cue_id: int, change: CueUpdate):
    job = store.get_or_404(job_id)
    if job.get("status") in {"processing", "queued", "uploading"}:
        raise HTTPException(409, "Pause the job before editing a line")
    cues = list(job.get("cues", []))
    cue = next((item for item in cues if int(item.get("id", -1)) == cue_id), None)
    if cue is None:
        raise HTTPException(404, "Voice line not found")
    values = change.model_dump(exclude_none=True)
    if "source" in values:
        cue["source"] = values["source"].strip()
        for key in ("faithful_translation", "literal_translation", "translation_qc",
                    "adaptation_confidence", "translation_candidates"):
            cue.pop(key, None)
    if "english" in values:
        cue["english"] = values["english"].strip()
        cue["adapted_dialogue"] = cue["english"]
        cue.pop("emotion_vector", None)
        cue.pop("translation_qc", None)
        cue.pop("adaptation_confidence", None)
    if "start" in values:
        cue["start"] = round(values["start"], 3)
    if "end" in values:
        cue["end"] = round(values["end"], 3)
    if float(cue["end"]) <= float(cue["start"]):
        raise HTTPException(400, "Line end must be after its start")
    if "speaker_id" in values:
        old_spk = cue.get("speaker_id")
        cue["speaker_id"] = values["speaker_id"]
        cue["speaker"] = "Uncertain voice" if values["speaker_id"] == 0 else f"Voice {values['speaker_id']}"
        cue["speaker_confidence"] = 1.0  # Explicit human assignment
        cue["speaker_assignment"] = "human correction"
        if old_spk != values["speaker_id"]:
            folder = Path(job["folder"])
            for spk in (old_spk, values["speaker_id"]):
                if spk is not None:
                    (folder / "speaker-references" / f"voice-{int(spk):03d}.wav").unlink(missing_ok=True)
                    (folder / "speaker-references" / f"voice-{int(spk):03d}.json").unlink(missing_ok=True)
    if "speaker_name" in values:
        new_name = values["speaker_name"].strip()
        speaker_id = cue.get("speaker_id")
        # A character name is project-wide. Rename every line already assigned
        # to the same voice instead of silently changing only one subtitle card.
        for item in cues:
            if speaker_id is not None and item.get("speaker_id") == speaker_id:
                item["speaker"] = new_name
        cue["speaker"] = new_name
    cue["status"] = "edited"
    cue["needs_review"] = False
    cue["review_reasons"] = []
    persist_job_cues(job, cues)
    job = store.update(job_id, cues=cues, status="paused", stage="Cue edit ready to regenerate", control=None)
    return public_job(job)


@app.post("/api/jobs/{job_id}/cues/{cue_id}/split")
async def split_cue(job_id: str, cue_id: int, change: CueSplit):
    job = store.get_or_404(job_id)
    if job.get("status") != "paused":
        raise HTTPException(409, "Pause the job before changing cue structure")
    cues = list(job.get("cues", []))
    index = next((i for i, item in enumerate(cues) if int(item.get("id", -1)) == cue_id), None)
    if index is None:
        raise HTTPException(404, "Voice line not found")
    cue = cues[index]
    if not float(cue["start"]) + .08 < change.at < float(cue["end"]) - .08:
        raise HTTPException(400, "Split time must be inside the cue")
    first = {**cue, "end": round(change.at, 3), "english": change.first_text.strip(),
             "adapted_dialogue": change.first_text.strip(), "status": "edited"}
    second = {**cue, "start": round(change.at, 3), "english": change.second_text.strip(),
              "adapted_dialogue": change.second_text.strip(), "status": "edited"}
    for item in (first, second):
        item.pop("qc", None); item.pop("translation_qc", None)
    cues[index:index + 1] = [first, second]
    for number, item in enumerate(cues, 1): item["id"] = number
    invalidate_all_takes(Path(job["folder"]))
    persist_job_cues(job, cues)
    return public_job(store.update(job_id, cues=cues, stage="Cue structure edited · regeneration required"))


@app.post("/api/jobs/{job_id}/cues/{cue_id}/merge-next")
async def merge_cue(job_id: str, cue_id: int):
    job = store.get_or_404(job_id)
    if job.get("status") != "paused":
        raise HTTPException(409, "Pause the job before changing cue structure")
    cues = list(job.get("cues", []))
    index = next((i for i, item in enumerate(cues) if int(item.get("id", -1)) == cue_id), None)
    if index is None or index + 1 >= len(cues):
        raise HTTPException(404, "There is no following cue to merge")
    merged = {**cues[index], "end": cues[index + 1]["end"],
              "english": f"{cues[index].get('english','')} {cues[index + 1].get('english','')}".strip(),
              "source": f"{cues[index].get('source','')} {cues[index + 1].get('source','')}".strip(),
              "status": "edited"}
    merged["adapted_dialogue"] = merged["english"]
    merged.pop("qc", None); merged.pop("translation_qc", None)
    cues[index:index + 2] = [merged]
    for number, item in enumerate(cues, 1): item["id"] = number
    invalidate_all_takes(Path(job["folder"]))
    persist_job_cues(job, cues)
    return public_job(store.update(job_id, cues=cues, stage="Cue structure edited · regeneration required"))


def invalidate_all_takes(folder: Path) -> None:
    for name in ("generated", "fitted", "acoustically-matched", "qwen-generated", "qwen-fitted",
                 "emotion-references", "references", "speaker-references"):
        shutil.rmtree(folder / name, ignore_errors=True)
    for name in ("english-dialogue.flac", "english-mix.flac", "dubbed-english.mkv",
                 "qc-report.json", "qc-report.html", "qc-backtranscription.json"):
        (folder / name).unlink(missing_ok=True)


@app.get("/api/jobs/{job_id}/cues/{cue_id}/takes")
async def list_cue_takes(job_id: str, cue_id: int):
    job = store.get_or_404(job_id)
    root = Path(job["folder"]) / "take-history" / f"{cue_id:06d}"
    return [{"id": path.name, "files": sorted(item.name for item in path.glob("*.wav"))}
            for path in sorted(root.glob("*"), reverse=True) if path.is_dir()]


@app.post("/api/jobs/{job_id}/cues/{cue_id}/takes/{take_id}/restore")
async def restore_cue_take(job_id: str, cue_id: int, take_id: str):
    job = store.get_or_404(job_id)
    if job.get("status") not in {"paused", "error", "complete", "needs_review"}:
        raise HTTPException(409, "Pause the job before restoring a prior take")
    cues = list(job.get("cues", []))
    cue_index = next((index for index, item in enumerate(cues) if int(item.get("id", -1)) == cue_id), None)
    if cue_index is None or not take_id.isdigit():
        raise HTTPException(404, "Prior take not found")
    folder = Path(job["folder"]).resolve(); line = cue_index + 1
    take = (folder / "take-history" / f"{line:06d}" / take_id).resolve()
    if folder not in take.parents or not take.is_dir():
        raise HTTPException(404, "Prior take not found")
    saved = {path.stem: path for path in take.glob("*.wav")}
    if not saved:
        raise HTTPException(404, "Prior take is empty")
    invalidate_cue_artifacts(folder, line)
    for directory, path in saved.items():
        target_dir = folder / directory; target_dir.mkdir(exist_ok=True)
        shutil.copy2(path, target_dir / f"{line:06d}.wav")
    cues[cue_index]["status"] = "waiting"; cues[cue_index].pop("qc", None)
    persist_job_cues(job, cues)
    job = store.update(job_id, cues=cues, status="queued", stage=f"Restoring prior take for line {cue_id}",
                       error=None, analysis_approved=True, output_path=None, qc=None)
    worker.submit(job_id)
    return public_job(job)


@app.post("/api/jobs/{job_id}/cues/{cue_id}/regenerate")
async def regenerate_cue(job_id: str, cue_id: int):
    job = store.get_or_404(job_id)
    if job.get("status") not in {"paused", "error", "complete", "needs_review"}:
        raise HTTPException(409, "Pause the job before regenerating a line")
    cues = list(job.get("cues", []))
    cue_index = next((index for index, item in enumerate(cues) if int(item.get("id", -1)) == cue_id), None)
    if cue_index is None:
        raise HTTPException(404, "Voice line not found")
    folder = Path(job["folder"]).resolve()
    if DATA not in folder.parents:
        raise HTTPException(400, "Invalid job workspace")
    line = cue_index + 1
    invalidate_cue_artifacts(folder, line)
    cues[cue_index].pop("qc", None)
    cues[cue_index]["status"] = "waiting"
    (folder / "cues.json").write_text(__import__('json').dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
    job = store.update(job_id, cues=cues, status="queued", stage=f"Regenerating line {cue_id}",
                       error=None, analysis_approved=True, output_path=None, qc=None)
    worker.submit(job_id)
    return public_job(job)


@app.get("/api/jobs/{job_id}/download")
async def download(job_id: str):
    job = store.get_or_404(job_id)
    output = Path(job.get("output_path", ""))
    if job.get("status") not in {"complete", "needs_review"} or not output.is_file():
        raise HTTPException(409, "The dubbed video is not ready")
    return FileResponse(output, filename=f"{Path(job['filename']).stem}.english.dub.mkv",
                        media_type="video/x-matroska")


@app.get("/api/jobs/{job_id}/qc")
async def download_qc(job_id: str):
    job = store.get_or_404(job_id)
    report = Path(job.get("qc_report_html", ""))
    if not report.is_file():
        raise HTTPException(409, "The quality-control report is not ready")
    return FileResponse(report, filename=f"{Path(job['filename']).stem}.qc.html", media_type="text/html")


@app.get("/api/jobs/{job_id}/export/{kind}")
async def download_export(job_id: str, kind: Literal["srt", "csv", "edl", "clips", "mix", "dialogue"]):
    job = store.get_or_404(job_id)
    path = Path((job.get("exports") or {}).get(kind, ""))
    if not path.is_file():
        raise HTTPException(409, "That export is not ready")
    media_type = {"srt": "application/x-subrip", "csv": "text/csv", "edl": "text/plain",
                  "clips": "application/zip", "mix": "audio/flac", "dialogue": "audio/flac"}[kind]
    return FileResponse(path, filename=path.name, media_type=media_type)


def normalized_options(values: dict) -> dict:
    subtitle_mode = values.get("subtitle_mode", "auto")
    audio_mode = values.get("audio_mode", "separate")
    engine = values.get("engine", os.getenv("DUB_ENGINE", "indextts"))
    def optional_seconds(name: str) -> float | None:
        value = values.get(name)
        if value in (None, ""):
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{name.replace('_', ' ').title()} is not a valid time")
        if seconds < 0:
            raise HTTPException(400, f"{name.replace('_', ' ').title()} cannot be negative")
        return round(seconds, 3)

    range_start = optional_seconds("range_start")
    range_end = optional_seconds("range_end")
    if range_start is not None and range_end is not None and range_end <= range_start:
        raise HTTPException(400, "The selected section end must be after its start")
    glossary_values = values.get("glossary") or {}
    if not isinstance(glossary_values, dict):
        raise HTTPException(400, "Glossary must be a term-to-pronunciation object")
    return {
        "subtitle_mode": subtitle_mode if subtitle_mode in {"auto", "embedded", "sidecar", "speech"} else "auto",
        "audio_mode": audio_mode if audio_mode in {"separate", "duck", "replace"} else "separate",
        "engine": engine if engine in {"indextts", "preview"} else "indextts",
        "emotion_mode": values.get("emotion_mode") if values.get("emotion_mode") in {"auto", "source", "text", "neutral"} else "auto",
        "source_language": str(values.get("source_language", "auto")),
        "target_language": str(values.get("target_language", "English")),
        "workflow_mode": values.get("workflow_mode") if values.get("workflow_mode") in {"automatic", "review", "approval"} else "automatic",
        "mastering_preset": values.get("mastering_preset") if values.get("mastering_preset") in {"cinema", "broadcast", "web", "preserve"} else "cinema",
        "whisper_model": str(values.get("whisper_model", os.getenv("WHISPER_MODEL", "turbo"))),
        "range_start": range_start,
        "range_end": range_end,
        "audio_stream_index": (int(values["audio_stream_index"])
                               if values.get("audio_stream_index") not in (None, "") else None),
        "subtitle_stream_index": (int(values["subtitle_stream_index"])
                                  if values.get("subtitle_stream_index") not in (None, "") else None),
        "glossary": {str(key).strip(): str(value).strip() for key, value in
                     glossary_values.items() if str(key).strip() and str(value).strip()},
        "voice_rights_confirmed": bool(values.get("voice_rights_confirmed")),
    }

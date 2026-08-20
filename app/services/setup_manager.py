from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import shutil
import signal
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.config import hf_token_summary, save_hf_token, validate_hf_token
from app.services.subprocess_control import terminate_process


@dataclass(frozen=True)
class SetupComponent:
    key: str
    name: str
    description: str
    checks: tuple[str, ...]
    estimated_gb: float
    required: bool = True
    needs_token: bool = False


COMPONENTS = (
    SetupComponent("media_tools", "Video tools", "Reads, trims, mixes and delivers your films.",
                   ("ffmpeg", "ffprobe"), .2),
    SetupComponent("voice_model", "English voice studio", "The main expressive voice-cloning model.",
                   ("model_ready",), 12.0),
    SetupComponent("separation", "Dialogue separation", "Keeps dialogue apart from music and effects.",
                   ("whisper_ready", "separator_ready", "recovery_ready", "roformer_ready"), 4.0),
    SetupComponent("speech", "Speech understanding", "Transcription and word-perfect timing alignment.",
                   ("asr_ready", "asr_escalation_ready", "aligner_ready"), 17.0),
    SetupComponent("language", "Translation and QC", "Natural English adaptation with an independent check.",
                   ("adapter_ready", "translation_qc_ready"), 10.0),
    SetupComponent("voice_tools", "Voice safety net", "Face/voice matching and a fallback voice engine.",
                   ("visual_speaker_ready", "tts_fallback_ready"), 13.0),
    SetupComponent("diarization", "Enhanced speaker detection",
                   "Optional gated model for difficult multi-speaker scenes.",
                   ("diarization_ready",), 1.0, required=False, needs_token=True),
    SetupComponent("lip_sync", "Selective lip sync", "Optional finishing model for a few clear close-ups.",
                   ("musetalk_ready",), 17.0, required=False),
)


def validate_huggingface_identity(token: str, timeout: float = 12.0) -> str:
    token = validate_hf_token(token)
    request = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "Dubline setup"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise ValueError("Hugging Face did not accept that token") from exc
        raise RuntimeError("Hugging Face could not verify the token right now") from exc
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not reach Hugging Face to verify the token") from exc
    name = str(payload.get("name") or payload.get("fullname") or "Hugging Face user").strip()
    access_request = urllib.request.Request(
        "https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/config.yaml",
        method="HEAD",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "Dubline setup"},
    )
    try:
        with urllib.request.urlopen(access_request, timeout=timeout) as response:
            if response.status >= 400:
                raise ValueError("Accept the speaker model's access conditions before connecting")
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise ValueError("Accept the speaker model's access conditions before connecting") from exc
        raise RuntimeError("Hugging Face could not check speaker-model access right now") from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Could not reach Hugging Face to check speaker-model access") from exc
    return name[:120]


class SetupManager:
    def __init__(self, base: Path, system_inspector: Callable[[], dict]):
        self.base = base.resolve()
        self.system_inspector = system_inspector
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen | None = None
        self.phase = "idle"
        self.active_component: str | None = None
        self.detail = "Checking this PC"
        self.progress = 0.0
        self.error: str | None = None
        self.logs: list[str] = []
        self.cancel_requested = False
        self.state_file = self.base / "data" / "setup-state.json"
        self.install_lock_file = self.base / "data" / "setup-install.lock"
        self._restore_state()

    def _restore_state(self) -> None:
        if not self.state_file.is_file():
            return
        try:
            saved = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if saved.get("phase") == "installing":
            self.detail = "Previous setup was interrupted — completed downloads are safe"
            self.logs = ["Ready to continue from the last completed component"]

    def _persist_state(self) -> None:
        with self.lock:
            payload = {
                "phase": self.phase, "active_component": self.active_component,
                "detail": self.detail, "progress": self.progress, "error": self.error,
                "logs": self.logs[-30:],
            }
            # start() and the installer thread can persist almost
            # simultaneously.  Keep the atomic replace inside the same lock so
            # Windows never sees two writers racing the destination handle.
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix="setup-state-", suffix=".json", dir=self.state_file.parent, text=True,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump(payload, output, ensure_ascii=False, indent=2)
                Path(temporary_name).replace(self.state_file)
            finally:
                Path(temporary_name).unlink(missing_ok=True)

    def _acquire_install_lock(self):
        self.install_lock_file.parent.mkdir(parents=True, exist_ok=True)
        handle = self.install_lock_file.open("a+b")
        handle.seek(0)
        if handle.read(1) == b"":
            handle.seek(0); handle.write(b"0"); handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise RuntimeError("Another Dubline setup is already running") from exc
        return handle

    @staticmethod
    def _release_install_lock(handle) -> None:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _component_ready(self, component: SetupComponent, system: dict) -> bool:
        if component.key == "media_tools":
            for name in ("ffmpeg", "ffprobe"):
                executable = shutil.which(name)
                if not executable:
                    return False
                try:
                    result = subprocess.run([executable, "-version"], capture_output=True,
                                            text=True, timeout=8, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    return False
                if result.returncode:
                    return False
        return all(bool(system.get(check)) for check in component.checks)

    def snapshot(self, system: dict | None = None) -> dict:
        system = system or self.system_inspector()
        token = hf_token_summary()
        os_name = platform.system().lower()
        architecture = platform.machine().lower()
        supported_os = os_name in {"windows", "linux"}
        supported_architecture = architecture in {"amd64", "x86_64"}
        platform_supported = supported_os and supported_architecture
        if os_name == "darwin":
            compatibility = (
                "This Mac uses Apple's Metal graphics platform. Dubline currently needs an NVIDIA CUDA GPU, "
                "so model installation is unavailable on macOS."
            )
        elif not supported_os:
            compatibility = "Dubline currently supports Windows and Linux with NVIDIA CUDA."
        elif not supported_architecture:
            compatibility = (
                f"{platform.machine()} was detected. This release supports 64-bit Intel/AMD Windows and Linux."
            )
        elif not system.get("cuda"):
            compatibility = "No working NVIDIA CUDA GPU was detected. Check the NVIDIA driver, then scan again."
        else:
            compatibility = "Compatible NVIDIA CUDA system detected."
        components = [{
            "key": component.key,
            "name": component.name,
            "description": component.description,
            "ready": self._component_ready(component, system),
            "required": component.required,
            "needs_token": component.needs_token,
            "estimated_gb": component.estimated_gb,
        } for component in COMPONENTS]
        required = [item for item in components if item["required"]]
        missing_required = [item for item in required if not item["ready"]]
        missing_gb = round(sum(float(item["estimated_gb"]) for item in missing_required), 1)
        disk_free_gb = round(shutil.disk_usage(self.base).free / 1024 ** 3, 1)
        system_ready = (platform_supported and bool(system.get("cuda"))
                        and disk_free_gb >= missing_gb + 3)
        with self.lock:
            phase = self.phase
            active = self.active_component
            detail = self.detail
            progress = self.progress
            error = self.error
            logs = list(self.logs[-30:])
            running = bool(self.thread and self.thread.is_alive())
        return {
            "phase": phase,
            "running": running,
            "ready": not missing_required and bool(system.get("cuda")) and platform_supported,
            "first_run": bool(missing_required),
            "can_install": system_ready and not running,
            "active_component": active,
            "detail": detail,
            "progress": progress,
            "error": error,
            "logs": logs,
            "components": components,
            "missing_download_gb": missing_gb,
            "token": token,
            "system": {
                "cuda": bool(system.get("cuda")),
                "gpu": system.get("gpu", "No CUDA GPU detected"),
                "disk_free_gb": disk_free_gb,
                "enough_disk": disk_free_gb >= missing_gb + 3,
            },
            "platform": {
                "os": "macOS" if os_name == "darwin" else platform.system(),
                "architecture": platform.machine(),
                "supported": platform_supported,
                "message": compatibility,
            },
        }

    def save_token(self, token: str) -> dict:
        account = validate_huggingface_identity(token)
        save_hf_token(token)
        with self.lock:
            self.logs.append(f"Hugging Face connected as {account}")
        return {**hf_token_summary(), "account": account}

    def start(self, *, include_diarization: bool, include_lip_sync: bool) -> dict:
        current = self.snapshot()
        if current["running"]:
            return current
        if not current["platform"]["supported"]:
            raise ValueError(current["platform"]["message"])
        if not current["system"]["cuda"]:
            raise ValueError("Dubline needs an NVIDIA CUDA GPU before models can be installed")
        if not current["system"]["enough_disk"]:
            raise ValueError(
                f"Free at least {current['missing_download_gb'] + 3:.1f} GB on this drive before installing"
            )
        token_configured = bool(current["token"]["configured"])
        if include_diarization and not token_configured:
            raise ValueError("Connect Hugging Face before adding enhanced speaker detection")
        selected = {component.key for component in COMPONENTS if component.required}
        if include_diarization:
            selected.add("diarization")
        if include_lip_sync:
            selected.add("lip_sync")
        pending = [component for component in COMPONENTS
                   if component.key in selected and not self._component_ready(component, self.system_inspector())]
        requested_gb = round(sum(component.estimated_gb for component in pending), 1)
        available_gb = round(shutil.disk_usage(self.base).free / 1024 ** 3, 1)
        if available_gb < requested_gb + 3:
            raise ValueError(
                f"This selection needs about {requested_gb + 3:.1f} GB free, but {available_gb:.1f} GB is available"
            )
        with self.lock:
            self.phase = "installing"
            self.active_component = pending[0].key if pending else None
            self.detail = "Preparing your local studio" if pending else "Everything is already installed"
            self.progress = 0.0 if pending else 100.0
            self.error = None
            self.logs = []
            self.cancel_requested = False
            self.thread = threading.Thread(
                target=self._install, args=(pending,), name="dubline-first-run-setup", daemon=True,
            )
            self.thread.start()
        self._persist_state()
        return self.snapshot()

    def cancel(self) -> dict:
        with self.lock:
            self.cancel_requested = True
            process = self.process
            self.detail = "Stopping safely"
        if process and process.poll() is None:
            self._terminate_process_tree(process)
        return self.snapshot()

    def shutdown(self) -> None:
        with self.lock:
            thread = self.thread
        if thread and thread.is_alive():
            self.cancel()
            thread.join(timeout=15)

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=8)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if process.poll() is None:
            terminate_process(process)

    def _install(self, pending: list[SetupComponent]) -> None:
        install_lock = None
        if not pending:
            with self.lock:
                self.phase = "complete"
                self.detail = "Your local dubbing studio is ready"
            self._persist_state()
            return
        token = os.getenv("HF_TOKEN", "").strip()
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        tools = self.base / "vendor" / "ffmpeg" / "bin"
        environment["PATH"] = str(tools) + os.pathsep + environment.get("PATH", "")
        try:
            install_lock = self._acquire_install_lock()
            for index, component in enumerate(pending):
                with self.lock:
                    if self.cancel_requested:
                        raise InterruptedError("Setup paused")
                    self.active_component = component.key
                    self.detail = f"Installing {component.name}"
                    self.progress = round(index / len(pending) * 100, 1)
                    self.logs.append(f"Starting {component.name}")
                command = [sys.executable, str(self.base / "scripts" / "install_models.py"), component.key]
                process = subprocess.Popen(
                    command, cwd=self.base, env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding="utf-8", errors="replace", bufsize=1,
                    start_new_session=os.name != "nt",
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
                )
                with self.lock:
                    self.process = process
                tail: list[str] = []
                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if token:
                        line = line.replace(token, "[hidden]")
                    tail.append(line)
                    tail = tail[-12:]
                    if line.startswith("::dubline-progress::"):
                        try:
                            event = json.loads(line.split("::", 2)[2])
                            local_progress = min(1.0, max(0.0, float(event.get("progress", 0))))
                            with self.lock:
                                self.detail = str(event.get("detail") or self.detail)[:240]
                                self.progress = round((index + local_progress) / len(pending) * 100, 1)
                        except (ValueError, TypeError, json.JSONDecodeError):
                            pass
                code = process.wait()
                with self.lock:
                    self.process = None
                if self.cancel_requested:
                    raise InterruptedError("Setup paused")
                if code:
                    useful = next((line for line in reversed(tail)
                                   if not line.startswith("Traceback") and not line.startswith("  File ")), "")
                    raise RuntimeError(useful or f"{component.name} could not be installed")
                with self.lock:
                    self.logs.append(f"{component.name} ready")
                    self.progress = round((index + 1) / len(pending) * 100, 1)
                self._persist_state()
                if component.key == "media_tools":
                    os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")
            missing = [component.name for component in pending
                       if not self._component_ready(component, self.system_inspector())]
            if missing:
                raise RuntimeError("The final check could not find: " + ", ".join(missing))
            with self.lock:
                self.phase = "complete"
                self.active_component = None
                self.detail = "Your local dubbing studio is ready"
                self.progress = 100.0
            self._persist_state()
        except InterruptedError:
            with self.lock:
                self.phase = "idle"
                self.active_component = None
                self.detail = "Setup paused — your completed downloads are safe"
                self.error = None
            self._persist_state()
        except Exception as exc:
            message = str(exc).strip() or "Setup stopped unexpectedly"
            if token:
                message = message.replace(token, "[hidden]")
            with self.lock:
                self.phase = "error"
                self.active_component = None
                self.detail = "Setup needs your attention"
                self.error = message[:600]
            self._persist_state()
        finally:
            with self.lock:
                self.process = None
            if install_lock is not None:
                self._release_install_lock(install_lock)

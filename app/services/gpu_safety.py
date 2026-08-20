from __future__ import annotations

"""Crash-evident supervision for every CUDA phase in the film pipeline."""

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


_lock = threading.RLock()
_state_write_lock = threading.Lock()
_watchdog_lock = threading.Lock()
_watchdog_active = False
_watchdog_failure: RuntimeError | None = None
_nvml_lock = threading.Lock()
_nvml_handle: object | None = None
_nvml_checked = False
_health_cache: tuple[float, dict] | None = None


class GPUStageUnsafe(RuntimeError):
    """A CUDA stage crossed a fail-closed hardware safety boundary."""


def _thermal_limit() -> int:
    try:
        return max(65, min(95, int(os.getenv("DUB_GPU_ABORT_TEMPERATURE_C", "90"))))
    except ValueError:
        return 90


def _watchdog_interval() -> float:
    try:
        return max(.5, min(5.0, float(os.getenv("DUB_GPU_WATCHDOG_SECONDS", ".5"))))
    except ValueError:
        return .5


def gpu_checkpoint() -> None:
    """Raise inside the controlling thread when the active watchdog trips.

    Model subprocesses are polled by ``controlled_lines`` every 250 ms, so this
    exception follows the normal terminate/wait path instead of resetting the
    driver or leaving a child process behind.
    """
    with _watchdog_lock:
        failure = _watchdog_failure if _watchdog_active else None
    if failure is not None:
        raise GPUStageUnsafe(str(failure))


def _start_watchdog(path: Path, stage: str) -> tuple[threading.Event, threading.Thread]:
    global _watchdog_active, _watchdog_failure
    stopped = threading.Event()
    with _watchdog_lock:
        _watchdog_active = True
        _watchdog_failure = None

    def monitor() -> None:
        global _watchdog_failure
        limit = _thermal_limit()
        while not stopped.wait(_watchdog_interval()):
            try:
                # Always a fresh reading: this is the fail-closed safety path,
                # and through NVML a measurement costs microseconds.
                health = query_nvidia()
                if health["temperature_c"] < limit:
                    continue
                message = (
                    f"CUDA stage '{stage}' reached {health['temperature_c']}°C "
                    f"(safety ceiling {limit}°C) and was stopped cleanly."
                )
            except Exception as exc:
                health = None
                message = f"CUDA stage '{stage}' lost NVIDIA health monitoring: {exc}"
            with _watchdog_lock:
                if _watchdog_failure is None:
                    _watchdog_failure = GPUStageUnsafe(message)
            state = _read_state(path)
            state.update({"status": "unsafe", "stage": stage,
                          "interrupted_at": time.time(), "error": message})
            if health is not None:
                state["health"] = health
            _write_state(path, state)
            return

    thread = threading.Thread(target=monitor, name="cuda-thermal-watchdog", daemon=True)
    thread.start()
    return stopped, thread


def _stop_watchdog(stopped: threading.Event, thread: threading.Thread) -> RuntimeError | None:
    global _watchdog_active, _watchdog_failure
    stopped.set()
    thread.join(timeout=max(2.0, _watchdog_interval() + .5))
    with _watchdog_lock:
        failure = _watchdog_failure
        _watchdog_active = False
        _watchdog_failure = None
    return failure


def parse_nvidia_status(value: str) -> dict:
    fields = [item.strip() for item in value.strip().split(",")]
    if len(fields) < 5:
        raise RuntimeError("NVIDIA health query returned incomplete data")
    return {"free_mb": int(fields[0]), "total_mb": int(fields[1]),
            "temperature_c": int(fields[2]), "utilization": int(fields[3]),
            "power_w": float(fields[4])}


def _nvml() -> object | None:
    """Bind NVML once, or None when the binding is not installed.

    Spawning nvidia-smi costs roughly 200 ms.  The watchdog polls twice a second
    for the whole length of every CUDA stage, so on a feature film that is over
    an hour of pure process-spawn overhead competing with the CPU side of
    inference.  NVML answers the same questions in microseconds.
    """
    global _nvml_handle, _nvml_checked
    with _nvml_lock:
        if _nvml_checked:
            return _nvml_handle
        _nvml_checked = True
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                int(os.getenv("DUB_GPU_INDEX", "0")))
            _nvml_module = pynvml
        except Exception:
            _nvml_handle = None
            return None
        globals()["_nvml_api"] = _nvml_module
        return _nvml_handle


def _query_nvml() -> dict | None:
    handle = _nvml()
    if handle is None:
        return None
    api = globals().get("_nvml_api")
    try:
        memory = api.nvmlDeviceGetMemoryInfo(handle)
        return {
            "free_mb": int(memory.free // 1024 // 1024),
            "total_mb": int(memory.total // 1024 // 1024),
            "temperature_c": int(api.nvmlDeviceGetTemperature(handle, api.NVML_TEMPERATURE_GPU)),
            "utilization": int(api.nvmlDeviceGetUtilizationRates(handle).gpu),
            "power_w": float(api.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
        }
    except Exception:
        # A driver hiccup must not be fatal here; the caller falls back to
        # nvidia-smi, which reports the same fields through a different path.
        with _nvml_lock:
            globals()["_nvml_handle"] = None
        return None


def _query_nvidia_smi() -> dict:
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=memory.free,memory.total,temperature.gpu,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8)
    if result.returncode or not result.stdout.strip():
        detail = (result.stdout + result.stderr).strip()[-600:]
        raise RuntimeError("The NVIDIA driver is unavailable" + (f": {detail}" if detail else ""))
    return parse_nvidia_status(result.stdout.splitlines()[0])


def query_nvidia(max_age: float = 0.0) -> dict:
    """Read GPU health, optionally reusing a reading newer than ``max_age``.

    Safety decisions pass 0 and always measure.  High-frequency pollers share a
    reading so that watching the device does not itself become a load on it.
    """
    global _health_cache
    if max_age > 0:
        cached = _health_cache
        if cached is not None and time.monotonic() - cached[0] <= max_age:
            return dict(cached[1])
    health = _query_nvml() or _query_nvidia_smi()
    _health_cache = (time.monotonic(), dict(health))
    return health


def _data_root(folder: Path) -> Path:
    folder = folder.resolve()
    for candidate in (folder, *folder.parents):
        if candidate.name.lower() == "jobs":
            return candidate.parent
    return Path(os.getenv("DUB_WORKDIR", "data")).resolve()


def _state_path(folder: Path) -> Path:
    return _data_root(folder) / "gpu-safety.json"


def gpu_safety_summary(data_root: Path | None = None) -> dict:
    root = (data_root or Path(os.getenv("DUB_WORKDIR", "data"))).resolve()
    state = _read_state(root / "gpu-safety.json")
    return {key: state.get(key) for key in (
        "status", "stage", "last_stage", "started_at", "interrupted_at",
        "last_completed_at", "last_canary_at", "health",
    ) if state.get(key) is not None}


def _boot_token() -> int:
    return int((time.time() - time.monotonic()) // 30)


def _read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _write_state(path: Path, value: dict) -> None:
    with _state_write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def _canary(path: Path, checkpoint: Callable[[], None]) -> None:
    checkpoint()
    _write_state(path, {"status": "canary", "stage": "CUDA safety probe",
                        "boot_token": _boot_token(), "pid": os.getpid(),
                        "started_at": time.time()})
    result = subprocess.run(
        [sys.executable, "-m", "app.services.gpu_canary_worker", "--seconds", "2"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    if result.returncode:
        _write_state(path, {"status": "unsafe", "stage": "CUDA safety probe",
                            "boot_token": _boot_token(), "failed_at": time.time(),
                            "error": (result.stdout + result.stderr)[-1200:]})
        raise RuntimeError("CUDA safety probe failed; the job was stopped before loading a production model")
    health = query_nvidia()
    _write_state(path, {"status": "idle", "boot_token": _boot_token(),
                        "last_canary_at": time.time(), "health": health})


def _handoff_target() -> int:
    try:
        return max(60, min(88, int(os.getenv("DUB_GPU_COOLDOWN_TARGET_C", "77"))))
    except ValueError:
        return 77


def _cooldown(checkpoint: Callable[[], None], seconds: float | None = None) -> dict:
    if seconds is None:
        seconds = float(os.getenv("DUB_GPU_HANDOFF_TIMEOUT_SECONDS", "45"))
    try:
        reserve_mb = max(200, int(os.getenv("DUB_GPU_HANDOFF_RESERVE_MB", "900")))
    except ValueError:
        reserve_mb = 900
    target = _handoff_target()
    deadline = time.monotonic() + max(0.0, seconds)
    latest = query_nvidia()
    while time.monotonic() < deadline:
        checkpoint()
        # A cool, idle, mostly released device is a safe hand-off boundary.
        # Polling is fine-grained because a health read is now effectively free,
        # so the hand-off ends the moment it is genuinely safe rather than on
        # the next whole second.
        if (latest["temperature_c"] <= target and latest["utilization"] <= 10
                and latest["free_mb"] >= latest["total_mb"] - reserve_mb):
            return latest
        time.sleep(.2)
        latest = query_nvidia()
    if latest["temperature_c"] >= _thermal_limit():
        raise RuntimeError(
            f"The NVIDIA GPU remained at {latest['temperature_c']}°C after its cooldown; "
            "the next CUDA stage was stopped instead of risking a driver reset."
        )
    if latest["free_mb"] < latest["total_mb"] - max(1400, reserve_mb + 500):
        raise RuntimeError(
            "The previous CUDA worker did not release enough VRAM; the pipeline stopped at a safe checkpoint."
        )
    return latest


@contextmanager
def gpu_stage(folder: Path, stage: str, checkpoint: Callable[[], None],
              *, minimum_free_mb: int = 5600) -> Iterator[None]:
    """Record an in-flight stage, validate CUDA, and enforce a cool hand-off.

    If Windows reboots while the body is active, the durable marker remains.
    The next CUDA phase must pass an isolated canary before production resumes.
    """
    path = _state_path(folder)
    with _lock:
        checkpoint()
        state = _read_state(path)
        # The process-wide lock makes any pre-existing active marker stale: no
        # legitimate second GPU stage can be entering concurrently.
        stale = state.get("status") in {"active", "canary", "unsafe", "interrupted"}
        if stale or not state.get("last_canary_at"):
            _canary(path, checkpoint)
            state = _read_state(path)
        health = query_nvidia()
        if health["temperature_c"] >= _thermal_limit():
            health = _cooldown(checkpoint)
        if health["free_mb"] < minimum_free_mb:
            raise RuntimeError(
                f"CUDA stage '{stage}' needs {minimum_free_mb} MB free, but only "
                f"{health['free_mb']} MB is available. Close other GPU applications and resume."
            )
        started = time.time()
        _write_state(path, {"status": "active", "stage": stage,
                            "boot_token": _boot_token(), "pid": os.getpid(),
                            "started_at": started, "health_before": health,
                            "last_canary_at": state.get("last_canary_at")})
        stopped, watchdog = _start_watchdog(path, stage)
        succeeded = False
        watchdog_failure: RuntimeError | None = None
        try:
            yield
            gpu_checkpoint()
            succeeded = True
        finally:
            watchdog_failure = _stop_watchdog(stopped, watchdog)
            if succeeded:
                try:
                    health_after = _cooldown(checkpoint)
                except BaseException:
                    _write_state(path, {"status": "interrupted", "boot_token": _boot_token(),
                                        "stage": stage, "interrupted_at": time.time(),
                                        "last_canary_at": state.get("last_canary_at")})
                    raise
                else:
                    _write_state(path, {"status": "idle", "boot_token": _boot_token(),
                                        "last_stage": stage, "last_completed_at": time.time(),
                                        "last_canary_at": state.get("last_canary_at"),
                                        "health": health_after})
            else:
                # A normal Python exception reaches here; a kernel crash does not.
                _write_state(path, {"status": "interrupted", "boot_token": _boot_token(),
                                    "stage": stage, "interrupted_at": time.time(),
                                    "error": str(watchdog_failure) if watchdog_failure else None,
                                    "last_canary_at": state.get("last_canary_at")})

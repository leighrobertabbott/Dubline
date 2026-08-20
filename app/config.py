from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
ENV_FILE = BASE / ".env"
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{10,512}$")
_TMDB_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{20,2048}$")


def load_local_environment(path: Path = ENV_FILE) -> None:
    """Load Dubline's small local .env file without overriding the shell."""
    raw_lines = path.read_text(encoding="utf-8-sig").splitlines() if path.is_file() else []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)
    scripts = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    runtime_defaults = {
        "QWEN_ASR_RUNTIME": BASE / "vendor" / "pyannote-env" / scripts / executable,
        "PYANNOTE_RUNTIME": BASE / "vendor" / "pyannote-env" / scripts / executable,
        "QWEN_TTS_RUNTIME": BASE / "vendor" / "qwen-tts-env" / scripts / executable,
        "MUSETALK_RUNTIME": BASE / "vendor" / "musetalk-env" / scripts / executable,
    }
    for name, value in runtime_defaults.items():
        os.environ.setdefault(name, str(value))
    bundled_tools = BASE / "vendor" / "ffmpeg" / "bin"
    if bundled_tools.is_dir():
        os.environ["PATH"] = str(bundled_tools) + os.pathsep + os.environ.get("PATH", "")


def validate_hf_token(token: str) -> str:
    token = token.strip()
    if not _TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Enter a valid Hugging Face access token")
    return token


def save_hf_token(token: str, path: Path = ENV_FILE) -> None:
    """Atomically persist the token while preserving every other setting."""
    token = validate_hf_token(token)
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.is_file() else []
    updated: list[str] = []
    replaced = False
    for line in lines:
        if re.match(r"^\s*HF_TOKEN\s*=", line):
            if not replaced:
                updated.append(f"HF_TOKEN={token}")
                replaced = True
            continue
        updated.append(line)
    if not replaced:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(f"HF_TOKEN={token}")
    _write_environment(path, updated)
    os.environ["HF_TOKEN"] = token


def validate_tmdb_token(token: str) -> str:
    token = token.strip()
    if not _TMDB_TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Enter a valid TMDB API Read Access Token")
    return token


def save_setting(name: str, value: str, path: Path = ENV_FILE) -> None:
    """Atomically save one allow-listed local setting without exposing its value."""
    validators = {"TMDB_TOKEN": validate_tmdb_token, "MEDIA_LOOKUP_ENABLED": lambda item: item if item in {"0", "1"} else (_ for _ in ()).throw(ValueError("Invalid media lookup setting"))}
    if name not in validators:
        raise ValueError("That setting cannot be changed here")
    value = validators[name](value.strip())
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.is_file() else []
    updated: list[str] = []
    replaced = False
    for line in lines:
        if re.match(rf"^\s*{re.escape(name)}\s*=", line):
            if not replaced:
                updated.append(f"{name}={value}")
                replaced = True
            continue
        updated.append(line)
    if not replaced:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(f"{name}={value}")
    _write_environment(path, updated)
    os.environ[name] = value


def _write_environment(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".env-", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(lines).rstrip() + "\n")
        if os.name != "nt":
            os.chmod(temporary_name, 0o600)
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def remove_hf_token(path: Path = ENV_FILE) -> None:
    if path.is_file():
        lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines()
                 if not re.match(r"^\s*HF_TOKEN\s*=", line)]
        _write_environment(path, lines)
    os.environ.pop("HF_TOKEN", None)


def remove_setting(name: str, path: Path = ENV_FILE) -> None:
    if name not in {"TMDB_TOKEN", "MEDIA_LOOKUP_ENABLED"}:
        raise ValueError("That setting cannot be changed here")
    if path.is_file():
        lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines()
                 if not re.match(rf"^\s*{re.escape(name)}\s*=", line)]
        _write_environment(path, lines)
    os.environ.pop(name, None)


def hf_token_summary() -> dict:
    token = os.getenv("HF_TOKEN", "").strip()
    return {
        "configured": bool(token),
        "display": f"{token[:4]}••••{token[-4:]}" if len(token) >= 10 else None,
    }


def media_lookup_summary() -> dict:
    personal_token = os.getenv("TMDB_TOKEN", "").strip()
    managed_token = os.getenv("DUBLINE_TMDB_TOKEN", "").strip()
    managed_key = os.getenv("DUBLINE_TMDB_API_KEY", "").strip()
    token = personal_token or managed_token
    return {
        "enabled": os.getenv("MEDIA_LOOKUP_ENABLED", "1") != "0",
        "configured": bool(token or managed_key),
        "managed": bool(managed_token or managed_key) and not bool(personal_token),
        "movie_lookup": bool(token or managed_key),
        "tv_lookup": True,
        "display": "Included with Dubline" if (managed_token or managed_key) and not personal_token else (f"••••{token[-6:]}" if len(token) >= 20 else None),
    }

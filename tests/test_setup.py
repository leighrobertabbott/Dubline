from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import hf_token_summary, remove_hf_token, save_hf_token
from app.main import app, setup as application_setup
from app.services.setup_manager import COMPONENTS, SetupManager
from scripts.install_models import download, extract_zip_safely


def system_state(value: bool = False) -> dict:
    keys = {check for component in COMPONENTS for check in component.checks}
    return {
        **{key: value for key in keys},
        "cuda": True,
        "gpu": "Test NVIDIA GPU, 12288 MiB",
        "disk_free_gb": 999,
    }


def supported_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.setup_manager.platform.system", lambda: "Windows")
    monkeypatch.setattr("app.services.setup_manager.platform.machine", lambda: "AMD64")


def test_first_run_status_is_complete_and_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    supported_windows(monkeypatch)
    monkeypatch.setattr("app.services.setup_manager.shutil.which", lambda _: None)
    manager = SetupManager(tmp_path, lambda: system_state(False))
    status = manager.snapshot()
    expected = round(sum(item.estimated_gb for item in COMPONENTS if item.required), 1)
    assert status["first_run"] and not status["ready"]
    assert status["missing_download_gb"] == expected
    assert status["platform"]["supported"]
    assert all({"name", "description", "ready", "estimated_gb"} <= item.keys()
               for item in status["components"])


def test_macos_is_detected_before_any_model_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.services.setup_manager.platform.system", lambda: "Darwin")
    monkeypatch.setattr("app.services.setup_manager.platform.machine", lambda: "arm64")
    monkeypatch.setattr("app.services.setup_manager.shutil.which", lambda _: None)
    manager = SetupManager(tmp_path, lambda: system_state(False))
    status = manager.snapshot()
    assert status["platform"]["os"] == "macOS"
    assert not status["platform"]["supported"]
    assert "Metal" in status["platform"]["message"]
    assert not status["can_install"]
    with pytest.raises(ValueError, match="Metal"):
        manager.start(include_diarization=False, include_lip_sync=False)


def test_hugging_face_token_is_masked_and_atomically_removable(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch):
    token = "hf_abcdefghijklmnopqrstuvwxyz1234"
    destination = tmp_path / ".env"
    destination.write_text("DUB_ENGINE=indextts\nHF_TOKEN=old-token-value\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    save_hf_token(token, destination)
    content = destination.read_text(encoding="utf-8")
    assert content.count("HF_TOKEN=") == 1 and token in content
    summary = hf_token_summary()
    assert summary["configured"] and token not in json.dumps(summary)
    assert summary["display"].startswith("hf_a") and summary["display"].endswith("1234")
    remove_hf_token(destination)
    assert "HF_TOKEN=" not in destination.read_text(encoding="utf-8")
    assert "HF_TOKEN" not in os.environ


def test_gated_component_cannot_start_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    supported_windows(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("app.services.setup_manager.shutil.which", lambda _: None)
    manager = SetupManager(tmp_path, lambda: system_state(False))
    with pytest.raises(ValueError, match="Hugging Face"):
        manager.start(include_diarization=True, include_lip_sync=False)


def test_interrupted_setup_state_recovers_without_claiming_it_is_running(tmp_path: Path):
    state = tmp_path / "data" / "setup-state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"phase": "installing", "progress": 42}), encoding="utf-8")
    manager = SetupManager(tmp_path, lambda: system_state(False))
    assert manager.phase == "idle"
    assert "interrupted" in manager.detail.lower()
    assert any("continue" in line.lower() for line in manager.logs)


def test_already_complete_setup_finishes_without_starting_downloads(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch):
    supported_windows(monkeypatch)
    manager = SetupManager(tmp_path, lambda: system_state(True))
    result = manager.start(include_diarization=False, include_lip_sync=False)
    assert result["running"] or result["phase"] == "complete"
    manager.thread.join(timeout=3)
    assert manager.phase == "complete" and manager.progress == 100


def test_direct_download_replaces_corrupt_file_and_verifies_hash(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"verified model payload")
    target = tmp_path / "target.bin"
    target.write_bytes(b"corrupt")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    download(source.as_uri(), target, start=0, end=1, label="test model", expected_hash=expected)
    assert target.read_bytes() == source.read_bytes()


def test_archive_extraction_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "hostile.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("../outside.txt", "no")
    destination = tmp_path / "extract"
    destination.mkdir()
    with pytest.raises(RuntimeError, match="outside"):
        extract_zip_safely(archive, destination)
    assert not (tmp_path / "outside.txt").exists()


def test_wizard_and_platform_launchers_are_shipped():
    root = Path(__file__).resolve().parents[1]
    page = (root / "web" / "index.html").read_text(encoding="utf-8")
    script = (root / "web" / "setup.js").read_text(encoding="utf-8")
    for required in ("setupWizard", "hfToken", "setupProgressBar", "setupRescan", "cancelSetup"):
        assert f'id="{required}"' in page
    assert "textContent = info.error" in script
    assert "innerHTML = info.error" not in script
    assert (root / "setup.ps1").is_file() and (root / "setup.sh").is_file()
    assert (root / "Start Dubline.cmd").is_file() and (root / "start-dubline.sh").is_file()
    assert "Darwin" in (root / "setup.sh").read_text(encoding="utf-8")
    assert "--frozen" in (root / "setup.ps1").read_text(encoding="utf-8")


def test_token_api_never_echoes_a_rejected_secret(monkeypatch: pytest.MonkeyPatch):
    secret = "hf_this_must_never_appear_in_an_error_response"

    def reject(_: str):
        raise ValueError("Hugging Face did not accept that token")

    monkeypatch.setattr(application_setup, "save_token", reject)
    response = TestClient(app).post("/api/setup/token", json={"token": secret})
    assert response.status_code == 400
    assert secret not in response.text


def test_setup_api_masks_configured_token(monkeypatch: pytest.MonkeyPatch):
    secret = "hf_api_response_must_mask_this_value_1234"
    monkeypatch.setenv("HF_TOKEN", secret)
    response = TestClient(app).get("/api/setup")
    assert response.status_code == 200
    assert secret not in response.text
    assert response.json()["token"]["configured"]


def test_local_ui_responses_have_browser_security_headers():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]

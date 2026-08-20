from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tarfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
VENDOR = BASE / "vendor"
DOWNLOADS = BASE / "data" / "setup-downloads"
VENDOR.mkdir(parents=True, exist_ok=True)
DOWNLOADS.mkdir(parents=True, exist_ok=True)

REVISIONS = {
    "indextts": "c39ce5ba981572cb187443877ff559dfb246ce63",
    "w2v_bert": "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
    "semantic_codec": "265c6cef07625665d0c28d2faafb1415562379dc",
    "campplus": "e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
    "bigvgan": "633ff708ed5b74903e86ff1298cf4a98e921c513",
    "qwen_asr_small": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    "qwen_asr_large": "7278e1e70fe206f11671096ffdd38061171dd6e5",
    "qwen_aligner": "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
    "translation": "ab8472660ac61fac25f1af43fac2599d52a8a775",
    "translation_qc": "7c41481f57cb95916b40956ab2f0b139b296d974",
    "qwen_tts": "fd4b254389122332181a7c3db7f27e918eec64e3",
    "pyannote": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
    "musetalk_models": "3ef28bc5cff08c90ad8178a25f1b570cd800170f",
    "sd_vae": "31f26fdeee1355a5c34592e401dd41e45d25a493",
    "whisper_tiny": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "dwpose": "1a7144101628d69ee7a3768d1ee3a094070dc388",
    "syncnet": "405eda8eab9f65c1a6e0c292a5dee5a08089e2ae",
    "face_parse": "0073b233a5a3c4b1377d4dbf49245017938a72b5",
}


def progress(value: float, detail: str) -> None:
    print("::dubline-progress::" + json.dumps({
        "progress": max(0.0, min(1.0, value)), "detail": detail,
    }), flush=True)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def download(url: str, target: Path, *, start: float, end: float, label: str,
             expected_hash: str | None = None, algorithm: str = "sha256") -> Path:
    if target.is_file() and target.stat().st_size:
        if expected_hash and file_hash(target, algorithm) != expected_hash.lower():
            target.unlink(missing_ok=True)
        else:
            progress(end, f"{label} already downloaded and verified")
            return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        received = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": "Dubline setup"}
        if received:
            headers["Range"] = f"bytes={received}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                resumed = received > 0 and getattr(response, "status", 200) == 206
                if not resumed:
                    received = 0
                content_length = int(response.headers.get("Content-Length", 0))
                total = received + content_length if content_length else 0
                with partial.open("ab" if resumed else "wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        received += len(chunk)
                        fraction = received / total if total else 0
                        progress(start + (end - start) * fraction,
                                 f"Downloading {label} · {received / 1024 ** 2:.0f} MB")
            last_error = None
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and partial.is_file():
                if expected_hash and file_hash(partial, algorithm) == expected_hash.lower():
                    last_error = None
                    break
                partial.unlink(missing_ok=True)
            last_error = exc
            progress(start, f"Download server interrupted {label} · retry {attempt}/3")
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            progress(start, f"Connection interrupted while downloading {label} · retry {attempt}/3")
    if last_error is not None:
        raise RuntimeError(f"The {label} download was interrupted after three attempts") from last_error
    partial.replace(target)
    if expected_hash and file_hash(target, algorithm) != expected_hash.lower():
        target.unlink(missing_ok=True)
        raise RuntimeError(f"The {label} download failed its integrity check")
    return target


def run(*command: str, cwd: Path = BASE) -> None:
    process = subprocess.run(command, cwd=cwd, check=False)
    if process.returncode:
        raise RuntimeError(f"A setup step stopped with exit code {process.returncode}")


def published_checksum(url: str, filename: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Dubline setup"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError("The published integrity information could not be downloaded") from exc
    for line in text.splitlines():
        parts = line.strip().replace("*", " ").split()
        if parts and len(parts[0]) == 64 and (len(parts) == 1 or parts[-1].endswith(filename)):
            return parts[0].lower()
    raise RuntimeError(f"No published checksum was found for {filename}")


def extract_zip_safely(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError("A downloaded archive tried to write outside its setup folder")
        package.extractall(destination)


def uv_command() -> list[str]:
    executable = shutil.which("uv")
    if executable:
        return [executable]
    try:
        import uv  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("The setup helper 'uv' is missing; run setup.ps1 again") from exc
    return [sys.executable, "-m", "uv"]


def env_python(folder: Path) -> Path:
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def verify_runtime(runtime: Path, modules: list[str], *, require_cuda: bool = True) -> None:
    imports = "; ".join(f"import {name}" for name in modules)
    cuda_check = "; import torch; assert torch.cuda.is_available(), 'CUDA is unavailable in this runtime'" if require_cuda else ""
    run(str(runtime), "-c", imports + cuda_check)


def ensure_venv(folder: Path, packages: list[str], *, torch: bool = False,
                python_version: str = "3.11") -> Path:
    uv = uv_command()
    runtime = env_python(folder)
    if not runtime.is_file():
        run(*uv, "venv", str(folder), "--python", python_version)
    if torch:
        run(*uv, "pip", "install", "--python", str(runtime), "torch==2.8.0", "torchaudio==2.8.0",
            "--index-url", "https://download.pytorch.org/whl/cu128")
    if packages:
        run(*uv, "pip", "install", "--python", str(runtime), *packages)
    return runtime


def install_archive(url: str, destination: Path, marker: Path, label: str) -> None:
    if marker.is_file():
        return
    archive = DOWNLOADS / f"{destination.name}.zip"
    download(url, archive, start=.05, end=.5, label=label)
    with tempfile.TemporaryDirectory(prefix="dubline-repo-") as temporary:
        root = Path(temporary)
        extract_zip_safely(archive, root)
        entries = [entry for entry in root.iterdir() if entry.is_dir()]
        if len(entries) != 1:
            raise RuntimeError(f"The {label} download had an unexpected layout")
        resolved = destination.resolve()
        if VENDOR.resolve() not in resolved.parents:
            raise RuntimeError("Refusing to replace a folder outside Dubline's vendor directory")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(entries[0]), str(destination))
    archive.unlink(missing_ok=True)


def media_tools() -> None:
    existing_tools = [shutil.which(name) for name in ("ffmpeg", "ffprobe")]
    if all(existing_tools):
        try:
            if all(subprocess.run([tool, "-version"], capture_output=True, timeout=15,
                                  check=False).returncode == 0 for tool in existing_tools):
                progress(1, "FFmpeg and FFprobe are ready")
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
    destination = VENDOR / "ffmpeg"
    if os.name == "nt":
        archive = DOWNLOADS / "ffmpeg-release-essentials.zip"
        checksum = published_checksum(
            "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip.sha256", archive.name,
        )
        download("https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip", archive,
                 start=.03, end=.82, label="video tools", expected_hash=checksum)
        archive_kind = "zip"
    elif sys.platform.startswith("linux"):
        architecture = platform.machine().lower()
        target = "linux64" if architecture in {"x86_64", "amd64"} else "linuxarm64"
        archive = DOWNLOADS / f"ffmpeg-master-latest-{target}-gpl.tar.xz"
        checksum = published_checksum(
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/checksums.sha256",
            archive.name,
        )
        download(
            f"https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/{archive.name}",
            archive, start=.03, end=.82, label="video tools", expected_hash=checksum,
        )
        archive_kind = "tar"
    else:
        raise RuntimeError("Dubline's CUDA pipeline is not available on this operating system")
    with tempfile.TemporaryDirectory(prefix="dubline-ffmpeg-") as temporary:
        root = Path(temporary)
        if archive_kind == "zip":
            extract_zip_safely(archive, root)
        else:
            with tarfile.open(archive, "r:xz") as package:
                package.extractall(root, filter="data")
        executable = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        candidates = [path for path in root.iterdir() if (path / "bin" / executable).is_file()]
        if len(candidates) != 1:
            raise RuntimeError("The video tools download had an unexpected layout")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(candidates[0]), str(destination))
    for executable in ("ffmpeg.exe", "ffprobe.exe") if os.name == "nt" else ("ffmpeg", "ffprobe"):
        result = subprocess.run([str(destination / "bin" / executable), "-version"],
                                capture_output=True, text=True, check=False, timeout=20)
        if result.returncode:
            raise RuntimeError(f"The installed {executable} tool did not pass its launch check")
    archive.unlink(missing_ok=True)
    progress(1, "Video tools ready")


def voice_model() -> None:
    progress(.03, "Preparing the expressive voice model")
    from huggingface_hub import hf_hub_download, snapshot_download
    model_dir = VENDOR / "index-tts" / "checkpoints"
    snapshot_download("IndexTeam/IndexTTS-2.5", revision=REVISIONS["indextts"], local_dir=model_dir)
    cache = model_dir / "hf_cache"
    progress(.68, "Preparing speech features")
    snapshot_download("facebook/w2v-bert-2.0", revision=REVISIONS["w2v_bert"],
                      local_dir=cache / "w2v-bert-2.0")
    progress(.78, "Preparing semantic voice features")
    semantic = hf_hub_download(
        "amphion/MaskGCT", "semantic_codec/model.safetensors",
        revision=REVISIONS["semantic_codec"], local_dir=cache,
    )
    shutil.copy2(semantic, cache / "semantic_codec_model.safetensors")
    progress(.84, "Preparing speaker matching")
    hf_hub_download(
        "funasr/campplus", "campplus_cn_common.bin", revision=REVISIONS["campplus"], local_dir=cache,
    )
    progress(.9, "Preparing the studio vocoder")
    bigvgan = cache / "bigvgan"
    for filename in ("config.json", "bigvgan_generator.pt"):
        hf_hub_download("nvidia/bigvgan_v2_22khz_80band_256x", filename,
                        revision=REVISIONS["bigvgan"], local_dir=bigvgan)
    required = ("config.yaml", "gpt.pth", "s2mel.pth", "codec.pth")
    if not all((model_dir / name).is_file() for name in required):
        raise RuntimeError("The expressive voice model did not pass its final file check")
    progress(1, "English voice studio ready")


def separation() -> None:
    progress(.02, "Preparing cinematic dialogue separation")
    bandit = VENDOR / "bandit-v2"
    install_archive(
        "https://github.com/kwatcharasupat/bandit-v2/archive/d5563d9031e95fdaa3e5a73d5020b9a0df61adb6.zip",
        bandit, bandit / "src" / "models" / "bandit" / "bandit.py", "Bandit separation engine",
    )
    checkpoint = bandit / "checkpoints" / "checkpoint-multi.ckpt"
    download("https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt?download=1", checkpoint,
             start=.12, end=.32, label="cinematic separation model")
    if file_hash(checkpoint, "md5").upper() != "FEA2868787551B0CFF36CFCF7C3622A3":
        checkpoint.unlink(missing_ok=True)
        raise RuntimeError("The cinematic separation model failed its integrity check")
    progress(.36, "Preparing speech recognition")
    import whisper
    whisper.load_model("turbo", device="cpu", download_root=str(VENDOR / "whisper"))
    progress(.58, "Preparing dialogue recovery")
    from demucs.pretrained import get_model
    get_model("htdemucs")
    progress(.72, "Preparing fine dialogue recovery")
    from mel_band_roformer.download import ensure_model_assets
    ensure_model_assets(models_dir=VENDOR / "melband-roformer", download_missing=True)
    progress(1, "Dialogue separation ready")


def speech() -> None:
    progress(.03, "Preparing the isolated speech runtime")
    runtime = ensure_venv(VENDOR / "pyannote-env", ["qwen-asr==0.0.6"], torch=True)
    progress(.26, "Downloading the everyday speech model")
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3-ASR-0.6B", revision=REVISIONS["qwen_asr_small"],
                      local_dir=VENDOR / "qwen3-asr-0.6b-qwen")
    progress(.5, "Downloading the difficult-speech model")
    snapshot_download("Qwen/Qwen3-ASR-1.7B", revision=REVISIONS["qwen_asr_large"],
                      local_dir=VENDOR / "qwen3-asr-1.7b-qwen")
    progress(.76, "Downloading precise word alignment")
    snapshot_download("Qwen/Qwen3-ForcedAligner-0.6B", revision=REVISIONS["qwen_aligner"],
                      local_dir=VENDOR / "qwen3-forced-aligner-0.6b-qwen")
    if not runtime.is_file():
        raise RuntimeError("The speech runtime could not be created")
    verify_runtime(runtime, ["qwen_asr"])
    progress(1, "Speech understanding ready")


def language() -> None:
    from huggingface_hub import hf_hub_download
    progress(.08, "Downloading natural English translation")
    hf_hub_download("tencent/Hy-MT2-7B-GGUF", "Hy-MT2-7B-Q4_K_M.gguf",
                    revision=REVISIONS["translation"],
                    local_dir=VENDOR / "hy-mt2-7b")
    progress(.54, "Downloading the independent translation check")
    hf_hub_download("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf",
                    revision=REVISIONS["translation_qc"],
                    local_dir=VENDOR / "qwen3-8b")
    progress(1, "Translation and QC ready")


def voice_tools() -> None:
    face = VENDOR / "opencv-face"
    progress(.03, "Preparing visual speaker matching")
    download(
        "https://raw.githubusercontent.com/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        face / "face_detection_yunet_2023mar.onnx", start=.03, end=.1, label="face detector",
        expected_hash="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    )
    download(
        "https://raw.githubusercontent.com/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        face / "face_recognition_sface_2021dec.onnx", start=.1, end=.18, label="face matcher",
        expected_hash="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    )
    progress(.2, "Preparing the fallback voice runtime")
    runtime = ensure_venv(VENDOR / "qwen-tts-env", ["qwen-tts==0.1.1", "soundfile==0.14.0"], torch=True)
    progress(.52, "Downloading the fallback voice model")
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", revision=REVISIONS["qwen_tts"],
                      local_dir=VENDOR / "qwen3-tts-1.7b-base")
    if not runtime.is_file():
        raise RuntimeError("The fallback voice runtime could not be created")
    verify_runtime(runtime, ["qwen_tts"])
    progress(1, "Voice safety net ready")


def diarization() -> None:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Connect a Hugging Face token before installing enhanced speaker detection")
    progress(.05, "Preparing enhanced speaker detection")
    runtime = ensure_venv(VENDOR / "pyannote-env", ["pyannote.audio==4.0.7", "soundfile==0.14.0"])
    progress(.36, "Downloading enhanced speaker detection")
    command = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('pyannote/speaker-diarization-community-1', revision='{REVISIONS['pyannote']}', local_dir=r'{VENDOR / 'pyannote-community-1'}')"
    )
    run(str(runtime), "-c", command)
    verify_runtime(runtime, ["pyannote.audio"])
    progress(1, "Enhanced speaker detection ready")


def lip_sync() -> None:
    progress(.02, "Preparing optional lip sync")
    repo = VENDOR / "MuseTalk"
    install_archive("https://github.com/TMElyralab/MuseTalk/archive/0a89dec45a0192b824e3cf4daf96c239440c5ed8.zip", repo,
                    repo / "scripts" / "inference.py", "MuseTalk")
    uv = uv_command()
    runtime = ensure_venv(VENDOR / "musetalk-env", [], python_version="3.10")
    run(*uv, "pip", "install", "--python", str(runtime), "torch==2.0.1", "torchvision==0.15.2",
        "torchaudio==2.0.2", "--index-url", "https://download.pytorch.org/whl/cu118")
    run(*uv, "pip", "install", "--python", str(runtime), "-r", str(repo / "requirements.txt"))
    run(*uv, "pip", "install", "--python", str(runtime), "openmim==0.3.9")
    run(*uv, "pip", "install", "--python", str(runtime), "chumpy==0.70", "--no-build-isolation")
    progress(.42, "Installing lip-sync finishing tools")
    run(str(runtime), "-m", "mim", "install", "mmengine==0.10.7", "mmcv==2.0.1", "mmdet==3.1.0")
    run(str(runtime), "-m", "mim", "install", "mmpose==1.1.0")
    progress(.58, "Downloading selective lip-sync models")
    from huggingface_hub import snapshot_download
    models = repo / "models"
    snapshot_download("TMElyralab/MuseTalk", revision=REVISIONS["musetalk_models"], local_dir=models)
    snapshot_download("stabilityai/sd-vae-ft-mse", revision=REVISIONS["sd_vae"], local_dir=models / "sd-vae",
                      allow_patterns=["config.json", "diffusion_pytorch_model.bin"])
    snapshot_download("openai/whisper-tiny", revision=REVISIONS["whisper_tiny"], local_dir=models / "whisper",
                      allow_patterns=["config.json", "pytorch_model.bin", "preprocessor_config.json"])
    snapshot_download("yzd-v/DWPose", revision=REVISIONS["dwpose"], local_dir=models / "dwpose",
                      allow_patterns=["dw-ll_ucoco_384.pth"])
    snapshot_download("ByteDance/LatentSync", revision=REVISIONS["syncnet"], local_dir=models / "syncnet",
                      allow_patterns=["latentsync_syncnet.pt"])
    snapshot_download("ManyOtherFunctions/face-parse-bisent", revision=REVISIONS["face_parse"],
                      local_dir=models / "face-parse-bisent",
                      allow_patterns=["79999_iter.pth", "resnet18-5c106cde.pth"])
    verify_runtime(runtime, ["torch", "cv2"])
    if not (models / "musetalkV15" / "unet.pth").is_file():
        raise RuntimeError("The selective lip-sync model did not pass its final file check")
    progress(1, "Selective lip sync ready")


INSTALLERS = {
    "media_tools": media_tools,
    "voice_model": voice_model,
    "separation": separation,
    "speech": speech,
    "language": language,
    "voice_tools": voice_tools,
    "diarization": diarization,
    "lip_sync": lip_sync,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in INSTALLERS:
        print("Choose one valid Dubline setup component", file=sys.stderr)
        return 2
    try:
        INSTALLERS[sys.argv[1]]()
        return 0
    except Exception as exc:
        print(f"Setup could not finish this component: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

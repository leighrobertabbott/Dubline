# Dubline

<div align="center">

**Fully Local, Unattended Cinematic Film-Dubbing Pipeline with Automatic Stem Separation, ASR, LLM Adaptation, Voice Cloning, and Cinema Mastering.**

*No cloud inference · media processing stays on-device · optional title-only catalogue lookup*

</div>

---

## Overview

**Dubline** is an end-to-end cinematic audio dubbing system designed to translate and dub full-length feature films and video programmes into English entirely offline on local CUDA hardware. 

Drop in any video file (with optional sidecar subtitles), select your workflow policy, and receive a broadcast-ready MKV with master-quality English dialogue and music/effects (M&E) stems, complete with automated quality-control (QC) reports.

```
┌─────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ Input Video │ ──► │ Cinematic Separation  │ ──► │ Multi-pass ASR        │
│ & Subtitles │     │ (Bandit v2 / RoFormer)│     │ & Forced Alignment    │
└─────────────┘     └───────────────────────┘     └───────────┬───────────┘
                                                              │
┌─────────────┐     ┌───────────────────────┐     ┌───────────▼───────────┐
│ Delivery    │ ◄── │ Studio Match EQ &     │ ◄── │ IndexTTS 2.5 /        │
│ (MKV & QC)  │     │ Cinema Mastering      │     │ Hy-MT2 Adaptation     │
└─────────────┘     └───────────────────────┘     └───────────────────────┘
```

---

## Key Features

- **Strict Process & VRAM Isolation:** Designed to run reliably on consumer GPUs (8 GB+ VRAM). Every heavy AI model (Bandit, RoFormer, pyannote, Qwen ASR, LLM, IndexTTS) executes in its own isolated subprocess where process exit guarantees clean CUDA memory release.
- **Cinematic Stem Separation:** Uses Bandit v2 for multilingual music, effects, and speech separation, with MelBand-RoFormer and HTDemucs vocal recovery gating.
- **Audiovisual Speaker Tracking:** Fuses whole-film `pyannote.audio` diarization and CAMPPlus voiceprints with OpenCV YuNet & SFace visual active-mouth tracking for accurate character identity assignment.
- **Measured Dialogue Adaptation:** Hy-MT2 7B generates meaning-preserving candidates, while real synthesized speech duration—not character, syllable, or invented viseme counts—controls retries and final selection.
- **Independent Bilingual QC Judge:** Translations are audited by an independent Qwen 8B bilingual judge to catch hallucinated facts, omissions, or tone shifts before synthesis.
- **Prosody & Emotion-Matched TTS:** Synthesizes speech with IndexTTS 2.5 using source cue audio as performance/emotion prompts, with automatic Qwen3-TTS fallback.
- **Non-Verbal Sound Preservation:** Retains natural human expressions (laughter, gasps, crying, grunts, sighs) by selectively removing only forced-aligned spoken word spans from the original performance track.
- **Cinema Mastering & Spatial Matching:** Applies 6-band dynamic Match EQ, early-reflection convolution matching, dialogue-gated loudness normalization (Cinema −27 LUFS, EBU R128 −23 LUFS, or Web −16 LUFS), and true-peak limiting.
- **Film & TV Project Library:** Parses scene/P2P/disc filenames, Kodi-style IDs and NFO files; conservatively matches film/TV catalogues; caches covers locally; and groups every full dub, range encode and retry beneath one collapsible project.

---

## How to Use

### 1. Ingesting Media
- **Local Path Input (Recommended for Films):** Paste the absolute path to a local media file (e.g. `D:\Movies\Film.mkv`). This bypasses browser upload limits and avoids redundant file copies.
- **Drag & Drop:** Drop consumer or professional media files directly into the UI.
  - *Supported Containers:* `mkv`, `mp4`, `mov`, `avi`, `webm`, `ts`, `m2ts`, `mxf`, `wav`, `flac`, `mp3`, `m4a`, `ogg`.
- **Subtitles:** Drop optional sidecar subtitle files (`.srt`, `.vtt`, `.sub`/`.idx`). If omitted, the pipeline runs automatic speech recognition (Qwen3-ASR & Whisper) and forced alignment.
- **Project identity and covers:** Local `<filename>.nfo`, `<filename>-poster.jpg`, `poster.jpg` and `folder.jpg` files take priority. With automatic matching enabled, Dubline sends only the cleaned title/year/episode identifier—not the media path, video or voices—and caches the selected artwork under the local work directory.

### 2. Time-Range Selection (Diagnostic / Scene Dubbing)
- **Full Film:** Leave start/end empty to process the complete runtime.
- **Time Slice:** Specify a custom timecode window (e.g. `20:20` to `22:00`). Dubline automatically trims and shifts subtitles and audio while maintaining full-film context for speaker identity clustering.

### 3. Automation Policies
Choose your preferred level of automation on the web dashboard:
- **Fully Automatic (Default):** Runs unattended from ingest to final delivery. Failed cues are retried automatically with alternative candidate phrasing or fallback TTS models. Any unresolved anomalies are flagged in the final report.
- **Review Flagged Lines:** Dubs the film and presents an editorial interface showing only cues that failed QC metrics (e.g. high WER, timing skew, or low speaker similarity) for one-click re-generation or manual editing.
- **Approval Workflow:** Performs stem separation, diarization, transcription, and translation/adaptation, then pauses before synthesis. Allows you to review character names, edit translated lines, split/merge cues, or import custom glossaries.

### 4. Audio Mastering Targets
Select the target loudness standard for the delivered English audio track:
- **Cinema (−27 LUFS / −2 dBTP):** Dialogue-gated theatrical dynamic range standard.
- **EBU R128 / Broadcast (−23 LUFS / −1 dBTP):** Standard television broadcast delivery.
- **Web / Streaming (−16 LUFS / −1 dBTP):** High-loudness target for online platforms.
- **Unmastered:** Raw balanced stems without final limiter/compressor coloration.

---

## Pipeline Architecture

1. **Preflight & Ingest:** Probes containers, streams, chapters, timebases, and audio layouts. Automatically trims and synchronizes sidecar subtitles or extracts internal tracks.
2. **Dialogue & M&E Separation:** Runs Bandit v2 to extract clean speech and cinema-grade Music & Effects (M&E) stems. Employs cue-level contamination gates to recover masked lines via RoFormer.
3. **Dialogue Mapping & Forced Alignment:** Uses Qwen3-ASR and Qwen3 Forced Aligner on CUDA for word-level timestamps, falling back to Whisper Large-v3-Turbo for boundary validation.
4. **Audiovisual Diarization:** Tracks speaking faces across scenes, building a persistent character voice-bank gated by strict confidence thresholds (≥ 0.62).
5. **Contextual Translation & Adaptation:** Translates scene-by-scene with Hy-MT2, evaluating up to six duration-constrained variants against source rhythm and mouth visibility.
6. **Performance Cloning & Generation:** Extracts energy contours, pitch, and cadence to drive IndexTTS 2.5 synthesis with BF16 precision.
7. **Generate–Measure–Reject Loop:** Trims outer silence, verifies active duration, and retries overlong lines. Applies non-linear Rubber Band time-stretching strictly to voiced phrases (< 8% skew, < 5% for visible mouth close-ups).
8. **Automated Quality Control (QC):** Back-transcribes every synthesized take with Whisper Turbo to calculate WER/CER, speaker cosine similarity, and M&E bed leakage.
9. **Acoustic Finish & Delivery:** Renders multichannel/stereo English FLAC delivery tracks, muxes with the original video container, and outputs comprehensive HTML/JSON QC reports.

---

## Restrictions & Scope

- **Local Video Files Only:** Designed for local DRM-free files. It does not ingest encrypted disc images (ISO/AACS), protected stream URLs, or DRM-wrapped media.
- **Target Language:** Currently specialized for dubbing foreign-language films into **English**.
- **Hardware Architecture:** Requires an NVIDIA CUDA-capable GPU. CPU inference is supported as a fallback for LLM adaptation, but real-time separation and TTS synthesis require a CUDA device.
- **Audio Channel Boundaries:** When a discrete 5.1/7.1 M&E bed is provided, it is preserved in the final mix. For stereo sources, the pipeline outputs an enhanced spatialized stereo delivery.

---

## Inherent Limitations

- **Extreme Dramatic Vocalizations:** While IndexTTS 2.5 and Qwen3-TTS capture natural speech nuances and prosody, extreme edge cases (screaming in terror, hysterical sobbing, guttural monster sounds, or fast multi-person banter) may still lack the dramatic depth of human voice actors.
- **Dense Mono Dialogue Overlaps:** When two characters speak simultaneously over a single mono microphone track, source separation models cannot always isolate 100% clean voiceprints. In such cases, Dubline preserves the natural background audio and flags the cue as `Needs review`.
- **Processing Time:** Processing a 2-hour feature film through multi-pass neural separation, speaker clustering, LLM adaptation, and multi-take TTS synthesis is computationally intensive and will take several hours on consumer GPUs.
- **Generative Lip-Sync Scope (MuseTalk):** The optional visual lip-sync pass is deliberately conservative. It only processes clear, single-face forward angles and skips complex multi-face shots, distance scenes, or HDR/10-bit color grades to protect master picture quality.

---

## Troubleshooting & Problem Areas

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **CUDA Out of Memory (OOM)** | Background apps or browser consuming VRAM on 8 GB cards | Close VRAM-heavy applications before starting long feature films. Set `DUB_LLAMA_GPU_LAYERS=20` to offload fewer LLM layers if needed. |
| **Rubberband Audio Filter Error** | FFmpeg build missing `librubberband` library | Ensure your local FFmpeg installation includes Rubber Band support for audio time-stretching. |
| **Pyannote Model Download Fails** | Hugging Face gated access requirement | Pyannote models require accepting user conditions on Hugging Face. Generate an access token and place it in `.env` under `HF_TOKEN=your_token`. |
| **Subtitle Alignment Drifts** | Frame rate mismatch or non-standard timebases | Check that sidecar subtitles match the film's frame rate (23.976 / 24 / 25 / 29.97 fps) or remove the sidecar to let automatic ASR forced-alignment handle timing. |
| **Slow Processing on Long Media** | Working drive I/O bottlenecks | Ensure `DUB_WORKDIR` points to a fast internal NVMe/SSD drive with at least 40 GB free space. |

---

## System Requirements

- **Operating System:** Windows 10/11 (64-bit) or x86-64 Linux. macOS is detected and explained, but the current CUDA pipeline cannot run on modern Metal-only Macs.
- **GPU:** NVIDIA GPU with CUDA support (8 GB VRAM minimum; 12 GB+ recommended)
- **CPU:** 6+ cores / 12+ threads recommended
- **RAM:** 16 GB minimum (32 GB recommended)
- **Storage:** 40 GB+ free disk space for models and working caches (SSD strongly recommended)
- **External Dependencies:** Python 3.11, Git, FFmpeg (with Rubber Band audio filter support), and recent NVIDIA GPU drivers.

---

## Quick Start

### Windows: one click

Download and extract the GitHub ZIP, then double-click **`Install Dubline.cmd`**. It prepares the isolated runtime and opens the first-run wizard; the wizard checks the GPU and storage, accepts an optional Hugging Face token, downloads every selected model with resume support, and verifies the installation. Later, use **`Start Dubline.cmd`**.

### Linux: one command

After extracting or cloning the repository, run:

```sh
chmod +x start-dubline.sh && ./start-dubline.sh
```

The launcher bootstraps the isolated runtime when it is missing, then opens the same browser-based setup wizard. A current NVIDIA driver and CUDA-capable GPU are required.

### Manual Windows setup (developers)

```powershell
# Clone the repository
git clone https://github.com/leighrobertabbott/Dubline.git
cd Dubline

# Run setup (installs dependencies and model weights)
.\setup.ps1
```

### Launch the Application

Start the local server and web interface:

```powershell
.\run.ps1
```

Open your browser at **`http://127.0.0.1:8000`**.

---

## Configuration

Dubline can be configured via environment variables or by creating a `.env` file (see [`.env.example`](.env.example)):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DUB_ENGINE` | `indextts` | Primary TTS engine (`indextts` or `qwen-tts`) |
| `DUB_WORKDIR` | `./data` | Working directory for job stems and temporary files |
| `DUB_DIARIZATION_DEVICE` | `cuda` | Hardware device for diarization (`cuda` or `cpu`) |
| `DUB_LLAMA_GPU_LAYERS` | `24` | CUDA-heavy partial offload for laptop-safe LLM adaptation (`-1` = full offload) |
| `DUB_GPU_ABORT_TEMPERATURE_C` | `90` | Last-resort stop above this laptop GPU's locally reported 87°C firmware target |
| `DUB_GPU_WATCHDOG_SECONDS` | `0.5` | Interval for active NVIDIA health monitoring during every CUDA stage |
| `DUB_GPU_SOFT_TEMPERATURE_C` | `72` | Begin yielding between synthesized lines before the hard watchdog is reached |
| `DUB_GPU_COOLDOWN_TARGET_C` | `69` | Resume line generation after proactive cooling reaches this target |
| `DUB_GPU_LINE_COOLDOWN_SECONDS` | `1.0` | Minimum idle handoff between synthesized dialogue lines |
| `INDEXTTS_DIFFUSION_STEPS` | `16` | Thermal-safe IndexTTS 2.5 flow refinement on this laptop (`25` restores upstream maximum) |
| `DUB_GPU_AUTO_RESUME_LIMIT` | `12` | Maximum automatic thermal resumptions before leaving a job paused for review |
| `DUB_GPU_AUTO_RESUME_TIMEOUT_SECONDS` | `600` | How long to wait for a cool, idle, released GPU before requiring manual resume |
| `BANDIT_CHECKPOINT` | `./vendor/...` | Path to Bandit v2 separation checkpoint |
| `MUSETALK_ENABLED` | `0` | Enable optional generative visual lip-sync pass (`0` or `1`) |
| `HF_TOKEN` | *None* | Optional Hugging Face token for gated models (e.g. pyannote) |
| `MEDIA_LOOKUP_ENABLED` | `1` | Use local metadata/artwork first, then optional title-only catalogue matching |
| `DUBLINE_TMDB_TOKEN` | release managed | Application catalogue credential injected by official builds; ordinary users do not need to provide it |

---

## Outputs & Deliverables

Each completed dubbing job produces a self-contained project workspace containing:

- **`dubbed-english.mkv`** — Lossless video with original streams preserved and new English FLAC delivery track.
- **`english-dialogue.flac`** — Clean, source-positioned English dialogue stem.
- **`english-mix.flac`** — Fully balanced mix (Dialogue + M&E).
- **`cues.json`** — Detailed timeline of every cue, transcript, timing, speaker ID, and QC metrics.
- **`qc-report.html` & `qc-report.json`** — Visual and machine-readable QC audit (WER, speaker similarity, loudness, clip warnings).
- **`subtitles.srt` & `cues.csv`** — Time-aligned English subtitle cards and editorial cue sheets.
- **Per-Character Stems & WAV takes** — Individual character dialogue tracks and ZIP archive of every synthesized line.

---

## Model Acknowledgments & References

Dubline brings together world-class open-source research and foundational models:

- **TTS & Voice Cloning:** [IndexTTS](https://github.com/index-tts/index-tts) · [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- **Audio Separation:** [Bandit v2](https://github.com/kwatcharasupat/bandit-v2) · [MelBand-RoFormer](https://github.com/lucidrains/mel-band-roformer) · [Demucs](https://github.com/facebookresearch/demucs)
- **ASR & Alignment:** [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) · [OpenAI Whisper](https://github.com/openai/whisper)
- **Translation & Adaptation:** [Hy-MT2](https://github.com/Tencent-Hunyuan/HunyuanTranslation-2) · [Qwen2.5 / Qwen3](https://github.com/QwenLM/Qwen2.5) · [llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Diarization & Voiceprints:** [pyannote.audio](https://github.com/pyannote/pyannote-audio) · [CAMPPlus (ModelScope)](https://github.com/modelscope/3D-Speaker)
- **Visual Sync & Face Models:** [OpenCV YuNet & SFace](https://github.com/opencv/opencv_zoo) · [MuseTalk](https://github.com/TMElyralab/MuseTalk)

---

## License

This project is licensed under the [MIT License](LICENSE) (or applicable component licenses for bundled third-party models). Please review the respective model licenses for commercial usage terms.

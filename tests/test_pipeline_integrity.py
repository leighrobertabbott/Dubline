from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from app.services.pipeline import (
    RETAKE_EMOTION_STRENGTH,
    SOURCE_EMOTION_STRENGTH,
    make_reference,
    match_acoustics,
)
from app.services.speakers import cluster_embeddings
from app.services.visual_speakers import fuse_visual_speakers
from app.services.diarization import assign_diarized_speakers


def test_retake_emotion_strength_preserves_speaker_identity():
    """Retakes must not escalate emotion strength to 0.8 which was proven to degrade speaker similarity."""
    assert RETAKE_EMOTION_STRENGTH <= 0.6
    assert SOURCE_EMOTION_STRENGTH == 0.6


def test_make_reference_no_silence_padding(tmp_path: Path):
    """make_reference should not pad seconds of dead air onto short reference lines."""
    import soundfile as sf
    sr = 24000
    audio_file = tmp_path / "source.wav"
    out_ref = tmp_path / "ref.wav"
    t = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(audio_file, samples, sr)

    cue = {"start": 0.0, "end": 0.6}
    make_reference(audio_file, cue, out_ref, media_length=0.6, minimum=0.5)

    assert out_ref.is_file()
    ref_data, ref_sr = sf.read(out_ref)
    duration = len(ref_data) / ref_sr
    assert duration < 1.0


def test_match_acoustics_clamps_eq_and_avoids_aecho(tmp_path: Path):
    """match_acoustics should not insert synthetic aecho or extreme match EQ."""
    import soundfile as sf
    sr = 24000
    fitted_dir = tmp_path / "fitted"
    matched_dir = tmp_path / "matched"
    fitted_dir.mkdir()
    line_file = fitted_dir / "000001.wav"

    t = np.linspace(0, 1.0, sr, endpoint=False)
    samples = 0.3 * np.sin(2 * np.pi * 300 * t)
    sf.write(line_file, samples, sr)

    cues = [{
        "id": 1,
        "start": 0.0,
        "end": 1.0,
        "source_performance": {"tail_ratio": 0.8, "distance": "reverberant", "rms": 0.1},
    }]

    match_acoustics(cues, fitted_dir, matched_dir)
    assert (matched_dir / "000001.wav").is_file()
    match_meta = cues[0].get("acoustic_match", {})
    assert "early_reflections" not in match_meta
    for gain in match_meta.get("match_eq_db", {}).values():
        assert abs(gain) <= 1.5


def test_visual_speaker_override_on_high_confidence():
    """Decisive visual evidence should override audio when visual confidence is high."""
    cues_with_anchor = [
        {
            "id": 0,
            "speaker_id": 2,
            "speaker_confidence": 0.95,
            "visual_speaker": {"active_face_id": 2, "active_speaker_confidence": 0.95},
        },
        {
            "id": 1,
            "start": 10.0,
            "end": 12.0,
            "speaker_id": 1,
            "speaker_confidence": 0.70,
        }
    ]
    visual_all = {
        0: {"active_face_id": 2, "active_speaker_confidence": 0.95, "mouth_visible": True},
        1: {"active_face_id": 2, "active_speaker_confidence": 0.90, "mouth_visible": True},
    }
    fuse_visual_speakers(cues_with_anchor, visual_all)
    assert cues_with_anchor[1]["speaker_id"] == 2


def test_diarization_assigns_temporal_overlap():
    """Diarization should record temporal_overlap explicitly."""
    cues = [{"start": 0.0, "end": 2.0}]
    result = {
        "diarization": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"}]
    }
    assign_diarized_speakers(cues, result)
    assert cues[0]["speaker_id"] == 1
    assert cues[0]["temporal_overlap"] == 1.0
    assert 0.4 < cues[0]["speaker_confidence"] < 1.0

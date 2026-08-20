from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from app.services.pipeline import SAMPLE_RATE, render_timeline
from app.services.pipeline import (allocate_dub_windows, hydrate_alignment_confidence,
                                   prepare_translation_revision, reconcile_subtitles_with_asr,
                                   second_asr_evidence_candidates, translation_dispute_candidates,
                                   apply_source_asr_agreement,
                                   source_asr_verification_candidates)
from app.services.dialogue import build_adaptive_dialogue, suppress_dialogue_leakage
from app.services.adapter_worker import (choose_candidate, faithful_pass, hard_line,
                                         parse_candidates)
from app.services.llm import (decode_json, gguf_block_count, gpu_layer_count,
                              response_tokens, scrape_string_fields)
from app.services.translation_qc_worker import parse_json as parse_translation_qc_json
from app.services.qc import (PERFORMANCE_RETAKE_THRESHOLD, inspect_cues,
                             measure_performance_similarity, register_correction,
                             voice_fidelity_summary)
from app.services.qc import edit_distance
from app.services.tts_worker import fit_audio, synthesis_input_signature
from app.services.subtitles import looks_english, parse_microdvd, parse_srt
from app.services.diarization import assign_diarized_speakers
from app.services.speakers import (build_reference_bank, load_reference_variants,
                                   reference_metrics, select_reference)
from app.services.visual_speakers import fuse_visual_speakers
from app.store import JobStore


def test_srt_and_language_detection(tmp_path: Path):
    path = tmp_path / "film.srt"
    path.write_text("1\n00:00:01,250 --> 00:00:03,000\n<i>What are you doing?</i>\n\n2\n00:00:04,000 --> 00:00:05,000\nI'm here.\n", encoding="utf-8")
    cues = parse_srt(path)
    assert [{key: cue[key] for key in ("start", "end", "text")} for cue in cues] == [
        {"start": 1.25, "end": 3.0, "text": "What are you doing?"},
        {"start": 4.0, "end": 5.0, "text": "I'm here."},
    ]
    assert looks_english(cues)


def test_microdvd_subtitle_timing(tmp_path: Path):
    path = tmp_path / "film.sub"
    path.write_text("{24}{48}First line|second line\n", encoding="utf-8")
    assert parse_microdvd(path, 24) == [{"start": 1.0, "end": 2.0, "text": "First line second line"}]


def test_job_store_recovery(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create({"id": "abc", "status": "processing", "stage": "Voice"})
    assert store.recover_interrupted() == ["abc"]
    assert store.get("abc")["status"] == "queued"


def test_live_processing_job_is_not_recovered_by_a_second_server(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    folder = tmp_path / "job"; folder.mkdir()
    store.create({"id": "abc", "status": "processing", "stage": "Voice", "folder": str(folder)})
    assert store.recover_interrupted(lambda job: True) == []
    assert store.get("abc")["status"] == "processing"


def test_sample_accurate_overlapping_timeline(tmp_path: Path):
    fitted = tmp_path / "fitted"
    fitted.mkdir()
    for index, amplitude in ((1, 1000), (2, 2000)):
        with wave.open(str(fitted / f"{index:06d}.wav"), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(SAMPLE_RATE)
            wav.writeframes(np.full(SAMPLE_RATE, amplitude, dtype="<i2").tobytes())
    output = tmp_path / "timeline.wav"
    render_timeline([
        {"start": 0.5, "end": 1.5},
        {"start": 1.0, "end": 2.0},
    ], fitted, output, 2.5)
    with wave.open(str(output), "rb") as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert len(samples) == round(2.5 * SAMPLE_RATE)
    assert samples[0] == 0
    assert samples[round(.75 * SAMPLE_RATE)] == 1000
    assert samples[round(1.25 * SAMPLE_RATE)] == 3000
    assert samples[round(1.75 * SAMPLE_RATE)] == 2000


def test_adaptive_dialogue_recovers_only_dropped_lines(tmp_path: Path):
    import soundfile as sf
    rate = 24_000
    primary = np.zeros(rate * 2, dtype=np.float32)
    recovery = np.zeros_like(primary)
    recovery[:rate] = 0.02 * np.sin(2 * np.pi * 180 * np.arange(rate) / rate)
    primary[rate:] = recovery[rate:] = 0.02 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)
    sf.write(tmp_path / "primary.flac", primary, rate)
    sf.write(tmp_path / "recovery.flac", recovery, rate)
    cues = [{"start": 0, "end": 1}, {"start": 1, "end": 2}]
    summary = build_adaptive_dialogue(tmp_path / "primary.flac", tmp_path / "recovery.flac",
                                      cues, tmp_path / "adaptive.flac")
    assert summary["recovered_cues"] == 1
    assert cues[0]["dialogue_source"] == "HTDemucs recovery"
    assert cues[1]["dialogue_source"] == "Bandit cinematic"


def test_reference_quality_rejects_loud_continuous_contamination():
    rate = 16_000
    clean = np.zeros(rate * 3, dtype=np.float32)
    voice = .08 * np.sin(2 * np.pi * 190 * np.arange(rate) / rate)
    clean[rate // 2:rate + rate // 2] = voice
    rng = np.random.default_rng(7)
    contaminated = rng.normal(0, .08, rate * 3).astype(np.float32)
    cue = {"speaker_confidence": .95, "overlapping_speech": False,
           "source_performance": {"tail_ratio": .1}}
    assert reference_metrics(clean, rate, cue)["score"] > reference_metrics(contaminated, rate, cue)["score"]


def test_correlated_foreign_dialogue_is_removed_from_background_bed(tmp_path: Path):
    import soundfile as sf
    rate = 24_000
    time = np.arange(rate * 2) / rate
    dialogue = (.06 * np.sin(2 * np.pi * 210 * time)).astype(np.float32)
    music = (.02 * np.sin(2 * np.pi * 620 * time)).astype(np.float32)
    bed = np.stack((music + dialogue * .55, music + dialogue * .42), axis=1)
    sf.write(tmp_path / "dialogue.wav", dialogue, rate)
    sf.write(tmp_path / "bed.wav", bed, rate)
    cues = [{"start": 0, "end": 2, "dialogue_leakage": .8,
             "words": [{"start": 0, "end": 2}]}]
    output = tmp_path / "clean.flac"
    suppress_dialogue_leakage(cues, tmp_path / "dialogue.wav", tmp_path / "bed.wav", output)
    cleaned, _ = sf.read(output, dtype="float32", always_2d=True)
    before = abs(float(np.corrcoef(bed[:, 0], dialogue)[0, 1]))
    after = abs(float(np.corrcoef(cleaned[:, 0], dialogue)[0, 1]))
    assert after < before * .35
    assert cues[0]["dialogue_leakage_after_suppression"] < cues[0]["dialogue_leakage"]


def test_subtitle_cards_use_word_timing_not_long_asr_segment():
    subtitles = [{"start": 10.0, "end": 12.0, "text": "In lane seven."}]
    asr = [{"start": 0, "end": 20, "source": "long source", "source_language": "ja",
            "transcription_confidence": .8,
            "alignment_confidence": .91,
            "words": [{"word": "鈴", "start": 10.2, "end": 10.5, "probability": .91},
                      {"word": "木", "start": 10.5, "end": 10.8, "probability": .91}]}]
    cue = reconcile_subtitles_with_asr(subtitles, asr)[0]
    assert cue["start"] == 10.2 and cue["end"] == 10.8
    assert cue["source"] == "鈴木"
    assert cue["literal_translation"] == "In lane seven."
    assert cue["alignment_confidence"] == .91


def test_unspoken_title_and_name_cards_are_not_synthesized_as_dialogue():
    subtitles = [
        {"start": 1.0, "end": 3.0, "text": "WATERBOYS"},
        {"start": 4.0, "end": 5.0, "text": "Suzuki"},
        {"start": 6.0, "end": 7.0, "text": "Stop!"},
    ]
    cues = reconcile_subtitles_with_asr(subtitles, [])
    assert [cue["english"] for cue in cues] == ["Stop!"]


def test_sign_caption_does_not_consume_a_short_spoken_interjection():
    subtitles = [{"start": 8.0, "end": 11.0,
                  "text": "Synchronized Swim Tickets - 500 Yen"}]
    asr = [{"start": 9.0, "end": 9.3, "source": "うん", "source_language": "Japanese",
            "transcription_confidence": .9, "alignment_confidence": .92,
            "words": [{"word": "うん", "start": 9.0, "end": 9.3, "probability": .92}]}]
    cues = reconcile_subtitles_with_asr(subtitles, asr)
    assert len(cues) == 1
    assert cues[0]["source"] == "うん"
    assert cues[0]["english"] == "うん"
    assert cues[0]["subtitle_gap_recovered_by_asr"] is True


def test_translation_revision_does_not_attach_overlapping_card_to_asr_only_cue(tmp_path: Path):
    (tmp_path / "selected-subtitles.srt").write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nUm, actually...\n", encoding="utf-8")
    cues = [{"id": 1, "start": 1.4, "end": 1.8, "source": "うん", "words": [{"word": "うん"}],
             "literal_translation": "Um, actually...", "english": "Um, actually..."}]
    assert prepare_translation_revision(tmp_path, cues)
    assert not cues[0]["translation_was_supplied"]
    assert "supplied_translation" not in cues[0]
    assert cues[0]["english"] == "うん"


def test_failed_low_confidence_subtitle_conflict_gets_selective_second_asr():
    cues = [
        {"id": 1, "source_language": "Japanese", "transcription_confidence": .84,
         "supplied_translation": "Pay you back.", "translation_qc": {"passed": False}},
        {"id": 2, "source_language": "Japanese", "transcription_confidence": .96,
         "supplied_translation": "Hello.", "translation_qc": {"passed": False}},
        {"id": 3, "source_language": "Japanese", "transcription_confidence": .82,
         "supplied_translation": "Thanks.", "translation_qc": {"passed": True}},
        {"id": 4, "source_language": "Japanese", "transcription_confidence": .72,
         "supplied_translation": "Stop.", "translation_qc": {"passed": False},
         "asr_second_opinion": {"text": "やめろ", "confidence": .95}},
    ]
    assert [cue["id"] for cue in translation_dispute_candidates(cues)] == [1]


def test_second_asr_is_translated_once_before_bilingual_rejudging():
    cues = [
        {"id": 1, "asr_second_opinion": {"text": "ちょっと足りないですよ"}},
        {"id": 2, "asr_second_opinion": {
            "text": "弁償します", "english_translation": "We'll reimburse you."}},
        {"id": 3},
    ]
    assert [cue["id"] for cue in second_asr_evidence_candidates(cues)] == [1]


def test_cached_alignment_and_short_dub_windows_use_retained_evidence():
    cues = [{"start": 1.0, "end": 1.22, "subtitle_end": 1.9,
             "words": [{"probability": .94}], "mouth_visible": False},
            {"start": 3.0, "end": 4.0, "words": [{"probability": .9}]}]
    assert hydrate_alignment_confidence(cues) == 2
    allocate_dub_windows(cues, 5.0)
    assert cues[0]["alignment_confidence"] == .94
    assert cues[0]["dub_end"] >= 1.62


def _pitched_cue(source_hz: float, generated_hz: float, length: float = 2.0) -> dict:
    return {"start": 0.0, "end": length,
            "source_performance": {"pitch_hz": source_hz},
            "qc": {"generated_performance": {"pitch_hz": generated_hz}}}


def test_register_correction_targets_the_actor_pitch():
    # 180 Hz against 202 Hz is two semitones: audible, and safely shiftable.
    correction = register_correction(_pitched_cue(180.0, 202.0))
    assert correction is not None
    semitones, scale = correction
    assert semitones == pytest.approx(-2.0, abs=.05)
    assert 202.0 * scale == pytest.approx(180.0, abs=.01)


def test_register_correction_declines_outside_its_safe_band():
    # Inaudible difference: leave the take alone rather than touch the signal.
    assert register_correction(_pitched_cue(180.0, 182.0)) is None
    # Nearly an octave: shifting would cost more in artefacts than it recovers.
    assert register_correction(_pitched_cue(110.0, 205.0)) is None
    # No reliable measurement on one side.
    assert register_correction(_pitched_cue(180.0, 0.0)) is None
    # Too short for a stable median pitch.
    assert register_correction(_pitched_cue(180.0, 202.0, length=0.7)) is None


def test_pitch_drift_is_flagged_well_below_an_octave(tmp_path: Path):
    import soundfile as sf
    rate = 24_000
    time = np.arange(int(rate * 2.0)) / rate
    # A take a perfect fourth above the actor used to pass the old 10-semitone gate.
    sf.write(tmp_path / "000001.wav", (.15 * np.sin(2 * np.pi * 240 * time)).astype(np.float32), rate)
    cues = [{"id": 1, "start": 0.0, "end": 2.0, "english": "Line.",
             "source_performance": {"pitch_hz": 180.0, "rms": .1},
             "qc": {}}]
    inspect_cues(cues, tmp_path)
    assert any("semitones from the source performance" in reason
               for reason in cues[0]["review_reasons"])


def test_pitch_error_dominates_the_performance_score(tmp_path: Path):
    import soundfile as sf
    rate = 24_000
    time = np.arange(int(rate * 2.0)) / rate
    source = {"pitch_hz": 180.0, "pause_ratio": .2,
              "energy_contour": [.5, .6, .7, .8, .9, 1.0, .9, .8, .7, .6]}
    scores = {}
    for name, frequency in (("matched", 182.0), ("drifted", 240.0)):
        path = tmp_path / name; path.mkdir()
        sf.write(path / "000001.wav", (.15 * np.sin(2 * np.pi * frequency * time)).astype(np.float32), rate)
        cues = [{"id": 1, "start": 0.0, "end": 2.0, "source_performance": dict(source)}]
        measure_performance_similarity(cues, path)
        scores[name] = cues[0]["qc"]["performance_similarity"]
    assert scores["matched"] > scores["drifted"]
    # A fourth off the actor must not still read as a good performance match.
    assert scores["drifted"] < PERFORMANCE_RETAKE_THRESHOLD


def test_voice_fidelity_summary_covers_every_line_not_just_failures():
    cues = [
        {"id": 1, "performance_source": "source performance",
         "qc": {"speaker_similarity": .81, "performance_similarity": .72,
                "pitch_delta_semitones": 1.2, "pitch_correction_semitones": -1.2}},
        {"id": 2, "performance_source": "source performance", "needs_review": True,
         "qc": {"speaker_similarity": .61, "performance_similarity": .49,
                "pitch_delta_semitones": 4.8, "performance_retake": True}},
        {"id": 3, "performance_source": "scene context", "qc": {}},
    ]
    cues[0]["source_asr_agreement"] = .97
    cues[1]["source_asr_agreement"] = .41
    cues[1]["source_asr_disputed"] = True
    summary = voice_fidelity_summary(cues)
    assert summary["lines"] == 3 and summary["source_performance_lines"] == 2
    # Transcript verification is reported over the lines that were verified,
    # so an unverified cue cannot be mistaken for an agreeing one.
    assert summary["source_lines_independently_verified"] == 2
    assert summary["source_lines_disputed"] == 1
    assert summary["lines_beyond_pitch_tolerance"] == 1
    assert summary["performance_retakes_kept"] == 1
    assert summary["register_corrected_lines"] == 1
    # Medians come from the measured lines, so an unmeasured cue cannot skew them.
    assert summary["median_speaker_similarity"] == 0.71


def test_cinema_gain_is_set_from_the_mix_not_the_isolated_stem(tmp_path: Path):
    from app.services.pipeline import dialogue_gated_lufs, master_audio, measured_lufs
    bed = tmp_path / "bed.wav"; stem = tmp_path / "stem.wav"; premaster = tmp_path / "pre.flac"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "anoisesrc=d=20:c=pink:a=0.02:r=48000", str(bed)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=frequency=300:duration=20:sample_rate=48000", "-af",
                    "volume='if(between(t,2,6)+between(t,10,15),0.25,0)':eval=frame",
                    str(stem)], check=True)
    # The mix graph scales every input by 1/n, so the stem is several dB adrift
    # from the level the dialogue actually sits at once mixed.
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(bed), "-i", str(stem),
                    "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest[m]",
                    "-map", "[m]", "-c:a", "flac", str(premaster)], check=True)
    cues = [{"start": 2.0, "end": 6.0}, {"start": 10.0, "end": 15.0}]
    gated = dialogue_gated_lufs(premaster, cues)
    assert gated is not None
    assert gated < measured_lufs(stem) - 3, "the stem must not stand in for the mixed dialogue"

    result = master_audio(premaster, tmp_path / "out.flac", stem, "cinema", cues)
    assert result["measurement_basis"] == "dialogue-gated mix"
    assert result["measured_dialogue_lufs"] == gated
    # The gain must actually close the distance to the target.
    assert abs(result["dialogue_lufs_after_gain"] - -27.0) < 0.01 or result["program_gain_db"] == 12


def test_cinema_gain_falls_back_to_the_stem_without_cues(tmp_path: Path):
    from app.services.pipeline import master_audio
    stem = tmp_path / "stem.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=frequency=300:duration=6:sample_rate=48000",
                    "-af", "volume=0.25", str(stem)], check=True)
    result = master_audio(stem, tmp_path / "out.flac", stem, "cinema", None)
    assert result["measurement_basis"] == "dialogue stem"


def test_confidently_aligned_lines_are_still_verified():
    # The escalation gate measures aligner fit, so a line the aligner was happy
    # with was never independently read. It must still be a candidate.
    cues = [
        {"id": 1, "start": 0, "end": 2, "source": "入場料を取る",
         "source_language": "japanese", "transcription_confidence": .95,
         "alignment_confidence": .97, "asr_escalation_consulted": False},
        {"id": 2, "start": 2, "end": 4, "source": "Hello there",
         "source_language": "english", "transcription_confidence": .5},
        {"id": 3, "start": 4, "end": 4.2, "source": "うん",
         "source_language": "japanese", "transcription_confidence": .3},
        {"id": 4, "start": 5, "end": 8, "source": "信じられるか",
         "source_language": "japanese", "transcription_confidence": .4,
         "alignment_confidence": .6, "asr_escalation_consulted": True},
    ]
    selected = [int(cue["id"]) for cue in source_asr_verification_candidates(cues)]
    assert 1 in selected, "a confidently aligned line is still unverified"
    assert 2 not in selected, "English source needs no translation check"
    assert 3 not in selected, "too short for a reliable second reading"
    assert selected == sorted(selected), "candidates keep cue order"


def test_verification_budget_goes_to_the_least_trustworthy_lines(monkeypatch):
    monkeypatch.setenv("DUB_SOURCE_VERIFY_MAX_LINES", "1")
    cues = [
        {"id": 1, "start": 0, "end": 3, "source": "あ", "source_language": "ja",
         "transcription_confidence": .97, "alignment_confidence": .96,
         "asr_escalation_consulted": True},
        {"id": 2, "start": 3, "end": 6, "source": "い", "source_language": "ja",
         "transcription_confidence": .31, "alignment_confidence": .40,
         "asr_escalation_consulted": True},
    ]
    assert [int(cue["id"]) for cue in source_asr_verification_candidates(cues)] == [2]


def test_disputed_transcript_stops_being_treated_as_settled():
    cues = [{"id": 1, "start": 0, "end": 3, "source": "入場料を取る",
             "transcription_confidence": .95},
            {"id": 2, "start": 3, "end": 6, "source": "信じられるか",
             "transcription_confidence": .95}]
    reliable = {"confidence": .9, "word_confidence": .93, "no_speech_probability": .02}
    disputed = apply_source_asr_agreement(cues, {
        1: {**reliable, "text": "入場料を取る"},
        2: {**reliable, "text": "入院料を捨てる"},
    })
    assert disputed == 1
    # Agreement leaves the confident line alone and carries the reading forward.
    assert not cues[0].get("source_asr_disputed")
    assert cues[0]["transcription_confidence"] == .95
    assert cues[0]["asr_second_opinion"]["text"]
    # Disagreement de-rates the transcript so downstream QC treats it as open.
    assert cues[1]["source_asr_disputed"] is True
    assert cues[1]["transcription_confidence"] <= .55
    # Both readings must be available to the translator as competing evidence.
    assert second_asr_evidence_candidates(cues) == cues


def test_a_perfect_no_speech_score_is_not_treated_as_the_worst_case():
    # 0.0 is the best possible no-speech probability and is falsy: a default-on-
    # falsy read would reject exactly the readings worth trusting.
    cues = [{"id": 1, "start": 0, "end": 3, "source": "勉強します",
             "transcription_confidence": .9}]
    disputed = apply_source_asr_agreement(cues, {1: {
        "text": "弁償します", "confidence": .86,
        "word_confidence": .96, "no_speech_probability": 0.0}})
    assert disputed == 1, "a confident reading must not be discarded by a falsy zero"
    assert cues[0]["source_asr_disputed"] is True


def test_near_identical_readings_are_still_contested_in_a_logographic_script():
    # One character separates "we will study" from "we will compensate"; a
    # character-similarity threshold would score this as agreement.
    cues = [{"id": 1, "start": 0, "end": 3,
             "source": "入場料を取って勉強します",
             "transcription_confidence": .9}]
    apply_source_asr_agreement(cues, {1: {
        "text": "入場料を取って弁償します",
        "confidence": .86, "word_confidence": .96, "no_speech_probability": 0.0}})
    assert cues[0]["source_asr_agreement"] > .8, "characters barely differ"
    assert cues[0]["source_asr_disputed"] is True, "but the meaning does"
    # De-rated enough to reach the judge, not so far it demands human eyes.
    assert .55 < cues[0]["transcription_confidence"] <= .7


def test_identical_readings_corroborate_rather_than_contest():
    cues = [{"id": 1, "start": 0, "end": 3, "source": "おい佐藤",
             "transcription_confidence": .9}]
    assert apply_source_asr_agreement(cues, {1: {
        "text": "おい、佐藤。", "confidence": .9,
        "word_confidence": .95, "no_speech_probability": 0.0}}) == 0
    assert cues[0]["transcription_confidence"] == .9


def test_a_wholesale_disagreement_de_rates_further_than_a_near_miss():
    def confidence_after(second: str) -> float:
        cues = [{"id": 1, "start": 0, "end": 3, "source": "おい佐藤",
                 "transcription_confidence": .9}]
        apply_source_asr_agreement(cues, {1: {
            "text": second, "confidence": .9, "word_confidence": .95,
            "no_speech_probability": 0.0}})
        return cues[0]["transcription_confidence"]
    # A hallucinated reading shares almost nothing with the transcript.
    assert confidence_after("ご視聴ありがとう") < confidence_after("おい佐藤さん")


def test_an_unreliable_second_reading_never_de_rates_a_transcript():
    cues = [{"id": 1, "start": 0, "end": 3, "source": "入場料を取る",
             "transcription_confidence": .95}]
    apply_source_asr_agreement(cues, {1: {
        "text": "totally different", "confidence": .2,
        "word_confidence": .3, "no_speech_probability": .8}})
    assert not cues[0].get("source_asr_disputed")
    assert cues[0]["transcription_confidence"] == .95


def test_qc_flags_large_stretch_and_word_mismatch(tmp_path: Path):
    import soundfile as sf
    fitted = tmp_path / "fitted"; fitted.mkdir()
    sf.write(fitted / "000001.wav", np.ones(SAMPLE_RATE, dtype=np.float32) * .02, SAMPLE_RATE)
    cues = [{"start": 0, "end": 1,
             "qc": {"stretch_percent": 12, "timing_pass": True, "padding_ms": 0, "truncated_ms": 0,
                    "word_similarity": .4, "wer": .5, "cer": .4,
                    "backtranscription": "wrong", "active_duration": .2},
             "speaker_confidence": .9, "reference_quality": .8, "timing_confidence": .9,
             "transcription_confidence": .9, "adaptation_confidence": .9, "alignment_confidence": .9,
             "translation_qc": {"available": True, "passed": True}}]
    summary = inspect_cues(cues, fitted)
    assert summary["flagged_count"] == 1
    assert len(cues[0]["review_reasons"]) == 2


def test_retained_english_translation_is_not_marked_uncertain():
    selected, confidence = choose_candidate("On your mark!", ["Yo-i!"], 0.79, True)
    assert selected == "On your mark!"
    assert confidence >= 0.75


def test_semantic_judge_rejects_short_but_wrong_adaptation():
    literal = "Sato, return the festival money tomorrow."
    candidates = ["Sato, bring it tomorrow.", "Sato, pay the festival money tomorrow."]
    semantic = [
        {"index": 0, "adequacy": .98, "terminology": 1.0, "register": .9},
        {"index": 1, "adequacy": .38, "terminology": .4, "register": .9},
        {"index": 2, "adequacy": .94, "terminology": .95, "register": .9},
    ]
    selected, _ = choose_candidate(literal, candidates, 2.1, True, semantic=semantic)
    assert selected != candidates[0]


def test_adaptation_candidates_survive_an_unescaped_quote():
    # The exact defect that aborted a run with JSONDecodeError: a quote inside
    # dialogue that the model never escaped.
    reply = ('{"natural":"He said "no" to the festival","compact":"He refused.",'
             '"fuller":"He told them no, plainly.","same_meaning":"He said no.",'
             '"rhythmic":"He said no.","literal":"He said no to it."}')
    candidates = parse_candidates(reply)
    assert 'He said "no" to the festival' in candidates
    assert "He refused." in candidates


def test_adaptation_candidates_ignore_surrounding_model_prose():
    reply = ('Sure! Here you go:\n```json\n'
             '{"natural":"Come here.","compact":"Here."}\n```\nHope that helps.')
    assert parse_candidates(reply) == ["Come here.", "Here."]


def test_broken_json_never_raises():
    assert decode_json("no json at all") is None
    assert decode_json('{"a": 1,,}') is None
    assert decode_json("") is None
    # A wanted type is found even when the model wrapped it in another container.
    assert decode_json('[{"id": 1}]', dict) == {"id": 1}
    assert decode_json('{"scores": [1, 2]}', list) == [1, 2]


def test_scrape_recovers_numbered_scene_lines():
    reply = '{"0":"We will charge "admission" at the gate.","1":"How can I count on that!"}'
    values = scrape_string_fields(reply, ["0", "1"])
    assert values["0"] == 'We will charge "admission" at the gate.'
    assert values["1"] == "How can I count on that!"


class _ScriptedLlama:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def create_chat_completion(self, messages, **options):
        self.prompts.append(messages[0]["content"])
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


def test_supplied_subtitle_is_reconciled_against_the_source(capsys):
    cues = [{"id": 1, "start": 0, "end": 3, "source": "文化祭で入場料を取る",
             "source_language": "japanese",
             "supplied_translation": "We'll perform at the festival."}]
    # The condensed subtitle drops the admission fee; the source states it.
    llm = _ScriptedLlama(['{"0":"We will charge admission at the cultural festival."}'])
    faithful_pass(llm, cues)
    capsys.readouterr()
    assert cues[0]["english"] == "We will charge admission at the cultural festival."
    assert cues[0]["translation_evidence"].startswith("source transcript reconciled")
    assert "SUPPLIED SUBTITLE" in llm.prompts[0] and "入場料" in llm.prompts[0]


def test_english_source_subtitle_is_kept_without_a_model_call(capsys):
    cues = [{"id": 1, "start": 0, "end": 2, "source": "Get in.",
             "source_language": "english", "supplied_translation": "Get in."}]
    llm = _ScriptedLlama([])
    faithful_pass(llm, cues)
    capsys.readouterr()
    assert llm.prompts == []
    assert cues[0]["english"] == "Get in."


def test_a_failed_reconciliation_keeps_the_subtitle(capsys):
    cues = [{"id": 1, "start": 0, "end": 3, "source": "行くぞ", "source_language": "japanese",
             "supplied_translation": "Let's go."}]
    # Unusable scene reply, then an unusable single-line retry.
    llm = _ScriptedLlama(["I cannot help with that.", "   "])
    faithful_pass(llm, cues)
    capsys.readouterr()
    assert cues[0]["english"] == "Let's go."


def _fake_gguf(path: Path, blocks: int, architecture: str = "llama") -> Path:
    """Write a GGUF v3 header carrying one string and one block_count value."""
    import struct

    def string(value: bytes) -> bytes:
        return struct.pack("<Q", len(value)) + value

    body = b""
    body += string(b"general.architecture") + struct.pack("<I", 8) + string(architecture.encode())
    body += string(f"{architecture}.block_count".encode()) + struct.pack("<I", 4)
    body += struct.pack("<I", blocks)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                     + struct.pack("<Q", 2) + body)
    return path


def test_gguf_block_count_is_read_from_the_header(tmp_path: Path):
    assert gguf_block_count(_fake_gguf(tmp_path / "m.gguf", 32)) == 32
    assert gguf_block_count(_fake_gguf(tmp_path / "q.gguf", 36, "qwen3")) == 36
    # Anything that is not a GGUF must degrade, never raise.
    (tmp_path / "not.gguf").write_bytes(b"nope")
    assert gguf_block_count(tmp_path / "not.gguf") is None
    assert gguf_block_count(tmp_path / "missing.gguf") is None


def test_gpu_layer_split_tracks_available_vram(tmp_path: Path, monkeypatch):
    from app.services import gpu_safety
    model = _fake_gguf(tmp_path / "m.gguf", 32)
    # Give the file a realistic weight size so the budget maths is meaningful.
    import os

    os.truncate(model, 4400 * 1024 * 1024)
    monkeypatch.delenv("DUB_LLAMA_GPU_LAYERS", raising=False)

    def at(free_mb: int):
        monkeypatch.setattr(gpu_safety, "query_nvidia",
                            lambda *a, **k: {"free_mb": free_mb, "total_mb": 8188,
                                             "temperature_c": 50, "utilization": 0, "power_w": 10.0})
        return gpu_layer_count(model, 8192)

    # A model that comfortably fits is fully offloaded.
    assert at(7900) == -1
    # A tighter card gets a partial split rather than an arbitrary fixed number,
    # and the split shrinks monotonically as VRAM does.
    splits = [at(free) for free in (6000, 4000, 2500)]
    assert all(0 <= value <= 32 for value in splits)
    assert splits == sorted(splits, reverse=True)
    # An explicit setting always wins over the measurement.
    monkeypatch.setenv("DUB_LLAMA_GPU_LAYERS", "12")
    assert gpu_layer_count(model, 8192) == 12


def test_reply_budget_grows_with_the_scene():
    short = response_tokens("a short line", floor=700, ceiling=3072)
    long = response_tokens("x" * 6000, floor=700, ceiling=3072)
    assert short == 700
    assert long == 3072
    assert response_tokens("y" * 2000, floor=700, ceiling=3072) > 700


def test_translation_qc_accepts_json_with_model_text_after_it():
    value = parse_translation_qc_json(
        '<think>done</think>\n[{"id": 1, "passed": true}]\nThe end.'
    )
    assert value == [{"id": 1, "passed": True}]


def test_timing_retry_can_limit_adaptation_to_failed_lines():
    cue = {"start": 0, "end": .5, "english": "This would normally be too long.",
           "_skip_adaptation": True}
    assert not hard_line(cue)


def test_tts_cache_signature_changes_when_edited_dialogue_changes():
    item = {"text": "The first translation.", "target": 1.2, "reference": "voice.wav",
            "emotion_strength": .6, "language": "EN", "fit_limit_percent": 8.0}
    first = synthesis_input_signature(item)
    item["text"] = "The corrected translation."
    assert synthesis_input_signature(item) != first


def test_visible_mouth_uses_tighter_timing_gate(tmp_path: Path):
    import soundfile as sf
    fitted = tmp_path / "fitted"; fitted.mkdir()
    tone = np.ones(SAMPLE_RATE, dtype=np.float32) * .02
    sf.write(fitted / "000001.wav", tone, SAMPLE_RATE)
    sf.write(fitted / "000002.wav", tone, SAMPLE_RATE)
    cues = [
        {"start": 0, "end": 1, "qc": {"stretch_percent": 6, "timing_pass": True, "padding_ms": 0,
          "truncated_ms": 0, "word_similarity": 1, "wer": 0, "cer": 0,
          "backtranscription": "ok", "active_duration": .2}, "speaker_confidence": .9,
         "reference_quality": .8, "mouth_visible": False, "timing_confidence": .9,
         "transcription_confidence": .9, "adaptation_confidence": .9, "alignment_confidence": .9,
         "translation_qc": {"available": True, "passed": True}},
        {"start": 0, "end": 1, "qc": {"stretch_percent": 6, "timing_pass": True, "padding_ms": 0,
          "truncated_ms": 0, "word_similarity": 1, "wer": 0, "cer": 0,
          "backtranscription": "ok", "active_duration": .2}, "speaker_confidence": .9,
         "reference_quality": .8, "mouth_visible": True, "timing_confidence": .9,
         "transcription_confidence": .9, "adaptation_confidence": .9, "alignment_confidence": .9,
         "translation_qc": {"available": True, "passed": True}},
    ]
    inspect_cues(cues, fitted)
    assert not cues[0]["needs_review"]
    assert cues[1]["needs_review"]
    assert "exceeds 5%" in cues[1]["review_reasons"][0]


def test_repeated_active_face_can_split_a_tentative_audio_cluster():
    cues = [
        {"id": 1, "start": 0, "end": .5, "speaker_id": 2, "speaker_confidence": .4},
        {"id": 2, "start": .6, "end": 1.1, "speaker_id": 2, "speaker_confidence": .45},
        {"id": 3, "start": 1.2, "end": 2.0, "speaker_id": 2, "speaker_confidence": .9},
    ]
    visual = {
        1: {"active_face_id": 8, "active_speaker_confidence": .96, "mouth_visible": True},
        2: {"active_face_id": 8, "active_speaker_confidence": .94, "mouth_visible": True},
        3: {"active_face_id": 5, "active_speaker_confidence": .95, "mouth_visible": True},
    }
    summary = fuse_visual_speakers(cues, visual)
    assert summary["created_visual_voices"] == 1
    assert cues[0]["speaker_id"] == cues[1]["speaker_id"] != cues[2]["speaker_id"]
    assert cues[0]["speaker_assignment"].startswith("repeated active face")


def test_silence_aware_fit_applies_only_bounded_slowing_and_fails_large_undershoot(tmp_path: Path):
    import soundfile as sf
    rate = 24_000
    audio = np.zeros(rate, dtype=np.float32)
    audio[round(.2 * rate):round(.5 * rate)] = .08 * np.sin(
        2 * np.pi * 180 * np.arange(round(.3 * rate)) / rate)
    source = tmp_path / "short.wav"; output = tmp_path / "fitted.wav"
    sf.write(source, audio, rate)
    metrics = fit_audio(source, output, 1.0)
    fitted, fitted_rate = sf.read(output)
    assert fitted_rate == rate and len(fitted) == rate
    assert -8.1 <= metrics["stretch_percent"] < 0
    assert metrics["padding_ms"] > 400
    assert not metrics["timing_pass"]


def test_error_rates_use_real_edit_distance():
    assert edit_distance(["in", "lane", "seven"], ["in", "line", "seven"]) == 1
    assert edit_distance(list("mark"), list("marks")) == 1


def test_index_native_rate_is_resampled_to_timeline_rate(tmp_path: Path):
    import soundfile as sf
    rate = 22_050
    values = .05 * np.sin(2 * np.pi * 190 * np.arange(rate) / rate)
    source = tmp_path / "native.wav"; output = tmp_path / "timeline.wav"
    sf.write(source, values, rate)
    fit_audio(source, output, .8)
    fitted, fitted_rate = sf.read(output)
    assert fitted_rate == SAMPLE_RATE
    assert len(fitted) == round(.8 * SAMPLE_RATE)


def test_full_context_diarization_preserves_four_scene_voices():
    cues = [{"start": 62.7, "end": 65.0}, {"start": 66.8, "end": 68.2},
            {"start": 85.0, "end": 88.4}, {"start": 99.9, "end": 100.7}]
    turns = [{"start": 62.65, "end": 65.0, "speaker": "SPEAKER_03"},
             {"start": 66.75, "end": 68.2, "speaker": "SPEAKER_02"},
             {"start": 84.96, "end": 88.42, "speaker": "SPEAKER_04"},
             {"start": 99.89, "end": 100.72, "speaker": "SPEAKER_00"}]
    assign_diarized_speakers(cues, {"diarization": turns, "exclusive_diarization": turns})
    assert [cue["speaker_id"] for cue in cues] == [1, 2, 3, 4]
    assert all(cue["speaker_assignment"] == "confident" for cue in cues)


def test_uncertain_speaker_does_not_contaminate_character_bank(tmp_path: Path):
    import soundfile as sf
    audio = tmp_path / "dialogue.wav"
    sf.write(audio, np.ones(SAMPLE_RATE * 3, dtype=np.float32) * .02, SAMPLE_RATE)
    cues = [{"start": 0.0, "end": 1.0, "speaker_id": 1, "speaker_confidence": .9,
             "reference_quality": .8, "overlapping_speech": False},
            {"start": 1.0, "end": 3.0, "speaker_id": 2, "speaker_confidence": .32,
             "reference_quality": .8, "overlapping_speech": False}]
    references = build_reference_bank(audio, cues, tmp_path / "references")
    assert set(references) == {1}


def test_character_bank_keeps_acoustically_distinct_takes(tmp_path: Path):
    import soundfile as sf
    rate = SAMPLE_RATE
    # One character heard twice: a low, quiet delivery and a high, loud one.
    def voiced(frequency: float, amplitude: float, seconds: float) -> np.ndarray:
        time = np.arange(int(rate * seconds)) / rate
        wave = sum(amplitude / (harmonic ** 1.4)
                   * np.sin(2 * np.pi * frequency * harmonic * time)
                   for harmonic in range(1, 12))
        return wave.astype(np.float32)

    audio = tmp_path / "dialogue.wav"
    # A real gap between deliveries, so neither read bleeds into the other.
    silence = np.zeros(int(rate * 1.0), dtype=np.float32)
    sf.write(audio, np.concatenate([voiced(110, .10, 4.0), silence, voiced(210, .30, 4.0)]), rate)
    cues = [{"start": 0.0, "end": 4.0, "speaker_id": 1, "speaker_confidence": .9,
             "reference_quality": .8, "overlapping_speech": False, "source": "Quiet line."},
            {"start": 5.0, "end": 9.0, "speaker_id": 1, "speaker_confidence": .9,
             "reference_quality": .8, "overlapping_speech": False, "source": "Loud line."}]
    references = build_reference_bank(audio, cues, tmp_path / "references")
    variants = load_reference_variants(tmp_path / "references")
    assert len(variants.get(1, [])) >= 2, "distinct deliveries must not collapse to one take"

    def chosen(pitch: float) -> str:
        path, basis = select_reference(
            {"source_performance": {"pitch_hz": pitch}}, variants[1], references[1])
        assert basis.startswith("nearest take")
        return path.name

    # Each line reaches for the take that matches how it was actually delivered.
    assert chosen(112.0) != chosen(208.0)

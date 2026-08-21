from __future__ import annotations

"""Two-pass scene translation and cue adaptation using local Hy-MT2."""

import argparse
import json
import math
import re
from pathlib import Path

from app.services.llm import decode_json, load_llama, response_tokens, scrape_string_fields


# Measured from real IndexTTS takes rendered at a natural rate (factor 0.95-1.10)
# on this pipeline: 3.39 words/second. The previous 2.65 constant overstated
# spoken length by ~28%, which made every duration target handed to the LLM too
# generous and left hard_line() firing on nearly every cue.
NATURAL_WORDS_PER_SECOND = 3.39
# Isometric dubbing keeps each line within +/-10% of the length that fits its
# window; IWSLT scores exactly this as "length compliance".
LENGTH_TOLERANCE = 0.10


def estimated_pace_seconds(text: str) -> float:
    """Predict spoken duration from text, calibrated against measured synthesis."""
    words = len(re.findall(r"[\w']+", text))
    return max(0.35, words / NATURAL_WORDS_PER_SECOND
               + text.count(",") * 0.06 + text.count(".") * 0.08)


def word_budget(target_seconds: float) -> tuple[int, int, int]:
    """Words that fit a window at a natural rate, with the isometric band."""
    ideal = max(1.0, target_seconds * NATURAL_WORDS_PER_SECOND)
    return (max(1, round(ideal * (1 - LENGTH_TOLERANCE))), round(ideal),
            max(1, round(ideal * (1 + LENGTH_TOLERANCE))))


def length_ratio(text: str, target_seconds: float) -> float:
    """How far this line is from fitting its window. 1.0 is exact."""
    return estimated_pace_seconds(text) / max(target_seconds, .05)


# Kept as a compatibility alias for cached manifests and external callers.
predicted_seconds = estimated_pace_seconds


def hard_line(cue: dict) -> bool:
    if cue.get("_skip_adaptation"):
        return False
    target = max(0.3, float(cue.get("dub_end", cue["end"]))
                 - float(cue.get("dub_start", cue["start"])))
    ratio = estimated_pace_seconds(cue.get("faithful_translation") or cue.get("literal_translation")
                              or cue.get("english", "")) / target
    return bool(cue.get("force_adaptation")) or ratio > 1.06 or not cue.get("translation_was_supplied", True)


CANDIDATE_NAMES = ("natural", "compact", "fuller", "same_meaning", "rhythmic", "literal")


def json_value(content: str, want: type | tuple[type, ...] = (dict, list)):
    """Tolerant decode that returns None instead of aborting the whole dub."""
    return decode_json(content, want)


def parse_candidates(content: str) -> list[str]:
    value = json_value(content, dict)
    if not isinstance(value, dict):
        # The reply was not decodable JSON, usually an unescaped quote inside a
        # line of dialogue.  Recover the fields rather than losing every variant.
        value = scrape_string_fields(content, CANDIDATE_NAMES)
    return list(dict.fromkeys(" ".join(value[name].split()) for name in CANDIDATE_NAMES
            if isinstance(value.get(name), str) and value[name].strip()
            and value[name].strip().lower() not in {"true", "false", "null"}))


def protected_names(text: str) -> set[str]:
    common = {"The", "A", "An", "In", "On", "At", "To", "From", "He", "She", "It",
              "We", "They", "I", "You", "This", "That", "What", "How", "Why", "Ready"}
    return {name for name in re.findall(r"\b[A-Z][A-Za-z'-]+\b", text) if name not in common}


def rhythm_score(text: str, cue: dict | None = None) -> float:
    cue = cue or {}
    punctuation_pauses = len(re.findall(r"[,;:—…]", text)) + text.count("...")
    expected = round(float(cue.get("source_performance", {}).get("pause_ratio", .15)) * 4)
    return max(0.0, 1.0 - abs(punctuation_pauses - expected) / 4)


def choose_candidate(literal: str, candidates: list[str], target: float,
                     translation_is_target: bool, semantic: list[dict] | None = None,
                     cue: dict | None = None) -> tuple[str, float]:
    required_names = protected_names(literal)
    cue = cue or {}
    all_candidates = list(dict.fromkeys([literal, *candidates]))
    semantic = semantic or []
    judged = {int(item.get("index", -1)): item for item in semantic if isinstance(item, dict)}
    best, best_score, best_quality = literal, float("inf"), 0.0
    for index, candidate in enumerate(all_candidates):
        evidence = judged.get(index, {})
        # Never use character overlap as a proxy for preserved meaning.  An
        # unjudged rewrite is ineligible; the faithful source remains safe.
        adequacy = float(evidence.get("adequacy", 1.0 if index == 0 and translation_is_target else 0.0))
        terminology = float(evidence.get("terminology", 1.0 if index == 0 else 0.0))
        register = float(evidence.get("register", .9 if index == 0 else 0.0))
        missing_names = len([name for name in required_names if name.lower() not in candidate.lower()])
        ratio = length_ratio(candidate, target)
        # Inside the isometric band costs nothing; outside it grows sharply,
        # because the synthesizer can only absorb so much before a line is
        # warped or truncated.
        duration_error = min(1.5, max(0.0, abs(ratio - 1.0) - LENGTH_TOLERANCE) * 3.0)
        rhythm_loss = 1 - rhythm_score(candidate, cue)
        mouth_visible = bool(cue.get("mouth_visible"))
        if mouth_visible:
            score = ((1 - adequacy) * .53 + duration_error * .15
                     + (1 - terminology) * .18 + (1 - register) * .10 + rhythm_loss * .04)
        else:
            score = ((1 - adequacy) * .58 + duration_error * .10 + (1 - terminology) * .18
                     + (1 - register) * .10 + rhythm_loss * .04)
        score += missing_names * 1.5
        acceptable = adequacy >= .62 and terminology >= .65 and missing_names == 0
        if acceptable and score < best_score:
            best, best_score, best_quality = candidate, score, adequacy
    if not math.isfinite(best_score):
        best, best_score, best_quality = literal, .35, .75
    confidence = max(0.0, min(1.0, .55 * (1 - best_score) + .45 * best_quality))
    if translation_is_target and best == literal:
        confidence = max(confidence, 0.75)
    return best, confidence


def judge_candidates(llm, cue: dict, faithful: str, candidates: list[str]) -> list[dict]:
    values = list(dict.fromkeys([faithful, *candidates]))
    numbered = "\n".join(f"{i}: {value}" for i, value in enumerate(values))
    prompt = f"""Judge English dubbing candidates against the source and faithful translation.
Return only a JSON array with one object per candidate:
{{"index":0,"adequacy":0.0,"terminology":0.0,"register":0.0}}
Scores are 0 to 1. Adequacy means all meaning and facts are preserved. Penalize additions,
omissions, wrong names and changed intent. Terminology covers names and scene terms. Register
covers tone and relationship. Do not prefer brevity by itself.
Source: {cue.get('source', '')}
Selective second ASR opinion: {json.dumps(cue.get('asr_second_opinion') or {}, ensure_ascii=False)}
Supplied subtitle evidence: {cue.get('supplied_translation', '')}
Independent rejection to resolve: {cue.get('translation_qc_feedback', '')}
Faithful English: {faithful}
Candidates:
{numbered}"""
    budget = 200 + 70 * len(values)
    value = json_value(ask(llm, prompt, budget), list)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def ask(llm, prompt: str, max_tokens: int = 900) -> str:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], temperature=0.05, top_p=0.9,
        max_tokens=max_tokens,
    )
    return str(response["choices"][0]["message"]["content"])


def scene_batches(cues: list[dict], size: int = 12) -> list[list[int]]:
    return [list(range(start, min(len(cues), start + size))) for start in range(0, len(cues), size)]


def retranslate_line(llm, cues: list[dict], index: int) -> str:
    """Translate one line that the scene reply missed, keeping its neighbours.

    The surrounding lines carry the names, pronouns and register the line needs;
    translating it alone is the single largest source of drifted meaning.
    """
    window = cues[max(0, index - 3):min(len(cues), index + 4)]
    context = chr(10).join(
        f"{offset + max(0, index - 3)}: {item.get('source', '')}"
        for offset, item in enumerate(window)
    )
    cue = cues[index]
    prompt = f"""Translate line {index} of this film scene into idiomatic English.
Keep the names, facts, polarity, register and continuity of the surrounding lines.
Return only the English sentence, with no numbering, quotes or commentary.
Supplied subtitle evidence (fallible): {cue.get('supplied_translation', '')}
Scene ({cue.get('source_language') or 'unknown'}):
{context}

Line {index}: {cue.get('source', '')}"""
    return ask(llm, prompt, response_tokens(cue.get("source", ""), floor=180, ceiling=400, factor=2.2))


FAITHFUL_SHARE = 0.35


def report(progress: float, index: int) -> None:
    print(json.dumps({"progress": max(0.0, min(1.0, progress)), "index": index}), flush=True)


def scene_line(cues: list[dict], index: int) -> str:
    """Render one scene line, refusing to privilege a transcript under dispute.

    When two transcribers disagree the primary is not the source of truth, it is
    one candidate reading.  Presenting it as SOURCE with the other buried in a
    JSON blob asks the model to translate the error faithfully, which is exactly
    what it then does.
    """
    cue = cues[index]
    language = cue.get("source_language") or "unknown"
    second = cue.get("asr_second_opinion") or {}
    rows = []
    if cue.get("source_asr_disputed") and str(second.get("text") or "").strip():
        rows.append(f"{index} CONTESTED ({language}): two transcribers disagree on this line."
                    " Neither reading is authoritative; choose the one the scene supports.")
        rows.append(f"{index}   READING A (primary): {cue.get('source', '')}")
        rows.append(f"{index}   READING B (independent, word confidence "
                    f"{float(second.get('word_confidence') or 0):.2f}): {second.get('text', '')}")
        english = str(second.get("english_translation") or "").strip()
        if english:
            rows.append(f"{index}   READING B already rendered in English: {english}")
        rows.append(f"{index}   The primary transcriber mishears similar-sounding words;"
                    " prefer the reading that is coherent in context.")
    else:
        rows.append(f"{index} SOURCE ({language}): {cue.get('source', '')}")
        if str(second.get("text") or "").strip():
            rows.append(f"{index} SECOND ASR OPINION (agrees): {second.get('text', '')}")
    if cue.get("supplied_translation"):
        rows.append(f"{index} SUPPLIED SUBTITLE (fallible evidence): {cue['supplied_translation']}")
    if cue.get("translation_qc_feedback"):
        rows.append(f"{index} PRIOR INDEPENDENT QC: {cue['translation_qc_feedback']}")
    return chr(10).join(rows)


def faithful_pass(llm, cues: list[dict]) -> None:
    batches = scene_batches(cues)
    for batch_number, batch in enumerate(batches, 1):
        untranslated = []
        for index in batch:
            cue = cues[index]
            # Only a real distributor subtitle counts as supplied evidence.
            # Back-filling this from english/literal_translation promoted the
            # model's own previous output to "independent evidence", which then
            # corroborated itself on every later pass and in the QC judge.
            supplied = cue.get("supplied_translation")
            cue["translation_was_supplied"] = bool(supplied or cue.get("translation_is_target", False))
            source_language = str(cue.get("source_language") or "").lower()
            source_needs_translation = source_language not in {"", "english"}
            correction_requested = bool(cue.get("force_translation_correction"))
            if supplied and not source_needs_translation:
                # The source is already English, so the subtitle card is a
                # transcript rather than a translation.  Nothing to reconcile.
                cue["faithful_translation"] = supplied
                cue["literal_translation"] = supplied
                cue["english"] = supplied
                cue["translation_is_target"] = True
                cue["translation_model"] = "English subtitle transcript v3"
                continue
            # A supplied subtitle is evidence, not the answer.  Distribution
            # subtitles are condensed for reading speed and routinely drop the
            # main action, so every one is reconciled against the source below.
            needs_faithful_pass = (source_needs_translation and (
                not cue.get("faithful_translation")
                or correction_requested
                or cue.get("translation_model") != "Hy-MT2-7B Q4 · source-faithful v3"
            )) or (not source_needs_translation and not cue.get("translation_is_target", True))
            if needs_faithful_pass:
                untranslated.append(index)
            elif cue["translation_was_supplied"] and not cue.get("faithful_translation"):
                cue["faithful_translation"] = cue.get("literal_translation") or cue.get("english", "")
        if not untranslated:
            report(FAITHFUL_SHARE * batch_number / max(1, len(batches)), batch[-1])
            continue
        numbered = chr(10).join(scene_line(cues, index) for index in batch)
        wanted = ", ".join(str(index) for index in untranslated)
        prompt = f"""Translate this complete film scene faithfully into idiomatic English.
Maintain names, terminology, facts, register, jokes, relationships and continuity across lines.
The source transcript can contain homophone/recognition errors. A supplied professional subtitle is
independent but fallible evidence: use scene continuity and both signals to resolve conflicts; do not
copy it blindly and do not reject it merely because of a plausible ASR homophone. Distribution subtitles
are condensed for reading speed: restore any action, object, quantity or clause the subtitle dropped but
the source states, and remove anything the subtitle added that the source does not support. A high-confidence
selective second ASR opinion is independent evidence and should resolve such a dispute. Do not shorten
for timing and never romanize instead of translating.
A line marked CONTESTED has no trusted transcript: weigh both readings against the scene and
translate the one that actually makes sense, rather than defaulting to the primary.
A line marked CONTESTED has no trusted transcript: weigh both readings against the scene and
translate the one that actually makes sense rather than defaulting to the primary reading.
Return only a JSON object mapping these requested line numbers to English strings: {wanted}
Scene:
{numbered}"""
        budget = response_tokens(numbered, floor=700, ceiling=2048, factor=0.6)
        reply = ask(llm, prompt, budget)
        value = json_value(reply, dict)
        if not isinstance(value, dict):
            value = scrape_string_fields(reply, [str(index) for index in untranslated])
        for index in untranslated:
            translated = value.get(str(index))
            if not isinstance(translated, str) or not translated.strip():
                # Retranslate the single line inside its scene rather than bare:
                # a context-free retry is where continuity and names get lost.
                translated = retranslate_line(llm, cues, index)
            translated = " ".join(str(translated).strip().strip('"').split())
            supplied = str(cues[index].get("supplied_translation") or "").strip()
            if not translated:
                # A failed reconciliation must never leave the line empty; the
                # condensed subtitle is still better than silence.
                translated = supplied or str(cues[index].get("english") or "").strip()
            if not translated:
                continue
            cues[index]["faithful_translation"] = translated
            cues[index]["english"] = translated
            cues[index]["translation_is_target"] = True
            cues[index]["translation_model"] = "Hy-MT2-7B Q4 · source-faithful v3"
            cues[index]["translation_evidence"] = (
                "source transcript reconciled with supplied subtitle" if supplied
                else "source transcript only")
            cues[index]["force_translation_correction"] = False
        report(FAITHFUL_SHARE * batch_number / max(1, len(batches)), batch[-1])


def adaptation_pass(llm, cues: list[dict], output: Path) -> None:
    difficult = [index for index, cue in enumerate(cues) if hard_line(cue)]
    if not difficult:
        report(1.0, max(0, len(cues) - 1))
    for position, index in enumerate(difficult):
        cue = cues[index]
        faithful = cue.get("faithful_translation") or cue.get("literal_translation") or cue.get("english", "")
        target = max(0.3, float(cue.get("dub_end", cue["end"]))
                     - float(cue.get("dub_start", cue["start"])))
        context = cues[max(0, index - 5):min(len(cues), index + 6)]
        context_text = "\n".join(
            f"{offset + max(0, index - 5)}: {item.get('faithful_translation') or item.get('english', '')}"
            for offset, item in enumerate(context)
        )
        prior_qc = cue.get("qc", {})
        previous_error = float(prior_qc.get("duration_error_percent") or 0.0)
        previous_padding = float(prior_qc.get("padding_ms") or 0.0)
        timing_limit = 5.0 if cue.get("mouth_visible") else 8.0
        too_long = previous_error > timing_limit
        too_short = previous_error < -timing_limit or previous_padding > 160
        urgency = (f"A measured spoken take was {abs(previous_error):.1f}% too long; include genuinely compact choices. "
                   if too_long else
                   f"A measured spoken take was {abs(previous_error):.1f}% too short; include fuller natural choices without adding meaning. "
                   if too_short else "")
        budget_low, budget_ideal, budget_high = word_budget(target)
        prompt = f"""Adapt one already-translated film line for dubbing. Do not translate it again.
Return strict JSON with six distinct versions:
{{"natural":"...","compact":"...","fuller":"...","same_meaning":"...","rhythmic":"...","literal":"..."}}
All values must be idiomatic English, never transliteration. Preserve meaning, names, facts and register.
When the source transcript and supplied subtitle conflict, explicitly resolve the independent rejection
below using the surrounding scene; do not merely paraphrase the rejected draft.
{urgency}LENGTH BUDGET: this line is spoken in {target:.2f} seconds, which at a natural delivery is {budget_low}-{budget_high} words (aim for {budget_ideal}). Every version must land in that range: a line outside it cannot be spoken naturally in its slot and will be rewritten or cut. Shorten by removing padding and redundancy, never by dropping meaning.
Mouth clearly visible: {bool(cue.get('mouth_visible'))}. When true, preserve the source line's starts, stops and emphasis points; do not claim phonetic lip matching.
Source transcript: {cue.get('source', '')}
Selective second ASR opinion: {json.dumps(cue.get('asr_second_opinion') or {}, ensure_ascii=False)}
Supplied subtitle evidence: {cue.get('supplied_translation', '')}
Independent rejection to resolve: {cue.get('translation_qc_feedback', '')}
Faithful English translation: {faithful}
Scene context:
{context_text}"""
        candidates = parse_candidates(
            ask(llm, prompt, response_tokens(faithful, floor=320, ceiling=1024, factor=9.0)))
        semantic = judge_candidates(llm, cue, faithful, candidates)
        selected, confidence = choose_candidate(faithful, candidates, target, True,
                                                 semantic=semantic, cue=cue)
        if too_long or too_short:
            required_names = protected_names(faithful)
            judged = {int(item.get("index", -1)): item for item in semantic if isinstance(item, dict)}
            # The judge numbered the deduplicated [faithful, *candidates] list.
            # Enumerating the raw candidates instead shifts every score by one
            # whenever a candidate repeats the faithful line.
            judged_order = list(dict.fromkeys([faithful, *candidates]))
            shorter = []
            for candidate in candidates:
                candidate_index = judged_order.index(candidate) if candidate in judged_order else -1
                evidence = judged.get(candidate_index, {})
                adequacy = float(evidence.get("adequacy", 0.0))
                terminology = float(evidence.get("terminology", 0.0))
                faithful_pace = estimated_pace_seconds(faithful)
                candidate_pace = estimated_pace_seconds(candidate)
                directional = (candidate_pace < faithful_pace * .90 if too_long
                               else candidate_pace > faithful_pace * 1.08)
                if (directional and adequacy >= .65 and terminology >= .65
                        and all(name.lower() in candidate.lower() for name in required_names)):
                    duration_fit = 1 - min(1.0, abs(candidate_pace - target) / target)
                    shorter.append((.65 * adequacy + .2 * terminology + .15 * duration_fit, candidate))
            if shorter:
                selected = max(shorter)[1]
                confidence = max(confidence, 0.7)
        cue["translation_candidates"] = candidates
        cue["candidate_semantic_scores"] = semantic
        cue["adaptation_scoring"] = "semantic + terminology + register; approximate pace is a low-weight prefilter; synthesized duration is measured"
        cue["adapted_dialogue"] = selected; cue["english"] = selected
        cue["adaptation_confidence"] = round(confidence, 3)
        cue["adaptation_attempts"] = 1 + len(candidates)
        cue["adaptation_model"] = "Hy-MT2-7B Q4 · duration pass"
        cue["force_adaptation"] = False
        if position % 10 == 9 or position == len(difficult) - 1:
            output.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"index": index, "difficult": len(difficult),
                          "progress": FAITHFUL_SHARE + (1 - FAITHFUL_SHARE)
                                      * (position + 1) / max(1, len(difficult))}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8")); cues = spec["cues"]
    # On Windows the CUDA llama wheel resolves cuBLAS from the already-tested
    # PyTorch runtime.  Importing torch first registers those dependent DLLs.
    import torch  # noqa: F401
    llm = load_llama(spec["model"], n_ctx=8192)
    faithful_pass(llm, cues)
    adaptation_pass(llm, cues, args.output)
    for cue in cues:
        cue.setdefault("faithful_translation", cue.get("literal_translation") or cue.get("english", ""))
        cue.setdefault("adapted_dialogue", cue.get("english", ""))
        cue.setdefault("adaptation_confidence", 0.9); cue.setdefault("adaptation_attempts", 1)
    args.output.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"progress": 1.0, "complete": True}), flush=True)


if __name__ == "__main__":
    main()

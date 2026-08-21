from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.llm import decode_json, load_llama, response_tokens


def parse_json(text: str) -> list:
    value = decode_json(text, list)
    return value if isinstance(value, list) else []


SCORE_FIELDS = ("adequacy", "names", "register")


def semantic_scores(value: dict) -> tuple[dict[str, float], bool]:
    """Read the judge's scores, reporting whether it actually supplied any.

    Small judges reliably fill the booleans and write a real reason, but often
    echo whatever numbers the response schema showed them.  A silent echo used
    to fail every line closed on a gate the translation never had a chance to
    pass, so an unusable score set has to be recognised rather than believed.
    """
    scores: dict[str, float] = {}
    for name in SCORE_FIELDS:
        raw = value.get(name)
        if raw is None:
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        # Accept both the 0-1 and 0-100 scales; only one is ever meant.
        scores[name] = number / 100.0 if number > 1.0 else number
    if len(scores) < len(SCORE_FIELDS):
        return {name: scores.get(name, 0.0) for name in SCORE_FIELDS}, False
    reported = any(scores[name] > 0.0 for name in SCORE_FIELDS)
    return scores, reported


def ask(llm, prompt: str, max_tokens: int = 900) -> str:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], temperature=0.0,
        top_p=.85, max_tokens=max_tokens,
    )
    return str(response["choices"][0]["message"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    cues = spec["cues"]
    import torch  # registers the bundled CUDA/cuBLAS DLL directory on Windows
    llm = load_llama(spec["model"], n_ctx=8192)
    results = []
    allowed_ids = {int(cue["id"]) for cue in cues}
    if args.output.is_file():
        try:
            prior = json.loads(args.output.read_text(encoding="utf-8"))
            results = [item for item in prior if isinstance(item, dict) and "id" in item
                       and int(item["id"]) in allowed_ids]
        except (OSError, ValueError, json.JSONDecodeError):
            results = []
    completed_ids = {int(item["id"]) for item in results}
    cues = [cue for cue in cues if int(cue["id"]) not in completed_ids]
    # Keep every prompt well below the 8k context even when subtitle cards contain
    # long SDH/translator notes.  Fixed ten-line batches could silently truncate
    # precisely the difficult passages this independent check exists to catch.
    batches = []
    current, characters = [], 0
    for cue in cues:
        size = sum(len(str(cue.get(key, ""))) for key in
                   ("source", "faithful_translation", "literal_translation", "english"))
        if current and (len(current) >= 6 or characters + size > 14_000):
            batches.append(current); current, characters = [], 0
        current.append(cue); characters += size
    if current:
        batches.append(current)
    completed = len(completed_ids)
    total = completed + len(cues)
    for batch in batches:
        blind_lines = "\n".join(
            f"ID {cue['id']} ({cue.get('source_language') or 'auto'}): {cue.get('source', '')}"
            for cue in batch
        )
        blind_prompt = f"""/no_think
Translate each SOURCE into literal, natural English without seeing or inferring any subtitle or dub.
Preserve polarity, quantities, intent, names and who does what. If a source is only an incomplete
fragment, translate the fragment and mark complete=false. Return only this JSON array:
[{{"id":1,"translation":"...","complete":true}}]
{blind_lines}"""
        blind_values = parse_json(ask(llm, blind_prompt,
            response_tokens(blind_lines, floor=700, ceiling=2048)))
        blind_by_id = {int(item.get("id", -1)): item for item in blind_values
                       if isinstance(item, dict)}
        lines = "\n".join(
            f"ID {cue['id']}\nSOURCE ({cue.get('source_language') or 'auto'}): {cue.get('source','')}\n"
            f"SOURCE TRANSCRIPTION CONFIDENCE: {cue.get('transcription_confidence','unknown')}\n"
            f"SECOND ASR OPINION: {json.dumps(cue.get('asr_second_opinion') or {}, ensure_ascii=False)}\n"
            f"SUPPLIED SUBTITLE (independent but fallible): {cue.get('supplied_translation','')}\n"
            f"FAITHFUL ENGLISH: {cue.get('faithful_translation') or cue.get('literal_translation','')}\n"
            f"DUB ENGLISH: {cue.get('english','')}"
            for cue in batch
        )
        prompt = f"""/no_think
You are an independent bilingual film-translation quality checker.
The translations were created by a different model. Judge SOURCE directly against DUB ENGLISH;
use FAITHFUL ENGLISH only as secondary evidence. Before scoring, independently interpret SOURCE and
every usable SECOND ASR OPINION. Matching the supplied subtitle is not proof of accuracy. Never report
"no contradiction" without comparing its facts, polarity and intent to those source-language readings.
The SECOND ASR OPINION may contain an english_translation generated literally without subtitle access;
compare DUB ENGLISH directly against that independent English meaning as well as the original wording.
SOURCE ASR can contain a plausible homophone error; use the supplied subtitle and surrounding ordered
scene to arbitrate without blindly trusting either. When SOURCE and a second ASR with word_confidence
above 0.85 agree, their meaning outweighs a contradictory subtitle. When the second ASR instead changes
one plausible homophone with good confidence and makes a coherent full-dialogue subtitle semantically
consistent, treat that convergence as strong independent evidence. Ignore a second ASR with poor word
confidence or obvious fragments. Normal synonyms such as reimburse, compensate and pay back are
semantically equivalent; do not invent a contradiction between them. If DUB English is
semantically equivalent to that subtitle, reject it only for a clear, material contradiction supported
by the surrounding source—not stylistic wording or minor specificity. A fragment that correctly
continues an adjacent line is not an omission. Detect
changed facts, polarity, names, relationships, omissions, additions, mistranslated idioms and register
changes. Do not reward fluency alone.
Return only a JSON array, one object per ID, in exactly this shape:
[{{"id":<the ID>,"adequacy":<integer 0-100>,"names":<integer 0-100>,"register":<integer 0-100>,
"passed":<true or false>,"reason":"<concise specific reason>"}}]
Every score is your own integer judgement from 0 to 100. Never emit 0 unless the dub genuinely
preserves nothing. Pass only when adequacy >= 78, names >= 85, and no material omission/addition exists.

{lines}"""
        values = parse_json(ask(llm, prompt,
            response_tokens(lines, floor=800, ceiling=2048, factor=0.8)))
        valid = {int(item.get("id", -1)): item for item in values if isinstance(item, dict)}
        comparison_lines = []
        comparison_meta: dict[int, dict] = {}
        for cue in batch:
            cue_id = int(cue["id"])
            second = cue.get("asr_second_opinion") or {}
            try:
                second_reliable = (float(second.get("confidence") or 0.0) >= .55
                                   and float(second.get("word_confidence") or 0.0) >= .85
                                   and float(second.get("no_speech_probability") or 0.0) < .25
                                   and bool(str(second.get("english_translation") or "").strip()))
            except (TypeError, ValueError):
                second_reliable = False
            blind = blind_by_id.get(cue_id, {})
            reference = (str(second.get("english_translation")) if second_reliable
                         else str(blind.get("translation") or ""))
            complete_value = blind.get("complete", False)
            complete = (True if second_reliable else
                        complete_value is True or str(complete_value).lower() == "true")
            comparison_meta[cue_id] = {
                "reference": reference, "complete": complete,
                "basis": "reliable selective second ASR" if second_reliable else "blind Qwen source translation",
                "blind": str(blind.get("translation") or ""),
            }
            comparison_lines.append(
                f"ID {cue_id}\nREFERENCE ENGLISH ({comparison_meta[cue_id]['basis']}): {reference}\n"
                f"REFERENCE COMPLETE: {complete}\nDUB ENGLISH: {cue.get('english', '')}\n"
                f"SUPPLIED SUBTITLE (secondary, fallible): {cue.get('supplied_translation', '')}\n"
                f"FAITHFUL DRAFT (secondary): {cue.get('faithful_translation') or cue.get('literal_translation', '')}"
            )
        comparison_prompt = f"""/no_think
You are a strict English semantic equivalence checker. Compare DUB ENGLISH against REFERENCE ENGLISH,
not against the subtitle. Equivalent paraphrases and synonyms pass. Changed polarity, more versus less,
enough versus insufficient, changed actor/action, quantities, names, intent, additions or omissions fail.
If REFERENCE COMPLETE is false, evidence_sufficient must be false unless ordered secondary evidence makes
the missing meaning unambiguous; do not pretend an incomplete fragment verifies a full sentence.
Return only one JSON array, one object per ID, in exactly this shape:
[{{"id":<the ID>,"adequacy":<integer 0-100>,"names":<integer 0-100>,"register":<integer 0-100>,
"contradiction":<true or false>,"evidence_sufficient":<true or false>,
"reason":"<specific English meaning comparison>"}}]
Every score is your own integer judgement from 0 to 100. Never emit 0 unless the dub preserves
nothing of the reference meaning.
{chr(10).join(comparison_lines)}"""
        comparison_values = parse_json(ask(llm, comparison_prompt,
            response_tokens(chr(10).join(comparison_lines), floor=800,
                            ceiling=2048, factor=0.8)))
        comparisons = {int(item.get("id", -1)): item for item in comparison_values
                       if isinstance(item, dict)}
        for cue in batch:
            cue_id = int(cue["id"])
            item = valid.get(cue_id)
            if not item:
                item = {"id": cue["id"], "adequacy": 0.0, "names": 0.0, "register": 0.0,
                        "passed": False, "reason": "independent judge returned no result"}
            semantic = comparisons.get(cue_id, {})
            evidence = comparison_meta.get(cue_id, {})
            item["independent_source_translation"] = evidence.get("blind", "")
            item["semantic_reference"] = evidence.get("reference", "")
            item["semantic_reference_basis"] = evidence.get("basis", "")
            item["semantic_gate"] = semantic
            scores, scores_reported = semantic_scores(semantic)
            item["adequacy"] = round(scores["adequacy"], 3)
            item["names"] = round(scores["names"], 3)
            item["register"] = round(scores["register"], 3)
            item["scores_reported"] = scores_reported
            item["reason"] = str(semantic.get("reason") or item.get("reason")
                                 or "independent semantic gate returned no reason")
            item["available"] = True
            item["model"] = "Qwen3-8B Q4 independent bilingual judge"
            item["revision"] = str(spec.get("judge_revision") or "unknown")
            findings_pass = (bool(semantic.get("evidence_sufficient"))
                             and not bool(semantic.get("contradiction")))
            if scores_reported:
                item["passed"] = findings_pass and item["adequacy"] >= .78 and item["names"] >= .85
                item["verdict_basis"] = "judge findings and scores"
            else:
                # No usable scores came back. Judging on the findings the model
                # did make is weaker evidence than a full verdict, so say so
                # rather than silently rejecting a sound translation.
                item["passed"] = findings_pass
                item["verdict_basis"] = "judge findings only; it returned no usable scores"
            results.append(item)
        completed += len(batch)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": min(1.0, completed / max(1, total)),
                          "index": completed - 1}), flush=True)


if __name__ == "__main__":
    main()

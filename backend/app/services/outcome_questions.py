"""
Outcome questions for ensemble runs.

Every replicate ends with a poll: each simulated agent answers 1-3 questions
about how the situation has come out. The answers are what the ensemble turns
into distributions, so the questions are part of an ensemble's definition.

Question schema::

    {"id": "q1", "text": "...", "type": "probability"}                       # 0-100
    {"id": "q2", "text": "...", "type": "choice", "options": ["a", "b", ...],  # 2-8 options
     "stance_map": {"a": "supportive", "b": "opposing"}}                       # optional
    {"id": "q3", "text": "...", "type": "number", "unit": "%"}               # unit required

``stance_map`` (choice questions only) maps options onto the stance vocabulary
used by the agents, which is what lets the aggregator count stance drift.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction
from ..utils.logger import get_logger

logger = get_logger('mirofish.outcome_questions')

MAX_QUESTIONS = 3
QUESTION_TYPES = ("probability", "choice", "number")
STANCES = ("supportive", "opposing", "neutral", "observer")
MIN_OPTIONS = 2
MAX_OPTIONS = 8
MAX_TEXT_CHARS = 500
MAX_OPTION_CHARS = 60
MAX_UNIT_CHARS = 20

_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
#: JSON key reserved by the poll reply format
_RESERVED_IDS = {"reason"}


def validate_outcome_questions(questions: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate and normalize a list of outcome questions.

    Returns ``(normalized, errors)``; the questions must not be used when
    ``errors`` is non-empty. Missing ids become ``q1``, ``q2`` ... An empty
    list is valid and means "no poll" (behavioural metrics only).
    """
    if not isinstance(questions, list):
        return [], ["outcome_questions must be a list"]
    if len(questions) > MAX_QUESTIONS:
        return [], [f"at most {MAX_QUESTIONS} outcome questions are allowed"]

    given = {
        q.get("id") for q in questions
        if isinstance(q, dict) and isinstance(q.get("id"), str)
    }
    counter = 0

    def fresh_id() -> str:
        nonlocal counter
        while True:
            counter += 1
            if f"q{counter}" not in given:
                given.add(f"q{counter}")
                return f"q{counter}"

    errors: List[str] = []
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for index, question in enumerate(questions):
        label = f"question {index + 1}"
        if not isinstance(question, dict):
            errors.append(f"{label}: must be an object")
            continue

        qid = question.get("id")
        if isinstance(qid, str) and qid.strip():
            qid = qid.strip()
        else:
            qid = fresh_id()
        if not _ID_PATTERN.match(qid) or qid in _RESERVED_IDS:
            errors.append(f"{label}: id '{qid}' must start with a letter, use letters/digits/underscores (max 32) and not be reserved")
            continue
        if qid in seen:
            errors.append(f"{label}: duplicate id '{qid}'")
            continue
        seen.add(qid)

        text = question.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{label}: text must be a non-empty string")
            continue
        text = text.strip()
        if len(text) > MAX_TEXT_CHARS:
            errors.append(f"{label}: text is longer than {MAX_TEXT_CHARS} characters")
            continue

        qtype = question.get("type")
        if qtype not in QUESTION_TYPES:
            errors.append(f"{label}: type must be one of {', '.join(QUESTION_TYPES)}")
            continue

        item: Dict[str, Any] = {"id": qid, "text": text, "type": qtype}

        if qtype == "choice":
            options = question.get("options")
            if (
                not isinstance(options, list)
                or not all(isinstance(o, str) and o.strip() for o in options)
            ):
                errors.append(f"{label}: options must be a list of non-empty strings")
                continue
            options = [o.strip() for o in options]
            if len({o.lower() for o in options}) != len(options):
                errors.append(f"{label}: options must be unique (ignoring case)")
                continue
            if not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
                errors.append(f"{label}: provide {MIN_OPTIONS}-{MAX_OPTIONS} options")
                continue
            if any(len(o) > MAX_OPTION_CHARS for o in options):
                errors.append(f"{label}: an option is longer than {MAX_OPTION_CHARS} characters")
                continue
            item["options"] = options

            stance_map = question.get("stance_map")
            if stance_map is not None:
                by_lower = {o.lower(): o for o in options}
                if not isinstance(stance_map, dict):
                    errors.append(f"{label}: stance_map must be an object")
                    continue
                cleaned_map = {}
                bad = False
                for option, stance in stance_map.items():
                    canonical = by_lower.get(str(option).strip().lower())
                    if canonical is None or stance not in STANCES:
                        errors.append(
                            f"{label}: stance_map entry '{option}': one of the options -> "
                            f"one of {', '.join(STANCES)}"
                        )
                        bad = True
                        break
                    cleaned_map[canonical] = stance
                if bad:
                    continue
                if cleaned_map:
                    item["stance_map"] = cleaned_map

        elif qtype == "number":
            unit = question.get("unit")
            if not isinstance(unit, str) or not unit.strip():
                errors.append(f"{label}: a number question needs a unit")
                continue
            if len(unit.strip()) > MAX_UNIT_CHARS:
                errors.append(f"{label}: unit is longer than {MAX_UNIT_CHARS} characters")
                continue
            item["unit"] = unit.strip()

        normalized.append(item)

    return normalized, errors


def default_outcome_questions(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A generic stance question, used when none were supplied and none could be derived.

    It is always meaningful for a simulation (it re-asks, at the end, the
    stance every agent started with) and it makes stance drift measurable.
    """
    topic = ((config.get("event_config") or {}).get("topic") or "").strip()
    if not topic:
        topic = str(config.get("simulation_requirement") or "this topic").strip()[:120]
    return [{
        "id": "q1",
        "text": f'What is your current stance on "{topic}"?',
        "type": "choice",
        "options": ["supportive", "neutral", "opposing"],
        "stance_map": {"supportive": "supportive", "neutral": "neutral", "opposing": "opposing"},
    }]


def _derivation_messages(config: Dict[str, Any]) -> List[Dict[str, str]]:
    requirement = str(config.get("simulation_requirement") or "")
    topic = ((config.get("event_config") or {}).get("topic") or "").strip()
    agents = "\n".join(
        f"- {a.get('entity_name')} ({a.get('entity_type')}), initial stance: {a.get('stance', 'neutral')}"
        for a in (config.get("agent_configs") or [])[:30]
    )
    system = (
        "You design the end-of-simulation poll for a social-media simulation. Every simulated "
        "participant will answer your questions once, as themselves, after the simulation. "
        "Return ONLY a JSON object."
    )
    user = f"""Simulation requirement:
{requirement}

Topic: {topic or '(not specified)'}

Participants:
{agents}

Write at most {MAX_QUESTIONS} outcome questions that capture how this situation has come out, as the
participants themselves would answer them. Prefer:
- one `probability` question (0-100) about the main outcome the requirement asks about, and
- one `choice` question about the participant's current stance, with 2-5 options and a `stance_map`
  that maps each option onto one of: supportive, opposing, neutral.
Do not leak any expected outcome into the wording of the questions.

Return JSON of this shape:
{{"questions": [
  {{"id": "q1", "text": "...", "type": "probability"}},
  {{"id": "q2", "text": "...", "type": "choice", "options": ["..."], "stance_map": {{"<option>": "supportive|opposing|neutral"}}}},
  {{"id": "q3", "text": "...", "type": "number", "unit": "..."}}
]}}

{get_language_instruction()}
Keep ids, types, options' stance_map values and field names exactly as specified; only the question
text, option labels and units use the language above."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def derive_outcome_questions(
    config: Dict[str, Any],
    llm_json: Optional[Callable[[List[Dict[str, str]]], Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Derive outcome questions from the simulation requirement with one LLM call.

    Returns ``(questions, source)`` where source is ``"llm"`` or ``"fallback"``.
    The reply is schema-validated (at most 3 questions); on any LLM or schema
    failure a generic stance question is returned instead, so creating an
    ensemble never fails just because the model misbehaved. The caller shows the
    questions to the user, who can edit them before the ensemble starts.
    """
    if llm_json is None:
        client = LLMClient()

        def llm_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
            return client.chat_json(messages, temperature=0.3, max_attempts=2)

    try:
        reply = llm_json(_derivation_messages(config))
        questions, errors = validate_outcome_questions(reply.get("questions"))
        if errors or not questions:
            raise ValueError("; ".join(errors) or "the model returned no questions")
        return questions, "llm"
    except Exception as error:
        logger.warning(f"Could not derive outcome questions, using the default: {error}")
        return default_outcome_questions(config), "fallback"

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

LABELS = frozenset({"supported", "contradicted", "unknown"})
MODEL_CLAIM_FIELDS = frozenset({"quote", "label", "fact_ids", "reason", "context_before", "context_after"})


class InvalidAgentOutput(ValueError):
    pass


def _reason_sentence_count(reason: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", reason.strip()) if part]) if reason.strip() else 0


def _find_occurrences(text: str, quote: str) -> list[int]:
    positions: list[int] = []
    offset = 0
    while True:
        position = text.find(quote, offset)
        if position < 0:
            return positions
        positions.append(position)
        offset = position + 1


def _locate_quote(
    text: str,
    quote: str,
    used: set[tuple[int, int]],
    context_before: str | None,
    context_after: str | None,
) -> tuple[int, int]:
    candidates = [position for position in _find_occurrences(text, quote) if (position, position + len(quote)) not in used]
    if context_before is not None:
        candidates = [position for position in candidates if text[:position].endswith(context_before)]
    if context_after is not None:
        candidates = [position for position in candidates if text[position + len(quote):].startswith(context_after)]
    if not candidates:
        raise InvalidAgentOutput("claim quote is not uniquely locatable in the source answer")
    if len(candidates) > 1 and context_before is None and context_after is None:
        raise InvalidAgentOutput("repeated claim quote requires source context")
    if len(candidates) > 1:
        raise InvalidAgentOutput("source context does not uniquely locate repeated claim quote")
    start = candidates[0]
    return start, start + len(quote)


def validate_model_claims(
    answer_id: str,
    answer_text: str,
    raw_claims: Sequence[Mapping[str, Any]],
    catalogue_fact_ids: frozenset[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw_claims, Sequence) or isinstance(raw_claims, (str, bytes)):
        raise InvalidAgentOutput("claims must be a list")
    accepted: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    previous_start = -1
    for raw in raw_claims:
        if not isinstance(raw, Mapping) or not {"quote", "label", "fact_ids", "reason"}.issubset(raw):
            raise InvalidAgentOutput("model claim is missing required fields")
        if not set(raw).issubset(MODEL_CLAIM_FIELDS):
            raise InvalidAgentOutput("model claim contains prohibited or unknown fields")
        quote = raw.get("quote")
        label = raw.get("label")
        reason = raw.get("reason")
        fact_ids = raw.get("fact_ids")
        context_before = raw.get("context_before")
        context_after = raw.get("context_after")
        if not isinstance(quote, str) or not quote or quote != quote.strip():
            raise InvalidAgentOutput("claim quote must be nonempty and have no edge whitespace")
        if quote[-1] not in ".!?":
            raise InvalidAgentOutput("claim quote must preserve its sentence-ending punctuation")
        if label not in LABELS:
            raise InvalidAgentOutput("claim label is invalid")
        if not isinstance(reason, str) or not reason.strip() or _reason_sentence_count(reason) > 2:
            raise InvalidAgentOutput("claim reason must contain one or two sentences")
        if not isinstance(fact_ids, list) or any(not isinstance(item, str) for item in fact_ids):
            raise InvalidAgentOutput("fact_ids must be a list of strings")
        if len(fact_ids) != len(set(fact_ids)) or not set(fact_ids).issubset(catalogue_fact_ids):
            raise InvalidAgentOutput("claim contains duplicate or unknown fact IDs")
        if context_before is not None and not isinstance(context_before, str):
            raise InvalidAgentOutput("context_before must be a string")
        if context_after is not None and not isinstance(context_after, str):
            raise InvalidAgentOutput("context_after must be a string")

        start, end = _locate_quote(answer_text, quote, used, context_before, context_after)
        if start < 0 or end > len(answer_text) or answer_text[start:end] != quote:
            raise InvalidAgentOutput("claim offsets or quote are invalid")
        if start < previous_start:
            raise InvalidAgentOutput("claims must be sorted by source position")
        if accepted and start < accepted[-1]["end_char"]:
            raise InvalidAgentOutput("claims must not overlap")
        used.add((start, end))
        previous_start = start
        accepted.append({
            "claim_id": f"{answer_id}:{start}:{end}",
            "start_char": start,
            "end_char": end,
            "quote": quote,
            "label": label,
            "fact_ids": list(fact_ids),
            "reason": reason.strip(),
        })
    return accepted
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

LABELS = ("supported", "contradicted", "unknown")
MATRIX_ROWS = (*LABELS, "no_gold")
MATRIX_COLUMNS = (*LABELS, "missing_or_error")


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _claim_id(claim: Mapping[str, Any]) -> str:
    return str(claim.get("claim_id") or f"{claim['answer_id']}:{claim['start_char']}:{claim['end_char']}")


def evaluate_split(
    answer_ids: Sequence[str],
    gold_claims: Sequence[Mapping[str, Any]] | None,
    answer_outcomes: Sequence[Mapping[str, Any]],
    answers: Mapping[str, str],
) -> dict[str, Any]:
    """Score accepted predictions against gold using exact source spans only.

    Pass ``gold_claims=None`` for an unlabeled check split. Outcome records must
    contain one terminal row per scheduled answer, including failed answers.
    """
    scheduled = list(answer_ids)
    if len(scheduled) != len(set(scheduled)):
        raise ValueError("scheduled answer IDs must be unique")

    outcome_by_id: dict[str, Mapping[str, Any]] = {}
    for outcome in answer_outcomes:
        answer_id = str(outcome["answer_id"])
        if answer_id in outcome_by_id:
            raise ValueError(f"duplicate answer outcome: {answer_id}")
        outcome_by_id[answer_id] = outcome
    if set(outcome_by_id) != set(scheduled):
        missing = sorted(set(scheduled) - set(outcome_by_id))
        unexpected = sorted(set(outcome_by_id) - set(scheduled))
        raise ValueError(f"answer outcomes do not match schedule; missing={missing}, unexpected={unexpected}")

    answer_errors = sorted(
        answer_id for answer_id, outcome in outcome_by_id.items()
        if str(outcome.get("status", "success")) != "success"
    )
    predictions: list[Mapping[str, Any]] = []
    for answer_id in scheduled:
        outcome = outcome_by_id[answer_id]
        claims = outcome.get("claims", [])
        if str(outcome.get("status", "success")) != "success" and claims:
            raise ValueError(f"failed answer has accepted claims: {answer_id}")
        for claim in claims:
            if str(claim["answer_id"]) != answer_id:
                raise ValueError(f"claim answer_id does not match its outcome: {answer_id}")
            if claim.get("label") not in LABELS:
                raise ValueError(f"invalid prediction label: {claim.get('label')}")
            body = answers.get(answer_id)
            if body is None:
                raise ValueError(f"missing source answer: {answer_id}")
            start, end = int(claim["start_char"]), int(claim["end_char"])
            if start < 0 or end < start or end > len(body) or body[start:end] != claim.get("quote"):
                raise ValueError(f"prediction quote does not match source span: {_claim_id(claim)}")
            if claim.get("claim_id") not in (None, f"{answer_id}:{start}:{end}"):
                raise ValueError(f"prediction claim_id does not match its source span: {_claim_id(claim)}")
            predictions.append(claim)

    if not scheduled:
        if gold_claims:
            raise ValueError("an empty schedule cannot include gold claims")
        return {
            "gold_count": 0,
            "prediction_count": 0,
            "matched_count": 0,
            "correct_count": 0,
            "extraction": None,
            "joint": None,
            "per_label": None,
            "macro_f1": None,
            "gold_claim_coverage": None,
            "confusion_matrix": None,
            "missed_gold_ids": None,
            "unmatched_prediction_ids": None,
            "wrong_label_matches": None,
            "answer_error_count": 0,
            "answer_error_ids": [],
            "per_answer": {},
        }

    if gold_claims is None:
        return {
            "gold_count": None,
            "prediction_count": len(predictions),
            "matched_count": None,
            "correct_count": None,
            "extraction": None,
            "joint": None,
            "per_label": None,
            "macro_f1": None,
            "gold_claim_coverage": None,
            "confusion_matrix": None,
            "missed_gold_ids": None,
            "unmatched_prediction_ids": None,
            "wrong_label_matches": None,
            "answer_error_count": len(answer_errors),
            "answer_error_ids": answer_errors,
            "per_answer": None,
        }

    gold_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for gold in gold_claims:
        answer_id = str(gold["answer_id"])
        start, end = int(gold["start_char"]), int(gold["end_char"])
        key = (answer_id, start, end)
        if key in gold_by_key:
            raise ValueError(f"duplicate gold span: {_claim_id(gold)}")
        if gold.get("expected_label", gold.get("label")) not in LABELS:
            raise ValueError(f"invalid gold label: {_claim_id(gold)}")
        body = answers.get(answer_id)
        if body is None or body[start:end] != gold.get("quote"):
            raise ValueError(f"gold quote does not match source span: {_claim_id(gold)}")
        gold_by_key[key] = gold

    pred_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    pred_ids: set[str] = set()
    for prediction in predictions:
        key = (str(prediction["answer_id"]), int(prediction["start_char"]), int(prediction["end_char"]))
        if key in pred_by_key:
            raise ValueError(f"duplicate prediction span: {_claim_id(prediction)}")
        prediction_id = _claim_id(prediction)
        if prediction_id in pred_ids:
            raise ValueError(f"duplicate prediction ID: {prediction_id}")
        pred_ids.add(prediction_id)
        pred_by_key[key] = prediction

    matched_keys = set(gold_by_key) & set(pred_by_key)
    missed_keys = set(gold_by_key) - matched_keys
    extra_keys = set(pred_by_key) - matched_keys
    correct_keys = {
        key for key in matched_keys
        if pred_by_key[key]["label"] == gold_by_key[key].get("expected_label", gold_by_key[key].get("label"))
    }
    wrong_label_matches = []
    for key in sorted(matched_keys - correct_keys):
        gold, prediction = gold_by_key[key], pred_by_key[key]
        wrong_label_matches.append({
            "answer_id": key[0],
            "gold_claim_id": _claim_id(gold),
            "prediction_claim_id": _claim_id(prediction),
            "expected_label": gold.get("expected_label", gold.get("label")),
            "predicted_label": prediction["label"],
        })

    gold_count, prediction_count = len(gold_by_key), len(pred_by_key)
    matched_count, correct_count = len(matched_keys), len(correct_keys)
    extraction_precision = _ratio(matched_count, prediction_count)
    extraction_recall = _ratio(matched_count, gold_count)
    joint_precision = _ratio(correct_count, prediction_count)
    joint_recall = _ratio(correct_count, gold_count)

    matrix = {row: {column: 0 for column in MATRIX_COLUMNS} for row in MATRIX_ROWS}
    for key, gold in gold_by_key.items():
        expected = gold.get("expected_label", gold.get("label"))
        prediction = pred_by_key.get(key)
        column = prediction["label"] if prediction is not None else "missing_or_error"
        matrix[expected][column] += 1
    for key in extra_keys:
        matrix["no_gold"][pred_by_key[key]["label"]] += 1

    per_label: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        true_positive = sum(
            1 for key in correct_keys
            if gold_by_key[key].get("expected_label", gold_by_key[key].get("label")) == label
        )
        predicted_positive = sum(1 for prediction in pred_by_key.values() if prediction["label"] == label)
        actual_positive = sum(
            1 for gold in gold_by_key.values()
            if gold.get("expected_label", gold.get("label")) == label
        )
        precision = _ratio(true_positive, predicted_positive)
        recall = _ratio(true_positive, actual_positive)
        per_label[label] = {
            "true_positive": true_positive,
            "predicted_positive": predicted_positive,
            "actual_positive": actual_positive,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }

    per_answer: dict[str, dict[str, Any]] = {}
    for answer_id in sorted(scheduled):
        answer_gold = {key: claim for key, claim in gold_by_key.items() if key[0] == answer_id}
        answer_pred = {key: claim for key, claim in pred_by_key.items() if key[0] == answer_id}
        answer_matches = set(answer_gold) & set(answer_pred)
        answer_correct = sum(
            answer_pred[key]["label"] == answer_gold[key].get("expected_label", answer_gold[key].get("label"))
            for key in answer_matches
        )
        answer_g, answer_p = len(answer_gold), len(answer_pred)
        answer_m = len(answer_matches)
        per_answer[answer_id] = {
            "status": outcome_by_id[answer_id].get("status", "success"),
            "error": outcome_by_id[answer_id].get("error"),
            "gold_count": answer_g,
            "prediction_count": answer_p,
            "matched_count": answer_m,
            "correct_count": answer_correct,
            "extraction_f1": _f1(_ratio(answer_m, answer_p), _ratio(answer_m, answer_g)),
            "joint_f1": _f1(_ratio(answer_correct, answer_p), _ratio(answer_correct, answer_g)),
            "missed_gold_ids": sorted(_claim_id(answer_gold[key]) for key in set(answer_gold) - answer_matches),
            "unmatched_prediction_ids": sorted(_claim_id(answer_pred[key]) for key in set(answer_pred) - answer_matches),
            "wrong_label_count": sum(
                answer_pred[key]["label"] != answer_gold[key].get("expected_label", answer_gold[key].get("label"))
                for key in answer_matches
            ),
        }

    return {
        "gold_count": gold_count,
        "prediction_count": prediction_count,
        "matched_count": matched_count,
        "correct_count": correct_count,
        "extraction": {
            "precision": extraction_precision,
            "recall": extraction_recall,
            "f1": _f1(extraction_precision, extraction_recall),
        },
        "joint": {
            "precision": joint_precision,
            "recall": joint_recall,
            "f1": _f1(joint_precision, joint_recall),
        },
        "per_label": per_label,
        "macro_f1": sum(float(per_label[label]["f1"]) for label in LABELS) / len(LABELS),
        "gold_claim_coverage": _ratio(correct_count, gold_count),
        "confusion_matrix": matrix,
        "missed_gold_ids": sorted(_claim_id(gold_by_key[key]) for key in missed_keys),
        "unmatched_prediction_ids": sorted(_claim_id(pred_by_key[key]) for key in extra_keys),
        "wrong_label_matches": wrong_label_matches,
        "answer_error_count": len(answer_errors),
        "answer_error_ids": answer_errors,
        "per_answer": per_answer,
    }
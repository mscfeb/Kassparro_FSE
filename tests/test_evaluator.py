import pytest

from app.evaluator import evaluate_split


def claim(answer_id, quote, start, label, claim_id=None):
    return {
        "answer_id": answer_id,
        "claim_id": claim_id or f"{answer_id}:{start}:{start + len(quote)}",
        "start_char": start,
        "end_char": start + len(quote),
        "quote": quote,
        "label": label,
    }


def gold(answer_id, quote, start, label):
    return {**claim(answer_id, quote, start, label), "expected_label": label}


def outcome(answer_id, claims=(), status="success", error=None):
    return {"answer_id": answer_id, "status": status, "claims": list(claims), "error": error}


def test_perfect_predictions_and_empty_claim_answer():
    answers = {"A01": "Alpha. Beta."}
    gold_claims = [gold("A01", "Alpha.", 0, "supported"), gold("A01", "Beta.", 7, "unknown")]
    result = evaluate_split(["A01", "A02"], gold_claims,
                            [outcome("A01", [claim("A01", "Alpha.", 0, "supported"), claim("A01", "Beta.", 7, "unknown")]), outcome("A02")],
                            {**answers, "A02": "Advice only."})
    assert result["extraction"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result["joint"]["f1"] == 1.0
    assert result["gold_claim_coverage"] == 1.0
    assert result["confusion_matrix"]["no_gold"]["missing_or_error"] == 0


def test_arithmetic_example_and_confusion_accounting():
    labels = ["supported", "contradicted", "unknown", "supported", "unknown", "contradicted", "supported", "unknown", "supported", "contradicted"]
    body = " ".join(f"C{i}." for i in range(10))
    spans = []
    offset = 0
    for index, label in enumerate(labels):
        quote = f"C{index}."
        spans.append((quote, offset, label))
        offset += len(quote) + 1
    gold_claims = [gold("A01", quote, start, label) for quote, start, label in spans]
    predicted = [claim("A01", quote, start, label if index != 0 else "unknown") for index, (quote, start, label) in enumerate(spans[:8])]
    predicted.append(claim("A01", "Advice.", len(body) + 1, "supported"))
    predicted.append(claim("A01", "Also.", len(body) + 9, "unknown"))
    result = evaluate_split(["A01"], gold_claims, [outcome("A01", predicted)], {"A01": body + " Advice. Also."})
    assert result["gold_count"] == result["prediction_count"] == 10
    assert (result["matched_count"], result["correct_count"]) == (8, 7)
    assert result["extraction"] == pytest.approx({"precision": 0.8, "recall": 0.8, "f1": 0.8})
    assert result["joint"] == pytest.approx({"precision": 0.7, "recall": 0.7, "f1": 0.7})
    assert result["confusion_matrix"]["no_gold"]["supported"] == 1
    assert result["confusion_matrix"]["no_gold"]["unknown"] == 1
    assert result["confusion_matrix"]["supported"]["unknown"] == 1
    assert sum(sum(row.values()) for row in result["confusion_matrix"].values()) == 12


def test_provider_error_leaves_gold_missed():
    result = evaluate_split(["A01"], [gold("A01", "Claim.", 0, "contradicted")],
                            [outcome("A01", status="error", error={"type": "ProviderError"})],
                            {"A01": "Claim."})
    assert result["answer_error_count"] == 1
    assert result["missed_gold_ids"] == ["A01:0:6"]
    assert result["confusion_matrix"]["contradicted"]["missing_or_error"] == 1


def test_wrong_label_counts_as_false_positive_and_false_negative():
    result = evaluate_split(["A01"], [gold("A01", "Claim.", 0, "supported")],
                            [outcome("A01", [claim("A01", "Claim.", 0, "unknown")])], {"A01": "Claim."})
    assert result["per_label"]["supported"]["recall"] == 0.0
    assert result["per_label"]["unknown"]["precision"] == 0.0
    assert result["wrong_label_matches"][0]["expected_label"] == "supported"


def test_check_split_quality_fields_are_null():
    result = evaluate_split(["A41"], None, [outcome("A41")], {"A41": ""})
    assert result["prediction_count"] == 0
    assert result["gold_count"] is None
    assert result["joint"] is None
    assert result["per_label"] is None


@pytest.mark.parametrize("outcomes", [[], [outcome("A01"), outcome("A01")]])
def test_missing_or_duplicate_outcomes_are_rejected(outcomes):
    with pytest.raises(ValueError):
        evaluate_split(["A01"], [], outcomes, {"A01": ""})


def test_invalid_prediction_quote_is_rejected_before_scoring():
    wrong = claim("A01", "Other.", 0, "supported")
    with pytest.raises(ValueError, match="quote does not match"):
        evaluate_split(["A01"], [], [outcome("A01", [wrong])], {"A01": "Claim."})
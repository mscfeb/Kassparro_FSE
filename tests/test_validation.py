import pytest

from app.validation import InvalidAgentOutput, validate_model_claims


FACT_IDS = frozenset({"f1"})


def raw(quote, **updates):
    return {"quote": quote, "label": "supported", "fact_ids": ["f1"], "reason": "The catalogue lists it.", **updates}


def test_valid_claim_gets_python_derived_unicode_offsets():
    answer = "Intro. \N{SNOWMAN} Product works."
    quote = "Product works."
    claim = validate_model_claims("A01", answer, [raw(quote)], FACT_IDS)[0]
    start = answer.index(quote)
    assert claim["start_char"] == start
    assert claim["end_char"] == start + len(quote)
    assert claim["claim_id"] == f"A01:{start}:{start + len(quote)}"


@pytest.mark.parametrize("raw_claim,answer", [
    (raw("Wrong."), "Claim."),
    (raw("Claim.", label="maybe"), "Claim."),
    (raw("Claim.", fact_ids=["invented"]), "Claim."),
    (raw("Claim.", reason="First. Second. Third."), "Claim."),
    (raw(" Claim."), "Claim."),
    (raw("Claim.", start_char=-1), "Claim."),
])
def test_invalid_output_is_rejected(raw_claim, answer):
    with pytest.raises(InvalidAgentOutput):
        validate_model_claims("A01", answer, [raw_claim], FACT_IDS)


def test_claims_must_be_ordered_and_nonoverlapping():
    answer = "One. Two."
    with pytest.raises(InvalidAgentOutput, match="sorted"):
        validate_model_claims("A01", answer, [raw("Two."), raw("One.")], FACT_IDS)
    with pytest.raises(InvalidAgentOutput, match="overlap"):
        validate_model_claims("A01", answer, [raw("One. Two."), raw("Two.")], FACT_IDS)


def test_repeated_quote_requires_context_and_can_be_disambiguated():
    answer = "A. Claim. B. Claim."
    with pytest.raises(InvalidAgentOutput, match="repeated"):
        validate_model_claims("A01", answer, [raw("Claim.")], FACT_IDS)
    selected = validate_model_claims("A01", answer, [raw("Claim.", context_before="B. ", context_after="")], FACT_IDS)
    assert selected[0]["start_char"] == answer.rindex("Claim.")


def test_empty_claims_are_valid():
    assert validate_model_claims("A01", "Advice only.", [], FACT_IDS) == []
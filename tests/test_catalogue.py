from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from app.catalogue import BrandCatalogue, UnknownBrandError


def test_typed_catalogue_values_and_inclusive_validity():
    catalogue = BrandCatalogue(Path("data/brand_facts.csv"))
    assert catalogue.get_brand_facts("cleartrail")
    assert any(isinstance(fact.value, bool) and fact.value is False for fact in catalogue.get_brand_facts("cleartrail"))
    assert any(isinstance(fact.value, Decimal) for fact in catalogue.get_brand_facts("cleartrail"))
    fact = next(fact for fact in catalogue.get_brand_facts("cleartrail") if fact.valid_from and fact.valid_to)
    assert fact.valid_on(fact.valid_from)
    assert fact.valid_on(fact.valid_to)
    assert not fact.valid_on(date.fromordinal(fact.valid_to.toordinal() + 1))


def test_unknown_brand_is_an_error():
    catalogue = BrandCatalogue(Path("data/brand_facts.csv"))
    with pytest.raises(UnknownBrandError):
        catalogue.tool_payload("not-a-brand")


def test_tool_payload_is_read_only_serializable_data():
    catalogue = BrandCatalogue(Path("data/brand_facts.csv"))
    payload = catalogue.tool_payload("cleartrail")
    assert payload["fact_version"] == "speaker-catalogue-v4"
    assert all(row["brand_id"] == "cleartrail" for row in payload["facts"])
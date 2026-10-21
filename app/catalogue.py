from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TypeAlias

FACT_VERSION = "speaker-catalogue-v4"
FactValue: TypeAlias = Decimal | bool | str


class UnknownBrandError(ValueError):
    pass


@dataclass(frozen=True)
class BrandFact:
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: FactValue
    value_type: str
    unit: str | None
    condition: str | None
    valid_from: date | None
    valid_to: date | None

    def valid_on(self, when: date) -> bool:
        return (self.valid_from is None or self.valid_from <= when) and (
            self.valid_to is None or when <= self.valid_to
        )

    def as_tool_record(self) -> dict[str, str | None]:
        if isinstance(self.value, Decimal):
            value = str(self.value)
        elif isinstance(self.value, bool):
            value = str(self.value).lower()
        else:
            value = self.value
        return {
            "fact_id": self.fact_id,
            "brand_id": self.brand_id,
            "model_id": self.model_id,
            "field": self.field,
            "value": value,
            "value_type": self.value_type,
            "unit": self.unit,
            "condition": self.condition,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
        }


def _parse_value(value: str, value_type: str) -> FactValue:
    if value_type == "string":
        return value
    if value_type == "boolean":
        normalized = value.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"invalid boolean catalogue value: {value!r}")
        return normalized == "true"
    if value_type == "number":
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"invalid numeric catalogue value: {value!r}") from exc
    raise ValueError(f"unsupported catalogue value_type: {value_type}")


def _parse_date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


class BrandCatalogue:
    def __init__(self, path: Path):
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "fact_id", "brand_id", "model_id", "field", "value", "value_type",
                "unit", "condition", "valid_from", "valid_to",
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("brand_facts.csv is missing required columns")
            self._facts = tuple(
                BrandFact(
                    fact_id=row["fact_id"],
                    brand_id=row["brand_id"],
                    model_id=row["model_id"],
                    field=row["field"],
                    value=_parse_value(row["value"], row["value_type"]),
                    value_type=row["value_type"],
                    unit=row["unit"] or None,
                    condition=row["condition"] or None,
                    valid_from=_parse_date(row["valid_from"]),
                    valid_to=_parse_date(row["valid_to"]),
                )
                for row in reader
            )
        if len({fact.fact_id for fact in self._facts}) != len(self._facts):
            raise ValueError("catalogue fact_id values must be unique")
        self._by_brand: dict[str, tuple[BrandFact, ...]] = {}
        for brand_id in {fact.brand_id for fact in self._facts}:
            self._by_brand[brand_id] = tuple(fact for fact in self._facts if fact.brand_id == brand_id)
        self._fact_ids = frozenset(fact.fact_id for fact in self._facts)

    @property
    def brands(self) -> frozenset[str]:
        return frozenset(self._by_brand)

    @property
    def fact_ids(self) -> frozenset[str]:
        return self._fact_ids

    def get_brand_facts(self, brand_id: str) -> tuple[BrandFact, ...]:
        if not isinstance(brand_id, str) or brand_id not in self._by_brand:
            raise UnknownBrandError("unknown brand_id")
        return self._by_brand[brand_id]

    def tool_payload(self, brand_id: str) -> dict[str, object]:
        return {
            "fact_version": FACT_VERSION,
            "brand_id": brand_id,
            "facts": [fact.as_tool_record() for fact in self.get_brand_facts(brand_id)],
        }
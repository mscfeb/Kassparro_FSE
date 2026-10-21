from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_VERSION = "long-brand-answers-50-v4"
EVALUATOR_VERSION = "sentence-spans-and-labels-v4"


@dataclass(frozen=True)
class Answer:
    answer_id: str
    split: str
    brand_id: str
    question: str
    ai_answer: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path.name}")
        return list(reader)


def load_answers(path: Path) -> list[Answer]:
    required = {"answer_id", "split", "brand_id", "question", "ai_answer"}
    rows = _read_csv(path)
    if not rows or not required.issubset(rows[0]):
        raise ValueError("answers.csv is empty or missing required columns")
    answers = [Answer(**{key: row[key] for key in required}) for row in rows]
    ids = [answer.answer_id for answer in answers]
    if len(ids) != len(set(ids)):
        raise ValueError("answer_id values must be unique")
    if any(answer.split not in {"development", "check"} for answer in answers):
        raise ValueError("answer split must be development or check")
    return answers


def load_gold(path: Path) -> list[dict[str, Any]]:
    required = {
        "answer_id", "claim_id", "start_char", "end_char", "quote",
        "expected_label", "evidence_fact_ids", "explanation", "difficulty_tags",
    }
    rows = _read_csv(path)
    if rows and not required.issubset(rows[0]):
        raise ValueError("development_gold.csv is missing required columns")
    return [
        {
            **row,
            "start_char": int(row["start_char"]),
            "end_char": int(row["end_char"]),
            "fact_ids": [value for value in row["evidence_fact_ids"].split("|") if value],
        }
        for row in rows
    ]


def dataset_hashes(data_dir: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("answers.csv", "brand_facts.csv", "development_gold.csv")
    }
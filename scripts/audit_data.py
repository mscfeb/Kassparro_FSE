import json
from pathlib import Path

from app.catalogue import BrandCatalogue
from app.data import dataset_hashes, load_answers, load_gold


def main() -> int:
    data_dir = Path("data")
    answers = load_answers(data_dir / "answers.csv")
    catalogue = BrandCatalogue(data_dir / "brand_facts.csv")
    gold = load_gold(data_dir / "development_gold.csv")
    by_id = {answer.answer_id: answer for answer in answers}
    errors = []
    spans_by_answer = {}
    for row in gold:
        answer = by_id.get(row["answer_id"])
        if answer is None or answer.ai_answer[row["start_char"]:row["end_char"]] != row["quote"]:
            errors.append(f"invalid source span: {row['claim_id']}")
        if row["claim_id"] != f"{row['answer_id']}:{row['start_char']}:{row['end_char']}":
            errors.append(f"invalid claim id: {row['claim_id']}")
        if not set(row["fact_ids"]).issubset(catalogue.fact_ids):
            errors.append(f"unknown gold evidence fact: {row['claim_id']}")
        spans_by_answer.setdefault(row["answer_id"], []).append(row)
    for answer_id, rows in spans_by_answer.items():
        ordered = sorted(rows, key=lambda row: row["start_char"])
        for left, right in zip(ordered, ordered[1:]):
            if left["end_char"] > right["start_char"]:
                errors.append(f"overlapping gold spans: {left['claim_id']} / {right['claim_id']}")
    split_counts = {split: sum(answer.split == split for answer in answers) for split in ("development", "check")}
    result = {
        "answer_count": len(answers),
        "split_counts": split_counts,
        "fact_count": len(catalogue.fact_ids),
        "gold_count": len(gold),
        "gold_span_errors": errors,
        "dataset_hashes": dataset_hashes(data_dir),
    }
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
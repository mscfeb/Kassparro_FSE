from app.data import load_answers


def test_csv_parser_preserves_embedded_newlines_and_quoted_commas(tmp_path):
    path = tmp_path / "answers.csv"
    original = 'answer_id,split,brand_id,question,ai_answer\nA01,development,cleartrail,"Question, with comma","First line.\nSecond, line!"\n'
    path.write_text(original, encoding="utf-8", newline="")
    answer = load_answers(path)[0]
    assert answer.question == "Question, with comma"
    assert answer.ai_answer == "First line.\nSecond, line!"
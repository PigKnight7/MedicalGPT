"""统一 CMExam 评估流程测试；仅使用 FakeTokenizer/FakeModel。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from evaluation.eval_cmexam import (
    EvaluationError,
    calculate_metrics,
    load_cmexam_records,
    main,
    parse_args,
    read_predictions,
    run,
    select_records,
)


def record(identifier: str, answer: str, *, split: str = "validation", metadata=None, long=False):
    question = ("很长的医学题干" * 20 if long else f"题目{identifier}") + "\nA. 甲\nB. 乙\nC. 丙\nD. 丁\nE. 戊"
    return {
        "id": identifier,
        "split": split,
        "question": question,
        "stem": f"题目{identifier}",
        "options": [{"label": label, "text": text} for label, text in zip("ABCDE", "甲乙丙丁戊")],
        "answer": answer,
        "answer_labels": list(answer),
        "is_multiple_choice": len(answer) > 1,
        "metadata": metadata or {},
    }


def write_data(root: Path, split: str, records: list[dict]) -> Path:
    path = root / split / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    return path


class FakeTokenizer:
    chat_template = "fake-chat-template"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "|".join(f"{x['role']}:{x['content']}" for x in conversation) + "|assistant:"

    @staticmethod
    def encode_text(text: str) -> list[int]:
        return [ord(character) + 2 for character in text]

    def __call__(self, text, *, return_tensors=None, padding=False, truncation=False,
                 max_length=None, add_special_tokens=True):
        if isinstance(text, str):
            ids = self.encode_text(text)
            if truncation and max_length is not None:
                ids = ids[-max_length:]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        rows = [self.encode_text(item) for item in text]
        if truncation and max_length is not None:
            rows = [item[-max_length:] for item in rows]
        width = max(len(item) for item in rows)
        padded = [[0] * (width - len(item)) + item for item in rows]
        masks = [[0] * (width - len(item)) + [1] * len(item) for item in rows]
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(masks)}

    def decode(self, ids, *, skip_special_tokens=True):
        values = ids.tolist() if hasattr(ids, "tolist") else ids
        return "".join(chr(value - 2) for value in values if value > 1)


class FakeModel:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.device = torch.device("cpu")
        self.eval_calls = 0
        self.generate_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        self.generate_calls += 1
        batch = input_ids.shape[0]
        current = [FakeTokenizer.encode_text(self.outputs.pop(0)) for _ in range(batch)]
        width = max(len(item) for item in current)
        new = torch.tensor([item + [0] * (width - len(item)) for item in current])
        return torch.cat([input_ids, new], dim=1)


def arguments(data_root: Path, output_dir: Path, *extra: str):
    return parse_args([
        "--model_name_or_path", "fake-base", "--data_root", str(data_root),
        "--output_dir", str(output_dir), "--batch_size", "2", *extra,
    ])


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_load_official_validation_and_duplicate_id(tmp_path: Path) -> None:
    path = write_data(tmp_path, "validation", [record("v1", "A"), record("v2", "AE")])
    assert len(load_cmexam_records(path, expected_split="validation")) == 2
    write_data(tmp_path, "validation", [record("v1", "A"), record("v1", "B")])
    with pytest.raises(EvaluationError, match="重复 id"):
        load_cmexam_records(path, expected_split="validation")


def test_test_split_requires_explicit_permission(tmp_path: Path) -> None:
    write_data(tmp_path, "test", [record("t1", "A", split="test")])
    with pytest.raises(EvaluationError, match="最终配置评估"):
        run(arguments(tmp_path, tmp_path / "out", "--split", "test", "--dry_run"))
    assert run(arguments(tmp_path, tmp_path / "out", "--split", "test", "--allow_test_evaluation", "--dry_run")) == 0


def test_max_samples_is_seeded() -> None:
    records = [record(str(i), "A") for i in range(20)]
    first = select_records(records, max_samples=5, seed=7)
    second = select_records(records, max_samples=5, seed=7)
    assert [x["id"] for x in first] == [x["id"] for x in second]
    assert len(first) == 5


def prediction(gold, predicted, *, valid=True, format_valid=True, truncated=False, metadata=None):
    return {
        "gold_answer": "".join(gold), "gold_labels": list(gold),
        "predicted_answer": "".join(predicted) if predicted else None,
        "predicted_labels": list(predicted) if predicted else None,
        "correct": set(gold) == set(predicted or ()), "valid_prediction": valid,
        "format_valid": format_valid, "is_multiple_choice": len(gold) > 1,
        "metadata": metadata or {}, "input_tokens": 10, "generated_tokens": 3,
        "latency_seconds": 0.5, "input_truncated": truncated,
        "parse_error": None if valid else "没有答案",
    }


def test_metrics_single_multiple_micro_counts_and_groups() -> None:
    items = [
        prediction(("A",), ("A",), metadata={"Department": "内科"}),
        prediction(("B",), ("C",), metadata={"Department": "外科"}),
        prediction(("A", "E"), ("E", "A"), metadata={"Department": "内科"}),
        prediction(("A", "E"), ("A",)),
        prediction(("A", "E"), ("A", "B", "E")),
        prediction(("C",), None, valid=False, format_valid=False, truncated=True),
    ]
    metrics = calculate_metrics(items, group_by=["Department", "Missing"])
    assert metrics["overall"]["records"] == 6
    assert metrics["overall"]["correct"] == 2
    assert metrics["overall"]["truncated_input_count"] == 1
    assert metrics["single_choice"]["records"] == 3
    assert metrics["multiple_choice"]["records"] == 3
    assert metrics["multiple_choice"]["exact_set_accuracy"] == pytest.approx(1 / 3)
    assert metrics["multiple_choice"]["micro_precision"] == pytest.approx(5 / 6)
    assert metrics["multiple_choice"]["micro_recall"] == pytest.approx(5 / 6)
    assert metrics["multiple_choice"]["micro_f1"] == pytest.approx(5 / 6)
    assert metrics["by_answer_label_count"]["1"]["records"] == 3
    assert metrics["by_answer_label_count"]["2"]["records"] == 3
    assert metrics["metadata_groups"]["Department"]["内科"]["records"] == 2
    assert metrics["missing_group_by_fields"] == ["Missing"]


def test_full_fake_evaluation_outputs_and_added_token_decoding(tmp_path: Path, capsys) -> None:
    records = [
        record("s-ok", "A", metadata={"Department": "内科"}),
        record("s-wrong", "B"),
        record("m-order", "AE"),
        record("m-missing", "AE"),
        record("m-extra", "AE"),
        record("invalid", "C", long=True),
    ]
    write_data(tmp_path / "data", "validation", records)
    outputs = ["<answer>A</answer>", "<answer>C</answer>", "<answer>EA</answer>",
               "<answer>A</answer>", "<answer>ABE</answer>", "没有明确答案"]
    model = FakeModel(outputs)
    args = arguments(
        tmp_path / "data", tmp_path / "out", "--max_input_length", "80",
        "--group_by", "Department", "--group_by", "Missing",
    )
    assert run(args, model=model, tokenizer=FakeTokenizer()) == 0
    predictions = read_jsonl(tmp_path / "out/predictions.jsonl")
    assert len(predictions) == 6
    assert predictions[0]["raw_output"] == "<answer>A</answer>"
    assert predictions[2]["correct"] is True and predictions[2]["predicted_answer"] == "AE"
    assert predictions[3]["correct"] is False and predictions[4]["correct"] is False
    assert predictions[5]["valid_prediction"] is False
    assert any(item["input_truncated"] for item in predictions)
    required = {"id", "gold_answer", "raw_output", "predicted_labels", "format_valid", "input_tokens", "latency_seconds"}
    assert required.issubset(predictions[0])
    errors = read_jsonl(tmp_path / "out/error_cases.jsonl")
    assert {item["id"] for item in errors} >= {"s-wrong", "m-missing", "m-extra", "invalid"}
    config_text = (tmp_path / "out/evaluation_config.json").read_text(encoding="utf-8").lower()
    assert "hf_token" not in config_text and "api_key" not in config_text
    assert "missing" in capsys.readouterr().err.lower()


def test_default_refuses_overwrite_and_overwrite_replaces(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    write_data(data, "validation", [record("one", "A")])
    assert run(arguments(data, out), model=FakeModel(["<answer>A</answer>"]), tokenizer=FakeTokenizer()) == 0
    with pytest.raises(FileExistsError):
        run(arguments(data, out), model=FakeModel(["<answer>A</answer>"]), tokenizer=FakeTokenizer())
    assert run(arguments(data, out, "--overwrite"), model=FakeModel(["<answer>B</answer>"]), tokenizer=FakeTokenizer()) == 0
    assert read_jsonl(out / "predictions.jsonl")[0]["predicted_answer"] == "B"


def test_resume_skips_completed_and_rejects_changed_config(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    write_data(data, "validation", [record("one", "A"), record("two", "B")])
    first = FakeModel(["<answer>A</answer>", "<answer>B</answer>"])
    run(arguments(data, out), model=first, tokenizer=FakeTokenizer())
    resumed = FakeModel([])
    assert run(arguments(data, out, "--resume"), model=resumed, tokenizer=FakeTokenizer()) == 0
    assert resumed.generate_calls == 0 and len(read_jsonl(out / "predictions.jsonl")) == 2
    with pytest.raises(EvaluationError, match="不兼容"):
        run(arguments(data, out, "--resume", "--seed", "99"), model=FakeModel([]), tokenizer=FakeTokenizer())


def test_corrupt_and_duplicate_predictions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"id":"a"}\n{broken\n', encoding="utf-8")
    with pytest.raises(EvaluationError, match="损坏"):
        read_predictions(path)
    path.write_text('{"id":"a"}\n{"id":"a"}\n', encoding="utf-8")
    with pytest.raises(EvaluationError, match="重复预测"):
        read_predictions(path)


def test_dry_run_does_not_load_model(tmp_path: Path, monkeypatch, capsys) -> None:
    data = tmp_path / "data"
    write_data(data, "validation", [record("one", "A"), record("two", "AE")])
    monkeypatch.setattr("evaluation.eval_cmexam.load_model_and_tokenizer", lambda args: pytest.fail("不应加载模型"))
    assert run(arguments(data, tmp_path / "out", "--dry_run")) == 0
    output = capsys.readouterr().out
    assert "完整数据：2；单选：1；多选：1" in output and "<answer>AE</answer>" in output
    assert not (tmp_path / "out/predictions.jsonl").exists()


def test_cli_dry_run_end_to_end(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_data(data, "validation", [record("one", "A")])
    script = Path(__file__).parents[1] / "evaluation/eval_cmexam.py"
    result = subprocess.run(
        [sys.executable, str(script), "--model_name_or_path", "never-loaded", "--data_root", str(data),
         "--output_dir", str(tmp_path / "out"), "--dry_run"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0
    assert "未加载模型" in result.stdout


def test_main_returns_zero_for_dry_run(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_data(data, "validation", [record("one", "A")])
    assert main(["--model_name_or_path", "fake", "--data_root", str(data),
                 "--output_dir", str(tmp_path / "out"), "--dry_run"]) == 0

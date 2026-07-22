"""CMExam GRPO 数据模块测试；不访问网络或加载模型。"""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from datasets import Dataset

from medicalgpt_ext.cmexam_grpo_data import (
    CMEXAM_GRPO_SYSTEM_PROMPT,
    CMExamGRPODataConfig,
    CMExamGRPODataError,
    build_cmexam_grpo_dataset,
    build_cmexam_grpo_prompt,
    build_cmexam_question_text,
    prepare_cmexam_grpo_examples,
    read_cmexam_grpo_jsonl,
    validate_cmexam_grpo_record,
)


def record(index: int = 1, labels: list[str] | None = None, **updates: object) -> dict[str, object]:
    labels = labels or ["A"]
    item: dict[str, object] = {
        "id": f"cmexam-train-{index:06d}",
        "split": "train",
        "question": f"第{index}题",
        "stem": f"第{index}题",
        "options": [
            {"label": "E", "text": "戊"},
            {"label": "C", "text": "丙"},
            {"label": "A", "text": "甲"},
            {"label": "D", "text": "丁"},
            {"label": "B", "text": "乙"},
        ],
        "answer": "".join(labels),
        "answer_labels": labels,
        "answer_texts": ["绝密标准答案文本"],
        "is_multiple_choice": len(labels) > 1,
        "metadata": {"source": "unit", "number": index},
    }
    item.update(updates)
    return item


def write_jsonl(path: Path, rows: list[object], *, prefix: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = prefix + "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows)
    path.write_text(text, encoding="utf-8")


def strict_path(tmp_path: Path) -> Path:
    return tmp_path / "cmexam/decontaminated/train/train.jsonl"


def config(path: Path, **kwargs: object) -> CMExamGRPODataConfig:
    return CMExamGRPODataConfig(
        train_file=path, require_decontaminated_train=False, **kwargs
    )


def test_config_defaults_and_validation() -> None:
    default = CMExamGRPODataConfig()
    assert default.train_file.parts[-3:] == ("decontaminated", "train", "train.jsonl")
    assert default.require_decontaminated_train and default.prompt_style == "answer_only"
    assert isinstance(CMExamGRPODataConfig(train_file="~/x", require_decontaminated_train=False).train_file, Path)
    for value in (0, -1, True, 1.2):
        with pytest.raises((TypeError, ValueError), match="max_samples"):
            CMExamGRPODataConfig(max_samples=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="answer_only"):
        CMExamGRPODataConfig(prompt_style="reasoning")
    with pytest.raises(TypeError, match="seed"):
        CMExamGRPODataConfig(seed=True)


def test_strict_path_allows_only_decontaminated_train(tmp_path: Path) -> None:
    good = strict_path(tmp_path)
    write_jsonl(good, [record()])
    assert len(read_cmexam_grpo_jsonl(good)[0]) == 1
    for path in (
        tmp_path / "official/train/train.jsonl",
        tmp_path / "decontaminated/validation/validation.jsonl",
        tmp_path / "decontaminated/test/test.jsonl",
    ):
        write_jsonl(path, [record()])
        with pytest.raises(CMExamGRPODataError, match="validation 和 test"):
            read_cmexam_grpo_jsonl(path)


def test_non_strict_path_and_record_split_guards(tmp_path: Path) -> None:
    path = tmp_path / "temporary.jsonl"
    write_jsonl(path, [record()])
    assert read_cmexam_grpo_jsonl(path, require_decontaminated_train=False)[0]
    for split in ("validation", "test"):
        write_jsonl(path, [record(split=split)])
        with pytest.raises(CMExamGRPODataError, match="validation 和 test"):
            read_cmexam_grpo_jsonl(path, require_decontaminated_train=False)


def test_jsonl_blank_lines_and_basic_read(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [record(), record(2, ["A", "E"])], prefix="\n  \n")
    rows, blanks = read_cmexam_grpo_jsonl(path, require_decontaminated_train=False)
    assert blanks == 2 and [item["id"] for item in rows] == ["cmexam-train-000001", "cmexam-train-000002"]
    assert rows[1]["answer"] == "AE" and rows[1]["is_multiple_choice"] is True


def test_jsonl_malformed_non_dict_empty_missing_and_directory(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("\n{bad}\n", encoding="utf-8")
    with pytest.raises(CMExamGRPODataError, match=r"bad.jsonl 第 2 行"):
        read_cmexam_grpo_jsonl(malformed, require_decontaminated_train=False)
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(CMExamGRPODataError, match="JSON 对象"):
        read_cmexam_grpo_jsonl(malformed, require_decontaminated_train=False)
    malformed.write_text(" \n", encoding="utf-8")
    with pytest.raises(CMExamGRPODataError, match="为空"):
        read_cmexam_grpo_jsonl(malformed, require_decontaminated_train=False)
    with pytest.raises(FileNotFoundError):
        read_cmexam_grpo_jsonl(tmp_path / "missing.jsonl", require_decontaminated_train=False)
    with pytest.raises(CMExamGRPODataError, match="不是文件"):
        read_cmexam_grpo_jsonl(tmp_path, require_decontaminated_train=False)


def test_duplicate_and_empty_ids(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [record(), record()])
    with pytest.raises(CMExamGRPODataError, match="重复 id"):
        read_cmexam_grpo_jsonl(path, require_decontaminated_train=False)
    for bad_id in ("", "   ", 3):
        with pytest.raises(CMExamGRPODataError, match="非空字符串 id"):
            validate_cmexam_grpo_record(record(id=bad_id))


def test_valid_single_multiple_and_input_not_modified() -> None:
    single = record()
    multiple = record(2, ["E", "A", "A"])
    before = copy.deepcopy(multiple)
    assert validate_cmexam_grpo_record(single)["answer_labels"] == ["A"]
    normalized = validate_cmexam_grpo_record(multiple)
    assert normalized["answer_labels"] == ["A", "E"] and normalized["answer"] == "AE"
    assert multiple == before


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (record(question=None, stem=None), "question"),
        (record(question=" ", stem=" "), "question"),
        ({key: value for key, value in record().items() if key != "options"}, "options"),
        (record(options="A.甲"), "options"),
        ({key: value for key, value in record().items() if key != "answer_labels"}, "answer_labels"),
        (record(answer_labels=["F"]), "A-E"),
        (record(answer_labels=["E"], options=[{"label": "A", "text": "甲"}, {"label": "B", "text": "乙"}]), "不存在"),
        (record(answer_labels=["A", "E"], is_multiple_choice=False), "不一致"),
        (record(metadata="bad"), "metadata"),
    ],
)
def test_invalid_record_fields(item: dict[str, object], message: str) -> None:
    with pytest.raises(CMExamGRPODataError, match=message):
        validate_cmexam_grpo_record(item, line_number=7)


def test_missing_metadata_is_normalized() -> None:
    item = record()
    del item["metadata"]
    assert validate_cmexam_grpo_record(item)["metadata"] == {}


def test_question_text_order_and_no_leakage() -> None:
    item = record(answer="SECRET", answer_labels=["C"], answer_texts=["GOLD-TEXT"])
    text = build_cmexam_question_text(item)
    assert text.splitlines() == ["第1题", "A. 甲", "B. 乙", "C. 丙", "D. 丁", "E. 戊"]
    assert "SECRET" not in text and "GOLD-TEXT" not in text and "source" not in text


def test_conversational_prompt_answer_only_and_independent() -> None:
    first = build_cmexam_grpo_prompt(record())
    second = build_cmexam_grpo_prompt(record())
    assert [message["role"] for message in first] == ["system", "user"]
    assert "<answer>A</answer>" in first[0]["content"]
    assert "不要输出解释" in first[0]["content"] and "分析过程" in first[0]["content"]
    assert CMEXAM_GRPO_SYSTEM_PROMPT == first[0]["content"]
    assert first == second and first is not second and first[0] is not second[0]
    first[0]["content"] = "changed"
    assert second[0]["content"] == CMEXAM_GRPO_SYSTEM_PROMPT
    assert all(message["role"] != "assistant" for message in second)


def test_output_fields_metadata_and_dataset_serialization(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    source = record(1, ["E", "A"])
    before = copy.deepcopy(source)
    write_jsonl(path, [source])
    examples, summary = prepare_cmexam_grpo_examples(config(path))
    example = examples[0]
    assert set(example) == {"id", "prompt", "answer", "answer_labels", "is_multiple_choice", "metadata", "question", "options"}
    assert example["answer"] == "AE" and example["answer_labels"] == ["A", "E"]
    assert example["metadata"] == source["metadata"] and source == before
    dataset = Dataset.from_list(examples)
    assert len(dataset) == 1 and dataset[0]["answer_labels"] == ["A", "E"]
    built = build_cmexam_grpo_dataset(config(path))
    assert isinstance(built, Dataset) and len(built) == summary.selected_records == 1


def make_many(path: Path, count: int = 20) -> None:
    write_jsonl(path, [record(i, ["A", "E"] if i % 4 == 0 else ["A"]) for i in range(1, count + 1)])


def test_sampling_all_seed_reproducibility_and_order(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    make_many(path)
    all_examples, _ = prepare_cmexam_grpo_examples(config(path, max_samples=None))
    a, sa = prepare_cmexam_grpo_examples(config(path, max_samples=6, seed=42))
    b, sb = prepare_cmexam_grpo_examples(config(path, max_samples=6, seed=42))
    c, _ = prepare_cmexam_grpo_examples(config(path, max_samples=6, seed=43))
    ids = [item["id"] for item in a]
    assert len(all_examples) == 20 and ids == [item["id"] for item in b] == list(sa.selected_ids)
    assert ids != [item["id"] for item in c]
    assert len(ids) == len(set(ids)) == 6 and ids == sorted(ids)
    assert sa == sb and sa.shuffled_before_select is True


def test_sampling_front_excess_and_global_random_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    make_many(path, 8)
    random.seed(987)
    state = random.getstate()
    front, front_summary = prepare_cmexam_grpo_examples(
        config(path, max_samples=3, shuffle_before_select=False)
    )
    assert random.getstate() == state
    assert [item["id"] for item in front] == [f"cmexam-train-{i:06d}" for i in (1, 2, 3)]
    assert front_summary.shuffled_before_select is False
    all_rows, summary = prepare_cmexam_grpo_examples(config(path, max_samples=99))
    assert len(all_rows) == 8 == len({item["id"] for item in all_rows})
    assert summary.requested_samples_exceeded_available is True


def test_summary_exact_values_and_plain_dict(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [record(1), record(2, ["A", "E"]), record(3, ["B", "C", "D"])], prefix="\n")
    examples, summary = prepare_cmexam_grpo_examples(config(path, shuffle_before_select=False))
    assert (summary.raw_records, summary.valid_records, summary.selected_records, summary.blank_lines) == (3, 3, 3, 1)
    assert (summary.single_choice_records, summary.multiple_choice_records) == (1, 2)
    assert summary.answer_label_count_distribution == {"1": 1, "2": 1, "3": 1}
    assert list(summary.selected_ids) == [item["id"] for item in examples]
    assert summary.single_choice_records + summary.multiple_choice_records == summary.selected_records
    plain = summary.to_dict()
    assert isinstance(plain, dict) and isinstance(plain["selected_ids"], list)
    assert asdict(summary)["source_file"] == str(path)


def run_cli(path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "medicalgpt_ext.cmexam_grpo_data", "--train_file", str(path),
         "--allow_non_decontaminated_train", *extra],
        cwd=Path(__file__).parents[1], text=True, capture_output=True, check=False,
    )


def test_cli_success_safe_examples_and_no_model_import(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [record(answer="GOLDSECRET", answer_texts=["TEXTSECRET"])])
    result = run_cli(path, "--show_examples", "3")
    assert result.returncode == 0 and '"selected_records": 1' in result.stdout
    assert "Prompt 示例 1" in result.stdout
    assert "GOLDSECRET" not in result.stdout and "TEXTSECRET" not in result.stdout
    assert '"answer_labels"' not in result.stdout
    assert "AutoTokenizer" not in result.stdout and "AutoModel" not in result.stdout


def test_cli_invalid_data_returns_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad}\n", encoding="utf-8")
    result = run_cli(path)
    assert result.returncode != 0 and "错误：" in result.stderr and "第 1 行" in result.stderr

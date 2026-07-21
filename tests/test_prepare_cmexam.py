"""tools.prepare_cmexam 的单元测试。"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.prepare_cmexam import (
    DataFormatError,
    main,
    normalize_answer,
    normalize_text,
    parse_options,
)


HEADERS = ["Question", "Options", "Answer", "Explanation"]


def row(question: str, answer: str = "A", options: str | None = None) -> dict[str, str]:
    """构造一条测试用 CMExam 原始记录。"""

    return {
        "Question": question,
        "Options": options or "A 选项甲\nB 选项乙\nC 选项丙\nD 选项丁\nE 选项戊",
        "Answer": answer,
        "Explanation": f"{question}的解析",
    }


def write_csv(path: Path, rows: list[dict[str, str]], *, annotated: bool = False) -> None:
    """写入小型 CSV；测试集可附带官方注释列。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = HEADERS + (["Disease Group"] if annotated else [])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for item in rows:
            output = dict(item)
            if annotated:
                output["Disease Group"] = "测试疾病组"
            writer.writerow(output)


def create_dataset(raw_dir: Path, *, invalid_train: bool = False) -> None:
    """创建包含三类跨划分重叠和一条多选题的数据集。"""

    train = [
        row("仅训练题", "A"),
        row("训练验证重叠题", "AE"),
        row("训练测试重叠题", "B"),
        row("训练内重复题", "C"),
        row("训练内重复题", "C"),
    ]
    if invalid_train:
        train.append(row("非法题", ""))
    validation = [row("仅验证题", "A"), row("训练验证重叠题", "AE"), row("验证测试重叠题", "C")]
    test = [row("仅测试题", "D"), row("训练测试重叠题", "B"), row("验证测试重叠题", "C")]
    write_csv(raw_dir / "train.csv", train)
    write_csv(raw_dir / "val.csv", validation)
    write_csv(raw_dir / "test_with_annotations.csv", test, annotated=True)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_main(raw_dir: Path, output_dir: Path, *extra: str) -> int:
    return main(["--raw_dir", str(raw_dir), "--output_dir", str(output_dir), *extra])


def test_normalize_text() -> None:
    assert normalize_text(" \r\n第一行\x00  \r\n\r\n第二行 \r ") == "第一行\n\n第二行"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("A", "A"), ("AE", "AE"), ("A,E", "AE"), ("答案：A、E", "AE"), ("AAE", "AE")],
)
def test_normalize_single_and_multiple_answers(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize("raw", ["F", "", "答案：A；答案：B"])
def test_invalid_or_conflicting_answer_raises(raw: str) -> None:
    with pytest.raises(DataFormatError):
        normalize_answer(raw)


@pytest.mark.parametrize(
    "punctuation",
    [".", "．", "、", ":", "：", ")", " "],
)
def test_option_punctuation_formats(punctuation: str) -> None:
    separator = punctuation if punctuation == " " else punctuation + " "
    options = parse_options(f"A{separator}甲\nB{separator}乙")
    assert options == [{"label": "A", "text": "甲"}, {"label": "B", "text": "乙"}]


def test_multiline_options_are_preserved() -> None:
    options = parse_options("A 第一行\n续行内容\n\nB 第二项\n继续说明")
    assert options[0]["text"] == "第一行\n续行内容"
    assert options[1]["text"] == "第二项\n继续说明"


def test_duplicate_option_label_raises() -> None:
    with pytest.raises(DataFormatError, match="重复"):
        parse_options("A 甲\nA 乙\nB 丙")


def test_fewer_than_two_options_raises() -> None:
    with pytest.raises(DataFormatError, match="不足两个"):
        parse_options("A 唯一选项")


def test_empty_option_content_raises() -> None:
    with pytest.raises(DataFormatError, match="内容为空"):
        parse_options("A.\nB. 有效")


def test_official_and_decontaminated_split_semantics(tmp_path: Path) -> None:
    raw_dir, output_dir = tmp_path / "raw", tmp_path / "out"
    create_dataset(raw_dir)
    assert run_main(raw_dir, output_dir) == 0
    train = read_jsonl(output_dir / "official/train/train.jsonl")
    validation = read_jsonl(output_dir / "official/validation/validation.jsonl")
    test = read_jsonl(output_dir / "official/test/test.jsonl")
    decontaminated = read_jsonl(output_dir / "decontaminated/train/train.jsonl")

    # official 三个划分均保留内部重复与跨划分重叠。
    assert len(train) == 5
    assert len(validation) == 3
    assert len(test) == 3
    assert [item["stem"] for item in decontaminated] == ["仅训练题", "训练内重复题", "训练内重复题"]

    report = json.loads((output_dir / "processing_report.json").read_text(encoding="utf-8"))
    assert report["cross_split_overlaps"] == {
        "train_validation": 1,
        "train_test": 1,
        "validation_test": 1,
    }
    assert report["decontaminated_train_removed_count"] == 2
    assert report["official_splits"]["train"]["within_split_duplicates"] == 1


def test_answer_not_in_options_is_filtered(tmp_path: Path) -> None:
    raw_dir, output_dir = tmp_path / "raw", tmp_path / "out"
    write_csv(raw_dir / "train.csv", [row("坏题", "E", "A 甲\nB 乙")])
    write_csv(raw_dir / "val.csv", [row("验证题")])
    write_csv(raw_dir / "test_with_annotations.csv", [row("测试题")])
    assert run_main(raw_dir, output_dir) == 0
    report = json.loads((output_dir / "processing_report.json").read_text(encoding="utf-8"))
    assert report["official_splits"]["train"]["answer_not_in_options"] == 1
    assert report["official_splits"]["train"]["invalid_rows"] == 1
    assert report["invalid_record_examples"]["train"][0]["row_number"] == 2


def test_single_multiple_statistics_distribution_and_fields(tmp_path: Path) -> None:
    raw_dir, output_dir = tmp_path / "raw", tmp_path / "out"
    write_csv(raw_dir / "train.csv", [row("单选", "C"), row("双选", "AE"), row("三选", "BCD")])
    write_csv(raw_dir / "val.csv", [row("验证", "ABCD")])
    write_csv(raw_dir / "test_with_annotations.csv", [row("测试", "ABCDE")], annotated=True)
    run_main(raw_dir, output_dir)
    report = json.loads((output_dir / "processing_report.json").read_text(encoding="utf-8"))
    stats = report["official_splits"]["train"]
    assert (stats["single_answer_records"], stats["multiple_answer_records"]) == (1, 2)
    assert report["answer_label_count_distribution"] == {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1}
    record = read_jsonl(output_dir / "official/train/train.jsonl")[1]
    assert set(record) == {
        "id", "split", "question", "stem", "options", "answer", "answer_labels",
        "answer_text", "answer_texts", "is_multiple_choice", "explanation", "metadata",
    }
    assert record["answer"] == "AE"
    assert record["answer_labels"] == ["A", "E"]
    assert record["answer_texts"] == ["选项甲", "选项戊"]
    assert record["answer_text"] == "选项甲；选项戊"
    assert record["is_multiple_choice"] is True


def test_overwrite_policy(tmp_path: Path) -> None:
    raw_dir, output_dir = tmp_path / "raw", tmp_path / "out"
    create_dataset(raw_dir)
    assert run_main(raw_dir, output_dir) == 0
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_main(raw_dir, output_dir)
    assert run_main(raw_dir, output_dir, "--overwrite") == 0


def test_drop_within_split_duplicates_is_explicit(tmp_path: Path) -> None:
    raw_dir, output_dir = tmp_path / "raw", tmp_path / "out"
    create_dataset(raw_dir)
    run_main(raw_dir, output_dir, "--drop_within_split_duplicates")
    assert len(read_jsonl(output_dir / "official/train/train.jsonl")) == 4


def test_strict_fails_on_first_invalid_record(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    create_dataset(raw_dir, invalid_train=True)
    with pytest.raises(DataFormatError, match="答案为空"):
        run_main(raw_dir, tmp_path / "out", "--strict")


def test_non_strict_records_invalid_sample(tmp_path: Path) -> None:
    raw_dir, output_dir = tmp_path / "raw", tmp_path / "out"
    create_dataset(raw_dir, invalid_train=True)
    assert run_main(raw_dir, output_dir, "--max_error_examples", "1") == 0
    report = json.loads((output_dir / "processing_report.json").read_text(encoding="utf-8"))
    assert report["official_splits"]["train"]["invalid_answer"] == 1
    assert len(report["invalid_record_examples"]["train"]) == 1


def test_complete_cli_end_to_end_and_cli_error(tmp_path: Path) -> None:
    raw_dir, output_dir = tmp_path / "raw", tmp_path / "out"
    create_dataset(raw_dir)
    script = Path(__file__).parents[1] / "tools/prepare_cmexam.py"
    command = [sys.executable, str(script), "--raw_dir", str(raw_dir), "--output_dir", str(output_dir)]
    success = subprocess.run(command, text=True, capture_output=True, check=False)
    assert success.returncode == 0
    assert "official/validation：3" in success.stdout
    failure = subprocess.run(command, text=True, capture_output=True, check=False)
    assert failure.returncode == 1
    assert "错误：" in failure.stderr and "--overwrite" in failure.stderr

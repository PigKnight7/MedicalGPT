"""tools.prepare_medical_sft 的单元测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.prepare_medical_sft import (
    DataFormatError,
    SFTRecord,
    SourceStats,
    iter_clean_records,
    load_unique_records,
    main,
    normalize_text,
    record_digest,
    reservoir_sample_unique,
)


def write_lines(path: Path, records: list[object | str]) -> None:
    """写入测试用 JSONL；字符串按原样写入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            if isinstance(record, str):
                file.write(record)
            else:
                json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def raw(instruction: object, input_text: object, output: object) -> dict[str, object]:
    return {"instruction": instruction, "input": input_text, "output": output}


def clean(path: Path, *, user_min: int = 1, assistant_min: int = 1):
    stats = SourceStats(source=str(path))
    records = list(
        iter_clean_records(
            path,
            min_user_chars=user_min,
            min_assistant_chars=assistant_min,
            stats=stats,
        )
    )
    return records, stats


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def create_dataset(raw_dir: Path) -> None:
    finetune = raw_dir / "finetune"
    write_lines(
        finetune / "valid_zh_0.json",
        [raw("验证问题", "", "验证回答"), raw("验证问题2", "", "验证回答2")],
    )
    write_lines(
        finetune / "test_zh_0.json",
        [
            raw("验证问题", "", "验证回答"),
            raw("测试问题", "补充", "测试回答"),
            raw("测试问题2", "", "测试回答2"),
        ],
    )
    training = [raw(f"训练问题{i}", "", f"训练回答{i}") for i in range(10)]
    training.extend(
        [raw("测试问题", "补充", "测试回答"), raw("训练问题0", "", "训练回答0")]
    )
    write_lines(finetune / "train_zh_0.json", training)


def test_normalize_text() -> None:
    assert normalize_text("  第一行\r\n第二行\r第三行  ") == "第一行\n第二行\n第三行"


def test_instruction_and_input_are_joined(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    write_lines(path, [raw("  问题\r\n二  ", "  补充\r二  ", " 回答 ")])
    records, _ = clean(path)
    assert records == [SFTRecord("问题\n二\n补充\n二", "回答")]


def test_empty_input_uses_instruction_only(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    write_lines(path, [raw("问题", " \r\n ", "回答")])
    records, _ = clean(path)
    assert records[0].user_text == "问题"


def test_empty_instruction_keeps_required_separator(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    write_lines(path, [raw(" ", " 补充 ", "回答")])
    records, _ = clean(path)
    assert records[0].user_text == "\n补充"


def test_empty_user_and_answer_are_filtered(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    write_lines(path, [raw(" ", "\r\n", "回答"), raw("问题", "", "  "), raw("有效", "", "有效")])
    records, stats = clean(path)
    assert records == [SFTRecord("有效", "有效")]
    assert stats.skipped_empty_user == 1
    assert stats.skipped_empty_assistant == 1


def test_blank_and_short_records_are_counted(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    write_lines(path, ["", raw("短", "", "足够"), raw("足够", "", "短")])
    records, stats = clean(path, user_min=2, assistant_min=2)
    assert records == []
    assert stats.total_lines == 3
    assert stats.skipped_blank_lines == 1
    assert stats.skipped_short_record == 2


def test_invalid_json_mentions_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    write_lines(path, [raw("正常", "", "回答"), "{broken"])
    with pytest.raises(DataFormatError, match=r"broken\.json 第 2 行不是合法JSON"):
        clean(path)


@pytest.mark.parametrize("field", ["instruction", "input", "output"])
def test_missing_field_is_rejected(tmp_path: Path, field: str) -> None:
    path = tmp_path / "missing.json"
    record = raw("问题", "", "回答")
    del record[field]
    write_lines(path, [record])
    with pytest.raises(DataFormatError, match=f"缺少必需字段 '{field}'"):
        clean(path)


@pytest.mark.parametrize("field,value", [("instruction", 1), ("input", None), ("output", [])])
def test_wrong_field_type_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    path = tmp_path / "type.json"
    record = raw("问题", "", "回答")
    record[field] = value
    write_lines(path, [record])
    with pytest.raises(DataFormatError, match=f"'{field}' 必须是字符串"):
        clean(path)


def test_non_object_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    write_lines(path, [[1, 2]])
    with pytest.raises(DataFormatError, match="应为JSON对象"):
        clean(path)


def test_internal_deduplication(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    write_lines(path, [raw(" 问题 ", "", "回答"), raw("问题", " ", " 回答 ")])
    records, hashes, stats = load_unique_records(
        path, min_user_chars=1, min_assistant_chars=1
    )
    assert len(records) == len(hashes) == 1
    assert stats.skipped_duplicate == 1


def test_cross_split_overlap_filtering(tmp_path: Path) -> None:
    validation = tmp_path / "validation.json"
    test = tmp_path / "test.json"
    train = tmp_path / "train.json"
    shared = raw("相同问题", "", "相同回答")
    write_lines(validation, [shared])
    write_lines(test, [shared, raw("测试", "", "回答")])
    write_lines(train, [shared, raw("测试", "", "回答"), raw("训练", "", "回答")])
    _, validation_hashes, _ = load_unique_records(
        validation, min_user_chars=1, min_assistant_chars=1
    )
    test_records, test_hashes, test_stats = load_unique_records(
        test,
        min_user_chars=1,
        min_assistant_chars=1,
        blocked_hashes=validation_hashes,
    )
    train_records, train_stats, eligible = reservoir_sample_unique(
        train,
        sample_size=1,
        seed=42,
        min_user_chars=1,
        min_assistant_chars=1,
        blocked_hashes=validation_hashes | test_hashes,
    )
    assert [item.user_text for item in test_records] == ["测试"]
    assert [item.user_text for item in train_records] == ["训练"]
    assert test_stats.skipped_overlap == 1
    assert train_stats.skipped_overlap == 2
    assert eligible == 1


def test_reservoir_size_seed_and_blocked_hashes(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    write_lines(path, [raw(f"问题{i}", "", f"回答{i}") for i in range(100)])
    blocked = {record_digest("问题7", "回答7")}
    kwargs = dict(
        sample_size=12,
        seed=123,
        min_user_chars=1,
        min_assistant_chars=1,
        blocked_hashes=blocked,
    )
    first, stats, eligible = reservoir_sample_unique(path, **kwargs)
    second, _, _ = reservoir_sample_unique(path, **kwargs)
    assert len(first) == 12
    assert first == second
    assert eligible == 99
    assert stats.skipped_overlap == 1
    assert all(item.user_text != "问题7" for item in first)


def test_too_few_training_records_raises(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    write_lines(path, [raw("问题", "", "回答")])
    with pytest.raises(ValueError, match="只有 1 条可用训练样本.*抽取 2 条"):
        reservoir_sample_unique(
            path,
            sample_size=2,
            seed=42,
            min_user_chars=1,
            min_assistant_chars=1,
            blocked_hashes=set(),
        )


def run_main(monkeypatch: pytest.MonkeyPatch, raw_dir: Path, output_dir: Path, *extra: str) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_medical_sft.py",
            "--raw_dir",
            str(raw_dir),
            "--output_dir",
            str(output_dir),
            "--train_samples",
            "4",
            "--min_user_chars",
            "1",
            "--min_assistant_chars",
            "1",
            *extra,
        ],
    )
    return main()


def test_end_to_end_and_exact_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"
    create_dataset(raw_dir)
    assert run_main(monkeypatch, raw_dir, output_dir) == 0
    train = read_jsonl(output_dir / "train" / "train.jsonl")
    validation = read_jsonl(output_dir / "validation" / "validation.jsonl")
    test = read_jsonl(output_dir / "test" / "test.jsonl")
    assert len(train) == 4
    assert len(validation) == 2
    assert len(test) == 2
    for item in train + validation + test:
        assert list(item) == ["conversations"]
        assert item["conversations"][0].keys() == {"from", "value"}
        assert item["conversations"][0]["from"] == "human"
        assert item["conversations"][1]["from"] == "gpt"
        assert len(item["conversations"]) == 2
    assert test[0] == {
        "conversations": [
            {"from": "human", "value": "测试问题\n补充"},
            {"from": "gpt", "value": "测试回答"},
        ]
    }
    report = json.loads((output_dir / "processing_report.json").read_text(encoding="utf-8"))
    assert report["sampling"]["train_eligible_unique_records"] == 10
    assert report["source_statistics"]["train"]["skipped_duplicate"] == 1
    assert report["source_statistics"]["train"]["skipped_overlap"] == 1
    assert report["outputs"]["train"]["records"] == 4


def test_overwrite_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"
    create_dataset(raw_dir)
    run_main(monkeypatch, raw_dir, output_dir)
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_main(monkeypatch, raw_dir, output_dir)
    assert run_main(monkeypatch, raw_dir, output_dir, "--overwrite") == 0
